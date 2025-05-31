import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
from CNN_BiLSTM_CTC import CNN_BiLSTM_CTC
from CNN import CNN
from lstm_utils import decode_label
import json
from tqdm import tqdm
import logging

# 自訂 Dataset 包裝 HuggingFace IAM-line
class IAMLineDataset(Dataset):
    def __init__(self, hf_dataset, char2idx):
        self.dataset = hf_dataset
        self.char2idx = char2idx
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        text = item["text"]

        image = self.transform(image)
        label_seq = [self.char2idx[c] for c in text if c in self.char2idx]

        return image, label_seq

# 打包 batch（CTC 格式需求）
def collate_fn(batch):
    images, labels = zip(*batch)
    image_tensors = torch.stack(images)
    targets = torch.cat([torch.tensor(label, dtype=torch.long) for label in labels])
    target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    return image_tensors, targets, target_lengths

# CTC 的 greedy decode
def greedy_decode(output, blank=0):
    output = output.argmax(2)
    results = []
    for seq in output:
        pred = []
        prev = blank
        for c in seq:
            c = c.item()
            if c != blank and c != prev:
                pred.append(c)
            prev = c
        results.append(pred)
    return results

# 評估模型準確率與損失
def evaluate(model, dataloader, criterion, device, idx2char):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, targets, target_lengths in tqdm(dataloader, desc="Evaluating"):
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = model(imgs)
            output_lengths = torch.full(size=(outputs.size(0),), fill_value=outputs.size(1), dtype=torch.long).to(device)

            loss = criterion(outputs.permute(1, 0, 2), targets, output_lengths, target_lengths)
            total_loss += loss.item()

            preds = greedy_decode(outputs)
            label_ptr = 0
            for i, pred in enumerate(preds):
                gt = targets[label_ptr:label_ptr + target_lengths[i].item()].tolist()
                pred_str = decode_label(pred, idx2char)
                gt_str = decode_label(gt, idx2char)

                if pred == gt:
                    correct += 1
                total += 1
                label_ptr += target_lengths[i].item()

                if total <= 5:
                    logging.info({
                        f"[GT  ] {gt_str}\n[Pred] {pred_str}\n"
                    })
                    print(f"[GT  ] {gt_str}")
                    print(f"[Pred] {pred_str}\n")

    acc = correct / total * 100 if total > 0 else 0.0
    avg_loss = total_loss / len(dataloader)
    print("Raw prediction indices:", preds[0])
    return avg_loss, acc

def main():
    logging.basicConfig(level=logging.INFO, filename="logger.log", filemode="w", 
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')
    
    num_classes = 63  # match char2idx.json
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Loading IAM-line dataset from HuggingFace...")
    print("Loading IAM-line dataset from HuggingFace...")
    hf_dataset = load_dataset("Teklia/IAM-line")

    with open("char2idx.json") as f:
        char2idx = json.load(f)
    idx2char = {v: k for k, v in char2idx.items()}

    train_dataset = IAMLineDataset(hf_dataset["train"], char2idx)
    test_dataset = IAMLineDataset(hf_dataset["test"], char2idx)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    cnn = CNN()
    model = CNN_BiLSTM_CTC(cnn_backbone=cnn, num_classes=num_classes).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for imgs, targets, target_lengths in progress_bar:
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = model(imgs)
            output_lengths = torch.full(size=(outputs.size(0),), fill_value=outputs.size(1), dtype=torch.long).to(device)

            loss = criterion(outputs.permute(1, 0, 2), targets, output_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}")
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}")

        test_loss, test_acc = evaluate(model, test_loader, criterion, device, idx2char)
        logging.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

if __name__ == '__main__':
    main()
