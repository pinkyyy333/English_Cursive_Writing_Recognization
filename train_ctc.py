import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
from CNN_BiLSTM_CTC import CNN_BiLSTM_CTC
from CNN import CNN  # 確保這是有 extract_features() 的版本
from lstm_utils import decode_label
from decode import ctc_beam_search_with_lm
import kenlm
import json
from tqdm import tqdm
from PIL import ExifTags
import logging

# Dataset 定義
class IAMLineDataset(Dataset):
    def __init__(self, hf_dataset, char2idx):
        self.dataset = hf_dataset
        self.char2idx = char2idx
        self.transform = transforms.Compose([
            # Change into grayscale
            transforms.Grayscale(),
            transforms.Resize((128, 512)),
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

def collate_fn(batch):
    images, labels = zip(*batch)
    image_tensors = torch.stack(images)
    targets = torch.cat([torch.tensor(label, dtype=torch.long) for label in labels])
    target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    return image_tensors, targets, target_lengths

# 評估（僅使用 Beam Search + LM）
def evaluate(model, dataloader, criterion, device, idx2char, lm=None, use_beam=True):
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

            label_ptr = 0
            for i in range(outputs.size(0)):
                log_probs = outputs[i].cpu()
                probs = log_probs.exp()
                topv, topi = probs.max(dim=-1)  # [T]
                print(f"[DEBUG] Top prob avg: {topv.mean().item():.4f}, Top idx sample: {topi[:20].tolist()}")
                if use_beam and lm is not None:
                    pred_str = ctc_beam_search_with_lm(
                        log_probs,
                        lm,
                        beam_width=10,
                        alpha=0.5,
                        beta=0.1,
                        blank=0,
                        idx2char=idx2char
                    )
                else:
                    pred_indices = torch.argmax(log_probs, dim=-1).numpy().tolist()
                    pred_str = decode_label(pred_indices, idx2char)

                gt = targets[label_ptr:label_ptr + target_lengths[i].item()].tolist()
                gt_str = decode_label(gt, idx2char)
                
                print(f"[DEBUG] GT index list: {gt}")
                print(f"[DEBUG] GT str       : '{gt_str}'")
                print(f"[DEBUG] PRED str     : '{pred_str}'")
                
                if pred_str == gt_str:
                    correct += 1
                total += 1
                label_ptr += target_lengths[i].item()

                if total <= 5:
                    logging.info(f"[GT     ] {gt_str}\n[Beam+LM] {pred_str}\n")
                    print(f"[GT     ] {gt_str}")
                    print(f"[Beam+LM] {pred_str}\n")

    acc = correct / total * 100 if total > 0 else 0.0
    avg_loss = total_loss / len(dataloader)
    return avg_loss, acc

# 主函式
def main():
    logging.basicConfig(level=logging.INFO, filename="lstm_logger.log", filemode="w", 
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')

    # print("Current Working Directory:", os.getcwd())
    # print("Files in this directory:", os.listdir("."))
    num_classes = 63  # char2idx 含空格共 62 + blank
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Loading IAM-line dataset from HuggingFace...")
    print("Loading IAM-line dataset from HuggingFace...")
    hf_dataset = load_dataset("Teklia/IAM-line")

    with open("char2idx.json") as f:
        char2idx = json.load(f)
    idx2char = [''] * (max(char2idx.values()) + 1)
    for char, idx in char2idx.items():
        idx2char[idx] = char

    logging.info("Loading KenLM language model...")
    print("Loading KenLM language model...")
    lm = kenlm.Model("corpus.arpa")

    train_dataset = IAMLineDataset(hf_dataset["train"], char2idx)
    test_dataset = IAMLineDataset(hf_dataset["test"], char2idx)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)


    cnn = CNN()
    pretrained_dict = torch.load("best_cnn_model.pth")
    cnn_dict = cnn.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in cnn_dict and 'fc' not in k and 'logSoftmax' not in k}
    cnn_dict.update(pretrained_dict)
    cnn.load_state_dict(cnn_dict)
    model = CNN_BiLSTM_CTC(cnn_backbone=cnn, num_classes=num_classes).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
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

        test_loss, test_acc = evaluate(model, test_loader, criterion, device, idx2char, lm=None, use_beam=False)
        logging.info(f"Test Loss: {test_loss:.4f}, Test Accuracy (Exact Match): {test_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy (Exact Match): {test_acc:.2f}%")

        # 儲存最佳模型
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            torch.save(model.state_dict(), "best_model_ctc.pth")
            logging.info(f"Saved best model at epoch {epoch+1} with loss {best_val_loss:.4f}")
            print(f"Saved best model at epoch {epoch+1} with loss {best_val_loss:.4f}")

if __name__ == '__main__':
    main()
