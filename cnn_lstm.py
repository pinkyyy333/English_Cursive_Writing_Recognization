import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import json
from tqdm import tqdm
from CNN import CNN

# ----- CRNN model (CNN + BiLSTM + CTC) -----
class CRNN(nn.Module):
    def __init__(self, cnn_backbone, lstm_hidden=256, num_classes=27):
        super(CRNN, self).__init__()
        self.cnn = cnn_backbone
        self.lstm = nn.LSTM(
            input_size=128,  # Assume final CNN output has 32 channels
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x):
        features = self.cnn.extract_features(x)  # [B, C, H, W]
        b, c, h, w = features.size()
        features = features.permute(0, 3, 1, 2).contiguous()  # [B, W, C, H]
        features = features.view(b, w, -1)  # [B, W, C*H]
        lstm_out, _ = self.lstm(features)
        logits = self.classifier(lstm_out)
        return logits.permute(1, 0, 2)  # [W, B, num_classes]

# ----- HuggingFace IAM-line Dataset Wrapper -----
class IAMLineDataset(Dataset):
    def __init__(self, hf_dataset, char2idx_path="char2idx.json"):
        self.dataset = hf_dataset
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((128, 512)),
            transforms.ToTensor()
        ])
        with open(char2idx_path, "r") as f:
            self.char2idx = json.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = self.transform(item["image"])
        text = item["text"].lower()
        label = [self.char2idx[c] for c in text if c in self.char2idx]
        return image, label

# ----- Collate Function -----
def collate_fn(batch):
    images, labels = zip(*batch)
    image_tensors = torch.stack(images)
    targets = torch.cat([torch.tensor(label, dtype=torch.long) for label in labels])
    target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    return image_tensors, targets, target_lengths

# ----- Main Training Function -----
def fine_tune_crnn():
    print("Loading IAM-line dataset...")
    ds = load_dataset("Teklia/IAM-line")
    train_dataset = IAMLineDataset(ds["train"])
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_cnn = CNN()
    base_cnn.load_state_dict(torch.load("best_cnn_model.pth"))
    model = CRNN(cnn_backbone=base_cnn).to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("Start fine-tuning CRNN with CTC...")
    model.train()
    for epoch in range(5):
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for images, targets, target_lengths in loop:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            logits = model(images)
            input_lengths = torch.full(size=(logits.size(1),), fill_value=logits.size(0), dtype=torch.long).to(device)

            loss = criterion(logits, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1}: Avg Loss = {total_loss / len(train_loader):.4f}")

if __name__ == "__main__":
    fine_tune_crnn()
