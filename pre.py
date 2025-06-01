import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
from CNN import CNN
from lstm_utils import TrainDataset

# ----- Step 2: CNN-LSTM with CTC for line recognition -----
class CRNN(nn.Module):
    def __init__(self, cnn_backbone, lstm_hidden=256, num_classes=27):  # 26 + CTC blank
        super(CRNN, self).__init__()
        self.cnn = cnn_backbone

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x):
        cnn_features = self.cnn(x)  # (B, 64, H', W')
        b, c, h, w = cnn_features.size()
        assert h == 1, f"Expected height 1 after CNN, got {h}"
        features = cnn_features.squeeze(2).permute(0, 2, 1)  # (B, W, C)
        cnn_features = self.cnn(x)
        lstm_out, _ = self.lstm(features)
        logits = self.classifier(lstm_out)  # (B, W, num_classes)
        return logits.permute(1, 0, 2)  # (W, B, num_classes) for CTC


def fine_tune_crnn():
    base_cnn = CNN()
    base_cnn.load_state_dict(torch.load("best_cnn_model.pth"))
    cnn_backbone = base_cnn.extract_features

    model = CRNN(cnn_backbone)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CTCLoss(blank=0)
    loader = DataLoader(TrainDataset(), batch_size=4, shuffle=True)

    model.train()
    for epoch in range(3):
        for images, targets, target_lengths in loader:
            optimizer.zero_grad()
            logits = model(images)
            input_lengths = torch.full((logits.size(1),), logits.size(0), dtype=torch.long)
            targets = torch.cat([t for t in targets])
            loss = criterion(logits, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")

# ----- Main pipeline -----
if __name__ == "__main__":
    print("Fine-tuning CRNN on line data...")
    fine_tune_crnn()
