import torch
import torch.nn as nn

class CNN_BiLSTM_CTC(nn.Module):
    def __init__(self, cnn_backbone: nn.Module, num_classes: int, lstm_hidden=256, lstm_layers=2):
        super(CNN_BiLSTM_CTC, self).__init__()

        self.cnn = cnn_backbone
        self.lstm = nn.LSTM(
            input_size=1792,
            hidden_size=256,
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
        out = self.classifier(lstm_out)  # [B, W, num_classes]
        return out.log_softmax(2)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x).argmax(2)