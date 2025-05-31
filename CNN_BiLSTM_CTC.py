import torch
import torch.nn as nn

class CNN_BiLSTM_CTC(nn.Module):
    def __init__(self, cnn_backbone: nn.Module, num_classes: int, lstm_hidden=256, lstm_layers=2):
        super(CNN_BiLSTM_CTC, self).__init__()

        self.cnn = cnn_backbone
        self._feature_size = None  # 等待 forward 初始化

        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.num_classes = num_classes

    def forward(self, x):
        features = self.cnn.extract_features(x)  # [B, C, H, W]
        b, c, h, w = features.size()
        features = features.permute(0, 3, 1, 2).contiguous()  # [B, W, C, H]
        features = features.view(b, w, -1)  # [B, W, C*H]

        if self._feature_size is None:
            self._feature_size = features.shape[2]
            self.lstm = nn.LSTM(
                input_size=self._feature_size,
                hidden_size=self.lstm_hidden,
                num_layers=self.lstm_layers,
                batch_first=True,
                bidirectional=True
            )
            self.classifier = nn.Linear(self.lstm_hidden * 2, self.num_classes)

        lstm_out, _ = self.lstm(features)
        out = self.classifier(lstm_out)
        return out.log_softmax(2)
