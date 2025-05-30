import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Parameter
batch_size = 1
timesteps = 3
input_dim = 10
hidden_size = 50
vocab_size = 26  # a-z
beam_width = 3

# Simulate the input data
input_data = torch.randn(batch_size, timesteps, input_dim)

# Model definition
class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, vocab_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)  # Bidirectional

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        probs = F.softmax(out, dim=2)
        return probs

model = SimpleLSTM(input_dim, hidden_size, vocab_size)
output_probs = model(input_data)[0].detach().numpy()  # shape: (timesteps, vocab_size)

# Decode function
def beam_search_decoder(probs, beam_width=3):
    # prob.shape = torch.tensor([timestamp, vocabulary_size])
    sequences = [([], 0.0)]  # (sequence, score)
    for t in range(probs.shape[0]):
        all_candidates = []
        for seq, score in sequences:
            for c in range(probs.shape[1]):
                log_probs = np.log(probs[t, c] + 1e-9)
                candidate = (seq + [c], score + log_probs)
                all_candidates.append(candidate)
        # Sort top beam_width
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        # Get the width we need
        sequences = ordered[:beam_width]
    return sequences

# Decode
results = beam_search_decoder(output_probs, beam_width=beam_width)

# Trabsform index into character
idx2char = [chr(i + ord('a')) for i in range(vocab_size)]

print("Beam Search Top 3:")
for seq, score in results:
    print("Sequence:", ''.join([idx2char[i] for i in seq]), "Score:", score)
