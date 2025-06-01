import torch
import torch.nn.functional as F
import kenlm
from collections import defaultdict
from math import log
import numpy as np
import json

class BeamEntry:
    def __init__(self):
        self.pr_blank = 0.0
        self.pr_non_blank = 0.0
        self.lm_score = 0.0
        self.total_score = float('-inf')
        self.sequence = []

    def score(self, alpha=0.5, beta=1.0):
        return log(self.pr_blank + self.pr_non_blank + 1e-10) + alpha * self.lm_score + beta * len(self.sequence)

def ctc_beam_search_with_lm(log_probs, lm, beam_width=10, alpha=0.5, beta=1.0, blank=0, idx2char=None):
    T, C = log_probs.shape
    beams = defaultdict(BeamEntry)
    beams[()].pr_blank = 1.0

    for t in range(T):
        next_beams = defaultdict(BeamEntry)
        for prefix, entry in beams.items():
            for c in range(C):
                p = torch.exp(log_probs[t, c]).item()
                new_prefix = prefix

                if c == blank:
                    be = next_beams[prefix]
                    be.sequence = prefix
                    be.pr_blank += entry.pr_blank * p
                    be.pr_blank += entry.pr_non_blank * p
                    be.pr_non_blank = be.pr_non_blank
                    be.lm_score = entry.lm_score
                else:
                    new_prefix = prefix + (c,)
                    last_char = prefix[-1] if prefix else None

                    be = next_beams[new_prefix]
                    be.sequence = new_prefix

                    if c != last_char:
                        be.pr_non_blank += entry.pr_blank * p
                    be.pr_non_blank += entry.pr_non_blank * p if c != last_char else 0.0

                    # LM 得分
                    text = ''.join([idx2char[i] for i in new_prefix])
                    be.lm_score = lm.score(text, bos=False, eos=False)

        # Beam pruning
        beam_list = sorted(next_beams.items(), key=lambda x: x[1].score(alpha, beta), reverse=True)
        beams = defaultdict(BeamEntry)
        for k, v in beam_list[:beam_width]:
            beams[k] = v

    best = max(beams.items(), key=lambda x: x[1].score(alpha, beta))
    final_seq = ''.join([idx2char[i] for i in best[0]])
    return final_seq

# === 測試 ===
if __name__ == "__main__":
    torch.manual_seed(0)
    T, C = 10, 6  # 10 time steps, 6 classes (0 is blank)
    logits = torch.randn(T, C)
    log_probs = F.log_softmax(logits, dim=-1)

    with open("char2idx.json", "r") as f:
        char2idx = json.load(f)

    idx2char = [''] * (max(char2idx.values()) + 1)
    for char, idx in char2idx.items():
        idx2char[idx] = char

    # 載入語言模型
    lm = kenlm.Model("corpus.arpa")

    result = ctc_beam_search_with_lm(log_probs, lm, beam_width=5, alpha=1.0, beta=0.5, blank=0, idx2char=idx2char)
    print("Decoded with LM:", result)