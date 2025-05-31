from torchvision import transforms
from torch.utils.data import Dataset
import os
import PIL
import json
from typing import Optional, Set

class TrainDataset(Dataset):
    def __init__(self, images, labels, char2idx_path="char2idx.json"):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((128, 512)),
            transforms.ToTensor()
        ])
        self.images = images
        self.labels = labels

        with open(char2idx_path, 'r') as f:
            self.char2idx = json.load(f)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = PIL.Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        label_str = self.labels[idx]
        label_seq = [self.char2idx[c] for c in label_str if c in self.char2idx]

        return image, label_seq

def load_iam_dataset(label_file: str, image_root: str, subset_ids: Optional[Set[str]] = None):
    images = []
    labels = []
    missing = []
    print("Debugging file path matching...")
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split(' ', 9)
            if len(parts) < 10:
                continue
            file_id = parts[0]
            ok_flag = parts[2]
            text = parts[-1].replace('|', ' ')
            if ok_flag != 'ok':
                continue
            if subset_ids and not any(file_id.startswith(sub_id) for sub_id in subset_ids):
                continue
            file_path = os.path.join(image_root, file_id + ".png")
            if os.path.exists(file_path):
                images.append(file_path)
                labels.append(text)
            else:
                missing.append((file_id, file_path))

    if not images:
        print("No matching images found.")
        print("Sample file_id + path attempts:", [m for m in missing[:5]])
    elif missing:
        print(f"Warning: {len(missing)} images listed not found. Sample:")
        print([m for m in missing[:5]])

    return images, labels

def load_subset_ids(filepath: str) -> Set[str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

def decode_label(label_seq, idx2char):
    return ''.join([idx2char[idx] for idx in label_seq if idx in idx2char])
