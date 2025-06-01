from torchvision import transforms
from torch.utils.data import Dataset
import os
import PIL
from typing import List, Tuple
import matplotlib.pyplot as plt
import csv
import json
from typing import Optional, Set

class TrainDataset(Dataset):
    def __init__(self, images, labels):
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = PIL.Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

class TestDataset(Dataset):
    def __init__(self, image):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.image = image

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image_path = self.image[idx]
        image = PIL.Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return image, base_name
    
def load_train_dataset(path: str='train_data/')->Tuple[List, List]:
    images = []
    labels = []
    
    folders = os.listdir(path)
    for f in folders:
        label = int(f)
        image_path = path + f + '/'
        
        pics = os.listdir(image_path)
        for pic in pics:
            images.append(image_path + "/" + pic)
            labels.append(label)
    
    return images, labels

def load_test_dataset(path: str='test_data/')->List:
    images = []
    pics = os.listdir(path)
    for pic in pics:
        images.append(path + pic)
    return images

def name_file(base_name, ext=".png"):
    i = 1.0
    filename = f"{base_name}{i}{ext}"
    while os.path.exists(filename):
        i += 0.01
        i = round(i, 2)
        filename = f"{base_name}{i}{ext}"
    return filename

def plot(train_losses: List, val_losses: List):
    # (TODO) Plot the training loss and validation loss of CNN, and save the plot to 'loss.png'
    #        xlabel: 'Epoch', ylabel: 'Loss'
    
    epoch = range(len(train_losses))
    plt.plot(epoch, train_losses, 's-c', label = 'Training Loss')
    plt.plot(epoch, val_losses, 'o-g', label='Val Losses')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss and Validation Loss of CNN')
    plt.grid(True)
    
    plt.legend()
    filename = name_file("plot_images/loss", ".png")
    plt.savefig(filename)
    print(f"Saved the plot to '{filename}'")
    return

def check_ans(answer: str, prediction: str):
    name = []
    ans = []
    check = {}

    with open(answer) as file:
        csvFile = csv.reader(file)
        next(csvFile)
        
        for lines in csvFile:
            check.update({lines[0]: lines[1]})

    total = 0
    correct = 0
    with open(prediction) as res:
        resFile = csv.reader(res)
        next(resFile)
        
        for lines in resFile:
            total += 1
            
            ind = lines[0]
            pred = lines[1]
            
            if(pred == check[ind]):
                correct += 1

    return (correct / total) * 100