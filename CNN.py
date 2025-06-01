import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
import pandas as pd

class CNN(nn.Module):
    def __init__(self, num_classes = 62):
        # (TODO) Design your CNN
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            # Change: 3 -> 1
            nn.Conv2d(1, 32, kernel_size=5, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  
        )
        
        # self.layer5 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3)
        # self.batchnorm5 = nn.BatchNorm2d(32)
        # self.relu5 = nn.ReLU()
        # self.max_pool5 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        # self.layer6 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)
        # self.batchnorm6 = nn.BatchNorm2d(32)
        # self.relu6 = nn.ReLU()
        # self.max_pool6 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        # compute the flattened size dynamically
        dummy_input = torch.zeros(1, 1, 224, 224)  # adjust to input image size
        x = self.layer1(dummy_input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        flattened_size = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flattened_size, 128)
        self.relu_last = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        #self.fc3 = nn.Linear(64, num_classes)
        self.logSoftmax = nn.LogSoftmax(dim = 1)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        # (TODO) Forward the model
        # pass through first set
        x = self.layer1(x)  # 包含 conv + batchnorm + relu + maxpool
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # pass through fifth set
        # x = self.layer5(x)
        # x = self.batchnorm5(x)
        # x = self.relu5(x)
        # x = self.max_pool5(x)
        
        # pass through sixth set
        # x = self.layer6(x)
        # x = self.batchnorm6(x)
        # x = self.relu6(x)
        # x = self.max_pool6(x)
        
        # flatten output and pass through fc -> relu layers
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        # x = self.relu_last(x)
        x = self.dropout(x)
        # pass to softmax classfier
        x = self.fc2(x)
        x = self.logSoftmax(x)
        
        return x
    
    def extract_features(self, x):
        x = self.layer1(x)  # [B, 32, H/2, W/2]
        x = self.layer2(x)  # [B, 64, H/4, W/4]
        x = self.layer3(x)  # [B, 128, H/8, W/8]
        x = self.layer4(x)  # [B, 128, H/16, W/16]
        return x
    

def train(model: CNN, train_loader: DataLoader, criterion, optimizer, device)->float:
    # (TODO) Train the model and return the average loss of the data, we suggest use tqdm to know the progress
    model.train()
    total_loss = 0.0
    steps = len(train_loader) # batch size
    
    model.to(device)
    
    # tqdm progress bar
    loop = tqdm(train_loader, desc = "Training", leave = False)
    
    for inputs, labels in loop:
        # move data to correct device
        inputs, labels = inputs.to(device), labels.to(device)
        
        # forward pass
        criterion = nn.CrossEntropyLoss()
        outputs = model(inputs)
        # print("CNN output shape: ", outputs.size())
        loss = criterion(outputs, labels)
        
        # zero the gradients, backward and optimize
        optimizer.zero_grad()
        loss.backward()
        # update model parameters
        optimizer.step()
        
        # update running loss
        total_loss += loss.item()
        
        # update tqdm description with current loss
        loop.set_postfix(loss = total_loss)
        
    avg_loss = total_loss / steps    
    # accuracy = correct / steps
    
    return avg_loss


def validate(model: CNN, val_loader: DataLoader, criterion, device)->Tuple[float, float]:
    # (TODO) Validate the model and return the average loss and accuracy of the data, we suggest use tqdm to know the progress
    total_loss = 0
    correct = 0
    total = 0
    
    # set model to evaluation mode
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        loop = tqdm(val_loader, desc = "Validating", leave = False)
        
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # get output and calculate loss
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss = total_loss)
    
    # this feels wrong
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total * 100
    
    return avg_loss, accuracy

def test(model: CNN, test_loader: DataLoader, criterion, device):
    # (TODO) Test the model on testing dataset and write the result to 'CNN.csv'
    # set model to evaluation mode
    model.eval()
    model.to(device)
    predictions = []
    name = []
    
    with torch.no_grad():
        loop = tqdm(test_loader, desc = "Validating", leave = False)
        
        for inputs, nm in loop:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            img, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            name.extend(nm)
            
    # Write predictions to CSV
    pred = []
    for idx, lb in enumerate(predictions):
        # id.append(lbl)
        pred.append(lb)
        
    data = {
        "id" : name,
        "prediction" : pred
    }
    
    df = pd.DataFrame(data)
    df.to_csv("CNN.csv", index = False)
            
    print(f"Predictions saved to 'CNN.csv'")
    return

