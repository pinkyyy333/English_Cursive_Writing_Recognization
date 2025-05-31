import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from typing import List, Tuple

"""
Notice:
    1) You can't add any additional package
    2) You can add or remove any function "except" fit, _build_tree, predict
    3) You can ignore the suggested data type if you want
"""

class ConvNet(nn.Module): # Don't change this part!
    def __init__(self):
        super(ConvNet, self).__init__()
        self.model = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=300)

    def forward(self, x):
        x = self.model(x)
        return x
    
class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.data_size = X.shape[0]
        total_steps = 2 ** self.max_depth
        self.progress = tqdm(total = total_steps, desc = "Growing tree", position = 0, leave = True)
        self.tree = self._build_tree(X, y, 0) # i added initial depth = 0
        self.progress.close()

    def _build_tree(self, X: pd.DataFrame, y: np.ndarray, depth: int):
        # (TODO) Grow the decision tree and return it
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            # return majority class
            values, counts = np.unique(y, return_counts = True)
            return {'value': values[np.argmax(counts)]}
        
        feature, threshold = self._best_split(X, y)
        if feature is None:
            values, counts = np.unique(y, return_counts = True)
            return {'value': values[np.argmax(counts)]}
        
        self.progress.update(1)
        
        X_left, y_left, X_right, y_right = self._split_data(X, y, feature, threshold)
        
        return {
            'feature': feature,
            'threshold': threshold,
            'left': self._build_tree(X_left, y_left, depth + 1),
            'right': self._build_tree(X_right, y_right, depth + 1)
        }

    def predict(self, X: pd.DataFrame)->np.ndarray:
        # (TODO) Call _predict_tree to traverse the decision tree to return the classes of the testing dataset
        predictions = np.array([self._predict_tree(x, self.tree) for x in X.values])
        
        return torch.tensor(predictions)

    def _predict_tree(self, x, tree_node):
        # (TODO) Recursive function to traverse the decision tree
        # return value of leaf nodes
        if "value" in tree_node:
            return tree_node["value"]
        
        feature_index = tree_node["feature"]
        threshold = tree_node["threshold"]

        if x[feature_index] <= threshold:
            return self._predict_tree(x, tree_node["left"])
        else:
            return self._predict_tree(x, tree_node["right"])

    def _split_data(self, X: pd.DataFrame, y: np.ndarray, feature_index: int, threshold: float):
        # (TODO) split one node into left and right node 
        feature_column = X.iloc[:, feature_index]
    
        left_mask = feature_column <= threshold
        right_mask = feature_column > threshold

        left_dataset_X = X[left_mask]
        left_dataset_y = y[left_mask]

        right_dataset_X = X[right_mask]
        right_dataset_y = y[right_mask]
        
        return left_dataset_X, left_dataset_y, right_dataset_X, right_dataset_y

    def _best_split(self, X: pd.DataFrame, y: np.ndarray):
        # (TODO) Use Information Gain to find the best split for a dataset
        best_feature_index = None
        best_threshold = None
        best_gain = -1  # initialize gain

        parent_entropy = self._entropy(y)

        features = X.shape[1]

        for feature_index in range(features):
            feature_values = X.iloc[:, feature_index].values
            thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = feature_values > threshold

                y_left = y[left_mask]
                y_right = y[right_mask]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue  # invalid split

                left_entropy = self._entropy(y_left)
                right_entropy = self._entropy(y_right)
                weighted_entropy = (len(y_left) / len(y)) * left_entropy + (len(y_right) / len(y)) * right_entropy

                info_gain = parent_entropy - weighted_entropy

                if info_gain > best_gain:
                    best_gain = info_gain
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def _entropy(self, y: np.ndarray)->float:
        # (TODO) Return the entropy
        values, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return -np.sum(probabilities * np.log2(probabilities))

def get_features_and_labels(model: ConvNet, dataloader: DataLoader, device)->Tuple[List, List]:
    # (TODO) Use the model to extract features from the dataloader, return the features and labels
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            outputs = model(imgs)  # forward pass
            features.append(outputs.cpu())
            labels.extend(lbls)

    features = torch.cat(features, dim = 0).numpy()
    features = pd.DataFrame(features)
    labels = np.array(labels)
    
    return features, labels # X is the features here

def get_features_and_paths(model: ConvNet, dataloader: DataLoader, device)->Tuple[List, List]:
    # (TODO) Use the model to extract features from the dataloader, return the features and path of the images
    model.eval()
    features = []
    paths = []

    with torch.no_grad():
        for imgs, img_paths in dataloader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            features.append(outputs.cpu())
            paths.extend(img_paths)

    features = torch.cat(features, dim = 0).numpy()
    features = pd.DataFrame(features)
    paths = np.array(paths)
    
    return features, paths