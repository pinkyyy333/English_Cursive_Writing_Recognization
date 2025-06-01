import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from loguru import logger
from sklearn.metrics import accuracy_score
import logging
#from CNN_transform import CNN_Transformer

from CNN import CNN, train, validate, test
from cnn_utils import TrainDataset, TestDataset, load_train_dataset, load_test_dataset, plot

"""
Notice:
    1) You can't add any additional package
    2) You can ignore the suggested data type if you want
"""

def main():
    logging.basicConfig(level=logging.INFO, filename="logger.log", filemode="w", 
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')
    
    """
    load data
    """
    logger.info("Start loading data")
    logging.info("Start loading data")
    images, labels = load_train_dataset()
    images, labels = shuffle(images, labels, random_state=777)
    train_len = int(0.8 * len(images))

    train_images, val_images = images[:train_len], images[train_len:]
    train_labels, val_labels = labels[:train_len], labels[train_len:]
    test_images = load_test_dataset()

    train_dataset = TrainDataset(train_images, train_labels)
    val_dataset = TrainDataset(val_images, val_labels)
    test_dataset = TestDataset(test_images)
    
    """
    CNN - train and validate
    """
    logger.info("Start training CNN")
    logging.info("Start training CNN")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # I added this line of device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()

    # Optimizer configuration
    base_params = [param for name, param in model.named_parameters() if param.requires_grad]
    optimizer = optim.Adam(base_params, lr=1e-4)

    train_losses = []
    val_losses = []
    max_acc = 0

    EPOCHS = 20
    for epoch in range(EPOCHS): #epoch
        train_loss = train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}")
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # (TODO) Print the training log to help you monitor the training process
        #        You can save the model for future usage
        log_msg = (
            f"Epoch [{epoch + 1}/{EPOCHS}] - "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_acc:.2f}%"
        )
        print(log_msg)
        logger.info(log_msg)
        logging.info(log_msg)

        # Save the best mode
        if val_acc > max_acc:
            max_acc = val_acc
            torch.save(model.state_dict(), "best_cnn_model.pth")
            logger.info(f"Saved new best model at epoch {epoch + 1} with accuracy {val_acc:.2f}%")
            logging.info(f"Saved new best model at epoch {epoch + 1} with accuracy {val_acc:.2f}%")


    logger.info(f"Best Accuracy: {max_acc:.4f}")
    logging.info(f"Best Accuracy: {max_acc:.4f}")

    """
    CNN - plot
    """
    plot(train_losses, val_losses)

    """
    CNN - test
    """
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # test(model, test_loader, criterion, device)

if __name__ == '__main__':
    main()
