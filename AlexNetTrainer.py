import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


class AlexNetTrainer:
    """
    Class that handles training, validation, and evaluation of an AlexNet model.

    Attributes:
        num_classes (int): Number of output classes.
        learning_rate (float): Learning rate used by the optimizer.
        dropout_rate (float): Dropout probability applied inside the model.
        batch_size (int): Batch size for training.
        device (str): Device on which the model is executed (e.g., "cuda" or "cpu").
        model (nn.Module): The AlexNet model instance being trained.
        criterion (nn.Module): Loss function used for optimization.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        history (dict): Logs containing per-epoch training and validation metrics.
    """
    def __init__(self, num_classes: int, learning_rate: float, batch_size: int, dropout_rate: float, device: str):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.device = device
        
        self.model = AlexNet(num_classes, self.dropout_rate)
        self.model = self.model.to(device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
    
    def fit(self, train_dataset: Dataset, val_dataset: Dataset, batch_size: int, epochs: int):
        """
        Train the model for multiple epochs.

        Args:
            train_dataset (Dataset): The training set.
            val_dataset (Dataset): The validation set.
            batch_size (int): Batch size for training.
            epochs (int): Number of epochs to train.

        Returns:
            None
        """
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 100)
            
            train_loss, train_acc = self.train(train_loader)            
            val_loss, val_acc = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
    
    def evaluate(self, test_ds: Dataset, batch_size: int):
        """
        Evaluate the trained model on a test dataset.

        Args:
            test_ds (Dataset): Test dataset supplying test batches.
            batch_size (int): Batch size for testing.

        Returns:
            tuple: A tuple ``(labels, predictions, accuracy)`` where:
                - labels (list[int]): Ground-truth labels.
                - predictions (list[int]): Model predictions.
                - accuracy (float): Percentage of correct predictions.
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Evaluating'):
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))

        return all_labels, all_preds, accuracy
            
    def train(self, train_loader: DataLoader):
        """
        Perform a single training epoch.

        Args:
            train_loader (DataLoader): DataLoader supplying training batches.

        Returns:
            tuple: A tuple ``(epoch_loss, epoch_acc)`` containing the average
                loss and accuracy for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100 * correct / total
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader: DataLoader):
        """
        Evaluate the model on the validation set.

        Args:
            val_loader (DataLoader): DataLoader supplying validation batches.

        Returns:
            tuple: A tuple ``(val_loss, val_acc)`` containing the average
                validation loss and accuracy.
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        return val_loss, val_acc

