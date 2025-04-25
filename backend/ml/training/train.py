import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os
import json
from datetime import datetime

from ..models.model import DeepfakeDetector
from ..utils.dataset import DeepfakeDataset

class Trainer:
    def __init__(self, 
                 model,
                 train_loader,
                 val_loader,
                 criterion,
                 optimizer,
                 device,
                 save_dir):
        """
        Initialize the trainer.
        Args:
            model: DeepfakeDetector model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            save_dir: Directory to save checkpoints and plots
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for frames, labels in pbar:
            frames, labels = frames.to(self.device), labels.to(self.device).float()
            
            self.optimizer.zero_grad()
            predictions, _ = self.model(frames)
            loss = self.criterion(predictions, labels.unsqueeze(1))
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            predicted = (predictions.squeeze() > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
        
        return total_loss / len(self.train_loader), correct / total
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for frames, labels in tqdm(self.val_loader, desc='Validating'):
                frames, labels = frames.to(self.device), labels.to(self.device).float()
                predictions, _ = self.model(frames)
                loss = self.criterion(predictions, labels.unsqueeze(1))
                
                total_loss += loss.item()
                predicted = (predictions.squeeze() > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return (total_loss / len(self.val_loader), 
                correct / total,
                np.array(all_preds),
                np.array(all_labels))
    
    def train(self, num_epochs):
        """
        Train the model.
        Args:
            num_epochs: Number of epochs to train for
        """
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_preds, val_labels = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f'Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 
                         os.path.join(self.save_dir, 'best_model.pth'))
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            
            # Plot and save metrics
            if (epoch + 1) % 5 == 0:
                self.plot_metrics()
                self.plot_roc_curve(val_preds, val_labels)
    
    def plot_metrics(self):
        """Plot and save training metrics."""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train')
        plt.plot(self.val_losses, label='Validation')
        plt.title('Loss vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train')
        plt.plot(self.val_accuracies, label='Validation')
        plt.title('Accuracy vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_metrics.png'))
        plt.close()
    
    def plot_roc_curve(self, predictions, labels):
        """Plot and save ROC curve."""
        fpr, tpr, _ = roc_curve(labels, predictions)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.save_dir, 'roc_curve.png'))
        plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Set up transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    dataset = DeepfakeDataset(
        data_dir='processed/combined',
        transform=transform,
        frame_count=32
    )
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # Initialize model
    model = DeepfakeDetector().to(device)
    
    # Set up training
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create save directory
    save_dir = os.path.join('checkpoints', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir, exist_ok=True)
    
    # Save dataset splits
    dataset.save_split(
        train_indices=train_dataset.indices,
        val_indices=val_dataset.indices,
        test_indices=test_dataset.indices,
        output_dir=save_dir
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_dir=save_dir
    )
    
    # Train model
    trainer.train(num_epochs=50)

if __name__ == '__main__':
    main() 