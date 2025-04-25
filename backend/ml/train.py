import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import wandb
import logging

from models.model import DeepFakeDetector
from data.dataset import VideoDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Train for one epoch.
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for frames, labels in progress_bar:
        frames, labels = frames.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs.squeeze(), labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(outputs.squeeze().cpu().detach().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate metrics
    epoch_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    auc = roc_auc_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds > 0.5)
    
    return epoch_loss, auc, accuracy

def validate(model, dataloader, criterion, device):
    """
    Validate the model.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for frames, labels in tqdm(dataloader, desc='Validation'):
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)
            loss = criterion(outputs.squeeze(), labels)
            
            total_loss += loss.item()
            all_preds.extend(outputs.squeeze().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    val_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    auc = roc_auc_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds > 0.5)
    
    return val_loss, auc, accuracy

def main():
    try:
        # Configuration
        config = {
            'data_dir': os.path.abspath('backend/ml/data/processed/celeb-df'),
            'num_epochs': 50,
            'batch_size': 8,
            'learning_rate': 1e-4,
            'num_frames': 32,
            'frame_interval': 4,
            'image_size': 224,
            'backbone': 'efficientnet_b0',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'save_dir': 'backend/ml/checkpoints'
        }
        
        logger.info(f"Starting training with config: {config}")
        
        # Initialize wandb
        # wandb.init(project='deepfake-detection', config=config)
        
        # Create save directory
        os.makedirs(config['save_dir'], exist_ok=True)
        
        # Initialize model
        logger.info("Initializing model...")
        model = DeepFakeDetector(backbone=config['backbone']).to(config['device'])
        transforms = model.get_transforms(image_size=config['image_size'])
        
        # Create datasets and dataloaders
        logger.info("Creating datasets...")
        train_dataset = VideoDataset(
            root_dir=config['data_dir'],
            transform=transforms['train'],
            num_frames=config['num_frames'],
            frame_interval=config['frame_interval'],
            mode='train'
        )
        
        val_dataset = VideoDataset(
            root_dir=config['data_dir'],
            transform=transforms['test'],
            num_frames=config['num_frames'],
            frame_interval=config['frame_interval'],
            mode='test'
        )
        
        logger.info("Creating dataloaders...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=VideoDataset.collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=VideoDataset.collate_fn
        )
        
        # Initialize loss function and optimizer
        logger.info("Initializing loss function and optimizer...")
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)
        
        # Training loop
        logger.info("Starting training loop...")
        best_val_auc = 0
        for epoch in range(config['num_epochs']):
            # Train
            train_loss, train_auc, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, config['device'], epoch
            )
            
            # Validate
            val_loss, val_auc, val_acc = validate(
                model, val_loader, criterion, config['device']
            )
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Log metrics
            # wandb.log({
            #     'train_loss': train_loss,
            #     'train_auc': train_auc,
            #     'train_accuracy': train_acc,
            #     'val_loss': val_loss,
            #     'val_auc': val_auc,
            #     'val_accuracy': val_acc,
            #     'learning_rate': optimizer.param_groups[0]['lr']
            # })
            
            # Save best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), os.path.join(config['save_dir'], 'best_model.pth'))
                logger.info(f'Saved new best model with validation AUC: {val_auc:.4f}')
            
            logger.info(f'Epoch {epoch}:')
            logger.info(f'Train - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, Acc: {train_acc:.4f}')
            logger.info(f'Val - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f}')
        
        # wandb.finish()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main() 