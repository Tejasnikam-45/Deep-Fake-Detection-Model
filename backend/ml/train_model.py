import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import logging
from tqdm import tqdm
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight
import json
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
DATA_PATH = os.path.join(BASE_DIR, "data/processed_data")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
NUM_EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EARLY_STOPPING_PATIENCE = 10
WARMUP_EPOCHS = 5
GRADIENT_CLIP = 1.0
CHECKPOINT_FREQUENCY = 5  # Save checkpoint every N epochs

# Data augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomErasing(p=0.2),
])

val_transform = transforms.Compose([])

# Dataset class
class DeepfakeDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.y[idx]

# Model class
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / len(train_loader), 100. * correct / total

# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / len(val_loader), 100. * correct / total

# Main training loop
def main():
    logging.info("Loading data...")
    try:
        X_train = np.load(os.path.join(DATA_PATH, "X_train.npy"))
        y_train = np.load(os.path.join(DATA_PATH, "y_train.npy"))
        X_val = np.load(os.path.join(DATA_PATH, "X_val.npy"))
        y_val = np.load(os.path.join(DATA_PATH, "y_val.npy"))
        X_train = X_train.transpose(0, 3, 1, 2)
        X_val = X_val.transpose(0, 3, 1, 2)
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = torch.FloatTensor(class_weights).to(DEVICE)
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        return
    train_dataset = DeepfakeDataset(X_train, y_train, transform=train_transform)
    val_dataset = DeepfakeDataset(X_val, y_val, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)
    model = DeepfakeDetector().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    logging.info(f"Starting training on {DEVICE}...")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        logging.info(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, "best_model.pth"))
    logging.info("Training complete!")

if __name__ == "__main__":
    main()