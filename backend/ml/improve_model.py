import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from mtcnn import MTCNN
from tqdm import tqdm
from test_model import DeepfakeDetector
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {DEVICE}")

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, "dataset")
real_videos_dir = os.path.join(dataset_dir, "real")
fake_videos_dir = os.path.join(dataset_dir, "fake")
models_dir = os.path.join(current_dir, "models")
os.makedirs(models_dir, exist_ok=True)

class VideoFaceDataset(Dataset):
    def __init__(self, real_videos_dir, fake_videos_dir, transform=None):
        self.real_videos = [os.path.join(real_videos_dir, f) for f in os.listdir(real_videos_dir) if f.endswith('.mp4')]
        self.fake_videos = [os.path.join(fake_videos_dir, f) for f in os.listdir(fake_videos_dir) if f.endswith('.mp4')]
        self.transform = transform
        self.detector = MTCNN()
        
        logging.info(f"Found {len(self.real_videos)} real videos and {len(self.fake_videos)} fake videos")
        
    def __len__(self):
        return len(self.real_videos) + len(self.fake_videos)
    
    def extract_faces(self, video_path):
        faces = []
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logging.info(f"Processing video: {video_path} with {total_frames} frames")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every 5th frame
            if frame_count % 5 == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detected_faces = self.detector.detect_faces(frame_rgb)
                
                for face in detected_faces:
                    x, y, w, h = face['box']
                    # Add padding to ensure we capture the full face
                    x = max(0, x - int(w * 0.1))
                    y = max(0, y - int(h * 0.1))
                    w = min(frame.shape[1] - x, w + int(w * 0.2))
                    h = min(frame.shape[0] - y, h + int(h * 0.2))
                    
                    face_img = frame_rgb[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (224, 224))
                    face_img = face_img / 255.0
                    faces.append(face_img)
            
            frame_count += 1
            
        cap.release()
        logging.info(f"Extracted {len(faces)} faces from {video_path}")
        return faces
    
    def __getitem__(self, idx):
        if idx < len(self.real_videos):
            video_path = self.real_videos[idx]
            label = 0  # Real
        else:
            video_path = self.fake_videos[idx - len(self.real_videos)]
            label = 1  # Fake
        
        faces = self.extract_faces(video_path)
        if not faces:
            # If no faces found, return a random face from another video
            logging.warning(f"No faces found in {video_path}, trying another video")
            return self.__getitem__((idx + 1) % len(self))
        
        # Randomly select one face
        face = random.choice(faces)
        face = torch.FloatTensor(face).permute(2, 0, 1)
        
        if self.transform:
            face = self.transform(face)
        
        return face, label

def train_model(model, train_loader, val_loader, num_epochs=10):
    """Train the model with early stopping."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        logging.info(f"Epoch {epoch+1}/{num_epochs}:")
        logging.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logging.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(models_dir, "improved_model.pth"))
            logging.info("Saved best model!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("Early stopping triggered!")
                break

def calibrate_model(model, val_loader):
    """Calibrate the model using temperature scaling."""
    model.eval()
    nll_criterion = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS([model.temperature], lr=0.01, max_iter=50)
    
    def eval():
        optimizer.zero_grad()
        total_loss = 0
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = nll_criterion(outputs, labels)
            total_loss += loss.item()
            loss.backward()
        return total_loss
    
    optimizer.step(eval)
    logging.info(f"Calibration complete. Temperature: {model.temperature.item():.3f}")

def main():
    # Create dataset
    dataset = VideoFaceDataset(real_videos_dir, fake_videos_dir)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders with num_workers=0 to avoid MTCNN serialization issues
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    # Load pre-trained model
    model = DeepfakeDetector()
    model_path = os.path.join(models_dir, "best_model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        logging.info("Loaded pre-trained model")
    else:
        logging.warning(f"Pre-trained model not found at {model_path}, starting from scratch")
    
    model = model.to(DEVICE)
    
    # Train model
    train_model(model, train_loader, val_loader)
    
    # Calibrate model
    calibrate_model(model, val_loader)
    
    logging.info("Model improvement completed!")

if __name__ == "__main__":
    main() 