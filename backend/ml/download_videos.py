import os
import logging
import requests
from tqdm import tqdm
import cv2
import numpy as np
from mtcnn import MTCNN
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, "dataset")
real_videos_dir = os.path.join(dataset_dir, "real")
fake_videos_dir = os.path.join(dataset_dir, "fake")

# Create directories if they don't exist
os.makedirs(real_videos_dir, exist_ok=True)
os.makedirs(fake_videos_dir, exist_ok=True)

def download_file(url, destination):
    """Download a file from a URL with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=destination,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

def create_sample_video(output_path, duration=5, fps=30):
    """Create a sample video with a face."""
    # Create a simple face drawing
    width, height = 640, 480
    frames = []
    
    for i in range(duration * fps):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw face outline
        cv2.circle(frame, (width//2, height//2), 100, (255, 255, 255), 2)
        
        # Draw eyes
        cv2.circle(frame, (width//2 - 30, height//2 - 20), 10, (255, 255, 255), -1)
        cv2.circle(frame, (width//2 + 30, height//2 - 20), 10, (255, 255, 255), -1)
        
        # Draw mouth
        cv2.ellipse(frame, (width//2, height//2 + 30), (40, 20), 0, 0, 180, (255, 255, 255), 2)
        
        frames.append(frame)
    
    # Save video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()

def create_fake_video(real_video_path, output_path):
    """Create a fake video by applying some transformations."""
    cap = cv2.VideoCapture(real_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Apply some transformations to make it look fake
        # 1. Slight color shift
        frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
        
        # 2. Slight blur
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        
        # 3. Slight distortion
        rows, cols = frame.shape[:2]
        M = np.float32([[1, 0.1, 0], [0, 1, 0]])
        frame = cv2.warpAffine(frame, M, (cols, rows))
        
        out.write(frame)
    
    cap.release()
    out.release()

def main():
    """Main function to prepare the dataset."""
    logging.info("Starting dataset preparation...")
    
    # Create 5 real videos
    for i in range(5):
        output_path = os.path.join(real_videos_dir, f"real_{i}.mp4")
        create_sample_video(output_path)
        logging.info(f"Created real video: {output_path}")
    
    # Create 5 fake videos
    for i in range(5):
        real_video_path = os.path.join(real_videos_dir, f"real_{i}.mp4")
        fake_video_path = os.path.join(fake_videos_dir, f"fake_{i}.mp4")
        create_fake_video(real_video_path, fake_video_path)
        logging.info(f"Created fake video: {fake_video_path}")
    
    logging.info("Dataset preparation completed!")

if __name__ == "__main__":
    main() 