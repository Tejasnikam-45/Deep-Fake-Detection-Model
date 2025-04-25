import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from typing import List, Tuple, Dict
import json

class DeepfakeDataset(Dataset):
    def __init__(self, 
                 data_dir: str,
                 transform=None,
                 frame_count: int = 32):
        """
        Initialize the DeepfakeDataset.
        Args:
            data_dir: Directory containing processed frames
            transform: Torchvision transforms to apply
            frame_count: Number of frames to use per video
        """
        self.data_dir = data_dir
        self.transform = transform
        self.frame_count = frame_count
        self.samples = self._load_dataset()

    def _load_dataset(self) -> List[Dict]:
        """
        Load dataset information.
        Returns:
            List of dictionaries containing sample information
        """
        samples = []
        
        # Walk through the real videos
        real_dir = os.path.join(self.data_dir, 'real')
        if os.path.exists(real_dir):
            for video_name in os.listdir(real_dir):
                video_path = os.path.join(real_dir, video_name)
                if os.path.isdir(video_path):
                    samples.append({
                        'path': video_path,
                        'label': 0,  # 0 for real
                        'name': video_name
                    })

        # Walk through the fake videos
        fake_dir = os.path.join(self.data_dir, 'fake')
        if os.path.exists(fake_dir):
            for video_name in os.listdir(fake_dir):
                video_path = os.path.join(fake_dir, video_name)
                if os.path.isdir(video_path):
                    samples.append({
                        'path': video_path,
                        'label': 1,  # 1 for fake
                        'name': video_name
                    })

        return samples

    def _load_frames(self, video_path: str) -> np.ndarray:
        """
        Load frames from a video directory.
        Args:
            video_path: Path to the directory containing frames
        Returns:
            Numpy array of frames
        """
        frame_files = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])
        
        if len(frame_files) == 0:
            raise ValueError(f"No frames found in {video_path}")
        
        # Select evenly spaced frames if we have more than we need
        if len(frame_files) > self.frame_count:
            indices = np.linspace(0, len(frame_files)-1, self.frame_count, dtype=int)
            frame_files = [frame_files[i] for i in indices]
        
        frames = []
        for frame_file in frame_files:
            frame_path = os.path.join(video_path, frame_file)
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        # Pad with zeros if we don't have enough frames
        while len(frames) < self.frame_count:
            frames.append(np.zeros_like(frames[0]))
        
        return np.stack(frames)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        Args:
            idx: Index of the sample
        Returns:
            Tuple of (frames, label)
        """
        sample = self.samples[idx]
        frames = self._load_frames(sample['path'])
        
        if self.transform:
            # Apply transform to each frame
            transformed_frames = []
            for frame in frames:
                transformed_frame = self.transform(frame)
                transformed_frames.append(transformed_frame)
            frames = torch.stack(transformed_frames)
        else:
            frames = torch.from_numpy(frames.transpose(0, 3, 1, 2)).float()
        
        return frames, sample['label']

    def save_split(self, train_indices: List[int], val_indices: List[int], 
                  test_indices: List[int], output_dir: str):
        """
        Save dataset split information.
        Args:
            train_indices: Indices for training set
            val_indices: Indices for validation set
            test_indices: Indices for test set
            output_dir: Directory to save split information
        """
        split_info = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }
        
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'split.json'), 'w') as f:
            json.dump(split_info, f) 