import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import decord
import logging
import glob

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_frames=32, frame_interval=4, mode='train', train_ratio=0.8):
        """
        Dataset class for loading video frames.
        Args:
            root_dir (str): Root directory containing 'real' and 'fake' subdirectories
            transform: Optional transform to be applied to frames
            num_frames (int): Number of frames to sample from each video
            frame_interval (int): Interval between sampled frames
            mode (str): 'train' or 'test'
            train_ratio (float): Ratio of data to use for training
        """
        self.root_dir = os.path.abspath(root_dir)
        self.transform = transform
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.mode = mode

        logger.info(f"Initializing dataset with root_dir: {self.root_dir}")
        logger.info(f"Mode: {mode}, num_frames: {num_frames}, frame_interval: {frame_interval}")

        # Get all video paths
        self.videos = []
        self.labels = []
        
        # Real videos
        real_dir = os.path.join(self.root_dir, 'real')
        if os.path.exists(real_dir):
            real_videos = glob.glob(os.path.join(real_dir, '*.mp4'))
            self.videos.extend(real_videos)
            self.labels.extend([0] * len(real_videos))  # 0 for real
            logger.info(f"Found {len(real_videos)} real videos in {real_dir}")
        else:
            logger.warning(f"Real directory not found: {real_dir}")
        
        # Fake videos
        fake_dir = os.path.join(self.root_dir, 'fake')
        if os.path.exists(fake_dir):
            fake_videos = glob.glob(os.path.join(fake_dir, '*.mp4'))
            self.videos.extend(fake_videos)
            self.labels.extend([1] * len(fake_videos))  # 1 for fake
            logger.info(f"Found {len(fake_videos)} fake videos in {fake_dir}")
        else:
            logger.warning(f"Fake directory not found: {fake_dir}")

        if not self.videos:
            raise ValueError(f"No videos found in {self.root_dir}. Please check the directory structure.")

        # Split train/test
        num_videos = len(self.videos)
        indices = list(range(num_videos))
        np.random.shuffle(indices)
        split = int(np.floor(train_ratio * num_videos))
        
        if mode == 'train':
            indices = indices[:split]
        else:
            indices = indices[split:]
            
        self.videos = [self.videos[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        
        logger.info(f"Dataset initialized with {len(self.videos)} videos in {mode} mode")
        logger.info(f"Sample video paths: {self.videos[:2]}")

    def __len__(self):
        return len(self.videos)

    def load_frames(self, video_path):
        """
        Load frames from a video file.
        Args:
            video_path (str): Path to the video file
        Returns:
            torch.Tensor: Tensor of frames
        """
        try:
            logger.debug(f"Loading frames from: {video_path}")
            vr = decord.VideoReader(video_path)
            total_frames = len(vr)
            
            # Calculate frame indices to sample
            if total_frames <= self.num_frames * self.frame_interval:
                # If video is too short, sample with replacement
                indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            else:
                # Randomly sample starting point and take consecutive frames
                max_start = total_frames - self.num_frames * self.frame_interval
                start_idx = np.random.randint(0, max_start)
                indices = np.arange(start_idx, start_idx + self.num_frames * self.frame_interval, self.frame_interval)
            
            # Read frames
            frames = vr.get_batch(indices).asnumpy()  # Convert NDArray to numpy array
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # Convert to tensor and permute dimensions
            
            if self.transform:
                frames = torch.stack([self.transform(frame) for frame in frames])
            
            return frames
        except Exception as e:
            logger.error(f"Error loading video {video_path}: {str(e)}")
            raise

    def __getitem__(self, idx):
        """
        Get a video and its label.
        Args:
            idx (int): Index
        Returns:
            tuple: (frames, label)
        """
        video_path = self.videos[idx]
        label = self.labels[idx]
        
        try:
            frames = self.load_frames(video_path)
            return frames, label
        except Exception as e:
            logger.error(f"Error in __getitem__ for video {video_path}: {str(e)}")
            # Return a random different video as fallback
            new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for the data loader.
        Args:
            batch: List of tuples (frames, label)
        Returns:
            tuple: (frames, labels)
        """
        frames = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch], dtype=torch.float)
        return frames, labels 