import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from typing import Tuple

class DeepFakeDetector(nn.Module):
    def __init__(self, backbone='efficientnet_b0', pretrained=True):
        """
        Initialize the DeepFake Detection model.
        Args:
            backbone (str): Name of the backbone model to use
            pretrained (bool): Whether to use pretrained weights
        """
        super(DeepFakeDetector, self).__init__()
        
        # Load the backbone model
        self.backbone = timm.create_model(backbone, pretrained=pretrained)
        
        # Get the number of features from the backbone
        if 'efficientnet' in backbone:
            n_features = self.backbone.classifier.in_features
            # Remove the original classifier
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),  # Binary classification
            nn.Sigmoid()
        )

        # Frame attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_frames, channels, height, width)
        Returns:
            torch.Tensor: Probability of the video being fake (0-1)
        """
        batch_size, num_frames = x.size(0), x.size(1)
        
        # Reshape input to process all frames
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        
        # Extract features from backbone
        features = self.backbone(x)  # Shape: (batch_size * num_frames, n_features)
        
        # Reshape features back to separate frames
        features = features.view(batch_size, num_frames, -1)  # Shape: (batch_size, num_frames, n_features)
        
        # Apply attention mechanism
        attention_weights = self.attention(features)  # Shape: (batch_size, num_frames, 1)
        weighted_features = torch.sum(features * attention_weights, dim=1)  # Shape: (batch_size, n_features)
        
        # Final classification
        output = self.classifier(weighted_features)  # Shape: (batch_size, 1)
        
        return output

    def predict_video(self, frames, device='cuda'):
        """
        Predict whether a video is real or fake.
        Args:
            frames (torch.Tensor): Video frames tensor of shape (1, num_frames, channels, height, width)
            device (str): Device to run prediction on
        Returns:
            float: Probability of the video being fake (0-1)
            list: Frame-wise attention weights
        """
        self.eval()
        with torch.no_grad():
            frames = frames.to(device)
            batch_size, num_frames = frames.size(0), frames.size(1)
            
            # Get backbone features
            x = frames.view(-1, frames.size(2), frames.size(3), frames.size(4))
            features = self.backbone(x)
            features = features.view(batch_size, num_frames, -1)
            
            # Get attention weights
            attention_weights = self.attention(features)
            weighted_features = torch.sum(features * attention_weights, dim=1)
            
            # Get final prediction
            output = self.classifier(weighted_features)
            
            return output.item(), attention_weights.squeeze().cpu().numpy().tolist()

    @staticmethod
    def get_transforms(image_size=224):
        """
        Get the required transforms for input images.
        Args:
            image_size (int): Size to resize images to
        Returns:
            dict: Dictionary containing train and test transforms
        """
        from torchvision import transforms
        
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return {
            'train': train_transform,
            'test': test_transform
        }

class DeepfakeClassifier:
    def __init__(self, model_path: str = None, device: str = None):
        """
        Initialize the classifier.
        Args:
            model_path: Path to saved model weights
            device: Device to run the model on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = DeepFakeDetector().to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.eval()

    def predict_video(self, frames: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """
        Predict whether a video is real or fake.
        Args:
            frames: Tensor of video frames [1, num_frames, channels, height, width]
        Returns:
            Tuple of (probability of being fake, attention weights)
        """
        frames = frames.to(self.device)
        with torch.no_grad():
            predictions, attention = self.model(frames)
        
        return predictions.item(), attention.cpu()

if __name__ == '__main__':
    # Test the model
    model = DeepFakeDetector()
    x = torch.randn(2, 32, 3, 224, 224)  # [batch_size, frames, channels, height, width]
    predictions, attention = model(x)
    print(f"Predictions shape: {predictions.shape}")  # [batch_size, 1]
    print(f"Attention shape: {attention.shape}")  # [batch_size, frames] 