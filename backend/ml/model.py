import torch
import torch.nn as nn

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        
        # First conv block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        # Second conv block
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        # Third conv block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        
        # Fourth conv block
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 14 * 14, 64)
        self.relu5 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 2)
        
        # Temperature scaling parameter
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # Fourth conv block
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        
        # Flatten and fully connected layers
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Apply temperature scaling
        x = x / self.temperature
        return x

    def load_state_dict(self, state_dict, strict=True):
        # Custom loading to handle the weight shape mismatch
        model_dict = self.state_dict()
        for k, v in state_dict.items():
            if k in model_dict:
                if v.shape != model_dict[k].shape:
                    # Handle the shape mismatch for BatchNorm layers
                    if 'weight' in k and 'conv_layers' in k:
                        if len(v.shape) == 1:
                            v = v.view(-1, 1, 1, 1)
                model_dict[k] = v
        super().load_state_dict(model_dict, strict=False) 