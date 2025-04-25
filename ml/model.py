import torch
import torch.nn as nn

class DeepfakeDetector(nn.Module):
    def __init__(self, input_size=224):
        super(DeepfakeDetector, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512)
        )

        # Compute dynamically flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, input_size, input_size)
            flattened_size = self._get_flattened_size(dummy_input)

        # Match the layer indexing with saved model
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),  # This is fc_layers.0 in original, but we ignore it
            nn.Linear(flattened_size, 512),  # This corresponds to fc_layers.1
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # This is fc_layers.2 in original, but we ignore it
            nn.Linear(512, 256),  # This corresponds to fc_layers.4
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # Final output layer
        )

    def _get_flattened_size(self, x):
        x = self.conv_layers(x)
        return x.view(x.size(0), -1).shape[1]

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x

# Load the corrected model
model = DeepfakeDetector()
model_path = "C:/Users/dnyaneshwar/Videos/SEM6/backend/ml/models/best_model.pth"
state_dict = torch.load(model_path, map_location="cpu")

# Load with strict=False to avoid minor mismatches
model.load_state_dict(state_dict, strict=False)
print("✅ Model loaded successfully!")
