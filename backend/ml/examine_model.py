import torch
import os

# Get absolute path to the model file
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "models", "best_model.pth")
print(f"Looking for model at: {model_path}")

if not os.path.exists(model_path):
    print(f"Model file not found: {model_path}")
    exit()

# Load the model weights
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
print("Keys in state_dict:", list(state_dict.keys()))

# Print the shape of each layer
for key in state_dict.keys():
    print(f"{key}: {state_dict[key].shape}") 