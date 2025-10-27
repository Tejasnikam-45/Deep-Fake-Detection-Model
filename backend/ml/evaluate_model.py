import torch
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix

# Define paths
DATA_PATH = os.path.join(os.getcwd(), "backend/ml/data/processed_data")
MODEL_PATH = os.path.join(os.getcwd(), "backend/ml/models/best_model.pth")

# Check if test data exists
X_test_path = os.path.join(DATA_PATH, "X_test.npy")
y_test_path = os.path.join(DATA_PATH, "y_test.npy")

if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
    raise FileNotFoundError(f"Test data files not found:\n - {X_test_path}\n - {y_test_path}")

# Load test data
X_test = np.load(X_test_path)
y_test = np.load(y_test_path)

X_test = torch.FloatTensor(X_test.transpose(0, 3, 1, 2))  # Reshape for PyTorch
y_test = torch.LongTensor(y_test)

# Import the DeepfakeDetector model
from train_model import DeepfakeDetector  # ✅ Make sure this matches your model file

# Initialize the model
model = DeepfakeDetector()

# Load model state_dict correctly
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))

# Put model in evaluation mode
model.eval()

# Predict on test set
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)

# Print evaluation results
print(classification_report(y_test.numpy(), predicted.numpy()))
print("Confusion Matrix:\n", confusion_matrix(y_test.numpy(), predicted.numpy()))
# Save results to a file
with open("backend/ml/evaluation_results.txt", "w") as f:
    f.write("Classification Report:\n")
    f.write(classification_report(y_test.numpy(), predicted.numpy()))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test.numpy(), predicted.numpy())))
