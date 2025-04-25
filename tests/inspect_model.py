import torch

model_path = "C:/Users/dnyaneshwar/Videos/SEM6/backend/ml/models/best_model.pth"
state_dict = torch.load(model_path, map_location="cpu")

# Print the actual layer names in the saved model
print("Saved model layers:")
for key in state_dict.keys():
    print(key)
