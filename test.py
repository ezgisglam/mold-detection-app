import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Upload model
model = torch.jit.load("mold_model_optimized.pt").to(device)
model.eval()

test_folder = "/your/path/here"

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

y_true = []
y_pred = []

# Get all of the images in the testset in order.
for label_folder in os.listdir(test_folder):
    class_folder = os.path.join(test_folder, label_folder)

    if not os.path.isdir(class_folder):
        continue

    for filename in os.listdir(class_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(class_folder, filename)
            image = Image.open(image_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image)
                prediction = 1 if output.item() > 0.5 else 0

            # lables by folder name
            true_label = 1 if label_folder.lower() == "mold" else 0

            y_true.append(true_label)
            y_pred.append(prediction)

            result = "Mold Detected" if prediction == 1 else "Clean"
            print(f"{filename} ({label_folder}) --> Prediction: {result}")

# Print Scores
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Clean", "Mold"]))
