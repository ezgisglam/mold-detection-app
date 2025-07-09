import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modeli yükle
model = torch.jit.load("mold_model_optimized.pt").to(device)
model.eval()

# Klasör yolu
test_folder = "/Users/ezgisaglam/Desktop/testset"

# Görsel işlemleri
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

y_true = []
y_pred = []

# Test setindeki tüm görselleri sırayla al
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

            # Gerçek etiket (klasör adına göre)
            true_label = 1 if label_folder.lower() == "mold" else 0

            y_true.append(true_label)
            y_pred.append(prediction)

            result = "Mold Detected" if prediction == 1 else "Clean"
            print(f"{filename} ({label_folder}) --> Prediction: {result}")

# Skorları yazdır
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Clean", "Mold"]))
