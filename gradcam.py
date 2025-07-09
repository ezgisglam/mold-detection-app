''' import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from main import MoldClassifier, device  # Modeli main.py'den import et

# Modeli yükle
model = MoldClassifier().to(device)
model.load_state_dict(torch.load("mold_model.pth", map_location=device))
model.eval()

# Grad-CAM için hook fonksiyonları
def hook_fn(module, input, output):
    global feature_maps
    feature_maps = output

def get_cam(image_path, model, target_layer):
    global feature_maps
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0).to(device)
    
    # Hook ekle
    handle = target_layer.register_forward_hook(hook_fn)
    output = model(image)
    handle.remove()
    
    # En yüksek skorlu sınıfı al
    output_class = torch.sigmoid(output).item()
    model.zero_grad()
    
    # Gradients hesapla
    output.backward()
    gradients = target_layer.weight.grad
    weights = torch.mean(gradients, dim=[0, 2, 3])
    
    # Feature map ile çarp
    cam = torch.zeros(feature_maps.shape[2:], dtype=torch.float32).to(device)
    for i, w in enumerate(weights):
        cam += w * feature_maps[0, i, :, :]
    
    cam = F.relu(cam).cpu().detach().numpy()
    cam = cv2.resize(cam, (128, 128))
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam, output_class

# Test görseli
image_path = "/Users/ezgisaglam/Desktop/testset/test5.jpg"  # Buraya test etmek istediğin görselin yolunu yaz
cam, output_class = get_cam(image_path, model, model.conv2)  # 2. Convolution katmanını seçtik

# Görselleştirme
original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlaid_image = cv2.addWeighted(original_image, 0.5, cam_heatmap, 0.5, 0)

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title("Orijinal Görsel")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cam, cmap='jet')
plt.title("Grad-CAM")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(overlaid_image)
plt.title("Grad-CAM Üst Üste")
plt.axis("off")

plt.suptitle(f"Tahmin: {'Mold Detected.' if output_class > 0.5 else 'Clean'}")
plt.show()
'''