import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Cihaz ve model yükleme
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load("mold_model_optimized.pt", map_location=device)
model.eval()

# Görüntü dönüştürme transform'u
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Kamera bağlantısı (Camo Studio açık ve iPhone bağlı olmalı)
cap = cv2.VideoCapture(0)  # Gerekirse 0, 1 veya 2 olarak değiştirin

if not cap.isOpened():
    print("❌ Kamera açılamadı. Camo Studio'nun açık ve telefonun bağlı olduğundan emin ol.")
    exit()

print("✅ Camo Camera ile canlı analiz başlatıldı. Çıkmak için 'q' tuşuna bas.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Görüntü alınamadı.")
        break

    # Görüntüyü PIL formatına dönüştür
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Model için hazırlık
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Tahmin yap
    with torch.no_grad():
        output = model(input_tensor)
        prob = output.item()
        prediction = 1 if prob > 0.3 else 0  # Eğitimdeki eşiği kullandık

    result_text = "Mold Detected" if prediction == 1 else "Clean"
    color = (0, 0, 255) if prediction == 1 else (0, 255, 0)

    # Sonucu görüntüye yaz
    cv2.putText(
        frame,
        f"{result_text} ({prob:.2f})",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        color,
        3
    )

    # Görüntüyü göster
    cv2.imshow("Live Mold Detection", frame)

    # Çıkmak için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
