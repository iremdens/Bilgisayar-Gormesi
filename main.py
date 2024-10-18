import torch
from ultralytics import YOLO

# Cihazı belirt (CPU için)
device = torch.device('cpu')

# Modeli yükle ve cihazı ayarla
model = YOLO('yolov8n.pt').to(device)  # Model dosya adını buraya yaz

# Resim dosyasının yolunu belirt
img_path = 'C:\\Users\\irem\\Downloads\\alagoz.webp'  # Buraya resim dosyanızın yolunu yaz

# Model ile tahmin yap
results = model(img_path)  # Resim dosyası yolunu direkt ver

# Sonuçları görüntüle
for result in results:  # Her bir sonuç için döngü
    result.show()  # Sonucu göster




