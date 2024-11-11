import torch
from roboflow import Roboflow
from ultralytics import YOLO

print("CUDA Version:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())

model = YOLO('yolov8n.pt')  # YOLOv8 Nano modeli

rf = Roboflow(api_key="2QQVXn8gEO7wtod1fbxQ")
project = rf.workspace("irem-kc8lh").project("kayisi-bu8jv")
version = project.version(1)
dataset = version.download("yolov8")

# Eğitim ve test işlemlerini ana kod bloğuna alıyoruz
# Modeli eğit
if __name__ == "__main__" :
    model.train(data=f"{dataset.location}/data.yaml", epochs=15)