import json
from ultralytics import YOLO

# Load a model
model = YOLO('./detection_multi/yolov8/total_aug/train/weights/best.pt')  # pretrained YOLOv8n model

image_path = "test/images"
# Run batched inference on a list of images
results = model(
    image_path, 
    save_txt=True,
    project="results",
    name="yolov8_multi",
    device='cuda:0', 
    exist_ok=True,
    stream=True,
    conf=0.1,
    imgsz=640, 
    batch=2, 
    iou=0.7, 
    max_det=None,
)  # return a generator of Results objects

for result in results:
    print("ok")