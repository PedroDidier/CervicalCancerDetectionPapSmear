from ultralytics import RTDETR

# Load a model
dataset_yaml = "./data.yaml"
model = RTDETR('detection_multi/rtdetr/base/train/weights/best.pt')

# Validate the model
metrics = model.val(
    data=dataset_yaml, 
    imgsz=640, 
    batch=2, 
    iou=0.7, 
    max_det=None,
    augment=True,
    # half=False,
    # dnn=False,
    device='cuda:0', 
    # rect=False,
    split='test'
)