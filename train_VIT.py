import torch
import yaml
import os
from ultralytics import RTDETR

project = "./detection_binary_anomaly/rtdetr/total_aug"

model = RTDETR("rtdetr-l.pt")
model.to('cuda')
# Train
dataset_yaml = "./data.yaml"
model.train(
    data=dataset_yaml, 
    epochs=1000, 
    imgsz=640, 
    batch=2, 
    project=project
)

# project = "./detection_binary_anomaly/rtdetr/total_aug/test"

metrics = model.val(
    data=dataset_yaml, 
    imgsz=640, 
    batch=2, 
    iou=0.7, 
    max_det=None,
    # half=False,
    # dnn=False,
    device='cuda:0', 
    # rect=False,
    split='test'
)