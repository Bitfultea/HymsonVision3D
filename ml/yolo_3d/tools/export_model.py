from ultralytics import YOLO
import torch

ckpt = "../runs/segment/defect_seg_v13/weights/best.pt"

# Load the YOLO11 model
model = YOLO(ckpt)

# Export the model to TensorRT format
# model.export(format="engine")  # creates trt engine'
model.export(format="onnx",simplify=True)  # creates onnx engine'
model.export(format="torchscript",simplify=True)  # creates torchscript'
