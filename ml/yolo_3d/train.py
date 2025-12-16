from ultralytics import YOLO
import torch
from pathlib import Path
import os
import argparse
import numpy as np
import cv2
import tifffile as tiff

DATA_YAML = "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/yolo_3d/dataset/data.yaml"
def train_model():
    # 1. 加载模型
    # 使用 yolov8s-seg (small版本)，速度和精度在你的 800x800 图上最平衡
    # 如果显存只有 4G/6G，可以改用 yolov8n-seg
    model = YOLO('yolov8m-seg.pt') 

    # 2. 开始训练
    results = model.train(
        data=DATA_YAML,        # 指定配置文件
        epochs=400,            # 工业数据通常 100-300 轮收敛
        imgsz=960,             # 你的图片大小是 800，直接设为 800 或 960
        batch=16,               # 根据显存调整，16G显存可开 16 或 32
        device=0,              # GPU ID
        workers=4,             # 数据加载线程
        patience=50,           # 50轮不提升则早停
        name='defect_seg_med_v2',  # 训练结果保存的文件夹名
        augment=True,          # 开启默认增强 (旋转、翻转等)
        
        # 针对工业缺陷的重要超参数微调
        degrees=180.0,          # 工业零件通常可以任意旋转，开启大角度旋转增强
        fliplr=0.5,            # 左右翻转
        flipud=0.5,            # 上下翻转
        mosaic=0.0,            # 开启马赛克增强 (对小目标有效)
        # close_mosaic=50,
        copy_paste=0.5,        # 开启复制粘贴增强
        mixup=0.15             # 开启混合增强
    )

if __name__ == '__main__':
    train_model()