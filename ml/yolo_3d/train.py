from ultralytics import YOLO
from ultralytics import RTDETR
import torch
from pathlib import Path
import os
import argparse
import numpy as np
import cv2
import tifffile as tiff
from collections import Counter

DATA_YAML = "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/yolo_3d/dataset/data.yaml"


def train_model():
    
    label_dir = "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/yolo_3d/dataset/labels/train"
    class_names = ['pinhole', 'crap', 'spatter']
    counts = Counter()

    for file in os.listdir(label_dir):
        with open(os.path.join(label_dir, file), 'r') as f:
            for line in f:
                cls_id = int(line.split()[0])
                counts[cls_id] += 1

    print("训练集标签分布：")
    for i, name in enumerate(class_names):
        print(f"{name}: {counts[i]} 个实例")

    # 1. 加载模型
    # 使用 yolov8s-seg (small版本)，速度和精度在你的 800x800 图上最平衡
    # 如果显存只有 4G/6G，可以改用 yolov8n-seg
    # model = YOLO('yolov8s-seg.pt') 
    model = YOLO('yolo11m-seg.pt') 
    # model = YOLO('runs/segment/yolov11122')
    # model = YOLO('yolov8s-seg-p2.yaml') 
    # model = RTDETR("rtdetr-l.pt")

    # 2. 开始训练
    results = model.train(
        data=DATA_YAML,        # 指定配置文件
        epochs=500,            # 工业数据通常 100-300 轮收敛
        imgsz=960,             # 你的图片大小是 800，直接设为 800 或 960
        batch=16,               # 根据显存调整，16G显存可开 16 或 32
        device=0,              # GPU ID
        workers=4,             # 数据加载线程
        patience=0,           # 50轮不提升则早停
        dropout = 0.4,
        name='yolov11',  # 训练结果保存的文件夹名
        augment=True,          # 开启默认增强 (旋转、翻转等)
        single_cls = False,
        
        # 针对工业缺陷的重要超参数微调
        degrees=180.0,          # 工业零件通常可以任意旋转，开启大角度旋转增强
        scale=0.8,
        fliplr=0.5,            # 左右翻转
        flipud=0.5,            # 上下翻转
        mosaic=1.00,            # 开启马赛克增强 (对小目标有效)
        close_mosaic=50,
        copy_paste=0.8,        # 开启复制粘贴增强
        cache=True,            # 开启复制粘贴增强
        # mixup=0.00,            # 开启混合增强
        mixup=0.0,          # 开启混合增强
        cutmix=0.40,
        
        # cls=2.0
        # dfl=2.0
        # rect=True,
    )
    
    model.val(
        conf=0.15,
        # plots = True
    )
    

if __name__ == '__main__':
    train_model()
    