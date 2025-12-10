import cv2
import numpy as np
import tifffile as tiff
import os
import shutil
from pathlib import Path


def robust_normalize(data, lower_percent=0.5, upper_percent=99.5, hard_limit=None):
    """
    鲁棒归一化：
    1. 去除 NaN/Inf
    2. 根据百分比或固定阈值截断数据 (Clamping)
    3. 归一化到 0-255
    """
    # A. 处理非数字值
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # B. 计算截断阈值
    if hard_limit is not None:
        # 方案1: 如果你明确知道物理范围 (例如 -5mm 到 5mm)
        lower, upper = hard_limit[0], hard_limit[1]
    else:
        # 方案2: 自动统计 (推荐)，截去头尾的噪点
        # 注意：计算分位数时最好只统计非零区域(假设0是背景)，防止背景拉低统计值
        # 这里简化为全局统计，通常也有效
        lower = np.percentile(data, lower_percent)
        upper = np.percentile(data, upper_percent)
    
    # 打印调试信息，帮助你确认是否截断了正确的数据范围
    # print(f"Clamping range: {lower:.2f} to {upper:.2f}")

    # C. 截断 (Clamping) - 关键步骤！
    # 所有小于 lower 的变成 lower，大于 upper 的变成 upper
    data_clipped = np.clip(data, lower, upper)
    
    # D. 归一化
    denominator = upper - lower
    if denominator == 0:
        return np.zeros_like(data, dtype=np.uint8)
        
    norm_data = ((data_clipped - lower) / denominator * 255).astype(np.uint8)
    return norm_data


def convert_tiff_to_3channel(tiff_path, output_path):
    """
    增强版预处理：加入截断逻辑
    """
    # 1. 读取原始 3D 数据
    raw_data = tiff.imread(tiff_path)
    
    # --- Channel 1: 高度图 (带截断) ---
    # 如果你知道具体的物理范围，可以用 hard_limit=(-5.0, 5.0) 替代百分比
    norm_height = robust_normalize(raw_data, 
                                   lower_percent=0.5, 
                                   upper_percent=99.5)

    # --- Channel 2: 梯度图 (Sobel) ---
    # 技巧：先截断再算梯度，还是先算梯度再截断？
    # 建议：直接对截断后的 float 数据算梯度，这样可以消除飞点带来的虚假强边缘
    
    # 先做一次 float 级别的 clip，保留精度用于算梯度
    lower = np.percentile(raw_data, 0.5)
    upper = np.percentile(raw_data, 99.5)
    raw_clipped = np.clip(raw_data, lower, upper)

    sobelx = cv2.Sobel(raw_clipped, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(raw_clipped, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    
    # 梯度图也需要截断归一化，因为边缘处的值可能会极大
    norm_grad = robust_normalize(gradient_mag, lower_percent=0, upper_percent=98.0) # 梯度通常只看大值，下限设0

    # --- Channel 3: CLAHE (纹理增强) ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_texture = clahe.apply(norm_height)

    # --- 合并与保存 ---
    merged_img = cv2.merge([norm_height, norm_grad, enhanced_texture])
    cv2.imwrite(output_path, merged_img)
    


# --- 批量处理示例 ---
if __name__ == "__main__":
    # 假设你的原始tiff在 'raw_tiffs' 文件夹，处理后存入 'dataset/images/train'
    input_dir = Path("/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/密封钉缺陷图片/collection")
    output_dir = Path("/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/密封钉缺陷图片/preprocessed")
    tiff_save = Path("/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/密封钉缺陷图片/rename")
    output_dir.mkdir(parents=True, exist_ok=True)
    id = 0
    for file in input_dir.glob("*.tif*"):
        save_name = output_dir / (str(id)+ ".png")
        #copy the file to destination
        save_tiff = tiff_save / (str(id)+ ".tif")
        shutil.copy(file, save_tiff)
        convert_tiff_to_3channel(str(file), str(save_name))
        id += 1