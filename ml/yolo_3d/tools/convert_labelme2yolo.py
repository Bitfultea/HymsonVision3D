import json
import os
import glob
import numpy as np
from pathlib import Path

# ================= 配置区域 =================

# 1. 定义你的缺陷类别名称 (必须与 LabelMe 中的标签完全一致)
# 这里的顺序决定了 class_id (0, 1, 2...)
CLASSES = ["pinhole", "crap", "spatter"] 

# 2. 路径设置
# JSON 文件所在的文件夹
INPUT_JSON_DIR = "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/yolo_3d/labelme_label" 
# 输出 TXT 文件的文件夹
OUTPUT_TXT_DIR = "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/yolo_3d/yolo_label" 

# ===========================================

def clip_value(val):
    """将坐标限制在 0.0 到 1.0 之间，防止越界"""
    return max(0.0, min(1.0, val))

def convert_labelme_json_to_yolo(json_dir, output_dir, classes):
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取类别映射字典: {"defect": 0, "scratch": 1}
    class_map = {name: i for i, name in enumerate(classes)}
    
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    print(f"Found {len(json_files)} JSON files in {json_dir}")

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 获取图像尺寸
        img_h = data.get("imageHeight")
        img_w = data.get("imageWidth")
        
        # 安全检查：防止 LabelMe 保存时没写入宽高
        if not img_h or not img_w:
            print(f"[Warning] Skipping {json_file}: Missing image size.")
            continue

        filename = Path(json_file).stem
        txt_path = os.path.join(output_dir, filename + ".txt")

        yolo_lines = []
        
        for shape in data["shapes"]:
            label = shape["label"]
            points = shape["points"] # List of [x, y]
            shape_type = shape["shape_type"]

            # 1. 检查类别是否在我们的列表中
            if label not in class_map:
                print(f"[Info] Skipping label '{label}' in {filename} (not in CLASSES list)")
                continue
            
            class_id = class_map[label]

            # 2. 检查形状类型 (必须是多边形 polygon)
            # 如果你用矩形框标注，这里会自动转为 4 个点的多边形，YOLO-Seg 也能吃
            if shape_type not in ["polygon", "rectangle"]:
                continue

            # 3. 归一化坐标并展平
            # 格式: class_id x1 y1 x2 y2 ... xn yn
            line_content = [str(class_id)]
            
            for pt in points:
                x, y = pt[0], pt[1]
                
                # 归一化 + 截断
                x_norm = clip_value(x / img_w)
                y_norm = clip_value(y / img_h)
                
                line_content.append(f"{x_norm:.6f}")
                line_content.append(f"{y_norm:.6f}")

            # 将列表转为字符串
            yolo_lines.append(" ".join(line_content))

        # 只有当有有效标注时才保存 TXT
        if yolo_lines:
            with open(txt_path, "w") as f_out:
                f_out.write("\n".join(yolo_lines))
        else:
            # 如果是空图片（无缺陷），可以选择生成空 txt 或不生成
            # YOLO 官方建议：负样本（无缺陷图）也生成一个空的 txt 文件
            open(txt_path, "w").close() 

    print(f"Conversion completed! Saved to {output_dir}")

if __name__ == "__main__":
    convert_labelme_json_to_yolo(INPUT_JSON_DIR, OUTPUT_TXT_DIR, CLASSES)