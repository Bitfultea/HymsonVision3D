from ultralytics import YOLO
import cv2
import pre_process_data
import numpy as np

# 1. 加载训练好的最佳权重
model = YOLO('runs/segment/defect_seg_v14/weights/best.pt')

def detect_defect(tiff_path):
    # A. 预处理：必须和训练时完全一致！
    # 我们不保存中间文件，直接在内存中处理
    temp_png_path = "temp_inference.png" 
    pre_process_data.convert_tiff_to_3channel(tiff_path, temp_png_path)
    
    # B. 读取处理后的图片
    img = cv2.imread(temp_png_path)
    
    # C. 推理
    # conf=0.4 表示置信度阈值，根据实际情况调整
    results = model.predict(img, imgsz=800, conf=0.2, save=False)
    
    # D. 解析结果
    result = results[0]
    
    if result.masks is not None:
        print(f"检测到 {len(result.masks)} 个缺陷")
        
        # 获取掩码 (Masks)
        masks = result.masks.data.cpu().numpy() # (N, H, W)
        # 获取边框 (Boxes)
        boxes = result.boxes.xyxy.cpu().numpy() # (N, 4)
        
        # 可视化展示
        annotated_frame = result.plot() # 画出结果
        cv2.imshow("Defect Detection", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("未检测到缺陷 (OK)")

if __name__ == "__main__":
    # tiff_file = "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/yolo_3d/rename_tiff/41.tif"
    tiff_file = "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/密封钉缺陷图片/├▄╖Γ╢ñ╚▒╧▌═╝╞¼/2025_11_23/CapTime_Dev7_2025_11_23_09_33_48_943_CurTime_09_33_49_318_0B5CBP2MBL0P8MFBK1009352.tiff"
    detect_defect(tiff_file)