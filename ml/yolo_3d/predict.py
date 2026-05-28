from ultralytics import YOLO
import cv2
import pre_process_data
import numpy as np
import os
from pathlib import Path

model = YOLO('runs/segment/defect_seg_v14/weights/best.pt')


def detect_single(tiff_path):
    """单张检测，显示结果窗口"""
    temp_png_path = "temp_inference.png"
    pre_process_data.convert_tiff_to_3channel(tiff_path, temp_png_path)
    img = cv2.imread(temp_png_path)
    results = model.predict(img, imgsz=800, conf=0.2, save=False)
    result = results[0]

    if result.masks is not None:
        print(f"检测到 {len(result.masks)} 个缺陷")
        annotated_frame = result.plot()
    else:
        print("未检测到缺陷 (OK)")
        annotated_frame = img

    cv2.imshow("Defect Detection", annotated_frame)
    while True:
        key = cv2.waitKey(100)
        if cv2.getWindowProperty("Defect Detection", cv2.WND_PROP_VISIBLE) < 1:
            break
        if key != -1 and key != 255:
            break
    cv2.destroyAllWindows()


def detect_folder(folder_path):
    """批量检测文件夹中所有 tiff 文件，键盘控制翻页

    Controls:
      n / →  — 下一张
      p / ←  — 上一张
      q / Esc — 退出
    """
    tiff_files = sorted(Path(folder_path).glob("*.tif*"))
    if not tiff_files:
        print(f"文件夹 {folder_path} 中没有找到 .tiff/.tif 文件")
        return

    total = len(tiff_files)
    print(f"找到 {total} 个文件")
    idx = 0

    while 0 <= idx < total:
        tiff_path = str(tiff_files[idx])
        filename = tiff_files[idx].name
        print(f"[{idx+1}/{total}] 正在检测: {filename}")

        temp_png_path = "temp_inference.png"
        pre_process_data.convert_tiff_to_3channel(tiff_path, temp_png_path)
        img = cv2.imread(temp_png_path)
        results = model.predict(img, imgsz=800, conf=0.2, save=False, verbose=False)
        result = results[0]

        if result.masks is not None:
            n_defects = len(result.masks)
            annotated = result.plot()
            status = f"Defects: {n_defects}"
        else:
            n_defects = 0
            annotated = img
            status = "OK (no defects)"

        # 叠加文件名和状态信息
        display = annotated.copy()
        h, w = display.shape[:2]
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)
        cv2.putText(display, f"File: {filename}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f"[{idx+1}/{total}]  {status}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if n_defects == 0 else (0, 0, 255), 2)
        cv2.putText(display, "n/->:next  p/<-:prev  q/ESC:quit", (10, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 2)

        cv2.imshow("Defect Detection", display)

        while True:
            raw_key = cv2.waitKey(100)
            if cv2.getWindowProperty("Defect Detection", cv2.WND_PROP_VISIBLE) < 1:
                idx = total
                break
            if raw_key == -1 or raw_key == 255:
                continue
            key = raw_key & 0xFF
            if key == ord('n') or key == 83:  # n 或 右箭头
                idx += 1
                break
            elif key == ord('p') or key == 81:  # p 或 左箭头
                idx -= 1
                break
            elif key == ord('q') or key == 27:  # q 或 ESC
                idx = total
                break
            else:
                # 其他任意键跳到下一张
                idx += 1
                break

    cv2.destroyAllWindows()
    print("检测结束")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Defect Detection")
    parser.add_argument("path", nargs="?", default=None,
                        help="TIFF 文件路径 (单张), 或文件夹路径 (批量)")
    args = parser.parse_args()

    if args.path is None:
        # 默认：批量检测文件夹
        folder = "/home/charles/Data/Dataset/Collected/针孔3D/3D"
        detect_folder(folder)
    elif os.path.isdir(args.path):
        detect_folder(args.path)
    else:
        detect_single(args.path)
