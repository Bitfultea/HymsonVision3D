import os
import numpy as np
from PIL import Image

def gen_txt_list(input_folder, output_file):
    txt_file = os.path.join(output_file, "list.txt")
    with open(txt_file, 'w') as f:
        for file in os.listdir(input_folder):
            if file.lower().endswith('.npy'):
                filenamewithext = os.path.splitext(file)[0]
                f.write(f"{filenamewithext}\n")

    print(f"TXT文件已生成：{txt_file}")

if __name__ == "__main__":
    input_folder = "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/分类缺陷/openpcdet_dataset/input"
    output_file = "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/分类缺陷/openpcdet_dataset/"
    gen_txt_list(input_folder, output_file)