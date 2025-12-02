import os
import numpy as np
import open3d as o3d


def generate_bbox(points):
    # bbox = o3d.geometry.OrientedBoundingBox.CreateFromPoints(points)
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    center = (min_bound + max_bound) / 2.0
    dimensions = max_bound - min_bound
                    
    x, y, z = center
    dx, dy, dz = dimensions
    
    # 对于轴对齐包围框，heading_angle为0
    heading_angle = 0.0
    
    # f.write(f"{x:.6f} {y:.6f} {z:.6f} {dx:.6f} {dy:.6f} {dz:.6f} {heading_angle:.6f} {category_name}\n")
    return np.array([x, y, z, dx, dy, dz, heading_angle])

def process_label_ply(label_input_folder, 
                      input_folder,
                      npy_folder, 
                      output_folder=None,
                      fea_dims = 3):
    for filename in os.listdir(label_input_folder):
        if not filename.endswith(".ply"):
            continue
        
        input_path = os.path.join(label_input_folder, filename)
        filenamewithext = os.path.splitext(filename)[0]
        label = filenamewithext.split('_')[0]
        id = filenamewithext.split('_')[1]
        try:
            pcd = o3d.io.read_point_cloud(input_path)
            for raw_ply in os.listdir(input_folder):
                if raw_ply.endswith(".ply"):
                    raw_file = os.path.splitext(raw_ply)[0]
                    id_raw = raw_file.split('_')[1]
                    if id_raw == id:
                        # print(f"raw_ply: {raw_ply}")
                        raw_ply_path = os.path.join(input_folder, raw_ply)
                        raw_ply_pcd = o3d.io.read_point_cloud(raw_ply_path)
                        if(fea_dims == 3):
                            raw_points = np.asarray(raw_ply_pcd.points)
                        else:
                            # add fake feature col to raw_points to make the feature 4 dims
                            pts = np.asarray(raw_ply_pcd.points)
                            intensity_values = np.ones((pts.shape[0], 1),dtype=np.float32)
                            print("intensity_values shape: ", intensity_values.shape)
                            print("pts shape: ", pts.shape)
                            raw_points = np.hstack([pts, intensity_values])
                        print("pointcloiud shape: ", raw_points.shape)
                        np.save(os.path.join(npy_folder, f"{id}.npy"), raw_points)
            
            # 获取点云数据
            points = np.asarray(pcd.points)
            bbox = generate_bbox(points)
            target_txt = os.path.join(output_folder, f"{id}.txt")
            with open(target_txt, 'a') as f:
                x, y, z, dx, dy, dz, heading_angle = bbox
                # 写入TXT文件
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {dx:.6f} {dy:.6f} {dz:.6f} {heading_angle:.6f} {label}\n")
                
                print(f"已处理: {filename} -> 中心: [{x:.3f}, {y:.3f}, {z:.3f}], 尺寸: [{dx:.3f}, {dy:.3f}, {dz:.3f}], 类别: {label}")
            
        except Exception as e:
            print(f"转换文件 {filename} 时出错: {str(e)}")

def process_txt_label(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    txt_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith('.txt'):
            txt_files.append(file)
    
    label_name = ['Pinhole', 'Spatter','FishScales']
    for idx, filename in enumerate(txt_files):
        try:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            with open(input_path, 'r') as infile:
                lines = infile.readlines()
            
            with open(output_path, 'w') as outfile:
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        try:
                            label_index = int(parts[-1])
                            if(label_index == 2): 
                                label_index = 1
                            if 0 <= label_index < len(label_name):
                                parts[-1] = label_name[label_index]
                            else:
                                print(f"警告: 文件 {filename} 中发现无效标签索引 {label_index}")
                        except ValueError:
                            # 如果最后一个字段不是数字，保持原样
                            pass
                    
                    # 写入处理后的行
                    outfile.write(' '.join(parts) + '\n')
            
            print(f"已处理标签文件: {filename}")
                    
                        
            
        except Exception as e:
            print(f"转换文件 {filename} 时出错: {str(e)}")

if __name__ == "__main__":
    # # 方法1: 处理单个文件夹
    # label_input_folder = "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/分类缺陷/label"
    # input_folder = "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/分类缺陷/ply"
    # npy_folder  = "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/分类缺陷/openpcdet_dataset/input"
    # output_file = "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/分类缺陷/openpcdet_dataset/label"
    # # category_name = "seal_defect"
    # fea_dims = 4
    
    # process_label_ply(label_input_folder, input_folder, npy_folder,output_file, fea_dims)
    
    input_dir = "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/2025-10/ply_more_labels"
    
    ouut_dir = "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/2025-10/label"
    
    process_txt_label(input_dir, ouut_dir)
    
