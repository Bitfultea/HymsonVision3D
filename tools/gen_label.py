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

def process_label_ply(label_input_folder, input_folder,npy_folder, output_folder=None):
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
                        raw_points = np.asarray(raw_ply_pcd.points)
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

if __name__ == "__main__":
    # 方法1: 处理单个文件夹
    label_input_folder = "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/分类缺陷/label"
    input_folder = "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/分类缺陷/ply"
    npy_folder  = "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/分类缺陷/openpcdet_dataset/input"
    output_file = "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/分类缺陷/openpcdet_dataset/label"
    # category_name = "seal_defect"
    
    process_label_ply(label_input_folder, input_folder, npy_folder,output_file)
    
