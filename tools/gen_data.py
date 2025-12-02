import os
import numpy as np
from PIL import Image
import re
import open3d as o3d

def convert_tiff_to_npy(input_folders, output_folder=None, fea_dim = 3):
    """
    读取文件夹下的TIFF文件转换成npy并且根据原来顺序重新命名
    
    Parameters:
    input_folder (str): 包含TIFF文件的输入文件夹路径
    output_folder (str): 输出NPY文件的文件夹路径，如果为None则在输入文件夹下创建npy文件夹
    """
    pre_idx = 0
    class_id = 0
    # 0-> 爆点凸起 1-> 翘钉 2-> 针孔凹坑
    for input_folder in input_folders:
        print(f"处理文件夹: {input_folder}")
        
        # 如果未指定输出文件夹，则在输入文件夹下创建npy子文件夹
        if output_folder is None:
            output_folder = os.path.join(input_folder, 'npy')
        
        # 创建输出文件夹（如果不存在）
        os.makedirs(output_folder, exist_ok=True)
        
        # 获取所有TIFF文件
        tiff_files = []
        for file in os.listdir(input_folder):
            if file.lower().endswith(('.tiff', '.tif')):
                tiff_files.append(file)
        
        # 按文件名字典序排序
        tiff_files.sort()
        
        # 转换每个TIFF文件为NPY格式
        for idx, filename in enumerate(tiff_files):
            try:
                # 构建完整的输入文件路径
                input_path = os.path.join(input_folder, filename)
                
                # 读取TIFF文件
                img = Image.open(input_path)
                
                # 转换为numpy数组
                img_array = np.array(img)
                # img_aray 转换成[HxW,3]
                height, width = img_array.shape
                x_coords = np.arange(width)
                y_coords = np.arange(height)
                x_grid, y_grid = np.meshgrid(x_coords, y_coords)
                
                # 将网格展平
                x_flat = x_grid.flatten()
                y_flat = y_grid.flatten()
                z_flat = img_array.flatten()
                scale_x = 1
                scale_y = 1
                scale_z = 1
                x_points = x_flat* scale_x
                y_points = y_flat* scale_y
                z_points = z_flat* scale_z
                
                if(fea_dim == 4):
                    intensity_values = np.ones_like(x_flat, dtype=np.float32)  # 全为1.0的强度值
                    points = np.stack([x_points, y_points, z_points,intensity_values], axis=1)
                else:
                    points = np.stack([x_points, y_points, z_points], axis=1)
                    
                # 生成新的文件名（保持原有文件名前缀，只更改扩展名）
                name_without_ext = os.path.splitext(filename)[0]
                output_filename = f"{class_id}_{pre_idx+idx}.npy"
                output_path = os.path.join(output_folder, output_filename)
                
                # 保存为NPY格式
                np.save(output_path, points)
                
                print(f"已转换: {filename} -> {output_filename}")
                
            except Exception as e:
                print(f"转换文件 {filename} 时出错: {str(e)}")
        
        pre_idx += len(tiff_files)
        class_id += 1
    
        print(f"转换完成！共处理 {len(tiff_files)} 个文件")

def convert_npy_to_ply(input_folder, output_folder=None, fea_dims = 3):
    os.makedirs(output_folder, exist_ok=True)
    npy_files = []
    for file in os.listdir(input_folder):
        if file.lower().endswith('.npy'):
            npy_files.append(file)
            
    npy_files.sort()
    
    # 转换每个NPY文件为PLY格式
    for idx, filename in enumerate(npy_files):
        try:
            # 构建完整的输入文件路径
            input_path = os.path.join(input_folder, filename)
            
            # 读取NPY文件
            data = np.load(input_path)
            
            # 创建Open3D点云对象
            pcd = o3d.geometry.PointCloud()

            # height, width = data.shape
            # x_coords = np.arange(width)
            # y_coords = np.arange(height)
            # x_grid, y_grid = np.meshgrid(x_coords, y_coords)
            
            # # 将网格展平
            # x_flat = x_grid.flatten()
            # y_flat = y_grid.flatten()
            # z_flat = data.flatten()
            
            # # # 创建掩码过滤无效深度值（假设深度为0表示无效）
            # # valid_mask = z_flat > 0
            
            # # 应用缩放因子并重新组织坐标
            # scale_x = 1
            # scale_y = 1
            # scale_z = 1
            # # x_points = x_flat[valid_mask] * scale_x
            # # y_points = y_flat[valid_mask] * scale_y
            # # z_points = z_flat[valid_mask] * scale_z    
            # x_points = x_flat* scale_x
            # y_points = y_flat* scale_y
            # z_points = z_flat* scale_z
            
            # # 在正视投影中，通常需要翻转y轴（因为图像坐标系原点在左上角）
            # # y_points = (height - y_points) if scale_y > 0 else y_points
            
            # # 组合成点云 (Nx3)
            # points = np.stack([x_points, y_points, z_points], axis=1)
            if(fea_dims == 4):
                points = data[:,:3]
            else:
                points = data
            pcd.points = o3d.utility.Vector3dVector(points)
            
            name_without_ext = os.path.splitext(filename)[0]
            output_filename = f"{name_without_ext}.ply"
            output_path = os.path.join(output_folder, output_filename)
            
            # 保存为PLY格式
            o3d.io.write_point_cloud(output_path, pcd)
            
            
        except Exception as e:
            print(f"转换文件 {filename} 时出错: {str(e)}")

def convert_ply_to_npy(input_folder, output_folder=None, fea_dims = 3):
    os.makedirs(output_folder, exist_ok=True)
    ply_files = []
    for file in os.listdir(input_folder):
        if file.lower().endswith('.ply'):
            ply_files.append(file)

    for idx, filename in enumerate(ply_files):
        try:
            input_path = os.path.join(input_folder, filename)
            data = o3d.io.read_point_cloud(input_path)
            points = np.asarray(data.points)
            
            if(fea_dims == 4):
                points = points[:,:3]
            else:
                points = data
            # extend the 3dim points to 4dim [n,4]
            points = np.hstack((points, np.ones((points.shape[0], 1))))
            print(points.shape)
            
            name_without_ext = os.path.splitext(filename)[0]
            name = name_without_ext.split("_")[1]
            output_filename = f"{name}.npy"
            print(output_filename)
            output_path = os.path.join(output_folder, output_filename)
            np.save(output_path, points)
            
        except Exception as e:
            print(f"转换文件 {filename} 时出错: {str(e)}")


# 使用示例
if __name__ == "__main__":
    # # 指定包含TIFF文件的文件夹路径
    # input_folders = ["/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/分类缺陷/爆点凸起",
    #                  "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/分类缺陷/翘钉",
    #                  "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/分类缺陷/针孔凹坑"]
    # # 替换为实际路径
    # output_folder = "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/分类缺陷/npy"
    # fea_dims = 4
    
    # # 调用函数进行转换
    # convert_tiff_to_npy(input_folders,output_folder, fea_dims)
    # ply_folder = "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/分类缺陷/ply"
    # convert_npy_to_ply(output_folder,ply_folder, fea_dims)
    
    
    input_dir = "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/2025-10/ply_more"
    output_dir = "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/2025-10/npy"
    convert_ply_to_npy(input_dir, output_dir, 4)