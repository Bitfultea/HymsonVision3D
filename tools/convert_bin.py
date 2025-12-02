import os
import numpy as np

def convert_npy_to_bin(input_folder, output_folder=None):
    """
    读取[nx4]的点云npy文件，并转换成bin文件
    
    参数:
    input_folder (str): 包含.npy文件的输入文件夹路径
    output_folder (str): 输出.bin文件的文件夹路径，如果为None则在输入文件夹下创建bin子文件夹
    """
    
    # 如果未指定输出文件夹，则在输入文件夹下创建bin子文件夹
    if output_folder is None:
        output_folder = os.path.join(input_folder, 'bin')
    
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有.npy文件
    npy_files = []
    for file in os.listdir(input_folder):
        if file.lower().endswith('.npy'):
            npy_files.append(file)
    
    # 按文件名字典序排序
    npy_files.sort()
    
    # 转换每个.npy文件为.bin格式
    for filename in npy_files:
        try:
            # 构建完整的输入文件路径
            input_path = os.path.join(input_folder, filename)
            
            # 读取.npy文件
            points = np.load(input_path)
            
            # 检查数据维度
            if points.ndim != 2 or points.shape[1] != 4:
                print(f"警告: 文件 {filename} 的形状 {points.shape} 不符合 [n, 4] 格式，跳过处理")
                continue
            
            # 生成输出文件名
            name_without_ext = os.path.splitext(filename)[0]
            output_filename = f"{name_without_ext}.bin"
            output_path = os.path.join(output_folder, output_filename)
            
            # 以二进制格式保存点云数据
            points.astype(np.float32).tofile(output_path)
            
            print(f"已转换: {filename} -> {output_filename} (点数: {points.shape[0]})")
            
        except Exception as e:
            print(f"转换文件 {filename} 时出错: {str(e)}")
    
    print(f"转换完成！共处理 {len(npy_files)} 个文件")


def convert_single_npy_to_bin(input_file, output_file):
    """
    转换单个.npy文件为.bin文件
    
    参数:
    input_file (str): 输入.npy文件路径
    output_file (str): 输出.bin文件路径
    """
    try:
        # 读取.npy文件
        points = np.load(input_file)
        
        # 检查数据维度
        print("points shape: ", points.shape)
        if points.ndim != 2 or points.shape[1] != 4:
            print(f"错误: 文件 {input_file} 的形状 {points.shape} 不符合 [n, 4] 格式")
            return False
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 以二进制格式保存点云数据
        points.astype(np.float32).tofile(output_file)
        
        print(f"已转换: {input_file} -> {output_file} (点数: {points.shape[0]})")
        return True
        
    except Exception as e:
        print(f"转换文件 {input_file} 时出错: {str(e)}")
        return False


def read_bin_file(bin_file):
    """
    读取.bin文件并返回点云数据（用于验证）
    
    参数:
    bin_file (str): .bin文件路径
    
    返回:
    numpy.ndarray: 点云数据 [n, 4]
    """
    try:
        # 读取二进制文件
        points = np.fromfile(bin_file, dtype=np.float32)
        
        # 重塑为 [n, 4] 形状
        points = points.reshape(-1, 4)
        
        print(f"成功读取 {bin_file}，点数: {points.shape[0]}")
        return points
        
    except Exception as e:
        print(f"读取文件 {bin_file} 时出错: {str(e)}")
        return None


# 使用示例
if __name__ == "__main__":
    # 批量转换文件夹中的所有.npy文件
    # input_folder = "/path/to/your/npy/files"  # 替换为实际路径
    # output_folder = "/path/to/your/bin/files"  # 替换为实际路径
    
    # 调用函数进行批量转换
    # convert_npy_to_bin(input_folder, output_folder)
    
    # 或者转换单个文件
    input_file = "/home/charles/Data/Repo/OpenPCDet/data/custom/points/2.npy"
    output_file = "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/分类缺陷/tt_3.bin"
    convert_single_npy_to_bin(input_file, output_file)
    
    # 验证转换结果
    # bin_file = "/path/to/your/file.bin"
    points = read_bin_file(output_file)
    if points is not None:
        print(f"点云数据形状: {points.shape}")
        print(f"前5个点:\n{points[:5]}")