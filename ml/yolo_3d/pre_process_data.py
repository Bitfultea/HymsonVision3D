import cv2
import numpy as np
import tifffile as tiff
import os
import shutil
from pathlib import Path


def read_tiff_height_intensity(tiff_path):
    """读取 HALCON 双通道 TIFF：主 IFD 为高度，SubIFD 为强度。"""
    with tiff.TiffFile(tiff_path) as tif:
        height = tif.pages[0].asarray()
        intensity = None
        subifds = getattr(tif.pages[0], "pages", None)
        if subifds is not None and len(subifds) > 0:
            intensity = subifds[0].asarray()
    return height, intensity


def write_tiff_height_intensity(output_path, height, intensity=None):
    """写出高度 TIFF；存在强度图时保留为 SubIFD，便于 HALCON decompose2。"""
    if intensity is None:
        tiff.imwrite(output_path, height)
        return

    if height.shape[:2] != intensity.shape[:2]:
        raise ValueError(
            f"height/intensity size mismatch: {height.shape} vs {intensity.shape}"
        )

    with tiff.TiffWriter(output_path) as tif:
        tif.write(
            height,
            photometric="minisblack",
            metadata=None,
            subifds=1,
        )
        tif.write(
            intensity,
            photometric="minisblack",
            metadata=None,
            subfiletype=1,
        )


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


def signed_robust_normalize(data, percentile=99.0, hard_limit=None):
    """将有正负方向的特征映射到 uint8，0 映射到 128。"""
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if hard_limit is None:
        limit = np.percentile(np.abs(data), percentile)
    else:
        limit = float(hard_limit)

    if limit <= 0:
        return np.full_like(data, 128, dtype=np.uint8)

    clipped = np.clip(data, -limit, limit)
    return ((clipped + limit) / (2 * limit) * 255).astype(np.uint8)


def preprocess_height_for_features(height, lower_percent=0.5, upper_percent=99.5):
    """清理高度图并截断飞点，保留 float 精度用于特征计算。"""
    height = np.asarray(height, dtype=np.float32)
    height = np.nan_to_num(height, nan=0.0, posinf=0.0, neginf=0.0)
    lower = np.percentile(height, lower_percent)
    upper = np.percentile(height, upper_percent)
    if upper <= lower:
        return height
    return np.clip(height, lower, upper)


def local_roughness(data, ksize=7):
    """用局部标准差近似粗糙度，对微小凹凸和纹理扰动敏感。"""
    if ksize % 2 == 0:
        ksize += 1
    data = np.asarray(data, dtype=np.float32)
    mean = cv2.blur(data, (ksize, ksize))
    mean_sq = cv2.blur(data * data, (ksize, ksize))
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(variance)


def height_defect_feature_channels(height,
                                   baseline_sigma=15,
                                   clip_percent=(0.5, 99.5),
                                   residual_percentile=99.0,
                                   roughness_ksize=7):
    """生成 2.5D 几何缺陷特征：residual、gradient、roughness。"""
    height = preprocess_height_for_features(height, *clip_percent)
    baseline = cv2.GaussianBlur(height, (0, 0), baseline_sigma)
    residual = height - baseline

    sobel_x = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(sobel_x, sobel_y)
    roughness = local_roughness(residual, roughness_ksize)

    residual_u8 = signed_robust_normalize(residual, residual_percentile)
    gradient_u8 = robust_normalize(gradient, 0, 99.0)
    roughness_u8 = robust_normalize(roughness, 0, 99.0)
    return residual_u8, gradient_u8, roughness_u8


def convert_tiff_to_3channel_defect_features(tiff_path,
                                             output_path,
                                             baseline_sigma=15,
                                             clip_percent=(0.5, 99.5),
                                             residual_percentile=99.0,
                                             roughness_ksize=7):
    """
    面向 YOLO 训练的纯几何 2.5D 预处理。

    B: 局部去背景后的 signed residual
    G: 梯度幅值
    R: 局部粗糙度/曲率近似
    """
    height, _ = read_tiff_height_intensity(tiff_path)
    residual_u8, gradient_u8, roughness_u8 = height_defect_feature_channels(
        height,
        baseline_sigma=baseline_sigma,
        clip_percent=clip_percent,
        residual_percentile=residual_percentile,
        roughness_ksize=roughness_ksize,
    )
    merged_img = cv2.merge([residual_u8, gradient_u8, roughness_u8])
    cv2.imwrite(output_path, merged_img)


def convert_tiff_to_3channel_intensity_features(tiff_path,
                                                output_path,
                                                third_channel="gradient",
                                                baseline_sigma=15,
                                                clip_percent=(0.5, 99.5),
                                                residual_percentile=99.0,
                                                roughness_ksize=7):
    """
    面向 YOLO 训练的高度 + 强度融合预处理。

    B: 局部去背景后的 signed residual
    G: 从 SubIFD 分离出的 intensity
    R: gradient 或 roughness，由 third_channel 指定
    """
    height, intensity = read_tiff_height_intensity(tiff_path)
    if intensity is None:
        raise ValueError(f"{tiff_path} does not contain SubIFD intensity")
    if height.shape[:2] != intensity.shape[:2]:
        raise ValueError(
            f"height/intensity size mismatch: {height.shape} vs {intensity.shape}"
        )

    residual_u8, gradient_u8, roughness_u8 = height_defect_feature_channels(
        height,
        baseline_sigma=baseline_sigma,
        clip_percent=clip_percent,
        residual_percentile=residual_percentile,
        roughness_ksize=roughness_ksize,
    )
    intensity_u8 = robust_normalize(intensity, 0.5, 99.5)

    if third_channel == "gradient":
        third_u8 = gradient_u8
    elif third_channel == "roughness":
        third_u8 = roughness_u8
    else:
        raise ValueError("third_channel must be 'gradient' or 'roughness'")

    merged_img = cv2.merge([residual_u8, intensity_u8, third_u8])
    cv2.imwrite(output_path, merged_img)


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
    

def convert_tiff_to_3channel_convex(tiff_path, output_path):
    """
    新版预处理：保留凹凸方向信息
    Channel 0: 高度图
    Channel 1: Sobel X (水平梯度)
    Channel 2: Sobel Y (垂直梯度)
    """
    # 1. 读取原始数据
    raw_data = tiff.imread(tiff_path)
    raw_data = np.nan_to_num(raw_data, nan=0.0)

    # 2. 预先截断 (去除飞点对梯度的干扰)
    # 如果不先截断，一个飞点会导致整张图的梯度极小
    lower_limit = np.percentile(raw_data, 0.1)
    upper_limit = np.percentile(raw_data, 99.9)
    raw_clamped = np.clip(raw_data, lower_limit, upper_limit)

    # --- Channel 1: 基础高度图 ---
    norm_height = robust_normalize(raw_clamped, 0, 100) # 既然已经clip过了，直接归一化

    # --- Channel 2: Sobel X (水平方向) ---
    # dx > 0 表示上坡，dx < 0 表示下坡
    # 结果包含正负小数
    sobel_x = cv2.Sobel(raw_clamped, cv2.CV_64F, 1, 0, ksize=3)
    # cv2.imwrite("sobel_x.png", sobel_x)
    
    # 关键点：如何把正负值映射到 0-255？
    # 直接使用 robust_normalize 会自动把最负的值映射为0，最正的映射为255
    # 平坦区域大约会落在 128 左右（灰色）
    norm_sobel_x = robust_normalize(sobel_x, 0.1, 99.9)

    # --- Channel 3: Sobel Y (垂直方向) ---
    sobel_y = cv2.Sobel(raw_clamped, cv2.CV_64F, 0, 1, ksize=3)
    norm_sobel_y = robust_normalize(sobel_y, 0.1, 99.9)

    # --- 合并与保存 ---
    # 这种组合下，原本"看起来一样"的圆环，现在会变成"一边亮一边暗"的立体球感
    merged_img = cv2.merge([norm_height, norm_sobel_x, norm_sobel_y])
    
    cv2.imwrite(output_path, merged_img)
    
def convert_tiff_to_3channel_convex_new(tiff_path, output_path):
    """
    新版预处理：保留凹凸方向信息
    Channel 0: 高度图
    Channel 1: Sobel X (水平梯度)
    Channel 3: CLAHE (纹理增强)
    """
    # 1. 读取原始数据
    raw_data = tiff.imread(tiff_path)
    raw_data = np.nan_to_num(raw_data, nan=0.0)

    # 2. 预先截断 (去除飞点对梯度的干扰)
    # 如果不先截断，一个飞点会导致整张图的梯度极小
    lower_limit = np.percentile(raw_data, 0.1)
    upper_limit = np.percentile(raw_data, 99.9)
    raw_clamped = np.clip(raw_data, lower_limit, upper_limit)

    # --- Channel 1: 基础高度图 ---
    norm_height = robust_normalize(raw_clamped, 0, 100) # 既然已经clip过了，直接归一化

    # --- Channel 2: Sobel X (水平方向) ---
    # dx > 0 表示上坡，dx < 0 表示下坡
    # 结果包含正负小数
    sobel_x = cv2.Sobel(raw_clamped, cv2.CV_64F, 1, 0, ksize=3)
    norm_sobel_x = robust_normalize(sobel_x, 0.1, 99.9)

    # --- Channel 3: Sobel Y (垂直方向) ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_texture = clahe.apply(norm_height)

    # --- 合并与保存 ---
    # 这种组合下，原本"看起来一样"的圆环，现在会变成"一边亮一边暗"的立体球感
    merged_img = cv2.merge([norm_height, norm_sobel_x, enhanced_texture])
    
    cv2.imwrite(output_path, merged_img)
    # print(f"Processed: {output_path}")

def convert_tiff_to_3channel_gradlimit(tiff_path, output_path):
    # 1. 读取数据
    raw_data = tiff.imread(tiff_path)
    raw_data = np.nan_to_num(raw_data, nan=0.0)

    # 2. 预处理：高度图 (Channel 1 - B)
    # 高度图依然使用之前的百分比截断，因为高度是相对的
    # lower_h = np.percentile(raw_data, 0.1)
    # upper_h = np.percentile(raw_data, 99.9)
    # raw_clamped = np.clip(raw_data, lower_h, upper_h)
    raw_clamped = raw_data
    min_pt = np.min(raw_data)
    max_pt = np.max(raw_data)
    
    denom = max_pt - min_pt
    if denom == 0: denom = 1
    norm_height = ((raw_clamped - min_pt) / denom * 255).astype(np.uint8)

    # 3. 关键修正：梯度图 (Channel 2 & 3 - G & R)
    # 我们不再动态统计 min/max，而是设置一个固定的物理阈值
    # 假设你的传感器数据单位，梯度超过一定数值（比如 5.0 或 10.0）就算陡峭边缘
    # 【重要】你需要根据实际数据调整这个 GRAD_LIMIT
    # 如果生成的图片看起来全是灰的，说明 LIMIT 太大了，改小点（如 2.0）
    # 如果生成的图片全是黑白的（对比度过饱和），说明 LIMIT 太小了，改大点（如 20.0）
    GRAD_LIMIT = 0.75

    # 计算梯度
    sobel_x = cv2.Sobel(raw_data, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(raw_data, cv2.CV_64F, 0, 1, ksize=3)

    def fixed_normalize(grad_data, limit):
        # 强制截断到 [-limit, +limit]
        # print("grad_data", grad_data)
        clipped = np.clip(grad_data, -limit, limit)
        # 线性映射：-limit -> 0, 0 -> 127.5, +limit -> 255
        norm = ((clipped + limit) / (2 * limit) * 255).astype(np.uint8)
        return norm

    norm_sobel_x = fixed_normalize(sobel_x, GRAD_LIMIT)
    norm_sobel_y = fixed_normalize(sobel_y, GRAD_LIMIT)

    # 4. 合并保存
    merged_img = cv2.merge([norm_height, norm_sobel_x, norm_sobel_y])
    cv2.imwrite(output_path, merged_img)


def normalize_to_rgb(vector_channel):
    """ 将 [-1, 1] 范围的法线分量映射到 [0, 255] 的 uint8 """
    # 公式: output = (input + 1) / 2 * 255
    normalized = ((vector_channel + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
    return normalized

def calculate_normal_map(height_data, grad_step=1.0):
    """ 从高度图计算法线贴图 """
    
    # 1. 计算 X 和 Y 方向的梯度 (使用 Sobel 算子也可以，但这里直接用 diff)
    # np.gradient 比 Sobel 更适合处理连续高度场
    dz_dx, dz_dy = np.gradient(height_data, grad_step, grad_step)
    
    # 2. 构建法线向量 N = (-dz/dx, -dz/dy, 1)
    N_x = -dz_dx
    N_y = -dz_dy
    N_z = 1.0  # Z分量固定为 1
    
    # 3. 归一化为单位向量 (计算法线长度，并除以长度)
    length = np.sqrt(N_x**2 + N_y**2 + N_z**2)
    
    unit_N_x = N_x / length
    unit_N_y = N_y / length
    unit_N_z = N_z / length
    
    # 4. 映射到 RGB 颜色空间 (R, G 通道对应 N_x, N_y)
    # N_x, N_y 范围都在 [-1, 1]
    R_channel = normalize_to_rgb(unit_N_x)
    G_channel = normalize_to_rgb(unit_N_y)
    B_channel = normalize_to_rgb(unit_N_z) # 理论上 N_z 接近 1，所以 B 通道会很亮

    # 5. 返回 N_x 和 N_y 作为两个特征通道
    return unit_N_x, unit_N_y

def convert_tiff_to_3channel_normal(tiff_path, output_path):
   # 1. 读取数据
    raw_data = tiff.imread(tiff_path)
    raw_data = np.nan_to_num(raw_data, nan=0.0)

    # 2. 预处理：高度图 (Channel 1 - B)
    # 高度图依然使用之前的百分比截断，因为高度是相对的
    lower_h = np.percentile(raw_data, 0.1)
    upper_h = np.percentile(raw_data, 99.9)
    raw_clamped = np.clip(raw_data, lower_h, upper_h)
    
    denom = upper_h - lower_h
    if denom == 0: denom = 1
    norm_height = ((raw_clamped - lower_h) / denom * 255).astype(np.uint8)

    # 计算法线分量
    unit_N_x, unit_N_y = calculate_normal_map(raw_data)
    
    # 将法线分量映射到 [0, 255]
    norm_Nx = normalize_to_rgb(unit_N_x)
    norm_Ny = normalize_to_rgb(unit_N_y)
    
    # 合并： [高度, Nx, Ny]
    merged_img = cv2.merge([norm_height, norm_Nx, norm_Ny]) # BGR 顺序
    cv2.imwrite(output_path, merged_img)

def crop_tiff(tiff_path, output_path, x, y, w, h):
    """裁切 TIFF 高度图的指定区域

    Args:
        tiff_path:  输入 TIFF 文件路径
        output_path: 输出 TIFF 文件路径
        x, y: 裁切区域左上角像素坐标
        w, h: 裁切区域的宽、高（像素）
    """
    data = tiff.imread(tiff_path)
    cropped = data[y:y+h, x:x+w]
    tiff.imwrite(output_path, cropped)
    print(f"Cropped {data.shape} -> {cropped.shape}, saved to {output_path}")


def crop_tiff_center(tiff_path, output_path, cx, cy, w, h):
    """以中心点裁切 TIFF

    Args:
        tiff_path:  输入 TIFF 文件路径
        output_path: 输出 TIFF 文件路径
        cx, cy: 裁切中心点像素坐标
        w, h: 裁切区域的宽、高（像素）
    """
    x = cx - w // 2
    y = cy - h // 2
    x = max(0, x)
    y = max(0, y)
    crop_tiff(tiff_path, output_path, x, y, w, h)


def batch_crop_tiff(input_dir, output_dir, x, y, w, h):
    """批量裁切文件夹中所有 TIFF 的同一区域

    Args:
        input_dir:  输入文件夹路径
        output_dir: 输出文件夹路径
        x, y, w, h: 裁切区域参数（同 crop_tiff）
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tiff_files = sorted(input_dir.glob("*.tif*"))
    if not tiff_files:
        print(f"文件夹 {input_dir} 中没有 TIFF 文件")
        return

    for f in tiff_files:
        out_path = output_dir / f.name
        crop_tiff(str(f), str(out_path), x, y, w, h)


def batch_strip_top_rows(input_dir, output_dir, n=5):
    """批量移除每个 TIFF 最上方的 n 行数据，并保留 SubIFD 强度图。

    Args:
        input_dir:  输入文件夹路径
        output_dir: 输出文件夹路径
        n: 要移除的行数（默认 5）
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tiff_files = sorted(input_dir.glob("*.tif*"))
    if not tiff_files:
        print(f"文件夹 {input_dir} 中没有 TIFF 文件")
        return

    for f in tiff_files:
        height, intensity = read_tiff_height_intensity(str(f))
        h, w = height.shape[:2]
        if n < 0 or n >= h:
            raise ValueError(f"n must be in [0, {h}), got {n}")

        cropped = height[n:h, 0:w]
        cropped_intensity = None
        if intensity is not None:
            ih, iw = intensity.shape[:2]
            if (ih, iw) != (h, w):
                raise ValueError(
                    f"{f.name}: height/intensity size mismatch "
                    f"{height.shape} vs {intensity.shape}"
                )
            cropped_intensity = intensity[n:ih, 0:iw]

        out_path = output_dir / f.name
        write_tiff_height_intensity(str(out_path), cropped, cropped_intensity)

        if cropped_intensity is None:
            print(f"{f.name}: height {height.shape} -> {cropped.shape}")
        else:
            print(
                f"{f.name}: height {height.shape} -> {cropped.shape}, "
                f"intensity {intensity.shape} -> {cropped_intensity.shape}"
            )


# --- 批量处理示例 ---
if __name__ == "__main__":
    # 假设你的原始tiff在 'raw_tiffs' 文件夹，处理后存入 'dataset/images/train'
    # input_dir = Path("/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/密封钉缺陷图片/collection")
    # output_dir = Path("/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/密封钉缺陷图片/preprocessed")
    
    batch_strip_top_rows("/home/charles/Data/Dataset/Collected/针孔3D/3D_rename",
                         "/home/charles/Data/Dataset/Collected/针孔3D/3D_rename_crop")
    
    input_dir = Path("/home/charles/Data/Dataset/Collected/针孔3D/3D_rename_crop")
    output_dir = Path("/home/charles/Data/Dataset/Collected/针孔3D/3d_preproces")
    gray_image_dir = Path("/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/yolo_3d/GrayImages")

    tiff_save = Path("/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/密封钉缺陷图片/rename")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # id = 0
    for file in input_dir.glob("*.tif*"):
        file_name = os.path.splitext(file)
        id = file_name[0].split("/")[-1]
        # print(id)
        save_name = output_dir / (str(id)+ ".png")
        #copy the file to destination
        # save_tiff = tiff_save / (str(id)+ ".tif")
        # shutil.copy(file, save_tiff)
        
        # convert_tiff_to_3channel(str(file), str(save_name))
        # convert_tiff_to_3channel_convex(str(file), str(save_name))
        # convert_tiff_to_3channel_convex_new(str(file), str(save_name))
        # convert_tiff_to_3channel_normal(str(file), str(save_name))

        convert_tiff_to_3channel_defect_features(str(file),str(save_name))
        # id += 1
