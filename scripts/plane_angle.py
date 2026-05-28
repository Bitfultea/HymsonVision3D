#!/usr/bin/env python3
"""计算两个平面的相对角度"""
import numpy as np
import math

def angle_between_planes(normal1, normal2):
    """计算两个平面的夹角（度）"""
    n1 = np.array(normal1, dtype=float)
    n2 = np.array(normal2, dtype=float)

    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)

    if n1_norm < 1e-10 or n2_norm < 1e-10:
        print("ERROR: 法向量模为0")
        return None

    cos_angle = np.dot(n1, n2) / (n1_norm * n2_norm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.abs(np.arccos(cos_angle))
    angle_deg = angle_rad * (180.0 / math.pi)
    return angle_deg

def orient_towards_positive_z(normal):
    """使法向量朝向正Z方向"""
    n = np.array(normal, dtype=float)
    z_ref = np.array([0.0, 0.0, 1.0])
    if np.dot(n, z_ref) < 0.0:
        n = -n
    return n

if __name__ == "__main__":
    # 你的数据
    central_normal = [0.0089174, -0.107703, 0.990176]
    bottom_normal  = [0.005004,  0.019015,  0.999807]

    # Orient both normals towards positive z
    central_oriented = orient_towards_positive_z(central_normal)
    bottom_oriented = orient_towards_positive_z(bottom_normal)

    print("=== 原始法向量 ===")
    print(f"Central: {central_normal}")
    print(f"Bottom:  {bottom_normal}")

    print("\n=== orient 后的法向量 ===")
    print(f"Central: {central_oriented}")
    print(f"Bottom:  {bottom_oriented}")

    print("\n=== orient 后的 z 分量 ===")
    print(f"Central z: {central_oriented[2]:.6f}  (朝上? {central_oriented[2] > 0})")
    print(f"Bottom z:  {bottom_oriented[2]:.6f}  (朝上? {bottom_oriented[2] > 0})")

    angle = angle_between_planes(central_oriented, bottom_oriented)

    print("\n=== 角度计算 ===")
    n1 = central_oriented
    n2 = bottom_oriented
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    dot = np.dot(n1, n2)
    cos_val = dot / (n1_norm * n2_norm)
    print(f"dot product:     {dot:.10f}")
    print(f"|central|:       {n1_norm:.10f}")
    print(f"|bottom|:        {n2_norm:.10f}")
    print(f"cos(angle):      {cos_val:.10f}")
    print(f"calculated angle: {angle:.5f} degrees")
