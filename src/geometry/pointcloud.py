# -*- coding: utf-8 -*-
import numpy as np
import cv2
import logging
from sklearn.neighbors import NearestNeighbors # Assuming sklearn is available as it was in original main.py

from src.utils.helpers import logger
from src.config import CAMERA_CONFIG, SKLEARN_AVAILABLE # Access camera_matrix_l and SKLEARN_AVAILABLE


def statistical_outlier_removal(points, k=20, std_ratio=2.0):
    """
    统计离群点去除。
    points: Nx3 array
    k: 近邻点数量
    std_ratio: 均值距离标准差的倍数
    """
    if len(points) < k or not SKLEARN_AVAILABLE: # Assume SKLEARN_AVAILABLE is a global or imported constant
        return points

    try:
        nbrs = NearestNeighbors(n_neighbors=min(k, len(points) - 1), algorithm='ball_tree').fit(points)
        distances, indices = nbrs.kneighbors(points)
        mean_distances = np.mean(distances, axis=1)
        threshold = np.mean(mean_distances) + std_ratio * np.std(mean_distances)
        inlier_mask = mean_distances < threshold
        return points[inlier_mask]
    except Exception as e:
        logger.debug(f"统计离群点去除失败 (sklean): {e}")
        return points

def texture_mapping(point_cloud, frame, camera_matrix=None):
    """
    将图像的纹理映射到点云上，返回 (points, colors)
    - points: Nx3 数组（坐标，单位 cm）
    - colors: Nx3 数组（颜色，RGB，0~1）

    修改内容：
    1. 原函数返回 Nx6 (XYZ+RGB)，改为分开返回 points 和 colors。
    2. 保证 points 始终 Nx3，避免下游函数崩溃。
    3. 如果像素越界或无效点，用默认灰色填充。
    """

    if point_cloud is None or len(point_cloud) == 0:
        return np.array([]).reshape(-1, 3), np.array([]).reshape(-1, 3)

    if camera_matrix is None:
        camera_matrix = CAMERA_CONFIG.camera_matrix_l

    frame_h, frame_w = frame.shape[:2]

    pts = np.asarray(point_cloud, dtype=np.float32).reshape(-1, 3)  # Nx3
    colors = np.zeros((len(pts), 3), dtype=np.float32)

    for i, p in enumerate(pts):
        X, Y, Z = p
        if Z is None or Z == 0 or np.isnan(Z) or np.isinf(Z):
            colors[i] = np.array([0.5, 0.5, 0.5])  # 灰色
            continue

        try:
            # OpenCV 坐标系 (u,v)
            # u = int((X * camera_matrix[0, 0] / Z) + camera_matrix[0, 2])
            # v = int((Y * camera_matrix[1, 1] / Z) + camera_matrix[1, 2])

            # 更通用的投影，考虑非零倾斜因子 (很少见，但严谨)
            # Pinhole camera model
            u = (X * camera_matrix[0,0] + Y * camera_matrix[0,1] + Z * camera_matrix[0,2]) / Z
            v = (X * camera_matrix[1,0] + Y * camera_matrix[1,1] + Z * camera_matrix[1,2]) / Z
            
            u = int(u)
            v = int(v)

        except Exception:
            colors[i] = np.array([0.5, 0.5, 0.5])
            continue

        if 0 <= u < frame_w and 0 <= v < frame_h:
            b, g, r = frame[v, u].astype(np.float32)
            colors[i] = np.array([r, g, b]) / 255.0  # 归一化到 0~1
        else:
            colors[i] = np.array([0.5, 0.5, 0.5])

    return pts, colors

def extract_object_pointcloud(xyz_map, box_local, class_name=None, max_points=20000, voxel_size_mm=5.0):
    """
    在原有函数基础上添加点云优化
    - 不改变原有接口，内部优化点云质量
    """
    if xyz_map is None:
        return np.empty((0, 3), dtype=np.float32), 0.0

    H, W = xyz_map.shape[:2]
    x1, y1, x2, y2 = map(int, box_local)
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 3), dtype=np.float32), 0.0

    roi = xyz_map[y1:y2 + 1, x1:x2 + 1, :].reshape(-1, 3)
    mask_valid = np.isfinite(roi).all(axis=1) & (roi[:, 2] > 0)
    if mask_valid.sum() == 0:
        return np.empty((0, 3), dtype=np.float32), 0.0

    pts = roi[mask_valid].astype(np.float32)

    # === 新增：点云质量优化 ===
    if pts.shape[0] > 10:
        try:
            # 1. 统计离群点去除
            pts_clean = statistical_outlier_removal(pts)
            if pts_clean.shape[0] > pts.shape[0] * 0.3:  # 确保保留足够点
                pts = pts_clean

            # 2. 单位检测和归一化（解决数值过大）
            z_median = np.median(pts[:, 2])
            if z_median > 1000:  # 检测到mm单位，转为cm
                pts = pts / 10.0
                logger.debug("检测到mm单位，已转换为cm")

        except Exception as e:
            logger.debug(f"点云优化失败: {e}")
    # === 优化结束 ===

    # 随机下采样控制点数（原有逻辑）
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]

    # 体素下采样（通用voxel_downsample_numpy，已移至helpers）
    try:
        from src.utils.helpers import voxel_downsample_numpy
        pts_ds = voxel_downsample_numpy(pts, voxel_size=voxel_size_mm)
        if pts_ds is None or pts_ds.size == 0:
            pts_ds = pts
    except Exception:
        # 简单体素近似 (fallback)
        vs = float(voxel_size_mm)
        keys = np.floor(pts / vs).astype(np.int64)
        uniq, idxs = np.unique(keys, axis=0, return_index=True)
        pts_ds = pts[idxs]

    valid_ratio = float(pts_ds.shape[0]) / max(1, roi.shape[0])
    return pts_ds.astype(np.float32), valid_ratio
