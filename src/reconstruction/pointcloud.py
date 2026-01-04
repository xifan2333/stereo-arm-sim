"""
点云处理工具函数
提供物体点云提取、离群点去除、3D信息计算等功能

遵循小步骤迭代原则，仅实现基础必要功能
"""

from typing import Tuple, Optional
import numpy as np
from src.utils.logger import get_logger

logger = get_logger()


def extract_object_pointcloud(
    xyz_pointcloud: np.ndarray, mask: Optional[np.ndarray] = None, bbox: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    从全局点云中提取物体点云

    Args:
        xyz_pointcloud: 全局3D点云 (HxWx3), 单位：毫米
        mask: 分割掩码 (HxW), True表示物体像素
        bbox: 边界框 [x1, y1, x2, y2] (如果没有mask则使用bbox)

    Returns:
        np.ndarray: 物体点云 (Nx3), 单位：毫米
    """
    H, W = xyz_pointcloud.shape[:2]

    # 使用 mask 提取点云（优先）
    if mask is not None:
        # 调整mask大小到点云尺寸
        if mask.shape != (H, W):
            import cv2
            mask_resized = cv2.resize(mask.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
            mask_bool = mask_resized > 0.5
        else:
            mask_bool = mask > 0.5

        # 提取mask内的点
        points = xyz_pointcloud[mask_bool]

    # 使用 bbox 提取点云（fallback）
    elif bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        # 提取bbox内的点
        points = xyz_pointcloud[y1:y2, x1:x2].reshape(-1, 3)

    else:
        raise ValueError("必须提供 mask 或 bbox")

    # 过滤无效点 (NaN, Inf)
    valid_mask = np.isfinite(points).all(axis=1)
    points = points[valid_mask]

    return points


def remove_outliers_mad(points: np.ndarray, threshold_scale: float = 3.0) -> np.ndarray:
    """
    使用 MAD (Median Absolute Deviation) 方法去除离群点

    Args:
        points: 点云 (Nx3), 单位：毫米
        threshold_scale: MAD 阈值倍数，默认3.0

    Returns:
        np.ndarray: 去除离群点后的点云 (Mx3)
    """
    if len(points) < 3:
        return points

    # 计算深度 (Z 坐标)
    z = points[:, 2]

    # 计算中位数
    median_z = np.median(z)

    # 计算 MAD
    mad = np.median(np.abs(z - median_z))

    # 设置阈值（至少 50mm）
    threshold = max(threshold_scale * mad, 50.0)

    # 过滤离群点
    valid_mask = np.abs(z - median_z) <= threshold
    inliers = points[valid_mask]

    if len(inliers) < len(points) * 0.1:
        logger.warning(f"MAD去噪后点数过少: {len(inliers)}/{len(points)}, 使用原始点云")
        return points

    return inliers


def calculate_object_3d_info(
    points: np.ndarray, method: str = "median"
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    计算物体的3D中心位置和尺寸

    Args:
        points: 物体点云 (Nx3), 单位：毫米
        method: 计算方法，"median" 或 "mean"

    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
            (中心位置(3,), 尺寸(3,), 置信度)
            单位：毫米
    """
    if len(points) == 0:
        return (
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            0.0,
        )

    # 去除离群点
    points_filtered = remove_outliers_mad(points)

    if len(points_filtered) < 3:
        logger.warning(f"过滤后点数过少: {len(points_filtered)}")
        return (
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            0.0,
        )

    # 计算中心（中位数更稳健）
    if method == "median":
        center = np.median(points_filtered, axis=0)
    else:
        center = np.mean(points_filtered, axis=0)

    # 计算尺寸（使用范围方法保持XYZ对应关系）
    dimensions = _estimate_dimensions_range(points_filtered)

    # 计算置信度（基于点数和深度变异系数）
    confidence = _calculate_confidence(points_filtered)

    return center, dimensions, confidence


def _estimate_dimensions_pca(points: np.ndarray) -> np.ndarray:
    """
    使用 PCA 估计物体尺寸

    Args:
        points: 点云 (Nx3)

    Returns:
        np.ndarray: 尺寸 [长, 宽, 高] (3,)
    """
    try:
        # 中心化
        centered = points - np.mean(points, axis=0)

        # 协方差矩阵
        cov = np.cov(centered.T)

        # 特征分解
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # 投影到主成分空间
        transformed = np.dot(centered, eigenvectors)

        # 计算每个主成分的范围
        dimensions = np.max(transformed, axis=0) - np.min(transformed, axis=0)

        # 取绝对值并排序（从大到小）
        dimensions = np.abs(dimensions)
        dimensions = np.sort(dimensions)[::-1]

        return dimensions

    except Exception as e:
        logger.warning(f"PCA 尺寸估计失败: {e}, 使用范围方法")
        return _estimate_dimensions_range(points)


def _estimate_dimensions_range(points: np.ndarray) -> np.ndarray:
    """
    使用简单范围方法估计物体尺寸

    Args:
        points: 点云 (Nx3)

    Returns:
        np.ndarray: 尺寸 [长, 宽, 高] (3,)
    """
    # 计算每个轴的范围
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    dimensions = max_vals - min_vals

    return dimensions


def _calculate_confidence(points: np.ndarray) -> float:
    """
    计算深度置信度

    Args:
        points: 点云 (Nx3)

    Returns:
        float: 置信度 (0~1)
    """
    if len(points) == 0:
        return 0.0

    # 基于点数的置信度（更多点 = 更高置信度）
    point_conf = min(len(points) / 100.0, 1.0)

    # 基于深度变异系数的置信度
    z = points[:, 2]
    z_mean = np.mean(z)
    z_std = np.std(z)

    if z_mean > 0:
        cv = z_std / z_mean  # 变异系数
        depth_conf = max(0.0, 1.0 - cv)  # 变异越小，置信度越高
    else:
        depth_conf = 0.0

    # 综合置信度
    confidence = 0.6 * point_conf + 0.4 * depth_conf

    return float(np.clip(confidence, 0.0, 1.0))
