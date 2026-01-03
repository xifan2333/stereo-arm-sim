# -*- coding: utf-8 -*-
import numpy as np
import logging

from src.utils.helpers import logger, apply_depth_calibration # Assuming apply_depth_calibration is used
from src.config import CAMERA_CONFIG, MIN_POINTS_FOR_3D, MIN_POINTS_FOR_ROUGH, \
    DEFAULT_CONF_LOW, DEFAULT_CONF_MED, DEFAULT_CONF_HIGH, \
    MIN_DEPTH_CM, MAX_DEPTH_CM, MIN_DIM_CM, MAX_DIM_CM, ASSUMED_HEIGHTS, SKLEARN_AVAILABLE

# Placeholder for sklearn availability, should be handled by config.py
try:
    from sklearn.decomposition import PCA
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def calculate_object_depth(object_points_3d, method='adaptive', bbox_2d=None, image_shape=None, class_name=None):
    """
    稳健版 calculate_object_depth
    - 输入:
        object_points_3d: Nx3 点云（单位可以是 cm 或 mm；函数会自动检测并转换为 cm）
        bbox_2d: 可选 [x1,y1,x2,y2] 用于 fallback
        image_shape: (h,w) 可选，用于 2D->3D 融合
        class_name: 可选，用于选择假设高度的类别特定值
    - 输出:
        (object_center_cm, dimensions_cm, depth_confidence)
        object_center_cm: np.array([X,Y,Z]) 单位 cm 或 None
        dimensions_cm: np.array([L,W,H]) 单位 cm 或 None
        depth_confidence: float 0..1
    重要：本函数保证在所有 fallback 路径上对深度与尺寸做上下限 clamp，
    并尽量返回尺寸（即便置信度低），避免上层拿到 None/NaN 导致显示“???”或天文数值。
    """
    try:
        # 规范输入点云为 numpy array
        if object_points_3d is None:
            pts = np.zeros((0, 3), dtype=np.float32)
        else:
            pts = np.asarray(object_points_3d, dtype=np.float32).reshape(-1, 3)
    except Exception:
        pts = np.zeros((0, 3), dtype=np.float32)

    # 过滤无效点
    if pts.size == 0:
        valid_pts = np.zeros((0, 3), dtype=np.float32)
    else:
        valid_mask = ~(np.isnan(pts).any(axis=1) | np.isinf(pts).any(axis=1))
        valid_pts = pts[valid_mask]

    # 如果点太少，尝试多级 fallback（每个 fallback 都 clamp）
    if len(valid_pts) < MIN_POINTS_FOR_ROUGH:
        logger.warning(f"calculate_object_depth: 有效点太少: {len(valid_pts)}")

        # fallback 1: 优先尝试使用全局 mono_depth_map_mm（若你程序维护了）
        # mono_depth_map_mm is a global variable from main.py, it should be passed as argument if needed
        fallback_point_cm = None
        # try:
        #     if 'mono_depth_map_mm' in globals() and mono_depth_map_mm is not None and bbox_2d is not None:
        #         x1, y1, x2, y2 = map(int, bbox_2d)
        #         cx = int((x1 + x2) / 2)
        #         cy = int((y1 + y2) / 2)
        #         if 0 <= cy < mono_depth_map_mm.shape[0] and 0 <= cx < mono_depth_map_mm.shape[1]:
        #             z_mm = mono_depth_map_mm[cy, cx]
        #             if np.isfinite(z_mm) and z_mm > 0:
        #                 z_cm = float(z_mm) / 10.0
        #                 if CAMERA_CONFIG.camera_matrix_l is not None:
        #                     fx = float(CAMERA_CONFIG.camera_matrix_l[0, 0]) if CAMERA_CONFIG.camera_matrix_l is not None else 1.0
        #                     fy = float(CAMERA_CONFIG.camera_matrix_l[1, 1]) if CAMERA_CONFIG.camera_matrix_l is not None else 1.0
        #                     cx_cam = float(CAMERA_CONFIG.camera_matrix_l[0, 2]) if CAMERA_CONFIG.camera_matrix_l is not None else (image_shape[1]/2 if image_shape else 0)
        #                     cy_cam = float(CAMERA_CONFIG.camera_matrix_l[1, 2]) if CAMERA_CONFIG.camera_matrix_l is not None else (image_shape[0]/2 if image_shape else 0)
        #                     X = (cx - cx_cam) * z_cm / (fx if fx != 0 else 1.0)
        #                     Y = (cy - cy_cam) * z_cm / (fy if fy != 0 else 1.0)
        #                     fallback_point_cm = np.array([X, Y, z_cm], dtype=np.float32)
        #                 else:
        #                     fallback_point_cm = np.array([0.0, 0.0, z_cm], dtype=np.float32)
        #                 logger.info("calculate_object_depth: 使用 mono_depth_map_mm 作为 fallback")
        # except Exception:
        #     fallback_point_cm = None

        # fallback 2: bbox 高度启发式估计（按类使用假设高度）
        if fallback_point_cm is None and bbox_2d is not None and image_shape is not None:
            try:
                x1, y1, x2, y2 = map(int, bbox_2d)
                bbox_h = max(1, y2 - y1)
                assumed_h = ASSUMED_HEIGHTS.get(class_name, 30.0)
                if CAMERA_CONFIG.camera_matrix_l is not None:
                    fy = float(CAMERA_CONFIG.camera_matrix_l[1, 1])
                    fx = float(CAMERA_CONFIG.camera_matrix_l[0, 0])
                    cx_cam = float(CAMERA_CONFIG.camera_matrix_l[0, 2])
                    cy_cam = float(CAMERA_CONFIG.camera_matrix_l[1, 2])
                else:
                    fy = fx = 1000.0
                    cx_cam = image_shape[1] / 2.0
                    cy_cam = image_shape[0] / 2.0
                depth_est_cm = (fy * assumed_h) / float(bbox_h)
                depth_est_cm = float(np.clip(depth_est_cm, MIN_DEPTH_CM, MAX_DEPTH_CM))
                cx_pixel = (x1 + x2) / 2.0
                cy_pixel = (y1 + y2) / 2.0
                X = (cx_pixel - cx_cam) * depth_est_cm / (fx if fx != 0 else 1.0)
                Y = (cy_pixel - cy_cam) * depth_est_cm / (fy if fy != 0 else 1.0)
                fallback_point_cm = np.array([X, Y, depth_est_cm], dtype=np.float32)
                logger.info("calculate_object_depth: 使用 bbox 启发式作为 fallback")
            except Exception:
                fallback_point_cm = None

        # 如果我们找到了 fallback_point_cm，则用 bbox 融合估计尺寸（保证返回 cm）
        if fallback_point_cm is not None:
            try:
                # 估尺寸：用 bbox 像素 + 深度进行简单透视估计
                est_dims = None
                if bbox_2d is not None and image_shape is not None:
                    x1, y1, x2, y2 = map(int, bbox_2d)
                    bbox_w = max(1, x2 - x1)
                    bbox_h = max(1, y2 - y1)
                    fx = float(CAMERA_CONFIG.camera_matrix_l[0, 0]) if CAMERA_CONFIG.camera_matrix_l is not None else 1.0
                    fy = float(CAMERA_CONFIG.camera_matrix_l[1, 1]) if CAMERA_CONFIG.camera_matrix_l is not None else 1.0
                    z = float(fallback_point_cm[2])
                    est_w_cm = (bbox_w * z) / (fx if fx != 0 else 1.0)
                    est_h_cm = (bbox_h * z) / (fy if fy != 0 else 1.0)
                    est_d_cm = z
                    est_dims = np.array([
                        float(np.clip(est_w_cm, MIN_DIM_CM, MAX_DIM_CM)),
                        float(np.clip(est_h_cm, MIN_DIM_CM, MAX_DIM_CM)),
                        float(np.clip(est_d_cm, MIN_DIM_CM, MAX_DIM_CM))
                    ], dtype=np.float32)
                return np.asarray(fallback_point_cm, dtype=np.float32), est_dims, float(DEFAULT_CONF_LOW)
            except Exception:
                return np.asarray(fallback_point_cm, dtype=np.float32), None, float(DEFAULT_CONF_LOW)

        # 没有 fallback 点，返回 None
        return None, None, 0.0

    # 至此 valid_pts 数量 >= MIN_POINTS_FOR_ROUGH
    pts = valid_pts

    # 将单位规范化到 cm：如果 z 值中位数 > 1000，则认为是 mm -> /10
    try:
        if np.nanmax(np.abs(pts)) > 1000.0:
            pts = pts / 10.0
    except Exception:
        pass

    # 剔除 z 离群点（IQR）
    try:
        z = pts[:, 2]
        q1 = np.percentile(z, 25)
        q3 = np.percentile(z, 75)
        iqr = q3 - q1
        lb = q1 - 1.5 * iqr
        ub = q3 + 1.5 * iqr
        inlier_mask = (z >= lb) & (z <= ub)
        inlier_pts = pts[inlier_mask]
        if len(inlier_pts) < max(MIN_POINTS_FOR_ROUGH, int(0.5 * len(pts))):
            inlier_pts = pts
    except Exception:
        inlier_pts = pts

    # 计算中心（中位数更稳）
    try:
        object_center = np.median(inlier_pts, axis=0).astype(np.float32)
    except Exception:
        object_center = np.mean(inlier_pts, axis=0).astype(np.float32)

    # clamp Z （加上可选校准映射）
    try:
        z_cm = float(object_center[2])
        # 转到 mm -> 校准 -> 回到 cm
        z_mm = z_cm * 10.0
        # Use apply_depth_calibration from helpers
        z_mm_corr = apply_depth_calibration(z_mm, class_name=class_name)
        object_center[2] = float(np.clip((z_mm_corr / 10.0), MIN_DEPTH_CM, MAX_DEPTH_CM))
    except Exception:
        object_center[2] = float(np.clip(object_center[2], MIN_DEPTH_CM, MAX_DEPTH_CM))

    # 尝试估尺寸：优先 PCA（当点足够多），否则 axis-aligned range 或 bbox 融合
    dims = None
    try:
        if len(inlier_pts) >= MIN_POINTS_FOR_3D:
            centroid = np.mean(inlier_pts, axis=0)
            centered = inlier_pts - centroid
            cov = np.cov(centered.T)
            eigenvals, eigenvecs = np.linalg.eig(cov)
            transformed = np.dot(centered, eigenvecs)
            mins = np.min(transformed, axis=0)
            maxs = np.max(transformed, axis=0)
            sizes = maxs - mins
            # sizes 对应主轴，可按绝对值排序，但我们返回一个稳定的三元（L,W,H）
            sizes = np.abs(sizes)
            # 最小保护并 clamp
            sizes = np.clip(sizes, MIN_DIM_CM, MAX_DIM_CM)
            # Try to produce [L,W,H] where H is vertical ~ use z-range
            z_range = np.max(inlier_pts[:,2]) - np.min(inlier_pts[:,2])
            dims = np.array([sizes[0], sizes[1], max(MIN_DIM_CM, z_range)], dtype=np.float32)
        else:
            # 点少，使用 axis-aligned box
            mins = np.min(inlier_pts, axis=0)
            maxs = np.max(inlier_pts, axis=0)
            sizes = maxs - mins
            dims = np.clip(np.abs(sizes), MIN_DIM_CM, MAX_DIM_CM).astype(np.float32)
            if dims.size < 3:
                dims = np.pad(dims, (0, 3 - dims.size), 'constant', constant_values=MIN_DIM_CM)
    except Exception:
        dims = None

    # 如果 dims 仍为空且 bbox 可用，使用 bbox 融合估计（保证返回）
    if dims is None and bbox_2d is not None and image_shape is not None:
        try:
            x1, y1, x2, y2 = map(int, bbox_2d)
            bbox_w = max(1, x2 - x1)
            bbox_h = max(1, y2 - y1)
            fx = float(CAMERA_CONFIG.camera_matrix_l[0, 0]) if CAMERA_CONFIG.camera_matrix_l is not None else 1000.0
            fy = float(CAMERA_CONFIG.camera_matrix_l[1, 1]) if CAMERA_CONFIG.camera_matrix_l is not None else 1000.0
            z = float(object_center[2])
            est_w_cm = (bbox_w * z) / (fx if fx != 0 else 1.0)
            est_h_cm = (bbox_h * z) / (fy if fy != 0 else 1.0)
            dims = np.array([
                float(np.clip(est_w_cm, MIN_DIM_CM, MAX_DIM_CM)),
                float(np.clip(est_h_cm, MIN_DIM_CM, MAX_DIM_CM)),
                float(np.clip(z, MIN_DIM_CM, MAX_DIM_CM))
            ], dtype=np.float32)
        except Exception:
            dims = None

    # 最终 clamp dims
    if dims is not None:
        dims = np.clip(dims, MIN_DIM_CM, MAX_DIM_CM).astype(np.float32)

    # depth confidence：根据点数和 bbox 融合情况给出一个保守估计
    if len(valid_pts) >= MIN_POINTS_FOR_3D:
        depth_conf = DEFAULT_CONF_HIGH
    elif len(valid_pts) >= MIN_POINTS_FOR_ROUGH:
        depth_conf = DEFAULT_CONF_MED
    else:
        depth_conf = DEFAULT_CONF_LOW

    return (object_center.astype(np.float32) if object_center is not None else None,
            (dims.astype(np.float32) if dims is not None else None),
            float(depth_conf))


def estimate_object_dimensions(point_cloud, method='pca'):
    """
    从三维点云估计物体的尺寸（长、宽、高）
    """

    # 简化的PCA实现，用于当scikit-learn不可用时
    def simple_pca(data):
        if data.shape[0] < 2:
            return np.eye(3), np.zeros(3)

        # 中心化数据
        centered_data = data - np.mean(data, axis=0)

        # 计算协方差矩阵
        cov_matrix = np.cov(centered_data.T)

        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # 按特征值大小排序
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]

        return eigenvectors, eigenvalues[sorted_indices]

    if point_cloud is None or len(point_cloud) < 10:
        return None, None, None

    # 移除无效点
    valid_mask = ~(np.isnan(point_cloud).any(axis=1) | np.isinf(point_cloud).any(axis=1))
    points = point_cloud[valid_mask]

    if len(points) < 10:
        return None, None, None

    logger.debug(f"点云数据范围: X={np.min(points[:, 0]):.2f}~{np.max(points[:, 0]):.2f}, "
                 f"Y={np.min(points[:, 1]):.2f}~{np.max(points[:, 1]):.2f}, "
                 f"Z={np.min(points[:, 2]):.2f}~{np.max(points[:, 2]):.2f}")

    if method == 'aabb':
        # 轴对齐包围盒
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        dimensions = max_coords - min_coords

        sorted_dims = np.sort(dimensions)[::-1]
        length, width, height = sorted_dims

        bbox_corners = np.array([
            [min_coords[0], min_coords[1], min_coords[2]],
            [max_coords[0], min_coords[1], min_coords[2]],
            [max_coords[0], max_coords[1], min_coords[2]],
            [min_coords[0], max_coords[1], min_coords[2]],
            [min_coords[0], min_coords[1], max_coords[2]],
            [max_coords[0], min_coords[1], max_coords[2]],
            [max_coords[0], max_coords[1], max_coords[2]],
            [min_coords[0], max_coords[1], max_coords[2]]
        ])

        return [length, width, height], None, bbox_corners

    elif method == 'pca' or method == 'obb':
        # 使用简化PCA实现
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid

        eigenvectors, eigenvalues = simple_pca(centered_points)

        # 按特征值大小排序（降序）
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]

        # 将点云转换到PCA坐标系
        transformed_points = np.dot(centered_points, eigenvectors)

        # 在PCA坐标系中计算最小-最大值
        min_vals = np.min(transformed_points, axis=0)
        max_vals = np.max(transformed_points, axis=0)
        dimensions = max_vals - min_vals

        # 对尺寸进行排序：长度 > 宽度 > 高度
        sorted_indices = np.argsort(dimensions)[::-1]
        dimensions = dimensions[sorted_indices]
        length, width, height = dimensions

        # 重新排序特征向量以匹配尺寸顺序
        eigenvectors = eigenvectors[:, sorted_indices]

        # 计算PCA坐标系中的包围盒角点
        corners_pca = np.array([
            [min_vals[0], min_vals[1], min_vals[2]],
            [max_vals[0], min_vals[1], min_vals[2]],
            [max_vals[0], max_vals[1], min_vals[2]],
            [min_vals[0], max_vals[1], min_vals[2]],
            [min_vals[0], min_vals[1], max_vals[2]],
            [max_vals[0], min_vals[1], max_vals[2]],
            [max_vals[0], max_vals[1], max_vals[2]],
            [min_vals[0], max_vals[1], max_vals[2]]
        ])

        # 将角点转换回世界坐标系
        bbox_corners = np.dot(corners_pca, eigenvectors.T) + centroid

        # 计算方向（欧拉角）
        R_mat = eigenvectors.T # Using R_mat to avoid conflict with config.R
        sy = np.sqrt(R_mat[0, 0] * R_mat[0, 0] + R_mat[1, 0] * R_mat[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R_mat[2, 1], R_mat[2, 2])
            y = np.arctan2(-R_mat[2, 0], sy)
            z = np.arctan2(R_mat[1, 0], R_mat[0, 0])
        else:
            x = np.arctan2(-R_mat[1, 2], R_mat[1, 1])
            y = np.arctan2(-R_mat[2, 0], sy)
            z = 0

        orientation = np.array([x, y, z])  # 弧度制的欧拉角

        return [length, width, height], orientation, bbox_corners

    return None, None, None


def estimate_width_from_2d_3d_fusion(bbox_2d, depth_value, camera_matrix, image_width):
    """
    将 bbox(像素) 与深度值融合估计物体宽度，返回值为 厘米(cm)
    depth_value: 可以是 cm 或 mm（函数会自动识别并转换）
    """
    if bbox_2d is None or depth_value is None:
        return None, 0.0
    try:
        x1, y1, x2, y2 = bbox_2d
        bbox_width_pixels = max(1, (x2 - x1))

        # 深度单位检测：如果 > 1000 认为是 毫米(mm)，否则认为是 厘米(cm)
        depth_val = float(depth_value)
        if depth_val > 1000.0:
            depth_cm = depth_val / 10.0
        else:
            depth_cm = depth_val

        fx = float(camera_matrix[0, 0]) if camera_matrix is not None else 1.0

        estimated_width_cm = (bbox_width_pixels * depth_cm) / fx

        # 置信度粗估：越靠近图中心越有信心
        bbox_center_x = (x1 + x2) / 2.0
        center_deviation = abs(bbox_center_x - image_width / 2.0) / (image_width / 2.0)
        confidence = max(0.0, 1.0 - center_deviation)

        return float(estimated_width_cm), float(confidence)
    except Exception:
        return None, 0.0


def fused_width_estimation(point_cloud, bbox_2d, depth_value, camera_matrix, image_size):
    """
    融合多种方法的宽度估计
    """
    methods = []
    widths = []
    confidences = []

    # 方法1: 基于3D点云的PCA估计
    if point_cloud is not None and len(point_cloud) >= 10:
        dimensions, _, _ = estimate_object_dimensions(point_cloud, method='pca')
        if dimensions is not None:
            _, width_pca, _ = dimensions
            methods.append('pca')
            confidences.append(min(1.0, len(point_cloud) / 1000))
            widths.append(width_pca)


    # 方法2: 基于2D-3D融合的估计
    if bbox_2d is not None and depth_value is not None:
        width_2d, confidence_2d = estimate_width_from_2d_3d_fusion(
            bbox_2d, depth_value, camera_matrix, image_size[0]
        )
        methods.append('2d_3d_fusion')
        confidences.append(confidence_2d)
        widths.append(width_2d)


    # 方法3: 简单轴对齐包围盒
    if point_cloud is not None and len(point_cloud) >= 5:
        dimensions, _, _ = estimate_object_dimensions(point_cloud, method='aabb')
        if dimensions is not None:
            _, width_aabb, _ = dimensions
            methods.append('aabb')
            confidences.append(0.5)
            widths.append(width_aabb)


    if not widths:
        return None, 0.0, "none"

    # 基于置信度加权融合
    total_confidence = sum(confidences)
    if total_confidence > 0:
        weighted_width = sum(w * c for w, c in zip(widths, confidences)) / total_confidence
        fused_confidence = total_confidence / len(confidences)

        method_info = sorted(zip(methods, confidences), key=lambda x: x[1], reverse=True)
        method_used = "+".join([f"{m}({c:.2f})" for m, c in method_info])

        return weighted_width, fused_confidence, method_used
    else:
        avg_width = sum(widths) / len(widths)
        return avg_width, 0.3, "average(" + "+".join(methods) + ")"


def calculate_3d_bounding_box_from_points(points, estimated_dimensions=None, center=None):
    """
    改进的3D包围盒计算，结合多方法融合
    - 替换原有函数，接口完全兼容
    - 解决包围盒抖动和数值过大的问题
    """
    if points is None:
        return None

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return None

    # 1. 改进的离群点去除 (from pointcloud.py)
    from src.geometry.pointcloud import statistical_outlier_removal
    pts_clean = statistical_outlier_removal(pts)
    if len(pts_clean) < 6:
        pts_clean = pts  # 回退

    # 2. 多方法包围盒计算
    bbox_candidates = []

    # 方法A: 改进PCA
    try:
        centroid = np.mean(pts_clean, axis=0)
        centered = pts_clean - centroid
        cov_matrix = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]

        transformed = np.dot(centered, eigenvectors)
        min_vals = np.min(transformed, axis=0)
        max_vals = np.max(transformed, axis=0)

        # 生成角点
        corners_pca = np.array([
            [min_vals[0], min_vals[1], min_vals[2]],
            [max_vals[0], min_vals[1], min_vals[2]],
            [max_vals[0], max_vals[1], min_vals[2]],
            [min_vals[0], max_vals[1], min_vals[2]],
            [min_vals[0], min_vals[1], max_vals[2]],
            [max_vals[0], min_vals[1], max_vals[2]],
            [max_vals[0], max_vals[1], max_vals[2]],
            [min_vals[0], max_vals[1], max_vals[2]]
        ])

        corners_world = np.dot(corners_pca, eigenvectors.T) + centroid

        # 计算质量分数
        min_bounds = np.min(corners_world, axis=0)
        max_bounds = np.max(corners_world, axis=0)
        inside_mask = np.all((pts_clean >= min_bounds) & (pts_clean <= max_bounds), axis=1)
        coverage_ratio = np.sum(inside_mask) / len(pts_clean)

        bbox_candidates.append({
            'center': centroid.astype(np.float32),
            'dimensions': (max_vals - min_vals).astype(np.float32),
            'corners': corners_world.astype(np.float32),
            'rotation': eigenvectors.T.astype(np.float32),
            'score': coverage_ratio * 0.7 + 0.3  # 基础分数
        })
    except Exception:
        pass

    # 方法B: 轴对齐包围盒（稳定备用）
    try:
        min_coords = np.min(pts_clean, axis=0)
        max_coords = np.max(pts_clean, axis=0)
        center_aabb = (min_coords + max_coords) / 2
        dimensions_aabb = max_coords - min_coords

        corners_aabb = np.array([
            [min_coords[0], min_coords[1], min_coords[2]],
            [max_coords[0], min_coords[1], min_coords[2]],
            [max_coords[0], max_coords[1], min_coords[2]],
            [min_coords[0], max_coords[1], min_coords[2]],
            [min_coords[0], min_coords[1], max_coords[2]],
            [max_coords[0], min_coords[1], max_coords[2]],
            [max_coords[0], max_coords[1], max_coords[2]],
            [min_coords[0], max_coords[1], max_coords[2]]
        ])

        inside_mask = np.all((pts_clean >= min_coords) & (pts_clean <= max_coords), axis=1)
        coverage_ratio = np.sum(inside_mask) / len(pts_clean)

        bbox_candidates.append({
            'center': center_aabb.astype(np.float32),
            'dimensions': dimensions_aabb.astype(np.float32),
            'corners': corners_aabb.astype(np.float32),
            'rotation': np.eye(3, dtype=np.float32),
            'score': coverage_ratio * 0.5 + 0.5  # AABB更稳定但不够精确
        })
    except Exception:
        pass

    # 3. 选择最优包围盒
    if not bbox_candidates:
        return None

    best_bbox = max(bbox_candidates, key=lambda x: x['score'])

    # 4. 应用尺寸估计（如果提供）
    if estimated_dimensions is not None:
        try:
            estimated = np.asarray(estimated_dimensions, dtype=np.float64)
            current_dims = best_bbox['dimensions']
            # 防止除零
            scale = np.ones_like(current_dims)
            nz = current_dims > 1e-6
            scale[nz] = estimated[nz] / (current_dims[nz] + 1e-6)
            scale = np.maximum(scale, 1.0)  # 只放大不缩小

            # 重新计算包围盒
            center = best_bbox['center']
            new_dims = current_dims * scale
            half_dims = new_dims / 2

            min_corner = center - half_dims
            max_corner = center + half_dims

            best_bbox['corners'] = np.array([
                [min_corner[0], min_corner[1], min_corner[2]],
                [max_corner[0], min_corner[1], min_corner[2]],
                [max_corner[0], max_corner[1], min_corner[2]],
                [min_corner[0], max_corner[1], min_corner[2]],
                [min_corner[0], min_corner[1], max_corner[2]],
                [max_corner[0], min_corner[1], max_corner[2]],
                [max_corner[0], max_corner[1], max_corner[2]],
                [min_corner[0], max_corner[1], max_corner[2]]
            ], dtype=np.float32)
            best_bbox['dimensions'] = new_dims.astype(np.float32)
        except Exception:
            pass

    # 5. 单位归一化（解决数值过大问题）
    try:
        # 检测单位：如果中位数>1000，可能是mm，需要转为cm
        median_norm = np.median(np.linalg.norm(best_bbox['corners'], axis=1))
        if median_norm > 1000:  # 单位是mm，转为cm
            best_bbox['center'] = best_bbox['center'] / 10.0
            best_bbox['dimensions'] = best_bbox['dimensions'] / 10.0
            best_bbox['corners'] = best_bbox['corners'] / 10.0
    except Exception:
        pass

    # 移除临时字段，保持原有接口
    result = {
        'center': best_bbox['center'],
        'dimensions': best_bbox['dimensions'],
        'corners': best_bbox['corners'],
        'rotation': best_bbox['rotation']
    }

    return result