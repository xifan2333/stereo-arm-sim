# -*- coding: utf-8 -*-
import numpy as np
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StereoVision")

# Open3D 导入和可用性检查 (仅做可用性判断，不强依赖)
try:
    import open3d as open3d
    O3D_VISUALIZATION_AVAILABLE = True
    logger.info("Open3D 可用，将使用3D可视化")
except ImportError as e:
    O3D_VISUALIZATION_AVAILABLE = False
    logger.warning(f"Open3D 不可用: {e}")

# 从 config 导入相关阈值 (现在已经在 config.py 中定义)
# from src.config import _UNIT_MM_THRESHOLD, _UNIT_M_THRESHOLD, DEPTH_CALIB, _SMOOTH_MAX_AGE_S

# 假定此处可以使用 config 中的常量，或者将这些常量传入函数
# 为了保持 helpers.py 的独立性，这里暂时硬编码，实际项目中应从 config 导入

_UNIT_MM_THRESHOLD = 1000.0  # if median z > 1000 -> input likely mm
_UNIT_M_THRESHOLD = 5.0      # if median z < 5 -> input likely meters

def convert_points_to_mm(pts):
    """
    将任意 pts (Nx3) 保守地转换为 毫米(mm) 单位并返回 (pts_mm, detected_unit)
    """
    if pts is None:
        return pts, 'none'
    arr = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    if arr.size == 0:
        return arr, 'empty'
    z = arr[:, 2]
    z_valid = z[np.isfinite(z) & (z > 0)]
    if z_valid.size == 0:
        return arr, 'unknown'
    zmed = float(np.median(z_valid))
    if zmed > _UNIT_MM_THRESHOLD:
        return arr.astype(np.float32), 'mm'
    if zmed < _UNIT_M_THRESHOLD:
        return (arr * 1000.0).astype(np.float32), 'm'
    # otherwise assume cm
    return (arr * 10.0).astype(np.float32), 'cm'

def ensure_xyz_in_mm(xyz):
    """
    强制把 HxWx3 或 Nx3 的 xyz 转换为 mm，并返回 (xyz_mm, detected_unit).
    """
    if xyz is None:
        return None, 'none'
    arr = np.asarray(xyz, dtype=np.float32)
    if arr.size == 0:
        return arr, 'empty'
    orig_shape = arr.shape
    if arr.ndim == 3 and arr.shape[2] == 3:
        flat = arr.reshape(-1, 3)
        conv, unit = convert_points_to_mm(flat)
        return conv.reshape(orig_shape).astype(np.float32), unit
    elif arr.ndim == 2 and arr.shape[1] == 3:
        conv, unit = convert_points_to_mm(arr)
        return conv.astype(np.float32), unit
    else:
        return arr, 'unknown'

def voxel_downsample_numpy(points, colors=None, voxel_size=5.0):
    """
    points: (N,3) array in mm (或与你点云单位一致)
    colors: (N,3) array (optional)
    voxel_size: 单位同 points（例如 mm），返回基于 voxel 的平均点
    返回: down_points (M,3), down_colors (M,3) or None
    """
    if points is None or len(points) == 0:
        return np.zeros((0,3), dtype=np.float32), None

    pts = np.asarray(points, dtype=np.float32)
    # 计算 voxel 索引
    inv = 1.0 / float(voxel_size)
    keys = np.floor(pts * inv).astype(np.int64)
    # 使用 dict 聚合
    voxels = {}
    if colors is None:
        for i, k in enumerate(map(tuple, keys)):
            if k not in voxels:
                voxels[k] = [pts[i], 1]
            else:
                voxels[k][0] += pts[i]
                voxels[k][1] += 1
        out = []
        for k, (s, c) in voxels.items():
            out.append(s / c)
        return np.array(out, dtype=np.float32), None
    else:
        cols = np.asarray(colors, dtype=np.float32)
        for i, k in enumerate(map(tuple, keys)):
            if k not in voxels:
                voxels[k] = [pts[i].copy(), cols[i].copy(), 1]
            else:
                voxels[k][0] += pts[i]
                voxels[k][1] += cols[i]
                voxels[k][2] += 1
        out_pts = []
        out_cols = []
        for k, (s, c, n) in voxels.items():
            out_pts.append(s / n)
            out_cols.append(c / n)
        return np.array(out_pts, dtype=np.float32), np.array(out_cols, dtype=np.float32)


# 全局深度校准结构：可保存全局或 per-class 校准系数（多项式）
_DEPTH_CALIB = { # Internal, will be handled by config system or passed
    "coeffs": None,
    "degree": None,
    "per_class": {}
}

def set_depth_calibration(measured_mm_list, true_mm_list, degree=1, class_name=None):
    """
    用若干已知点拟合校准多项式 measured_mm -> true_mm。
    """
    mm = np.asarray(measured_mm_list, dtype=np.float64)
    tt = np.asarray(true_mm_list, dtype=np.float64)
    if mm.size < 2 or mm.size != tt.size:
        raise ValueError("需要至少 2 个测量点且长度匹配以进行拟合")
    coeffs = np.polyfit(mm, tt, degree)
    if class_name:
        _DEPTH_CALIB['per_class'][str(class_name)] = coeffs
    else:
        _DEPTH_CALIB['coeffs'] = coeffs
        _DEPTH_CALIB['degree'] = degree
    return coeffs

def apply_depth_calibration(z_mm, class_name=None):
    """
    把 z_mm（毫米）通过拟合函数映射到校准后的值（毫米）。
    如果没有校准参数，直接返回原值。
    """
    try:
        if class_name and str(class_name) in _DEPTH_CALIB.get('per_class', {}):
            coeffs = _DEPTH_CALIB['per_class'][str(class_name)]
            return float(np.polyval(coeffs, float(z_mm)))
        if _DEPTH_CALIB.get('coeffs') is None:
            return float(z_mm)
        coeffs = _DEPTH_CALIB['coeffs']
        return float(np.polyval(coeffs, float(z_mm)))
    except Exception:
        return float(z_mm)

_DEPTH_SMOOTH_CACHE = {}
_SMOOTH_MAX_AGE_S = 2.0  # 缓存年龄上限（秒）

def smooth_depth_by_key(key, z_mm, alpha=0.7, now_ts=None):
    """
    EMA平滑深度缓存 - 强制统一为毫米单位
    key: 唯一标识字符串
    z_mm: 深度值，单位必须为毫米
    alpha: EMA系数
    now_ts: 时间戳
    返回：平滑后的深度值，单位毫米
    """
    now = now_ts if now_ts is not None else time.time()

    try:
        z_mm = float(z_mm)
    except Exception:
        return z_mm

    ent = _DEPTH_SMOOTH_CACHE.get(key)
    if ent is None or (now - ent.get('t', 0.0)) > _SMOOTH_MAX_AGE_S:
        _DEPTH_SMOOTH_CACHE[key] = {'z_mm': float(z_mm), 't': now}
        return float(z_mm)

    prev_mm = float(ent['z_mm'])
    z_new_mm = float(alpha) * float(z_mm) + (1.0 - float(alpha)) * prev_mm
    _DEPTH_SMOOTH_CACHE[key] = {'z_mm': z_new_mm, 't': now}
    return float(z_new_mm)

def clear_depth_smooth_cache():
    _DEPTH_SMOOTH_CACHE.clear()

def get_object_id(obj):
    """
    统一获取/生成物体的稳定 ID。
    优先返回 obj['id']，否则返回 'track_<track_id>'，否则根据 class_name + bbox center 合成，
    最后兜底返回 class_name 或 'obj_unknown'。
    """
    try:
        if obj is None:
            return "obj_unknown"
        # 1) 直接使用已有 id
        if 'id' in obj and obj['id'] is not None:
            return obj['id']
        # 2) 使用 track_id（如果存在）
        if 'track_id' in obj and obj['track_id'] is not None:
            try:
                return f"track_{int(obj['track_id'])}"
            except Exception:
                return f"track_{obj['track_id']}"
        # 3) 基于 bbox 合成（若 bbox 存在）
        bbox = obj.get('bbox') or obj.get('box') or None
        if bbox and len(bbox) >= 4:
            try:
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
                return f"{obj.get('class_name','obj')}_{cx}_{cy}"
            except Exception:
                pass
        # 4) 兜底
        return obj.get('class_name', 'obj_unknown')
    except Exception:
        return "obj_unknown"

