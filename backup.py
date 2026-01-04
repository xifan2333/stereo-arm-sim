# -*- coding: utf-8 -*-
# 首先集中所有导入
import threading
import cv2
import numpy as np
import math
import os
import logging
import torch
from ultralytics import YOLO
from collections import deque, defaultdict
import time
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from scipy.optimize import minimize
import collections
import logging
from enum import Enum
from scipy import ndimage
import random

# sklearn导入
try:
    from sklearn.decomposition import PCA
    import sklearn  # 单独导入sklearn以访问版本信息

    SKLEARN_AVAILABLE = True
    print("scikit-learn可用，版本:", sklearn.__version__)
except ImportError as e:
    SKLEARN_AVAILABLE = False
    print("scikit-learn不可用:", e)


_logger = logging.getLogger(__name__)

# 保守的阈值用于检测传入点云的单位
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




def calculate_object_depth_mm(pts_mm, method='median', bbox_2d=None, image_shape=None, class_name=None):
    """
    计算物体深度和尺寸 - 严格毫米单位
    输入：pts_mm (Nx3 numpy array)，单位毫米
    返回：center_mm (3,), dims_mm (3,), confidence (0..1)
    """
    try:
        pts = np.asarray(pts_mm, dtype=np.float32).reshape(-1, 3)
    except Exception:
        pts = np.zeros((0, 3), dtype=np.float32)

    if pts.size == 0:
        return None, None, 0.0

    # 过滤无效点（毫米单位下的合理范围）
    mask = (~np.isnan(pts).any(axis=1)) & (~np.isinf(pts).any(axis=1)) & (pts[:, 2] > 10.0) & (pts[:, 2] < 10000.0)
    valid_pts = pts[mask]
    n = valid_pts.shape[0]

    if n == 0:
        return None, None, 0.0

    if n >= 20:
        center = np.median(valid_pts, axis=0)
        mins = np.percentile(valid_pts, 10, axis=0)
        maxs = np.percentile(valid_pts, 90, axis=0)
        dims = (maxs - mins)
        dims = np.clip(dims, a_min=5.0, a_max=10000.0)
        conf = min(1.0, float(n) / 200.0)
        return center.astype(np.float32), dims.astype(np.float32), float(conf)

    # 点太少时的fallback
    center = np.median(valid_pts, axis=0)
    return center.astype(np.float32), np.array([50.0, 50.0, 50.0], dtype=np.float32), 0.1

def voxel_downsample(points_mm, voxel_size_mm=5.0):
    if points_mm is None:
        return points_mm
    pts = np.asarray(points_mm, dtype=np.float32).reshape(-1,3)
    if pts.size == 0:
        return pts
    idx = np.floor(pts / float(voxel_size_mm)).astype(np.int64)
    uniq = {}
    for i, key in enumerate(map(tuple, idx)):
        if key not in uniq:
            uniq[key] = pts[i]
    out = np.stack(list(uniq.values()), axis=0)
    return out.astype(np.float32)

def simple_point_cloud_visualization_mm(ax_3d, detected_objects, voxel_size_mm=5.0):
    total_drawn = 0
    for obj in detected_objects:
        pts = obj.get('pointcloud') or obj.get('points') or None
        if pts is None:
            continue
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        if pts.size == 0:
            continue
        mask = (~np.isnan(pts).any(axis=1)) & (~np.isinf(pts).any(axis=1))
        pts = pts[mask]
        if pts.size == 0:
            continue
        pts_ds = voxel_downsample(pts, voxel_size_mm=voxel_size_mm)
        if pts_ds.size == 0:
            continue
        pts_adj = np.stack([pts_ds[:,0], pts_ds[:,2], -pts_ds[:,1]], axis=1)
        try:
            ax_3d.scatter(pts_adj[:,0], pts_adj[:,1], pts_adj[:,2], s=1, alpha=0.8)
            total_drawn += pts_adj.shape[0]
        except Exception:
            continue
    return total_drawn
# ------------------- END: 单位统一与核心工具 -------------------


# Open3D 导入和可用性检查
try:
    import open3d as o3d

    O3D_VISUALIZATION_AVAILABLE = True
    print("Open3D 可用，将使用3D可视化")
except ImportError as e:
    O3D_VISUALIZATION_AVAILABLE = False
    print("Open3D 不可用:", e)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StereoVision")

# 常量定义
imageWidth, imageHeight = 640, 480  # 相机分辨率
imageSize = (imageWidth, imageHeight)

# 相机标定参数（单位：像素+毫米）
# 注意：这些参数需要根据实际相机重新标定
cameraMatrixL = np.array([[410.3084, -0.2777, 309.3976],
                          [0, 410.2129, 262.0564],
                          [0, 0, 1.0000]])
distCoeffL = np.array([0.0027, 0.6848, -0.0043, 0.0066, -2.0220], dtype=np.float64)

cameraMatrixR = np.array([[409.2531, -0.3487, 299.1536],
                          [0, 408.9821, 265.7137],
                          [0, 0, 1.0000]])
distCoeffR = np.array([0.0357, 0.3211, -0.0042, 0.0035, -0.7268], dtype=np.float64)

# 立体标定结果（平移向量T单位：毫米，基线长度≈61.4mm）
T = np.array([-61.114637167122600, -0.044597478744966, 0.583576573837856])  # T[0]为基线，单位毫米
R = np.array([[0.999780788646075, 0.0004806193456666200, 0.020931881407937],
              [-0.0004322695666685383, 0.999997228514369, -0.0006092913311984787],
              [-0.020932935705560, 0.002304770839986, 0.999778225525102]])

# 立体校正结果
Rl, Rr, Pl, Pr, Q, mapLx, mapLy, mapRx, mapRy = [None] * 9

# 设备选择
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


model = YOLO('yolo11s.pt')
try:
    # 把模型固定到 device（避免每次推理内部迁移引入延迟）
    model.to(device)
    logger.info(f"YOLO 模型已加载并移动到设备: {device}")
except Exception as e:
    logger.warning(f"无法将 YOLO 模型移动到 {device}，将使用默认设备。错误: {e}")

# 预热一次模型以减少首帧卡顿（这会短暂阻塞一次，但能显著降低第一次的长延迟）
try:
    # 生成一个小的规范化张量做 warmup（一次性）
    dummy_img = np.zeros((1, 3, 640, 640), dtype=np.float32)
    with torch.inference_mode():
        # 这里用 model.predict 做一次短推理（verbose=False 关闭多余输出）
        _ = model.predict(torch.from_numpy(dummy_img).to(device), device=device, verbose=False, conf=0.1, iou=0.45)
    logger.info("YOLO 模型预热完成")
except Exception as e:
    # 如果预热失败也无妨（仅作为优化）
    logger.debug(f"YOLO 模型预热失败（可忽略）: {e}")


# 测量参数
MEASUREMENT_WINDOW_SIZE = 11
DEPTH_CONFIDENCE_THRESHOLD = 0.8
# ===== 深度校准与轻量平滑（插入点：在 DEPTH_CONFIDENCE_THRESHOLD 定义之后） =====
# 全局深度校准结构：可保存全局或 per-class 校准系数（多项式）
DEPTH_CALIB = {
    "coeffs": None,        # 全局 polycoeffs (高->低): 用 np.polyfit 拟合 measured_mm -> true_mm
    "degree": None,
    "per_class": {}        # dict: class_name -> coeffs
}

def set_depth_calibration(measured_mm_list, true_mm_list, degree=1, class_name=None):
    """
    用若干已知点拟合校准多项式 measured_mm -> true_mm。
    - measured_mm_list: list/np.array of measured depths (mm)
    - true_mm_list: list/np.array of ground-truth depths (mm)
    - degree: 1 (线性) 或 2 (二次) 等
    - class_name: 若指定则保存为该类别的校准（per-class）
    返回拟合系数（numpy array）
    """
    import numpy as _np
    mm = _np.asarray(measured_mm_list, dtype=_np.float64)
    tt = _np.asarray(true_mm_list, dtype=_np.float64)
    if mm.size < 2 or mm.size != tt.size:
        raise ValueError("需要至少 2 个测量点且长度匹配以进行拟合")
    coeffs = _np.polyfit(mm, tt, degree)
    if class_name:
        DEPTH_CALIB['per_class'][str(class_name)] = coeffs
    else:
        DEPTH_CALIB['coeffs'] = coeffs
        DEPTH_CALIB['degree'] = degree
    return coeffs

def apply_depth_calibration(z_mm, class_name=None):
    """
    把 z_mm（毫米）通过拟合函数映射到校准后的值（毫米）。
    如果没有校准参数，直接返回原值。
    """
    import numpy as _np
    try:
        if class_name and str(class_name) in DEPTH_CALIB.get('per_class', {}):
            coeffs = DEPTH_CALIB['per_class'][str(class_name)]
            return float(_np.polyval(coeffs, float(z_mm)))
        if DEPTH_CALIB.get('coeffs') is None:
            return float(z_mm)
        coeffs = DEPTH_CALIB['coeffs']
        return float(_np.polyval(coeffs, float(z_mm)))
    except Exception:
        return float(z_mm)

# ---------- 替换：smooth_depth_by_key（统一为 mm） ----------
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
    import time as _time
    now = now_ts if now_ts is not None else _time.time()

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

# 保持原有缓存清理函数
def clear_depth_smooth_cache():
    _DEPTH_SMOOTH_CACHE.clear()


POSITION_SMOOTHING_FACTOR = 0.7

# 中文字符串
LABELS = {
    'coords': '坐标', 'invalid': '无效', 'distance': '距离',
    'cm': '厘米', 'world_coords': '三维坐标', 'cleanup': '清理资源中...',
    'cleanup_complete': '清理完成', 'user_quit': '用户请求退出',
    'user_interrupt': '用户中断程序', 'error_occurred': '发生错误',
    'frame_error': '无法获取图像帧', 'program_end': '程序结束'
}

# 可视化参数
L_section = 300.0  # 每节长度300mm
total_L = 3 * L_section  # 总长度900mm
r = 16.0  # 绳孔到中心距离
rope_offsets = [0, np.pi / 2, np.pi, 3 * np.pi / 2]  # 四线90°间隔
rope_colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00']  # 绳子颜色 (红,绿,蓝,黄)
rope_linewidth = 2  # 绳子粗细
disc_interval = 60.0  # 圆盘间隔改为60mm
disc_diameter = 40.0  # 圆盘直径40mm
disc_radius = disc_diameter / 2
x_range = (-1500, 1500)  # X轴范围
y_range = (-1500, 1500)  # Y轴范围
z_range = (-1000, 1500)  # Z轴范围扩展为(-1000, 1500)
total_frames = 100  # 帧数

# 三节圆盘颜色
section_colors = ['#A0D2FF', '#90EE90', '#FFB6C1']  # 浅蓝, 浅绿, 浅粉

# 绳子名称
rope_names = ["绳1", "绳2", "绳3", "绳4", "绳5", "绳6", "绳7", "绳8", "绳9", "绳10", "绳11", "绳12"]

# 全局变量
target_xyz = np.array([300, 200, 600])  # 初始目标位置
planning_result = None  # 规划结果
animation_running = False  # 动画运行状态
detected_objects = []  # 存储检测到的物体
detected_objects_lock = threading.Lock()  # 用于同步检测到的物体列表
selected_object_id = None  # 选中的物体ID
object_bbox_points = {}  # 存储物体的边界框点
stabilization_complete = False  # 稳定检测完成标志
stabilization_counter = 0  # 稳定计数器
STABILIZATION_THRESHOLD = 10  # 稳定阈值，连续10帧稳定
MAX_DEPTH = 1500.0  # 最大深度限制（毫米）

# 新增全局变量
last_left_frame_bgr = None
mono_depth_map_mm = None
last_mono_frame_idx = 0
object_bbox_rects = {}  # 存储物体的边界框矩形
object_3d_bboxes = {}  # 存储物体的3D包围盒


# 立体匹配算法类型
class StereoAlgorithm(Enum):
    SGBM = 1
    BM = 2
    ELAS = 3


# 配置参数
STEREO_ALGORITHM = StereoAlgorithm.SGBM
USE_GPU_ACCELERATION = torch.cuda.is_available()
ENABLE_DEPTH_FILTERING = True
ENABLE_POST_PROCESSING = True





def extract_foreground_by_depth_mode(pts_cm, bin_size_cm=1.0, min_peak_count=10):
    """
    从 pts_cm (cm) 中选出“最近的深度峰”对应的点云（去掉桌面/背景）。
    - pts_cm: Nx3 np.array 单位 cm
    - bin_size_cm: 用于 depth histogram 的 bin 宽度（cm）
    - min_peak_count: 一个 bin 至少需要这个点数才被认为是峰
    返回：filtered_pts_cm (same unit cm)
    逻辑：
      - 在 z 方向做 histogram（近 - 小 z），找到第一个（最靠近相机）且 count >= min_peak_count 的 bin，
        然后选取该 bin 或周围少数 bin 的点作为前景。
      - 这种方法对“桌面 + 小物体”场景非常稳健：桌面通常是一个宽平峰，而近处的物体会有较集中的近峰。
    """
    import numpy as _np
    try:
        if pts_cm is None:
            return pts_cm
        pts = _np.asarray(pts_cm, dtype=_np.float32)
        if pts.size == 0:
            return pts
        z = pts[:, 2]
        z_valid = z[_np.isfinite(z)]
        if z_valid.size == 0:
            return _np.empty((0,3), dtype=_np.float32)
        zmin, zmax = float(z_valid.min()), float(z_valid.max())
        if zmax - zmin < 1e-6:
            return pts
        # build histogram bins
        nbins = max(8, int(_np.ceil((zmax - zmin) / float(bin_size_cm))))
        hist, edges = _np.histogram(z_valid, bins=nbins)
        # find the first (lowest z) bin index with count >= min_peak_count (robust to noise)
        candidate_idxs = _np.where(hist >= min_peak_count)[0]
        if candidate_idxs.size == 0:
            # 若没有足够点的 bin，退回到峰值较大的那个 bin（但要求不为空）
            peak_idx = int(_np.argmax(hist))
        else:
            peak_idx = int(candidate_idxs.min())  # 最近（z 最小）且点数足够的 bin
        z_low, z_high = edges[peak_idx], edges[min(peak_idx + 1, len(edges)-1)]
        # expand a bit to include neighbors
        pad = max(bin_size_cm*1.0, 0.5)
        mask = (z >= (z_low - pad)) & (z <= (z_high + pad))
        out = pts[mask]
        # 如果过滤后点太少，则退回原始 pts（防止把物体全部过滤掉）
        if out.shape[0] < min(4):
            return pts
        return out
    except Exception:
        try:
            return _np.asarray(pts_cm, dtype=_np.float32)
        except Exception:
            return pts_cm



class DepthFilter:
    """深度滤波器类，用于平滑深度测量值"""

    def __init__(self, alpha=0.3, max_deviation=15.0):
        self.alpha = alpha
        self.max_deviation = max_deviation
        self.filtered_depth = None
        self.initialized = False

    def update(self, new_depth):
        if new_depth is None or np.isnan(new_depth).any():
            return self.filtered_depth
        if not self.initialized:
            self.filtered_depth = new_depth
            self.initialized = True
            return self.filtered_depth
        deviation = np.linalg.norm(new_depth - self.filtered_depth)
        if deviation > self.max_deviation:
            return self.filtered_depth
        self.filtered_depth = self.alpha * new_depth + (1 - self.alpha) * self.filtered_depth
        return self.filtered_depth


class PositionStabilizer:
    """对每个 track 的位置做时间平滑（滑动窗口或 EMA）"""
    def __init__(self, window_size=5, ema_alpha=None):
        self.window_size = int(max(1, window_size))
        self.ema_alpha = float(ema_alpha) if ema_alpha is not None else None
        self.history = {}    # track_id -> list of np.array
        self.ema = {}        # track_id -> np.array

    def update(self, track_id, pos):
        """
        pos: numpy array (3,) 或可被转为 np.array 的坐标（单位与 pipeline 保持一致）
        返回平滑后坐标（np.array）
        """
        if pos is None:
            return None
        pos = np.asarray(pos, dtype=np.float32)

        if self.ema_alpha is not None:
            prev = self.ema.get(track_id)
            if prev is None:
                self.ema[track_id] = pos.copy()
            else:
                self.ema[track_id] = self.ema_alpha * pos + (1.0 - self.ema_alpha) * prev
            return self.ema[track_id].copy()

        lst = self.history.setdefault(track_id, [])
        lst.append(pos.copy())
        if len(lst) > self.window_size:
            lst.pop(0)
        arr = np.stack(lst, axis=0)
        return np.mean(arr, axis=0)

# 全局实例（放在头部，确保 detect 等函数能直接使用）
stabilizer = PositionStabilizer(window_size=5, ema_alpha=None)



# ====== 体素降采样（numpy 实现，避免对 Open3D 强依赖） ======
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


class DepthEstimationSystem:
    """完整的深度估计系统"""

    def __init__(self):
        self.deep_model_available = False
        self.model = None
        model_path = "E:/python/yolov11/new/midas_v21_small-70d6b9c8.pt"

        try:
            # 检查本地模型文件是否存在
            if os.path.exists(model_path):
                logging.info(f"加载本地模型: {model_path}")
                checkpoint = torch.load(model_path, map_location=device)

                # 创建 MiDaS 模型实例
                self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=False)

                # 加载权重
                if isinstance(checkpoint, dict):
                    state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
                    self.model.load_state_dict(state_dict, strict=False)
                    logging.info("MiDaS模型从本地state_dict加载成功")
                else:
                    raise RuntimeError("权重文件格式不正确，加载失败")
            else:
                logging.error(f"未找到本地MiDaS模型文件：{model_path}")
                return

            # 将模型转移到设备并设为评估模式
            self.model.to(device)
            self.model.eval()
            self.deep_model_available = True
        except Exception as e:
            logging.error(f"MiDaS模型加载失败: {e}")
            self.deep_model_available = False


class SimpleTracker:
    """
    轻量跟踪器。输入 detections 列表（每项 dict，包含 'bbox'=[x1,y1,x2,y2], 'class_name'）。
    update(detections) 会在每个 detection 上增加 'track_id' 字段并返回修改后的 detections 列表。
    """
    def __init__(self, iou_thresh=0.35, max_missed=5):
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks = {}  # id -> {'bbox':..., 'class_name':..., 'missed':0}

    @staticmethod
    # def _iou(boxA, boxB):
    #        xA = max(boxA[0], boxB[0])
    #        yA = max(boxA[1], boxB[1])
    #        xB = min(boxA[2], boxB[2])
    #        yB = min(boxA[3], boxB[3])
    #        interW = max(0, xB - xA)
    #        interH = max(0, yB - yA)
    #        interArea = interW * interH
    #        boxAArea = max(1e-6, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    #        boxBArea = max(1e-6, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    #        return interArea / (boxAArea + boxBArea - interArea + 1e-9)

    def update(self, detections):
        """
        detections: list of dicts, each must have 'bbox' and 'class_name'.
        Adds 'track_id' to each detection.
        """
        matched_ids = set()
        # 逐个检测与现有 track 匹配（贪心）
        for det in detections:
            best_iou = 0.0
            best_id = None
            b = det.get('bbox')
            cls = det.get('class_name', None)
            if b is None:
                continue
            for tid, tr in self.tracks.items():
                if tr['class_name'] != cls:
                    continue
                iou = SimpleTracker._iou(b, tr['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid
            if best_iou >= self.iou_thresh and best_id is not None:
                det['track_id'] = best_id
                # 更新 track 信息
                self.tracks[best_id]['bbox'] = det['bbox']
                self.tracks[best_id]['missed'] = 0
                matched_ids.add(best_id)
            else:
                # 新 track
                new_id = self.next_id
                self.next_id += 1
                det['track_id'] = new_id
                self.tracks[new_id] = {'bbox': det['bbox'], 'class_name': cls, 'missed': 0}
                matched_ids.add(new_id)

        # 增加未匹配 track 的 missed 计数并删除长期丢失的 track
        for tid in list(self.tracks.keys()):
            if tid not in matched_ids:
                self.tracks[tid]['missed'] += 1
                if self.tracks[tid]['missed'] > self.max_missed:
                    del self.tracks[tid]

        return detections

        # 初始化跟踪器（模块全局，main 外）
tracker = SimpleTracker(iou_thresh=0.35, max_missed=5)

class PositionTracker:
    def __init__(self, history_size=30):
        self.positions = defaultdict(lambda: {
            'left_positions': deque(maxlen=history_size),
            'right_positions': deque(maxlen=history_size),
            'left_confidences': deque(maxlen=history_size),
            'right_confidences': deque(maxlen=history_size),
            'fused_positions': deque(maxlen=history_size),
            'last_update': 0,
            'class_id': -1
        })
        self.R = R
        self.T = T
        self.object_counter = 0
        self.tracked_ids = {}
        self.last_print_time = 0
        self.coordinate_system = "X:右为正, Y:下为正, Z:前为正"
        self.depth_filters = {}

    def calculate_confidence(self, surrounding_points):
        if len(surrounding_points) < 3:
            return 0.0
        std_dev = np.std(surrounding_points, axis=0)
        mean_std = np.mean(std_dev)
        confidence = np.exp(-mean_std / 50.0)
        return min(max(confidence, 0.0), 1.0)

    def transform_to_left_coordinates(self, right_position):
        return np.dot(self.R, right_position) + self.T

    def get_or_create_id(self, class_id, position):
        if class_id not in self.tracked_ids:
            self.object_counter += 1
            self.tracked_ids[class_id] = self.object_counter
            return self.object_counter
        return self.tracked_ids[class_id]

    def update_position(self, obj_id, position, confidence, class_id, is_left=True):
        if np.isnan(position).any() or confidence < 0.1:
            return
        if not is_left:
            position = self.transform_to_left_coordinates(position)
        if obj_id not in self.depth_filters:
            self.depth_filters[obj_id] = DepthFilter()
        filtered_position = self.depth_filters[obj_id].update(position)
        if is_left:
            self.positions[obj_id]['left_positions'].append(filtered_position)
            self.positions[obj_id]['left_confidences'].append(confidence)
        else:
            self.positions[obj_id]['right_positions'].append(filtered_position)
            self.positions[obj_id]['right_confidences'].append(confidence)
        self.positions[obj_id]['class_id'] = class_id
        self.positions[obj_id]['last_update'] = time.time()
        self.fuse_positions(obj_id)

    def fuse_positions(self, obj_id):
        data = self.positions[obj_id]
        if len(data['left_positions']) == 0 or len(data['right_positions']) == 0:
            return
        left_pos = data['left_positions'][-1]
        left_conf = data['left_confidences'][-1]
        right_pos = data['right_positions'][-1]
        right_conf = data['right_confidences'][-1]
        total_conf = left_conf + right_conf
        if total_conf < 0.01:
            fused_pos = (left_pos + right_pos) / 2
        else:
            fused_pos = (left_pos * left_conf + right_pos * right_conf) / total_conf
        data['fused_positions'].append(fused_pos)

    def get_fused_position(self, obj_id):
        if obj_id in self.positions and len(self.positions[obj_id]['fused_positions']) > 0:
            positions = list(self.positions[obj_id]['fused_positions'])
            weights = [POSITION_SMOOTHING_FACTOR ** i for i in range(len(positions))]
            weights = np.array(weights[::-1]) / sum(weights)
            smooth_position = np.zeros(3)
            for i, pos in enumerate(positions):
                smooth_position += weights[i] * pos
            return smooth_position
        return None

    def get_position_change(self, obj_id):
        if obj_id in self.positions and len(self.positions[obj_id]['fused_positions']) > 1:
            initial_pos = self.positions[obj_id]['fused_positions'][0]
            current_pos = self.get_fused_position(obj_id)
            return current_pos - initial_pos
        return None

    def get_tracked_objects(self):
        tracked_objects = []
        current_time = time.time()
        for obj_id, data in self.positions.items():
            if current_time - data['last_update'] < 5.0:
                fused_pos = self.get_fused_position(obj_id)
                position_change = self.get_position_change(obj_id)
                if fused_pos is not None:
                    tracked_objects.append({
                        'id': obj_id, 'class_id': data['class_id'],
                        'position': fused_pos, 'change': position_change
                    })
        return tracked_objects

    def is_stable(self, obj_id):
        """检查物体位置是否稳定"""
        if obj_id in self.positions and len(self.positions[obj_id]['fused_positions']) > 5:
            recent_positions = list(self.positions[obj_id]['fused_positions'])[-5:]
            # 计算最近5个位置的标准差
            positions_array = np.array(recent_positions)
            std_dev = np.std(positions_array, axis=0)
            mean_std = np.mean(std_dev)
            return mean_std < 2.0  # 2cm稳定性阈值
        return False

    def print_positions(self):
        current_time = time.time()
        if current_time - self.last_print_time > 0.5:
            self.last_print_time = current_time
            tracked_objects = self.get_tracked_objects()
            if tracked_objects:
                logger.info("===== 物体位置信息 =====")
                logger.info(f"坐标系定义: {self.coordinate_system}")
                for obj in tracked_objects:
                    pos = obj['position']
                    class_name = model.names[obj['class_id']]
                    logger.info(f"物体ID: {get_object_id(obj)}, 类别: {class_name}")
                    logger.info(f"位置: X={pos[0]:.1f}, Y={pos[1]:.1f}, Z={pos[2]:.1f} cm")
                    if obj['change'] is not None:
                        change = obj['change']
                        logger.info(f"位置变化: ΔX={change[0]:.1f}, ΔY={change[1]:.1f}, ΔZ={change[2]:.1f} cm")
                    logger.info("-" * 40)


# 初始化位置跟踪器
position_tracker = PositionTracker()
xyz_lock = threading.Lock()

def safe_str(val, fmt="{:.1f}"):
    """将可能为 None/nan 的数值安全格式化为字符串（用于显示）"""
    try:
        if val is None:
            return "?"
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return "?"
        return fmt.format(v)
    except Exception:
        try:
            # 如果是数组
            arr = np.asarray(val)
            if arr.size > 0:
                return "(" + ",".join(safe_str(x, fmt) for x in arr.flatten()[:3]) + ")"
            else:
                return "?"
        except Exception:
            return "?"

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


def match_detections_between_views(dets_left, dets_right, iou_thresh=0.25):
    """
    改进匹配：IoU + class + y-center (epipolar 行一致性) + size_ratio 的加权评分。
    输入与原来一致：dets_left/dets_right = list of tuples (class_id, conf, [x1,y1,x2,y2])
    返回 matches 列表：[(left_item, right_item), ...]
    """
    def _iou(a, b):
        xA = max(a[0], b[0]); yA = max(a[1], b[1])
        xB = min(a[2], b[2]); yB = min(a[3], b[3])
        inter_w = max(0, xB - xA); inter_h = max(0, yB - yA)
        inter = inter_w * inter_h
        areaA = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        areaB = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        union = areaA + areaB - inter
        return inter / union if union > 0 else 0.0

    matches = []
    used_r = set()

    # y-center 容忍阈（像素），利用 imageHeight 全局变量（若不可用则回退到 8 px）
    try:
        y_thresh_px = int(max(8, 0.02 * imageHeight))
    except Exception:
        y_thresh_px = 8

    for i, (clsL, confL, boxL) in enumerate(dets_left):
        best_j = None
        best_score = 0.0
        x1L, y1L, x2L, y2L = boxL
        cxL = (x1L + x2L) / 2.0
        cyL = (y1L + y2L) / 2.0
        wL = max(1.0, (x2L - x1L))

        for j, (clsR, confR, boxR) in enumerate(dets_right):
            if j in used_r:
                continue
            if clsL != clsR:
                continue

            x1R, y1R, x2R, y2R = boxR
            cxR = (x1R + x2R) / 2.0
            cyR = (y1R + y2R) / 2.0
            wR = max(1.0, (x2R - x1R))

            # 1) y-center 约束（立体校正后同一物体 y 应接近）
            ydiff = abs(cyL - cyR)
            if ydiff > (y_thresh_px * 3):  # 绝对错位（非常大）直接跳过
                continue

            # 2) IoU（基本重合度）
            iou_val = _iou(boxL, boxR)

            # 3) 宽度相似性（越接近越好）
            size_ratio = min(wL, wR) / max(wL, wR)

            # 综合评分（权重可调整）
            # - IoU 占主导（0.6），y-center 一致性占 0.25，size_ratio 占 0.15
            y_score = max(0.0, 1.0 - (ydiff / (y_thresh_px + 1e-6)))
            score = 0.60 * iou_val + 0.25 * y_score + 0.15 * size_ratio

            if score > best_score:
                best_score = score
                best_j = j

        # 匹配阈（确保不会用微弱相似度匹配错误对象）
        if best_j is not None and best_score >= 0.35 and _iou(boxL, dets_right[best_j][2]) >= (0.05 if iou_thresh < 0.05 else iou_thresh):
            matches.append((dets_left[i], dets_right[best_j]))
            used_r.add(best_j)

    return matches


def stereo_rectify():
    """立体校正，生成校正映射"""
    global Rl, Rr, Pl, Pr, Q, mapLx, mapLy, mapRx, mapRy
    Rl, Rr, Pl, Pr, Q, _, _ = cv2.stereoRectify(
        cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR,
        imageSize, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1, newImageSize=imageSize
    )
    mapLx, mapLy = cv2.initUndistortRectifyMap(
        cameraMatrixL, distCoeffL, Rl, Pl, imageSize, cv2.CV_32FC1
    )
    mapRx, mapRy = cv2.initUndistortRectifyMap(
        cameraMatrixR, distCoeffR, Rr, Pr, imageSize, cv2.CV_32FC1
    )
    # 调试输出：打印 Q 的关键信息（便于核查 baseline 与单位）
    try:
        if Q is not None:
            q_32 = float(Q[3,2]) if Q.shape == (4,4) else None
            baseline_est_mm = (1.0 / q_32) if (q_32 is not None and abs(q_32) > 1e-9) else None
            logger.info(f"stereo_rectify: Q[3,2]={q_32}, baseline_est (mm) ~ {baseline_est_mm}, Q[2,3]={float(Q[2,3])}")
    except Exception:
        pass



def verify_calibration():
    """验证标定参数合理性"""
    det_R = np.linalg.det(R)
    if abs(det_R - 1.0) > 0.01:
        logger.warning(f"警告: 旋转矩阵行列式应为1 (当前: {det_R:.6f})")
    I = np.eye(3)
    diff = np.abs(np.dot(R, R.T) - I)
    if np.max(diff) > 0.01:
        logger.warning("警告: 旋转矩阵不是正交矩阵")
    logger.info("标定验证完成")


def calculate_image_quality_metrics(image):
    """计算图像质量指标，用于自适应参数调整"""
    # 计算图像对比度（使用标准差）
    contrast = np.std(image)

    # 计算图像亮度（使用平均值）
    brightness = np.mean(image)

    # 计算图像噪声（使用拉普拉斯算子的方差）
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()

    # 计算图像梯度幅值（用于评估纹理丰富度）
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.mean(np.sqrt(sobelx ** 2 + sobely ** 2))

    return {
        'contrast': contrast,
        'brightness': brightness,
        'laplacian_var': laplacian_var,
        'gradient_magnitude': gradient_magnitude
    }


def adaptive_preprocess_images(imgL, imgR):
    """
    自适应图像预处理管道
    根据图像质量指标动态调整预处理参数
    """
    # 计算左右图像的质量指标
    metricsL = calculate_image_quality_metrics(imgL)
    metricsR = calculate_image_quality_metrics(imgR)

    # 使用左右图像指标的平均值
    avg_contrast = (metricsL['contrast'] + metricsR['contrast']) / 2
    avg_brightness = (metricsL['brightness'] + metricsR['brightness']) / 2
    avg_gradient = (metricsL['gradient_magnitude'] + metricsR['gradient_magnitude']) / 2

    # 根据指标自适应调整参数
    # 低对比度 -> 增加CLAHE的clipLimit
    clahe_clip = 2.0 + (40 - min(avg_contrast, 40)) / 40 * 3.0  # 2.0-5.0

    # 低亮度 -> 增加gamma校正
    gamma = 1.0
    if avg_brightness < 60:
        gamma = 1.0 + (60 - avg_brightness) / 60 * 0.8  # 1.0-1.8

    # 弱纹理 -> 增加梯度增强
    enhance_gradients = avg_gradient < 10

    # 1. 伽马校正（针对亮度问题）
    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        imgL = cv2.LUT(imgL, table)
        imgR = cv2.LUT(imgR, table)

    # 2. 自适应直方图均衡化 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    imgL_clahe = clahe.apply(imgL)
    imgR_clahe = clahe.apply(imgR)

    # 3. 光度标准化
    meanL = np.mean(imgL_clahe)
    meanR = np.mean(imgR_clahe)
    stdL = np.std(imgL_clahe)
    stdR = np.std(imgR_clahe)

    imgR_normalized = (imgR_clahe - meanR) * (stdL / (stdR + 1e-8)) + meanL
    imgR_normalized = np.clip(imgR_normalized, 0, 255).astype(np.uint8)
    imgL_normalized = imgL_clahe  # 左图保持不变

    # 4. 联合双边滤波
    # 根据图像噪声水平调整滤波强度
    noise_level = min(metricsL['laplacian_var'], metricsR['laplacian_var'])
    d = 5
    sigmaColor = 10 + (50 - min(noise_level, 50)) / 50 * 20  # 10-30
    sigmaSpace = 10 + (50 - min(noise_level, 50)) / 50 * 20  # 10-30

    try:
        # 尝试使用联合双边滤波
        imgR_guided = cv2.ximgproc.jointBilateralFilter(imgL_normalized, imgR_normalized, d, sigmaColor, sigmaSpace)
        imgL_guided = cv2.ximgproc.jointBilateralFilter(imgL_normalized, imgL_normalized, d, sigmaColor, sigmaSpace)
    except:
        # 回退到标准双边滤波
        imgL_guided = cv2.bilateralFilter(imgL_normalized, d, sigmaColor, sigmaSpace)
        imgR_guided = cv2.bilateralFilter(imgR_normalized, d, sigmaColor, sigmaSpace)

    # 5. 可选：梯度增强（针对弱纹理场景）
    if enhance_gradients:
        # 使用自定义核进行锐化
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        imgL_preprocessed = cv2.filter2D(imgL_guided, -1, kernel)
        imgR_preprocessed = cv2.filter2D(imgR_guided, -1, kernel)
    else:
        imgL_preprocessed = imgL_guided
        imgR_preprocessed = imgR_guided

    # 记录使用的参数（用于调试）
    params_used = {
        'clahe_clip': clahe_clip,
        'gamma': gamma,
        'sigmaColor': sigmaColor,
        'sigmaSpace': sigmaSpace,
        'enhance_gradients': enhance_gradients
    }

    return imgL_preprocessed, imgR_preprocessed, params_used

def postprocess_disparity(disp):
    """
    输入 disp (float32, 单位像素)，返回同尺寸的 float32 disparity（空洞和噪声被平滑/填充）。
    使用方法：在你 compute() 后立即调用，例如:
        disp = postprocess_disparity(disp)
    """
    if disp is None or disp.size == 0:
        return disp

    # 复制，避免原地修改
    d = disp.astype(np.float32).copy()

    # 1) 中值滤波去孤点
    try:
        d_med = cv2.medianBlur(d, 5)
    except Exception:
        d_med = d

    # 2) 双边滤波保持边缘
    try:
        d_bi = cv2.bilateralFilter(d_med, 9, 75, 75)
    except Exception:
        d_bi = d_med

    # 3) 小孔洞填充（局部均值法）
    mask_invalid = (d_bi <= 0) | (~np.isfinite(d_bi))
    if mask_invalid.all():
        return d_bi

    kernel = np.ones((5,5), dtype=np.float32)
    valid_mask = (~mask_invalid).astype(np.float32)
    # 局部和与计数，用于局部均值
    sum_local = cv2.filter2D((d_bi * valid_mask).astype(np.float32), -1, kernel)
    count_local = cv2.filter2D(valid_mask, -1, kernel)

    fill_vals = np.zeros_like(d_bi, dtype=np.float32)
    nonzero = count_local > 0
    fill_vals[nonzero] = sum_local[nonzero] / count_local[nonzero]

    # 仅用局部均值填充那些失效且有可用邻居的位置
    fill_mask = mask_invalid & (count_local > 0)
    d_bi[fill_mask] = fill_vals[fill_mask]

    # 4) 最后再一个小的 median 以收尾
    try:
        d_out = cv2.medianBlur(d_bi, 3)
    except Exception:
        d_out = d_bi

    return d_out.astype(np.float32)



def stereo_match(rectifyImageL, rectifyImageR):
    """
    改进版立体匹配（SGBM + optional WLS），并保证始终返回左右视差与左右点云。
    输入：rectifyImageL, rectifyImageR - 单通道灰度 uint8
    输出：
      disp_color_left: (H,W,3) uint8 可视化伪彩色图
      xyz_left_mm: HxWx3 float32 点云（单位：毫米，保证输出为 mm）
      confidence_left: HxW float32 (0~1)
      disp_left: HxW float32 视差（像素）
      disp_right: HxW float32 视差（像素）
      xyz_right_mm: HxWx3 float32 点云（单位：毫米，基于左点云 remap 得到右点云）
    说明：此函数尽量保证在没有 ximgproc 时也能产生右视差用于显示，并在返回前做单位自适应转换（确保输出为 mm）。
    """
    try:
        global Q, imageWidth, imageHeight
        if rectifyImageL is None or rectifyImageR is None:
            H = imageHeight; W = imageWidth
            empty_xyz = np.full((H, W, 3), np.nan, dtype=np.float32)
            return np.zeros((H, W, 3), dtype=np.uint8), empty_xyz, np.zeros((H, W), dtype=np.float32), \
                   np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32), empty_xyz

        left = rectifyImageL
        right = rectifyImageR
        # ensure uint8 single channel
        if left.dtype != np.uint8:
            left = cv2.normalize(left, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if right.dtype != np.uint8:
            right = cv2.normalize(right, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # CLAHE 预处理
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            left_eq = clahe.apply(left)
            right_eq = clahe.apply(right)
        except Exception:
            left_eq = left
            right_eq = right

        # SGBM 参数（可调）
        window_size = 5
        min_disp = 0
        num_disp = 16 * 8  # 128
        P1 = 8 * 3 * window_size ** 2
        P2 = 32 * 3 * window_size ** 2

        left_matcher = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=window_size,
            P1=P1,
            P2=P2,
            disp12MaxDiff=2,
            uniquenessRatio=12,
            speckleWindowSize=200,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # 尝试使用 ximgproc 的右匹配器 + WLSFilter（若不可用则回退为独立的 SGBM 右匹配）
        use_wls = False
        wls_filter = None
        try:
            right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
            wls_filter.setLambda(8000.0)
            wls_filter.setSigmaColor(1.5)
            use_wls = True
        except Exception:
            # 回退：用另一个 StereoSGBM 实例做右目匹配（保证 disp_right 可视化）
            right_matcher = cv2.StereoSGBM_create(
                minDisparity=min_disp,
                numDisparities=num_disp,
                blockSize=window_size,
                P1=P1,
                P2=P2,
                disp12MaxDiff=10,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=2,
                preFilterCap=63,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )
            use_wls = False

        # 计算原始视差（SGBM 输出需要 /16）
        disp_left_raw = left_matcher.compute(left_eq, right_eq).astype(np.float32) / 16.0
        disp_right_raw = None
        try:
            # 右 matcher 输入顺序 swapped
            disp_right_raw = right_matcher.compute(right_eq, left_eq).astype(np.float32) / 16.0
        except Exception:
            # 极端兜底（不应发生，但保证变量存在）
            disp_right_raw = np.zeros_like(disp_left_raw, dtype=np.float32)

        # WLS 或 后处理
        if use_wls and wls_filter is not None:
            try:
                disp_filtered = wls_filter.filter(disp_left_raw, left_eq, None, disp_right_raw).astype(np.float32)
            except Exception:
                disp_filtered = postprocess_disparity(disp_left_raw)
        else:
            disp_filtered = postprocess_disparity(disp_left_raw)

        # 同步后处理右目视差供显示（后处理保证无负值/NaN）
        disp_right_proc = postprocess_disparity(disp_right_raw)

        # 清理 NaN 与负值
        disp_filtered[~np.isfinite(disp_filtered)] = 0.0
        disp_filtered[disp_filtered < 0] = 0.0
        disp_right_proc[~np.isfinite(disp_right_proc)] = 0.0
        disp_right_proc[disp_right_proc < 0] = 0.0

        # 可视化左视差（伪彩色）
        if np.max(disp_filtered) > 0:
            disp_vis = (disp_filtered / (np.max(disp_filtered) + 1e-9) * 255.0).clip(0, 255).astype(np.uint8)
            disp_color_left = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        else:
            disp_color_left = np.zeros((left.shape[0], left.shape[1], 3), dtype=np.uint8)

        # 简单置信度图（基于视差是否有效与幅值归一）
        conf = np.clip(disp_filtered / (np.max(disp_filtered) + 1e-9), 0.0, 1.0).astype(np.float32)

        # 使用 Q 做 3D 反投影（返回 xyz 单位会根据 Q/外参决定，我们后面 normalize）
        H, W = disp_filtered.shape
        if 'Q' not in globals() or Q is None:
            xyz_left = np.full((H, W, 3), np.nan, dtype=np.float32)
        else:
            try:
                # OpenCV 要求输入为 single-channel float
                xyz_left = cv2.reprojectImageTo3D(disp_filtered.astype(np.float32), Q)
                xyz_left = np.asarray(xyz_left, dtype=np.float32)
            except Exception:
                # 兜底
                xyz_left = np.full((H, W, 3), np.nan, dtype=np.float32)

        # 单位自适应：根据 Z 中位数自动将 xyz 归一到 毫米(mm)
        def ensure_xyz_in_mm(xyz_map):
            """
            统一的 ensure_xyz_in_mm（保守版，不做隐式尺度缩放）
            目的：确保传入的 xyz_map 是 numpy array、dtype=float32，且不随意做 mm<->cm 自动转换。
            返回值：原样以 mm 为单位的 numpy array（或原始输入的 np.array 形式）。
            使用说明：上游应保证 reprojectImageTo3D / stereo 输出为 mm（如果不是，请在生成处显式转换）。
            """
            import numpy as _np
            try:
                if xyz_map is None:
                    return None
                arr = _np.asarray(xyz_map, dtype=_np.float32)
                if arr.size == 0:
                    return arr
                # 简单检测：Z 的中位数是否在合理 mm 范围（10 mm ~ 10000 mm = 10m）
                try:
                    z = arr[..., 2]
                    mask = _np.isfinite(z) & (z > 0)
                    if mask.sum() == 0:
                        # 没有有效深度点，直接返回原数组
                        return arr
                    zmed = float(_np.median(z[mask]))
                    # 只做警告，不做自动缩放
                    if zmed < 10.0 or zmed > 20000.0:
                        # 打印 debug 信息以便人工校准（不会修改数据）
                        try:
                            logger.warning(
                                f"ensure_xyz_in_mm: median Z appears suspicious: {zmed:.3f} (expected mm). No autoscale applied.")
                        except Exception:
                            pass
                except Exception:
                    # 任何异常都不做修改
                    pass
                return arr
            except Exception:
                # 兜底：返回原始输入（尽量不抛异常）
                try:
                    return xyz_map
                except Exception:
                    return None

        xyz_left = ensure_xyz_in_mm(xyz_left)

        # 构造右目的 xyz：优先使用 remap xyz_left -> right（这个方法和后续点云提取/融合兼容）
        try:
            xs, ys = np.meshgrid(np.arange(W), np.arange(H))
            # 映射：右视图的像素 (x_r) 对应左视图的 x_l = x_r + disp_left(x_l) 。
            # 要把左->右：map_x = xs - disp_filtered
            map_x = (xs - disp_filtered).astype(np.float32)
            map_y = ys.astype(np.float32)
            xyz_right = np.full_like(xyz_left, np.nan, dtype=np.float32)
            for c in range(3):
                ch = xyz_left[:, :, c].astype(np.float32)
                rem = cv2.remap(ch, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
                xyz_right[:, :, c] = rem
            invalid_mask = (disp_filtered <= 0) | (~np.isfinite(xyz_left[:, :, 2]))
            xyz_right[invalid_mask, :] = np.nan
        except Exception:
            xyz_right = np.full_like(xyz_left, np.nan, dtype=np.float32)

        # 把最新的视差与点云导出成全局变量，供后续诊断用（不会影响原逻辑）
        try:
            global last_disp_left, last_disp_right, last_xyz_left, last_xyz_right
            last_disp_left = disp_filtered.astype(np.float32).copy() if 'disp_filtered' in locals() else None
            last_disp_right = disp_right_proc.astype(np.float32).copy() if 'disp_right_proc' in locals() else None
            try:
                last_xyz_left = xyz_left.copy()
            except Exception:
                last_xyz_left = None
            try:
                last_xyz_right = xyz_right.copy()
            except Exception:
                last_xyz_right = None
        except Exception:
            # 不影响主流程：若导出失败，则忽略
            pass

        # 返回（注意：disp 返回 float32 像素单位）
        return disp_color_left, xyz_left, conf, disp_filtered.astype(np.float32), disp_right_proc.astype(
            np.float32), xyz_right

    except Exception as e:
        logger.error(f"stereo_match: 异常: {e}")
        H = imageHeight if 'imageHeight' in globals() else 480
        W = imageWidth if 'imageWidth' in globals() else 640
        empty_xyz = np.full((H, W, 3), np.nan, dtype=np.float32)
        return np.zeros((H, W, 3), dtype=np.uint8), empty_xyz, np.zeros((H, W), dtype=np.float32), \
               np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32), empty_xyz



def segment_object_with_yolo(frame, bbox_2d):
    """使用YOLO检测框创建粗略的分割掩码"""
    x1, y1, x2, y2 = bbox_2d
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # 确保边界框在图像范围内
    height, width = frame.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width - 1, x2)
    y2 = min(height - 1, y2)

    # 创建与图像相同大小的掩码
    mask = np.zeros((height, width), dtype=np.uint8)

    # 在边界框内创建椭圆形状的掩码
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    ellipse_width = (x2 - x1) // 2
    ellipse_height = (y2 - y1) // 2

    # 绘制椭圆
    cv2.ellipse(mask, (center_x, center_y), (ellipse_width, ellipse_height),
                0, 0, 360, 255, -1)

    return mask


def enhanced_segmentation(frame, bbox_2d, class_name):
    """增强的物体分割方法，结合边缘检测和区域生长"""
    x1, y1, x2, y2 = bbox_2d
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # 提取ROI区域
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    # 转换为灰度图
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 使用自适应阈值处理
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # 边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 结合边缘和阈值结果
    combined = cv2.bitwise_or(binary, edges)

    # 形态学操作
    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    # 区域生长 - 从中心点开始
    center_x, center_y = (x2 - x1) // 2, (y2 - y1) // 2
    mask = np.zeros((roi.shape[0] + 2, roi.shape[1] + 2), np.uint8)
    cv2.floodFill(combined, mask, (center_x, center_y), 255)

    # 创建完整图像大小的掩码
    full_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = combined

    return full_mask



def texture_mapping(point_cloud, frame, bbox_2d, camera_matrix=None):
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
        camera_matrix = cameraMatrixL

    frame_h, frame_w = frame.shape[:2]

    pts = np.asarray(point_cloud, dtype=np.float32).reshape(-1, 3)  # Nx3
    colors = np.zeros((len(pts), 3), dtype=np.float32)

    for i, p in enumerate(pts):
        X, Y, Z = p
        if Z is None or Z == 0 or np.isnan(Z) or np.isinf(Z):
            colors[i] = np.array([0.5, 0.5, 0.5])  # 灰色
            continue

        try:
            u = int((X * camera_matrix[0, 0] / Z) + camera_matrix[0, 2])
            v = int((Y * camera_matrix[1, 1] / Z) + camera_matrix[1, 2])
        except Exception:
            colors[i] = np.array([0.5, 0.5, 0.5])
            continue

        if 0 <= u < frame_w and 0 <= v < frame_h:
            b, g, r = frame[v, u].astype(np.float32)
            colors[i] = np.array([r, g, b]) / 255.0  # 归一化到 0~1
        else:
            colors[i] = np.array([0.5, 0.5, 0.5])

    return pts, colors


def refined_postprocessing(disparity_map, left_image=None, right_image=None):
    """
    改进的视差图后处理管道
    包含多种后处理技术，可根据需要启用或禁用
    """
    # 参数配置 - 可以根据实际效果调整这些参数
    params = {
        'enable_median_filter': True,  # 中值滤波
        'enable_speckle_filter': True,  # 散斑滤波
        'enable_hole_filling': True,  # 空洞填充
        'enable_bilateral_filter': False,  # 双边滤波（保持边缘）
        'enable_subpixel_refinement': True,  # 亚像素优化
        'enable_lr_check': True,  # 左右一致性检查

        # 滤波器参数
        'median_kernel_size': 5,
        'speckle_window_size': 100,
        'speckle_range': 2,
        'bilateral_d': 5,
        'bilateral_sigma_color': 10,
        'bilateral_sigma_space': 10,

        # 空洞填充参数
        'max_hole_size': 10,  # 最大填充空洞大小
        'fill_method': 'median'  # 填充方法：'median', 'mean', 'nearest'
    }

    disp_processed = disparity_map.copy()

    # 1. 散斑滤波
    disp_int16 = (disp_processed * 16).astype(np.int16)
    cv2.filterSpeckles(disp_int16, 0, 100, 2)
    disp_processed = disp_int16.astype(np.float32) / 16.0

    # 2. 高斯滤波
    disp_processed = cv2.GaussianBlur(disp_processed, (5, 5), 0)

    # 3. 简单的空洞填充
    holes = disp_processed == 0
    if np.any(holes):
        # 使用邻域中值填充空洞
        from scipy import ndimage
        filled = ndimage.median_filter(disp_processed, size=3)
        disp_processed[holes] = filled[holes]

    return disp_processed


def left_right_consistency_check(disp_left, left_img, right_img):
    """
    左右一致性检查：通过比较左右视差图来检测遮挡和错误匹配
    """
    # 为简化起见，这里使用一个简化的实现
    # 实际应用中可能需要计算右视差图并进行精确比较

    # 创建一个一致性掩码（简化版）
    # 在实际实现中，应该计算右视差图并检查 |d_left(x,y) - d_right(x-d_left(x,y), y)| < threshold
    height, width = disp_left.shape

    # 使用简单的梯度一致性作为代理检查
    grad_x_left = cv2.Sobel(left_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_x_right = cv2.Sobel(right_img, cv2.CV_64F, 1, 0, ksize=3)

    # 计算梯度差异
    grad_diff = np.abs(grad_x_left - grad_x_right)

    # 在梯度差异大的区域，视差可能不可靠，进行平滑处理
    unreliable_mask = grad_diff > np.percentile(grad_diff, 75)  # 差异最大的25%区域
    disp_processed = disp_left.copy()

    # 对不可靠区域的视差进行中值滤波
    if np.any(unreliable_mask):
        disp_unreliable = disp_processed[unreliable_mask]
        if len(disp_unreliable) > 0:
            median_val = np.median(disp_unreliable[disp_unreliable > 0])
            disp_processed[unreliable_mask] = median_val

    return disp_processed


def subpixel_refinement(disparity_map):
    """
    亚像素优化：通过二次插值提高视差精度
    """
    disp_refined = disparity_map.copy().astype(np.float32)
    height, width = disparity_map.shape

    # 只对有效视差点进行优化
    valid_mask = disparity_map > 0

    if not np.any(valid_mask):
        return disparity_map

    # 为简化实现，使用一个基于邻域的子像素优化近似
    # 在实际应用中，可以使用更精确的二次曲线拟合方法

    # 对每个有效点，检查其邻域并计算加权平均
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if valid_mask[y, x]:
                # 获取3x3邻域
                neighborhood = disparity_map[y - 1:y + 2, x - 1:x + 2]
                valid_neighbors = neighborhood[neighborhood > 0]

                if len(valid_neighbors) >= 5:  # 至少有5个有效邻域点
                    # 计算加权平均（中心点权重更高）
                    weights = np.array([0.7, 1.0, 0.7,
                                        1.0, 2.0, 1.0,
                                        0.7, 1.0, 0.7])[:len(valid_neighbors)]
                    weights = weights / np.sum(weights)
                    disp_refined[y, x] = np.sum(valid_neighbors * weights)

    return disp_refined


def advanced_hole_filling(disparity_map, max_size=10, method='median'):
    """
    高级空洞填充算法
    """
    disp_filled = disparity_map.copy()
    height, width = disparity_map.shape

    # 找到所有空洞（视差为0的区域）
    holes = disparity_map == 0
    labeled_array, num_features = ndimage.label(holes)

    # 处理每个连通的空洞区域
    for i in range(1, num_features + 1):
        # 获取当前空洞的掩码
        hole_mask = labeled_array == i
        hole_size = np.sum(hole_mask)

        # 只处理小于最大尺寸的空洞
        if hole_size <= max_size:
            # 找到空洞区域的边界
            dilated_mask = ndimage.binary_dilation(hole_mask, iterations=1)
            boundary_mask = dilated_mask & ~hole_mask

            # 获取边界上的有效视差值
            boundary_values = disparity_map[boundary_mask]
            valid_boundary_values = boundary_values[boundary_values > 0]

            if len(valid_boundary_values) > 0:
                # 根据选择的方法计算填充值
                if method == 'median':
                    fill_value = np.median(valid_boundary_values)
                elif method == 'mean':
                    fill_value = np.mean(valid_boundary_values)
                elif method == 'nearest':
                    # 找到最近的边界点
                    y, x = np.where(hole_mask)
                    if len(y) > 0 and len(x) > 0:
                        center_y, center_x = np.mean(y), np.mean(x)
                        # 简化实现：使用最近边界点的值
                        distances = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
                        nearest_idx = np.argmin(distances)
                        nearest_y, nearest_x = y[nearest_idx], x[nearest_idx]
                        # 寻找最近的有效边界点
                        boundary_y, boundary_x = np.where(boundary_mask)
                        if len(boundary_y) > 0:
                            distances = np.sqrt((boundary_y - nearest_y) ** 2 + (boundary_x - nearest_x) ** 2)
                            closest_boundary_idx = np.argmin(distances)
                            fill_value = disparity_map[
                                boundary_y[closest_boundary_idx], boundary_x[closest_boundary_idx]]

                # 填充空洞
                disp_filled[hole_mask] = fill_value

    return disp_filled


def calculate_enhanced_depth(disparity_map, Q_matrix, confidence_map=None, min_depth=100.0, max_depth=5000.0):
    """
    增强的深度计算函数，提供更可靠和准确的深度估计

    Args:
        disparity_map: 视差图（浮点型）
        Q_matrix: 重投影矩阵
        confidence_map: 可选，视差置信度图
        min_depth: 最小有效深度（mm）
        max_depth: 最大有效深度（mm）

    Returns:
        enhanced_xyz: 增强的3D坐标图
        depth_confidence: 深度置信度图
    """
    height, width = disparity_map.shape

    # 1. 使用OpenCV标准函数计算初始3D坐标
    xyz = cv2.reprojectImageTo3D(disparity_map, Q_matrix, handleMissingValues=True)

    # 2. 创建深度置信度图（如果没有提供）
    if confidence_map is None:
        # 基于视差值的简单置信度估计
        confidence_map = np.ones_like(disparity_map, dtype=np.float32)
        # 低视差异常低置信度
        confidence_map[disparity_map < 1.0] = 0.1
        # 视差平滑区域的较低置信度（可能缺乏纹理）
        gradient_x = cv2.Sobel(disparity_map, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(disparity_map, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        low_texture_mask = gradient_magnitude < 0.5
        confidence_map[low_texture_mask] *= 0.7

    # 3. 计算深度值（Z坐标）
    depth_map = xyz[:, :, 2]  # Z坐标就是深度

    # 4. 应用深度相关的误差模型进行校准
    # 深度误差通常与深度平方成正比：error ~ k * Z^2
    # 我们可以使用这个模型来估计深度置信度
    calibrated_depth = depth_map.copy()
    depth_confidence = confidence_map.copy()

    # 深度相关的置信度调整
    z = depth_map.flatten()
    valid_z = z[(z > min_depth) & (z < max_depth)]

    if len(valid_z) > 10:
        # 估计深度误差模型参数（简化版）
        # 在实际系统中，这应该通过标定来确定
        k = 1e-7  # 误差系数，需要通过实验标定

        # 调整置信度：深度越大，置信度越低
        depth_confidence = depth_confidence / (1 + k * depth_map ** 2)

    # 5. 创建增强的XYZ地图
    enhanced_xyz = xyz.copy()

    # 6. 应用深度范围过滤
    invalid_depth_mask = (depth_map < min_depth) | (depth_map > max_depth)
    enhanced_xyz[invalid_depth_mask] = np.nan
    depth_confidence[invalid_depth_mask] = 0.0

    # 7. 可选：基于深度的自适应滤波
    # 近处物体需要更少的平滑，远处物体需要更多的平滑
    if np.any(depth_map > 0):
        avg_depth = np.nanmedian(depth_map[depth_map > 0])
        if avg_depth > 1000:  # 如果平均深度较大，应用额外平滑
            kernel_size = max(3, min(7, int(avg_depth / 500)))
            if kernel_size > 3:
                valid_mask = ~np.isnan(enhanced_xyz[:, :, 0])
                if np.sum(valid_mask) > 100:
                    for channel in range(3):
                        channel_data = enhanced_xyz[:, :, channel]
                        # 只对有效数据应用滤波
                        channel_data_filtered = cv2.medianBlur(
                            channel_data.astype(np.float32), kernel_size
                        )
                        enhanced_xyz[:, :, channel] = np.where(
                            valid_mask, channel_data_filtered, channel_data
                        )

    return enhanced_xyz, depth_confidence


def calibrate_depth_estimation(depth_map, known_distances, known_positions):
    """
    深度估计校准函数
    使用已知距离的参考点来校准深度估计

    Args:
        depth_map: 原始深度图
        known_distances: 已知的真实距离列表（mm）
        known_positions: 对应的图像位置列表[(x1, y1), (x2, y2), ...]

    Returns:
        calibrated_depth: 校准后的深度图
        calibration_params: 校准参数
    """
    if len(known_distances) < 3 or len(known_distances) != len(known_positions):
        logger.warning("已知点数量不足，跳过深度校准")
        return depth_map, None

    measured_depths = []
    true_depths = []

    # 提取测量深度值
    for pos, true_depth in zip(known_positions, known_distances):
        x, y = pos
        if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
            measured_depth = depth_map[y, x, 2]  # Z coordinate
            if not np.isnan(measured_depth) and measured_depth > 0:
                measured_depths.append(measured_depth)
                true_depths.append(true_depth)

    if len(measured_depths) < 3:
        logger.warning("有效校准点不足，跳过深度校准")
        return depth_map, None

    # 拟合校准模型：true_depth = a * measured_depth + b * measured_depth^2 + c
    measured_depths = np.array(measured_depths)
    true_depths = np.array(true_depths)

    # 构建特征矩阵
    X = np.column_stack([measured_depths, measured_depths ** 2, np.ones_like(measured_depths)])

    # 使用加权最小二乘法拟合（权重基于深度，远处点权重较低）
    weights = 1.0 / (measured_depths ** 2)  # 深度越大，权重越小
    try:
        params = np.linalg.lstsq(X * weights[:, np.newaxis], true_depths * weights, rcond=None)[0]
        a, b, c = params

        # 应用校准到整个深度图
        calibrated_depth = depth_map.copy()
        valid_mask = ~np.isnan(depth_map[:, :, 2]) & (depth_map[:, :, 2] > 0)
        measured_z = depth_map[:, :, 2][valid_mask]

        # 应用二次校准模型
        calibrated_z = a * measured_z + b * measured_z ** 2 + c
        calibrated_depth[:, :, 2][valid_mask] = calibrated_z

        # 同时更新X和Y坐标（保持比例关系）
        scale_factors = calibrated_z / measured_z
        calibrated_depth[:, :, 0][valid_mask] = depth_map[:, :, 0][valid_mask] * scale_factors
        calibrated_depth[:, :, 1][valid_mask] = depth_map[:, :, 1][valid_mask] * scale_factors

        logger.info(f"深度校准完成: 参数 a={a:.6f}, b={b:.9f}, c={c:.1f}")
        return calibrated_depth, params

    except np.linalg.LinAlgError:
        logger.warning("深度校准矩阵奇异，跳过校准")
        return depth_map, None


def create_precise_mask_from_bbox(frame, bbox_2d, class_name):
    """基于物体类别创建精确的分割掩码"""
    x1, y1, x2, y2 = bbox_2d
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # 创建基础掩码
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # 确保边界框在图像范围内
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1] - 1, x2)
    y2 = min(frame.shape[0] - 1, y2)

    if x2 <= x1 or y2 <= y1:
        return mask

    # 根据物体类别创建不同形状的掩码
    if class_name in ['cup', 'bottle', 'can']:
        # 圆柱形物体 - 椭圆掩码
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        width, height = (x2 - x1) // 2, (y2 - y1) // 2
        cv2.ellipse(mask, (center_x, center_y), (width, height), 0, 0, 360, 255, -1)

    elif class_name in ['chair', 'table', 'sofa']:
        # 家具类 - 矩形掩码，但稍微缩小以避免背景污染
        padding = 5
        cv2.rectangle(mask, (x1 + padding, y1 + padding), (x2 - padding, y2 - padding), 255, -1)

    else:
        # 默认 - 使用椭圆掩码
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        width, height = (x2 - x1) // 2, (y2 - y1) // 2
        cv2.ellipse(mask, (center_x, center_y), (width, height), 0, 0, 360, 255, -1)

    return mask


def extract_object_pointcloud(xyz_map, box_local, src_img=None, class_name=None, max_points=20000, voxel_size_mm=5.0):
    """
    【修改函数】在原有函数基础上添加点云优化
    - 不改变原有接口，内部优化点云质量
    """
    import numpy as np

    if xyz_map is None:
        return np.empty((0, 3), dtype=np.float32), 0.0

    H, W = xyz_map.shape[:2]
    x1, y1, x2, y2 = map(int, box_local)
    x1 = max(0, min(W - 1, x1));
    x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1));
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
            def remove_outliers(points, k=15):
                """内置离群点去除"""
                if len(points) < k:
                    return points
                try:
                    from sklearn.neighbors import NearestNeighbors
                    nbrs = NearestNeighbors(n_neighbors=min(k, len(points) - 1), algorithm='ball_tree').fit(points)
                    distances, indices = nbrs.kneighbors(points)
                    mean_distances = np.mean(distances, axis=1)
                    threshold = np.mean(mean_distances) + 2.0 * np.std(mean_distances)
                    return points[mean_distances < threshold]
                except Exception:
                    return points

            pts_clean = remove_outliers(pts)
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

    # 体素下采样（原有逻辑）
    try:
        if 'voxel_downsample' in globals():
            pts_ds = voxel_downsample(pts, voxel_size_mm=voxel_size_mm)
            if pts_ds is None or pts_ds.size == 0:
                pts_ds = pts
        else:
            # 简单体素近似
            vs = float(voxel_size_mm)
            keys = np.floor(pts / vs).astype(np.int64)
            uniq, idxs = np.unique(keys, axis=0, return_index=True)
            pts_ds = pts[idxs]
    except Exception:
        pts_ds = pts

    valid_ratio = float(pts_ds.shape[0]) / max(1, roi.shape[0])
    return pts_ds.astype(np.float32), valid_ratio


def enhanced_point_cloud_visualization(ax_3d, detected_objects, voxel_size_mm=8.0, max_points_per_obj=3000):
    """
    【新增函数】改进的点云可视化，解决"一坨"问题
    - 替代原有的 simple_point_cloud_visualization
    - 提供更好的点云分布和可视化效果
    """
    import numpy as np
    total_drawn = 0

    for obj in detected_objects:
        pts_cm = None
        if 'pointcloud' in obj and obj['pointcloud'] is not None:
            pts_cm = obj['pointcloud']
        elif 'points' in obj and obj['points'] is not None:
            pts_cm = obj['points']
        else:
            continue

        pts = np.asarray(pts_cm, dtype=np.float32)
        if pts.size == 0:
            continue

        # 过滤无效点
        valid = ~(np.isnan(pts).any(axis=1) | np.isinf(pts).any(axis=1))
        pts = pts[valid]
        if pts.shape[0] == 0:
            continue

        # === 新增：点云预处理 ===
        if pts.shape[0] > 100:
            try:
                # 1. 进一步降采样以获得更好分布
                if pts.shape[0] > max_points_per_obj:
                    # 使用更智能的采样：先体素下采样，再随机采样
                    try:
                        # 体素下采样
                        vs = float(voxel_size_mm) / 10.0  # mm转cm
                        keys = np.floor(pts / vs).astype(np.int64)
                        _, unique_indices = np.unique(keys, axis=0, return_index=True)
                        pts_sampled = pts[unique_indices]

                        # 如果还是太多，随机采样
                        if pts_sampled.shape[0] > max_points_per_obj:
                            idx = np.random.choice(pts_sampled.shape[0], max_points_per_obj, replace=False)
                            pts = pts_sampled[idx]
                        else:
                            pts = pts_sampled
                    except Exception:
                        # 回退到随机采样
                        idx = np.random.choice(pts.shape[0], min(max_points_per_obj, pts.shape[0]), replace=False)
                        pts = pts[idx]
            except Exception:
                pass
        # === 预处理结束 ===

        # cm -> mm 并调整坐标系
        pts_mm = pts * 10.0
        pts_adj = np.zeros_like(pts_mm)
        pts_adj[:, 0] = pts_mm[:, 0]  # X不变
        pts_adj[:, 1] = pts_mm[:, 2]  # Z -> Y（前）
        pts_adj[:, 2] = -pts_mm[:, 1]  # -Y -> Z（上）

        if pts_adj.shape[0] == 0:
            continue

        # 绘制 - 使用更小的点和大透明度避免"一坨"
        try:
            # 根据物体类别选择颜色
            class_name = obj.get('class_name', '').lower()
            color = '#87CEEB'  # 默认浅蓝色

            if 'person' in class_name:
                color = '#FF6B6B'  # 红色
            elif 'cup' in class_name or 'bottle' in class_name:
                color = '#4ECDC4'  # 青色
            elif 'chair' in class_name:
                color = '#45B7D1'  # 蓝色
            elif 'table' in class_name:
                color = '#96CEB4'  # 绿色

            # 使用更小的点尺寸和适当透明度
            scatter = ax_3d.scatter(
                pts_adj[:, 0], pts_adj[:, 1], pts_adj[:, 2],
                c=color, s=1, alpha=0.7, edgecolors='none', depthshade=True
            )
            total_drawn += pts_adj.shape[0]

        except Exception as e:
            logger.debug(f"点云绘制失败: {e}")
            continue

    return total_drawn




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
    # 参数（根据室内场景设置）
    MIN_POINTS_FOR_3D = 10
    MIN_POINTS_FOR_ROUGH = 3
    DEFAULT_CONF_LOW = 0.15
    DEFAULT_CONF_MED = 0.6
    DEFAULT_CONF_HIGH = 0.9
    MIN_DEPTH_CM = 5.0        # 最小深度 5cm
    MAX_DEPTH_CM = 1000.0     # 最大深度 1000cm = 10m（室内足够）
    MIN_DIM_CM = 0.5
    MAX_DIM_CM = 2000.0

    # 类别默认真实高度（可根据需要扩充）
    ASSUMED_HEIGHTS = {
        'person': 170.0,
        'chair': 80.0,
        'sofa': 90.0,
        'table': 75.0,
        'book': 3.0,
        'cup': 8.0,
        'keyboard': 2.0
    }

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
        fallback_point_cm = None
        try:
            if 'mono_depth_map_mm' in globals() and mono_depth_map_mm is not None and bbox_2d is not None:
                x1, y1, x2, y2 = map(int, bbox_2d)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                if 0 <= cy < mono_depth_map_mm.shape[0] and 0 <= cx < mono_depth_map_mm.shape[1]:
                    z_mm = mono_depth_map_mm[cy, cx]
                    if np.isfinite(z_mm) and z_mm > 0:
                        z_cm = float(z_mm) / 10.0
                        # 相机内参存在则反投影
                        if 'cameraMatrixL' in globals() and cameraMatrixL is not None:
                            fx = float(cameraMatrixL[0, 0]) if cameraMatrixL is not None else 1.0
                            fy = float(cameraMatrixL[1, 1]) if cameraMatrixL is not None else 1.0
                            cx_cam = float(cameraMatrixL[0, 2]) if cameraMatrixL is not None else (image_shape[1]/2 if image_shape else 0)
                            cy_cam = float(cameraMatrixL[1, 2]) if cameraMatrixL is not None else (image_shape[0]/2 if image_shape else 0)
                            X = (cx - cx_cam) * z_cm / (fx if fx != 0 else 1.0)
                            Y = (cy - cy_cam) * z_cm / (fy if fy != 0 else 1.0)
                            fallback_point_cm = np.array([X, Y, z_cm], dtype=np.float32)
                        else:
                            fallback_point_cm = np.array([0.0, 0.0, z_cm], dtype=np.float32)
                        logger.info("calculate_object_depth: 使用 mono_depth_map_mm 作为 fallback")
        except Exception:
            fallback_point_cm = None

        # fallback 2: bbox 高度启发式估计（按类使用假设高度）
        if fallback_point_cm is None and bbox_2d is not None and image_shape is not None:
            try:
                x1, y1, x2, y2 = map(int, bbox_2d)
                bbox_h = max(1, y2 - y1)
                assumed_h = ASSUMED_HEIGHTS.get(class_name, 30.0)
                if 'cameraMatrixL' in globals() and cameraMatrixL is not None:
                    fy = float(cameraMatrixL[1, 1])
                    fx = float(cameraMatrixL[0, 0])
                    cx_cam = float(cameraMatrixL[0, 2])
                    cy_cam = float(cameraMatrixL[1, 2])
                else:
                    # 兜底值
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
                    fx = float(cameraMatrixL[0, 0]) if 'cameraMatrixL' in globals() and cameraMatrixL is not None else 1.0
                    fy = float(cameraMatrixL[1, 1]) if 'cameraMatrixL' in globals() and cameraMatrixL is not None else 1.0
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
        if 'apply_depth_calibration' in globals():
            z_mm_corr = apply_depth_calibration(z_mm, class_name=class_name)
        else:
            z_mm_corr = z_mm
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
            fx = float(cameraMatrixL[0, 0]) if 'cameraMatrixL' in globals() and cameraMatrixL is not None else 1000.0
            fy = float(cameraMatrixL[1, 1]) if 'cameraMatrixL' in globals() and cameraMatrixL is not None else 1000.0
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

    # 添加调试信息（仅新增此行及以下3行）
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
        R = eigenvectors.T
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
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
            widths.append(width_pca)
            confidence_pca = min(1.0, len(point_cloud) / 1000)
            confidences.append(confidence_pca)

    # 方法2: 基于2D-3D融合的估计
    if bbox_2d is not None and depth_value is not None:
        width_2d, confidence_2d = estimate_width_from_2d_3d_fusion(
            bbox_2d, depth_value, camera_matrix, image_size[0]
        )
        methods.append('2d_3d_fusion')
        widths.append(width_2d)
        confidences.append(confidence_2d)

    # 方法3: 简单轴对齐包围盒
    if point_cloud is not None and len(point_cloud) >= 5:
        dimensions, _, _ = estimate_object_dimensions(point_cloud, method='aabb')
        if dimensions is not None:
            _, width_aabb, _ = dimensions
            methods.append('aabb')
            widths.append(width_aabb)
            confidences.append(0.5)

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
    【替换函数】改进的3D包围盒计算，结合多方法融合
    - 替换原有函数，接口完全兼容
    - 解决包围盒抖动和数值过大的问题
    """
    if points is None:
        return None

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return None

    # 1. 改进的离群点去除
    def statistical_outlier_removal(points, k=20, std_ratio=2.0):
        """统计离群点去除 - 内置函数避免未解析"""
        if len(points) < k:
            return points

        try:
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=min(k, len(points) - 1), algorithm='ball_tree').fit(points)
            distances, indices = nbrs.kneighbors(points)
            mean_distances = np.mean(distances, axis=1)
            threshold = np.mean(mean_distances) + std_ratio * np.std(mean_distances)
            inlier_mask = mean_distances < threshold
            return points[inlier_mask]
        except Exception:
            # 失败时返回原数据
            return points

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

def draw_3d_bounding_box(ax, bbox_3d, color='green', linewidth=2):
    """在matplotlib 3D轴上绘制3D包围盒"""
    if bbox_3d is None:
        return

    corners = bbox_3d['corners']

    # 定义包围盒的12条边
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7]  # 侧面
    ]

    # 绘制所有边
    for edge in edges:
        ax.plot3D(
            [corners[edge[0]][0], corners[edge[1]][0]],
            [corners[edge[0]][1], corners[edge[1]][1]],
            [corners[edge[0]][2], corners[edge[1]][2]],
            color=color, linewidth=linewidth
        )


def draw_object_info(ax, obj, color='green'):
    """在3D可视化中绘制物体信息和尺寸"""
    if obj.get('bbox_3d') is None or obj.get('position') is None:
        return

    # 绘制3D包围盒
    draw_3d_bounding_box(ax, obj['bbox_3d'], color=color)

    # 显示物体信息和尺寸
    pos = obj['position'] * 10  # 转换回毫米
    text_x, text_y, text_z = pos[0], pos[1], pos[2] + 50

    # 创建信息文本
    info_text = f"{obj['class_name']} ID:{obj['id']}"

    if obj.get('dimensions') is not None:
        dim = obj['dimensions'] * 10
        info_text += f"\n尺寸: {dim[0]:.1f}×{dim[1]:.1f}×{dim[2]:.1f}mm"

    info_text += f"\n位置: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"
    info_text += f"\n置信度: {obj['confidence']:.2f}"

    ax.text(text_x, text_y, text_z, info_text, color=color, fontsize=8,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))


def letterbox_image(image, target_size):
    """图像预处理：调整大小并填充"""
    h, w = image.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded_image = np.full((target_size[1], target_size[0], 3), 128, dtype=np.uint8)
    dw, dh = (target_size[0] - new_w) // 2, (target_size[1] - new_h) // 2
    padded_image[dh:dh + new_h, dw:dw + new_w] = resized_image
    return padded_image, scale, dw, dh


def put_chinese_text(img, text, position, font_size=32, color=(0, 0, 0)):
    """绘制中文文本"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", font_size)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def calculate_surrounding_points(xyz, cx, cy, search_radius=None):
    """
    兼容性增强版 calculate_surrounding_points
    - 兼容原调用（xyz,cx,cy）以及旧调用带 search_radius 的情况
    - 返回:
        - 有效点 -> numpy array shape (3,) （与 xyz 保持单位一致，脚本中通常是 mm）
        - 无有效点 -> 空的一维 numpy array (shape (0,))
    - 不会返回 [0,0,0] 这样的伪值（上层会根据 len==0 判断无效）
    - 如果提供 search_radius（int），会用作半窗口大小；否则使用全局 MEASUREMENT_WINDOW_SIZE
    """
    try:
        # 兼容 search_radius 参数（可能是 None 或者数字）
        if search_radius is None:
            window_size = int(MEASUREMENT_WINDOW_SIZE)
        else:
            try:
                window_size = max(1, int(search_radius))
            except Exception:
                window_size = int(MEASUREMENT_WINDOW_SIZE)
        # 防止窗口超过边界过大
        window_size = min(window_size, max(1, min(xyz.shape[0] // 2, xyz.shape[1] // 2)))
    except Exception:
        # 兜底
        window_size = int(MEASUREMENT_WINDOW_SIZE) if 'MEASUREMENT_WINDOW_SIZE' in globals() else 3

    # 边界坐标（注意 cx,cy 的顺序：cx 为 x 列，cy 为 y 行）
    y_start = max(0, cy - window_size)
    y_end = min(xyz.shape[0], cy + window_size + 1)
    x_start = max(0, cx - window_size)
    x_end = min(xyz.shape[1], cx + window_size + 1)

    surrounding_points = []
    total_weight = 0.0

    # 先尝试中心点（防止索引越界）
    try:
        if 0 <= cy < xyz.shape[0] and 0 <= cx < xyz.shape[1]:
            center_point = xyz[cy, cx]
            if (not np.isnan(center_point).any()) and (not np.isinf(center_point).any()):
                surrounding_points.append((center_point, 1.0))
                total_weight += 1.0
    except Exception:
        pass

    # 遍历邻域采样（加权）
    for y in range(y_start, y_end):
        for x in range(x_start, x_end):
            # 跳过中心点已处理的情况
            if x == cx and y == cy:
                continue
            try:
                pt = xyz[y, x]
            except Exception:
                continue
            if np.isnan(pt).any() or np.isinf(pt).any():
                continue
            # 只接受合理的 z 值（>0）
            try:
                if hasattr(pt, '__len__') and len(pt) >= 3:
                    z = float(pt[2])
                    if z <= 0:
                        continue
                else:
                    continue
            except Exception:
                continue
            # 距离权重（近的权重大）
            dist = math.hypot(x - cx, y - cy)
            # 高斯/指数衰减权重：参数可根据需要调整
            weight = math.exp(-dist / (max(1.0, window_size)))
            surrounding_points.append((pt, weight))
            total_weight += weight

    if total_weight <= 0 or len(surrounding_points) == 0:
        # 没有任何有效点 -> 返回空数组，上游会做 fallback 处理
        return np.array([])

    # 加权平均
    weighted = np.zeros(3, dtype=np.float32)
    for p, w in surrounding_points:
        # 防御性：确保 p 可转为数组
        try:
            pp = np.asarray(p, dtype=np.float32).reshape(3,)
        except Exception:
            continue
        weighted += pp * float(w)

    # 防止除以 0（理论上 total_weight>0）
    if total_weight <= 0:
        return np.array([])

    result = (weighted / float(total_weight)).astype(np.float32)
    # 最终再校验一下 z 是否合理
    if not np.isfinite(result).all() or result[2] <= 0:
        return np.array([])

    return result




def calculate_bbox_points(xyz, x1, y1, x2, y2):
    """计算 bbox 的代表三维点（返回 list，每项为 [X(mm),Y(mm),Z(mm)]）"""
    corners = [
        ((x1 + x2) // 2, y2),  # 底部中心
        (x1, y1),  # 左上
        (x2, y1),  # 右上
        (x1, y2),  # 左下
        (x2, y2)   # 右下
    ]

    bbox_points = []
    for cx, cy in corners:
        if 0 <= cx < xyz.shape[1] and 0 <= cy < xyz.shape[0]:
            p = calculate_surrounding_points(xyz, cx, cy)
            if p is not None and getattr(p, 'size', 0) == 3 and np.isfinite(p).all() and p[2] > 0:
                bbox_points.append(p.astype(np.float32))  # **保持 mm 单位**
    return bbox_points


class ObjectCutter3DReconstructor:
    """
    【新增类】基于实际物体切割的3D重建器
    从视频中切割实际物体，结合几何和纹理信息进行3D重建
    """

    def __init__(self):
        self.background_model = None
        self.background_initialized = False
        self.bg_samples = []

    def initialize_background(self, frame, num_samples=30):
        """初始化背景模型"""
        try:
            if len(self.bg_samples) < num_samples:
                self.bg_samples.append(frame.copy())
                return False

            # 使用中值法创建背景模型
            self.background_model = np.median(np.array(self.bg_samples), axis=0).astype(np.uint8)
            self.background_initialized = True
            logger.info("背景模型初始化完成")
            return True
        except Exception as e:
            logger.error(f"背景初始化失败: {e}")
            return False

    def extract_foreground_object(self, frame, bbox_2d=None):
        """
        从图像中精确切割前景物体
        结合背景减除和颜色分割
        """
        try:
            # 如果没有背景模型，使用基于颜色的简单分割
            if not self.background_initialized:
                return self.simple_color_based_segmentation(frame, bbox_2d)

            # 1. 背景减除
            fg_mask = self.background_subtraction(frame)

            # 2. 如果有bbox，使用它来约束分割区域
            if bbox_2d is not None:
                constrained_mask = self.constrain_mask_to_bbox(fg_mask, bbox_2d)
                fg_mask = constrained_mask

            # 3. 形态学操作优化掩码
            refined_mask = self.refine_mask(fg_mask)

            # 4. 提取最大连通区域
            final_mask = self.extract_largest_component(refined_mask)

            return final_mask

        except Exception as e:
            logger.error(f"前景提取失败: {e}")
            return self.simple_color_based_segmentation(frame, bbox_2d)

    def background_subtraction(self, frame):
        """背景减除"""
        try:
            # 转换为HSV颜色空间，对光照变化更鲁棒
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv_bg = cv2.cvtColor(self.background_model, cv2.COLOR_BGR2HSV)

            # 计算颜色差异
            color_diff = cv2.absdiff(hsv_frame, hsv_bg)

            # 分离通道
            h_diff, s_diff, v_diff = cv2.split(color_diff)

            # 组合差异（重点关注饱和度和亮度变化）
            combined_diff = s_diff * 0.7 + v_diff * 0.3

            # 二值化
            _, fg_mask = cv2.threshold(combined_diff, 25, 255, cv2.THRESH_BINARY)

            return fg_mask.astype(np.uint8)

        except Exception:
            # fallback: 使用RGB空间的简单背景减除
            diff = cv2.absdiff(frame, self.background_model)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, fg_mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
            return fg_mask

    def simple_color_based_segmentation(self, frame, bbox_2d):
        """基于颜色的简单分割（当背景模型不可用时）"""
        try:
            if bbox_2d is None:
                return np.zeros(frame.shape[:2], dtype=np.uint8)

            x1, y1, x2, y2 = bbox_2d
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                return np.zeros(frame.shape[:2], dtype=np.uint8)

            # 转换为HSV颜色空间
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # 计算ROI内的主要颜色范围
            h_mean = np.mean(hsv_roi[:, :, 0])
            s_mean = np.mean(hsv_roi[:, :, 1])
            v_mean = np.mean(hsv_roi[:, :, 2])

            # 创建颜色范围掩码
            lower_bound = np.array([max(0, h_mean - 20), max(0, s_mean - 40), max(0, v_mean - 40)])
            upper_bound = np.array([min(180, h_mean + 20), min(255, s_mean + 40), min(255, v_mean + 40)])

            # 在整个图像中应用颜色分割
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            color_mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

            # 约束到bbox区域
            constrained_mask = self.constrain_mask_to_bbox(color_mask, bbox_2d)

            return self.refine_mask(constrained_mask)

        except Exception as e:
            logger.error(f"颜色分割失败: {e}")
            return np.zeros(frame.shape[:2], dtype=np.uint8)

    def constrain_mask_to_bbox(self, mask, bbox_2d):
        """将掩码约束到边界框区域"""
        x1, y1, x2, y2 = bbox_2d
        constrained_mask = np.zeros_like(mask)
        constrained_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        return constrained_mask

    def refine_mask(self, mask):
        """优化掩码质量"""
        try:
            # 形态学操作：先开运算去除噪声，再闭运算填充空洞
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # 高斯模糊平滑边缘
            mask = cv2.GaussianBlur(mask, (3, 3), 0)

            return mask
        except Exception:
            return mask

    def extract_largest_component(self, mask):
        """提取最大连通区域"""
        try:
            # 找到连通区域
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

            if num_labels <= 1:
                return mask

            # 找到面积最大的区域（跳过背景）
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

            # 创建只包含最大区域的掩码
            largest_component = np.uint8(labels == largest_label) * 255

            return largest_component
        except Exception:
            return mask

    def create_textured_3d_object(self, points, colors, mask, rgb_image):
        """
        创建带纹理的3D物体
        结合几何形状和实际颜色
        """
        try:
            if len(points) < 50:
                return None

            # 1. 创建彩色点云
            textured_points = points.copy()

            # 2. 如果Open3D可用，创建网格
            mesh = None
            if O3D_VISUALIZATION_AVAILABLE:
                mesh = self.create_textured_mesh(points, colors, mask, rgb_image)

            # 3. 计算3D包围盒
            bbox_3d = self.calculate_oriented_bounding_box(points)

            return {
                'points': textured_points,
                'colors': colors,
                'mesh': mesh,
                'bbox_3d': bbox_3d,
                'mask': mask
            }

        except Exception as e:
            logger.error(f"纹理3D物体创建失败: {e}")
            return None

    def create_textured_mesh(self, points, colors, mask, rgb_image):
        """创建带纹理的网格"""
        if not O3D_VISUALIZATION_AVAILABLE:
            return None

        try:
            import open3d as o3d

            # 创建点云
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # 估计法向量
            pcd.estimate_normals()
            pcd.orient_normals_consistent_tangent_plane(10)

            # 泊松表面重建
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

            # 移除低密度区域
            if len(densities) > 0:
                density_threshold = np.quantile(densities, 0.01)
                vertices_to_remove = densities < density_threshold
                mesh.remove_vertices_by_mask(vertices_to_remove)

            return mesh

        except Exception:
            return None

    def calculate_oriented_bounding_box(self, points):
        """计算定向包围盒"""
        if len(points) < 10:
            return None

        try:
            # 使用PCA找到主方向
            centroid = np.mean(points, axis=0)
            centered = points - centroid
            cov_matrix = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

            # 按特征值排序
            sorted_indices = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, sorted_indices]

            # 投影到主成分空间
            transformed = np.dot(centered, eigenvectors)
            min_vals = np.min(transformed, axis=0)
            max_vals = np.max(transformed, axis=0)

            # 生成包围盒角点
            corners_local = np.array([
                [min_vals[0], min_vals[1], min_vals[2]],
                [max_vals[0], min_vals[1], min_vals[2]],
                [max_vals[0], max_vals[1], min_vals[2]],
                [min_vals[0], max_vals[1], min_vals[2]],
                [min_vals[0], min_vals[1], max_vals[2]],
                [max_vals[0], min_vals[1], max_vals[2]],
                [max_vals[0], max_vals[1], max_vals[2]],
                [min_vals[0], max_vals[1], max_vals[2]]
            ])

            # 转换回世界坐标系
            corners_world = np.dot(corners_local, eigenvectors.T) + centroid

            return {
                'center': centroid.astype(np.float32),
                'dimensions': (max_vals - min_vals).astype(np.float32),
                'corners': corners_world.astype(np.float32),
                'rotation': eigenvectors.T.astype(np.float32)
            }
        except Exception:
            return None


# 全局实例
object_cutter = ObjectCutter3DReconstructor()

def detect_objects_with_distance(frame, xyz_left, xyz_right, confidence_left=None, confidence_right=None):
    """
    完整恢复双视图检测和所有原有功能
    """
    import numpy as np
    global detected_objects, detected_objects_lock, model, device, imageWidth, position_tracker, tracker
    global last_detection_time, detection_interval

    # 初始化全局变量
    if 'last_detection_time' not in globals():
        last_detection_time = 0
    if 'detection_interval' not in globals():
        detection_interval = 2  # 每2帧检测一次

    if frame is None or frame.size == 0:
        return frame

    current_time = time.time()
    frame_count = getattr(detect_objects_with_distance, 'frame_count', 0) + 1
    detect_objects_with_distance.frame_count = frame_count

    # 1. 切分左右视图
    try:
        left_img = frame[:, :imageWidth].copy()
        right_img = frame[:, imageWidth:].copy()
    except Exception:
        return frame

    # 2. 智能检测频率控制
    should_run_detection = (frame_count % detection_interval == 0) or (current_time - last_detection_time > 0.5)

    if should_run_detection:
        last_detection_time = current_time

        # 运行YOLO检测
        def run_yolo_on_view(view_frame, conf_thres=0.25):
            boxes_out = []
            try:
                img_padded, scale, dw, dh = letterbox_image(view_frame, (640, 640))
                img_t = img_padded.transpose(2, 0, 1)
                img_t = np.ascontiguousarray(img_t, dtype=np.float32)
                import torch
                t = torch.from_numpy(img_t).to(device) / 255.0
                if t.ndim == 3:
                    t = t.unsqueeze(0)
                res = model.predict(t, conf=conf_thres, iou=0.45, verbose=False)[0]
                if hasattr(res, 'boxes') and hasattr(res.boxes, 'data'):
                    arr = res.boxes.data.cpu().numpy()
                    H_orig, W_orig = view_frame.shape[:2]
                    for a in arr:
                        try:
                            x1p, y1p, x2p, y2p = float(a[0]), float(a[1]), float(a[2]), float(a[3])
                            conf = float(a[4]) if a.size > 4 else 0.0
                            cls = int(a[5]) if a.size > 5 else 0
                            x1o = int(np.clip((x1p - dw) / max(scale, 1e-6), 0, W_orig - 1))
                            x2o = int(np.clip((x2p - dw) / max(scale, 1e-6), 0, W_orig - 1))
                            y1o = int(np.clip((y1p - dh) / max(scale, 1e-6), 0, H_orig - 1))
                            y2o = int(np.clip((y2p - dh) / max(scale, 1e-6), 0, H_orig - 1))
                            boxes_out.append((cls, float(conf), [x1o, y1o, x2o, y2o]))
                        except Exception:
                            continue
            except Exception:
                return []
            return boxes_out

        dets_left = run_yolo_on_view(left_img)
        dets_right_local = run_yolo_on_view(right_img)

        # 将右目检测框转换为全图坐标
        dets_right = []
        for cls, conf, box in dets_right_local:
            x1, y1, x2, y2 = box
            dets_right.append((cls, conf, [x1 + imageWidth, y1, x2 + imageWidth, y2]))

        # 3. 准备跟踪器输入
        detections_for_tracking = []
        for cls, conf, box in dets_left:
            cls_name = model.names[cls] if hasattr(model, 'names') else str(cls)
            detections_for_tracking.append({
                'bbox': box,
                'class_name': cls_name,
                'confidence': conf
            })

        # 应用跟踪器更新
        if 'tracker' in globals():
            try:
                tracked_detections = tracker.update(detections_for_tracking)
                # 存储跟踪结果供后续帧使用
                detect_objects_with_distance.last_tracked_detections = tracked_detections
                detect_objects_with_distance.last_dets_left = dets_left
                detect_objects_with_distance.last_dets_right = dets_right
            except Exception:
                tracked_detections = detections_for_tracking
        else:
            tracked_detections = detections_for_tracking
    else:
        # 使用上一帧的跟踪结果
        tracked_detections = getattr(detect_objects_with_distance, 'last_tracked_detections', [])
        dets_left = getattr(detect_objects_with_distance, 'last_dets_left', [])
        dets_right = getattr(detect_objects_with_distance, 'last_dets_right', [])

    # 4. 左右视图匹配（完整恢复）
    def _iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        boxAArea = max(1e-6, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = max(1e-6, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        return interArea / (boxAArea + boxBArea - interArea + 1e-9)

    matches = []
    used_r = set()
    for i, (clsL, confL, boxL) in enumerate(dets_left):
        best_j = None
        best_iou = 0.0
        for j, (clsR, confR, boxR) in enumerate(dets_right):
            if j in used_r:
                continue
            if clsL != clsR:
                continue
            iou_val = _iou(boxL, boxR)
            if iou_val > best_iou:
                best_iou = iou_val
                best_j = j
        if best_j is not None and best_iou > 0.30:
            matches.append((i, best_j))
            used_r.add(best_j)
        else:
            matches.append((i, None))

    # 5. 点云提取函数（优化但不删减功能）
    def extract_points_from_xyz(xyz, box, max_points=3000):
        if xyz is None:
            return np.empty((0, 3), dtype=np.float32)
        H, W = xyz.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, min(W - 1, x1));
        x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1));
        y2 = max(0, min(H - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 3), dtype=np.float32)

        roi = xyz[y1:y2 + 1, x1:x2 + 1, :].reshape(-1, 3)
        if roi.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        mask = np.isfinite(roi).all(axis=1) & (roi[:, 2] > 0)
        pts = roi[mask]

        if pts.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        # 随机下采样控制点数
        if pts.shape[0] > max_points:
            idx = np.random.choice(pts.shape[0], max_points, replace=False)
            pts = pts[idx]

        return pts.astype(np.float32)

    frame_out = frame.copy()
    results_for_tracker = []

    # 6. 处理匹配的检测对（完整恢复所有逻辑）
    for idx_left, matched_right_idx in matches:
        clsL, confL, boxL = dets_left[idx_left]
        boxR = None
        confR = 0.0
        if matched_right_idx is not None:
            clsR, confR, boxR = dets_right[matched_right_idx]

        # 查找对应的跟踪ID
        track_id = None
        for det in tracked_detections:
            if (abs(det['bbox'][0] - boxL[0]) < 10 and
                abs(det['bbox'][1] - boxL[1]) < 10 and
                det['class_name'] == model.names[clsL] if hasattr(model, 'names') else str(clsL)):
                track_id = det.get('track_id')
                break

        if track_id is None:
            track_id = f"{model.names[clsL] if hasattr(model, 'names') else clsL}_{(boxL[0] + boxL[2]) // 2}_{(boxL[1] + boxL[3]) // 2}"

        # 提取左右视图点云（毫米）
        ptsL_mm = extract_points_from_xyz(xyz_left, boxL)
        ptsR_mm = np.empty((0, 3), dtype=np.float32)
        if boxR is not None:
            # 将右目box转换回右图局部坐标
            bx_local = [boxR[0] - imageWidth, boxR[1], boxR[2] - imageWidth, boxR[3]]
            bx_local = [int(max(0, b)) for b in bx_local]
            ptsR_mm = extract_points_from_xyz(xyz_right, bx_local)

        # 合并点云并进行去噪（完整恢复原逻辑）
        if ptsL_mm.size and ptsR_mm.size:
            pts_comb_mm = np.vstack([ptsL_mm, ptsR_mm])
        elif ptsL_mm.size:
            pts_comb_mm = ptsL_mm.copy()
        elif ptsR_mm.size:
            pts_comb_mm = ptsR_mm.copy()
        else:
            pts_comb_mm = np.empty((0, 3), dtype=np.float32)

        # 点云去噪（基于Z值的MAD去噪）- 完整恢复
        if pts_comb_mm.size:
            z = pts_comb_mm[:, 2]
            med = float(np.median(z))
            mad = float(np.median(np.abs(z - med))) + 1e-6
            thr = max(3.0 * mad, 50.0)  # 50mm下限
            mask = np.abs(z - med) <= thr
            pts_comb_mm = pts_comb_mm[mask]
            if pts_comb_mm.shape[0] < 6:
                # 回退到未过滤的点云
                pts_comb_mm = np.vstack([p for p in [ptsL_mm, ptsR_mm] if p.size])

        # 计算中心位置和尺寸（毫米）- 完整恢复所有计算逻辑
        center_mm = None
        dims_mm = None
        depth_conf = 0.0
        bbox_3d = None

        if pts_comb_mm.size:
            center_mm = np.median(pts_comb_mm, axis=0).astype(np.float32)

            # 完整的尺寸估计算法 - 恢复所有方法
            if pts_comb_mm.shape[0] >= 10:
                try:
                    # 方法1: PCA估计
                    if 'estimate_object_dimensions' in globals():
                        dimensions, orientation, bbox_corners = estimate_object_dimensions(pts_comb_mm / 10.0,
                                                                                           method='pca')
                        if dimensions is not None:
                            dims_mm = np.array(dimensions) * 10.0
                            depth_conf = min(1.0, pts_comb_mm.shape[0] / 100.0)

                            # 计算3D包围盒
                            if bbox_corners is not None:
                                bbox_3d = {
                                    'center': center_mm / 10.0,
                                    'dimensions': dims_mm / 10.0,
                                    'corners': bbox_corners,
                                    'rotation': orientation
                                }
                except Exception:
                    pass

                # 方法2: 分位数方法（如果PCA失败）
                if dims_mm is None:
                    try:
                        lo = np.percentile(pts_comb_mm, 5, axis=0)
                        hi = np.percentile(pts_comb_mm, 95, axis=0)
                        dims_mm = (hi - lo).astype(np.float32)
                        dims_mm = np.clip(dims_mm, 10.0, 2000.0)
                        depth_conf = min(1.0, pts_comb_mm.shape[0] / 50.0)
                    except Exception:
                        pass
            else:
                # 方法3: 简单范围
                try:
                    lo = np.min(pts_comb_mm, axis=0)
                    hi = np.max(pts_comb_mm, axis=0)
                    dims_mm = (hi - lo).astype(np.float32)
                    depth_conf = min(1.0, pts_comb_mm.shape[0] / 30.0)
                except Exception:
                    pass

        # Fallback：使用bbox中心点 - 完整恢复
        if center_mm is None:
            x1, y1, x2, y2 = boxL
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if 0 <= cx < xyz_left.shape[1] and 0 <= cy < xyz_left.shape[0]:
                point = xyz_left[cy, cx]
                if np.isfinite(point).all() and point[2] > 0:
                    center_mm = point.astype(np.float32)
                    # 基于bbox的启发式尺寸估计
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    if center_mm[2] > 0 and 'cameraMatrixL' in globals():
                        fx = cameraMatrixL[0, 0]
                        fy = cameraMatrixL[1, 1]
                        est_width = (bbox_width * center_mm[2]) / fx
                        est_height = (bbox_height * center_mm[2]) / fy
                        dims_mm = np.array([est_width, est_height, center_mm[2]], dtype=np.float32)
                        depth_conf = 0.1
                    else:
                        dims_mm = np.array([100.0, 100.0, 100.0], dtype=np.float32)
                        depth_conf = 0.05

        # 深度平滑（毫米单位）
        if center_mm is not None and np.isfinite(center_mm[2]) and center_mm[2] > 0:
            try:
                center_mm[2] = smooth_depth_by_key(track_id, float(center_mm[2]), alpha=0.7)
            except Exception:
                pass

        # 更新位置跟踪器
        if center_mm is not None:
            try:
                if 'position_tracker' in globals():
                    center_cm = center_mm / 10.0
                    position_tracker.update_position(track_id, center_cm, depth_conf,
                                                     class_id=clsL, is_left=True)
            except Exception:
                pass

        # 绘制检测结果 - 左右视图都绘制
        x1, y1, x2, y2 = boxL
        color = (255, 180, 0) if boxR is not None else (180, 180, 255)
        cv2.rectangle(frame_out, (x1, y1), (x2, y2), color, 2)

        cls_name = model.names[clsL] if hasattr(model, 'names') else str(clsL)
        depth_text = f"{int(center_mm[2])}mm" if center_mm is not None else "??mm"
        dims_text = f"{int(dims_mm[0])}x{int(dims_mm[1])}x{int(dims_mm[2])}mm" if dims_mm is not None else "??mm"
        label = f"{cls_name} {max(confL, confR):.2f} {depth_text}"
        cv2.putText(frame_out, label, (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # 如果有匹配的右视图，也在右视图上绘制
        if boxR is not None:
            x1r, y1r, x2r, y2r = boxR
            cv2.rectangle(frame_out, (x1r, y1r), (x2r, y2r), color, 2)
            cv2.putText(frame_out, label, (x1r, max(12, y1r - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # 存储完整结果
        obj = {
            'id': track_id,
            'class_id': clsL,
            'class_name': cls_name,
            'bbox': [x1, y1, x2, y2],
            'position': center_mm / 10.0 if center_mm is not None else None,
            'dimensions': dims_mm / 10.0 if dims_mm is not None else None,
            'confidence': float(max(confL, confR)),
            'points': pts_comb_mm / 10.0 if pts_comb_mm.size > 0 else None,
            'pointcloud': pts_comb_mm / 10.0 if pts_comb_mm.size > 0 else None,
            'bbox_3d': bbox_3d,
            'depth_confidence': float(depth_conf),
            'track_id': track_id
        }
        results_for_tracker.append(obj)

    # 7. 处理未匹配的右目检测（完整恢复）
    unmatched_right_indices = [j for j in range(len(dets_right)) if j not in used_r]
    for j in unmatched_right_indices:
        cls, conf, box = dets_right[j]
        x1g, y1, x2g, y2 = box
        bx_local = [x1g - imageWidth, y1, x2g - imageWidth, y2]
        bx_local = [int(max(0, b)) for b in bx_local]
        pts_mm = extract_points_from_xyz(xyz_right, bx_local)

        # 处理右目单独检测
        center_mm = None
        dims_mm = None
        if pts_mm.size:
            center_mm = np.median(pts_mm, axis=0).astype(np.float32)
            # 简单尺寸估计
            if pts_mm.shape[0] >= 5:
                lo = np.percentile(pts_mm, 5, axis=0)
                hi = np.percentile(pts_mm, 95, axis=0)
                dims_mm = (hi - lo).astype(np.float32)

        # 深度平滑
        if center_mm is not None and np.isfinite(center_mm[2]):
            cls_name = model.names[cls] if hasattr(model, 'names') else str(cls)
            track_id = f"{cls_name}_{(x1g + x2g) // 2}_{(y1 + y2) // 2}"
            center_mm[2] = smooth_depth_by_key(track_id, float(center_mm[2]), alpha=0.65)

        # 绘制右目单独检测（红色标识）
        cv2.rectangle(frame_out, (x1g, y1), (x2g, y2), (0, 0, 255), 2)
        cls_name = model.names[cls] if hasattr(model, 'names') else str(cls)
        depth_text = f"{int(center_mm[2])}mm" if center_mm is not None else "??mm"
        label = f"{cls_name} {conf:.2f} {depth_text}"
        cv2.putText(frame_out, label, (x1g, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # 存储右目单独检测结果
        obj = {
            'id': track_id,
            'class_id': cls,
            'class_name': cls_name,
            'bbox': [x1g, y1, x2g, y2],
            'position': center_mm / 10.0 if center_mm is not None else None,
            'dimensions': dims_mm / 10.0 if dims_mm is not None else np.array([100.0, 100.0, 100.0],
                                                                              dtype=np.float32) / 10.0,
            'confidence': float(conf),
            'points': pts_mm / 10.0 if pts_mm.size > 0 else None,
            'pointcloud': pts_mm / 10.0 if pts_mm.size > 0 else None,
            'depth_confidence': 0.1
        }
        results_for_tracker.append(obj)

    # 8. 更新全局检测结果
    try:
        with detected_objects_lock:
            detected_objects = results_for_tracker
    except Exception:
        detected_objects = results_for_tracker

    return frame_out





class PersistentVisualization:
    """确保3D可视化持续显示的系统"""

    def __init__(self):
        self.detected_objects_history = []
        self.max_history_size = 10  # 保留最近10帧的检测结果

    def update_detection_history(self, current_objects):
        """更新检测历史"""
        if current_objects:
            self.detected_objects_history.append(current_objects.copy())
            # 保持历史大小
            if len(self.detected_objects_history) > self.max_history_size:
                self.detected_objects_history.pop(0)

    def get_all_visible_objects(self):
        """
        获取所有应该显示的物体（合并历史并去重）。
        使用 get_object_id 作为唯一键，确保每个对象都带有稳定的 'id' 字段。
        """
        all_objects = []

        # 合并历史中的物体，去除重复（使用稳定的对象 ID）
        seen_ids = set()
        for frame_objects in reversed(self.detected_objects_history):
            for obj in frame_objects:
                try:
                    oid = get_object_id(obj)
                except Exception:
                    oid = "obj_unknown"
                if oid not in seen_ids:
                    # 确保 downstream 可以直接使用 obj['id']
                    if 'id' not in obj or obj['id'] is None:
                        try:
                            obj['id'] = oid
                        except Exception:
                            obj['id'] = str(oid)
                    all_objects.append(obj)
                    seen_ids.add(oid)

        return all_objects

    def visualize_scene(self, ax_3d, planning_result=None, progress=0):
        """可视化整个场景，包括历史和当前状态"""
        # 获取所有要显示的物体
        all_objects = self.get_all_visible_objects()

        # 清空场景
        ax_3d.clear()
        ax_3d.set_xlim(x_range)
        ax_3d.set_ylim(y_range)
        ax_3d.set_zlim(z_range)

        # 绘制坐标轴
        axis_length = 200
        ax_3d.quiver(0, 0, 0, axis_length, 0, 0, color='r', linewidth=2, label='X')
        ax_3d.quiver(0, 0, 0, 0, axis_length, 0, color='g', linewidth=2, label='Y')
        ax_3d.quiver(0, 0, 0, 0, 0, axis_length, color='b', linewidth=2, label='Z')
        ax_3d.legend()

        # 绘制所有物体
        for obj in all_objects:
            self.draw_object(ax_3d, obj)

        # 如果正在运行动画，绘制机械臂
        if planning_result is not None:
            self.draw_robot_arm(ax_3d, planning_result, progress)

        plt.draw()

    def draw_object(self, ax_3d, obj):
        """绘制单个物体"""
        # 绘制3D包围盒
        if obj.get('bbox_3d') is not None:
            bbox_3d = obj['bbox_3d']
            corners = bbox_3d['corners']

            # 调整坐标系：厘米转毫米，并调整坐标系
            corners_mm = corners * 10  # 厘米转毫米
            corners_adjusted = []
            for c in corners_mm:
                # 调整坐标系：X右为正，Y前为正，Z上为正
                corners_adjusted.append([c[0], c[2], -c[1]])

            corners_adjusted = np.array(corners_adjusted)

            # 定义包围盒的12条边
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
                [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
                [0, 4], [1, 5], [2, 6], [3, 7]  # 侧面
            ]

            # 绘制所有边
            for edge in edges:
                ax_3d.plot3D(
                    [corners_adjusted[edge[0]][0], corners_adjusted[edge[1]][0]],
                    [corners_adjusted[edge[0]][1], corners_adjusted[edge[1]][1]],
                    [corners_adjusted[edge[0]][2], corners_adjusted[edge[1]][2]],
                    color='green', linewidth=2, alpha=0.8
                )

        # 绘制物体点云（如果存在）
        if obj.get('pointcloud') is not None and len(obj['pointcloud']) > 0:
            points = obj['pointcloud']
            points_mm = points * 10  # 厘米转毫米
            points_adjusted = np.array([[p[0], p[2], -p[1]] for p in points_mm])  # 调整坐标系

            ax_3d.scatter(
                points_adjusted[:, 0], points_adjusted[:, 1], points_adjusted[:, 2],
                c='red', s=10, alpha=0.7, edgecolors='none'
            )

        # 绘制物体标签
        if obj.get('position') is not None:
            pos = obj['position'] * 10  # 厘米转毫米
            pos_adjusted = np.array([pos[0], pos[2], -pos[1]])
            label = f"{obj['class_name']} ID:{obj['id']}"
            if obj.get('dimensions') is not None:
                dim = obj['dimensions'] * 10  # 厘米转毫米
                label += f"\n尺寸: {dim[0]:.1f}×{dim[1]:.1f}×{dim[2]:.1f}mm"

            ax_3d.text(
                pos_adjusted[0], pos_adjusted[1], pos_adjusted[2] + 50,
                label, fontsize=8, color='white',
                bbox=dict(facecolor='green', alpha=0.7, edgecolor='none')
            )

    def draw_robot_arm(self, ax_3d, planning_result, progress):
        """绘制机械臂"""
        # 计算当前帧的数据
        rod_trajs, disc_data, rope_points, end_pos = calc_three_section_data(planning_result, progress)

        # 绘制机械臂杆
        for i, rod_traj in enumerate(rod_trajs):
            color = section_colors[i % len(section_colors)]
            line_width = 8 if len(rod_trajs) == 1 else 6
            ax_3d.plot(rod_traj[:, 0], rod_traj[:, 1], rod_traj[:, 2],
                       color=color, linewidth=line_width, alpha=0.8)

        # 绘制圆盘
        for disc in disc_data:
            section_idx = disc.get('section', 0)
            color = section_colors[section_idx % len(section_colors)]
            create_disc(ax_3d, disc['center'], disc_radius, disc['v1'], disc['v2'],
                        color=color, alpha=0.7)

        # 绘制绳子
        for i in range(4):
            if len(rope_points[i]) > 0:
                rope_array = np.array(rope_points[i])
                ax_3d.plot(rope_array[:, 0], rope_array[:, 1], rope_array[:, 2],
                           color=rope_colors[i], linewidth=rope_linewidth, linestyle='--', alpha=0.6)

        # 更新末端位置
        if end_pos is not None:
            ax_3d.scatter([end_pos[0]], [end_pos[1]], [end_pos[2]], c='blue', s=50, marker='o')
            error = np.linalg.norm(end_pos - target_xyz)
            ax_3d.text(end_pos[0] + 20, end_pos[1], end_pos[2] - 40,
                       f'({end_pos[0]:.1f}, {end_pos[1]:.1f}, {end_pos[2]:.1f}) 误差: {error:.2f}mm',
                       fontsize=10, color='blue')

class Open3DVisualizer:
    """Open3D 3D可视化系统 - 完整实现"""

    def __init__(self):
        self.vis = None
        self.geometries = {}  # 存储几何体引用
        self.initialized = False
        self.running = False
        self.pointclouds = {}  # 存储物体点云
        self.bboxes = {}  # 存储3D包围盒

    def initialize(self):
        """初始化Open3D可视化器"""
        if not O3D_VISUALIZATION_AVAILABLE:
            logger.warning("Open3D不可用，无法初始化3D可视化")
            return False

        try:
            # 设置Open3D渲染选项
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window("三维物体识别与重建系统", width=1400, height=900, visible=True)

            # 设置渲染选项
            render_option = self.vis.get_render_option()
            render_option.background_color = np.array([0.1, 0.1, 0.1])
            render_option.point_size = 3.0
            render_option.line_width = 2.0
            render_option.mesh_show_back_face = True

            # 添加坐标系
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
            self.vis.add_geometry(coordinate_frame)
            self.geometries['coordinate_frame'] = coordinate_frame

            # 初始渲染
            self.vis.poll_events()
            self.vis.update_renderer()

            self.initialized = True
            self.running = True
            logger.info("Open3D可视化系统初始化成功")
            return True

        except Exception as e:
            logger.error(f"Open3D可视化系统初始化失败: {e}")
            return False

    def add_coordinate_frame(self):
        """添加坐标系"""
        if not self.initialized or not self.vis:
            return

        try:
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
            self.vis.add_geometry(coordinate_frame)
            self.geometries['coordinate_frame'] = coordinate_frame
        except Exception as e:
            logger.error(f"添加坐标系失败: {e}")

    def update_scene(self, detected_objects):
        """更新场景中的物体 - 修复缺失的方法"""
        if not self.initialized:
            return

        try:
            # 清除之前的物体
            self.clear_objects()

            # 添加新的物体
            for obj in detected_objects:
                self.add_object_visualization(obj)

            # 更新渲染
            self.vis.poll_events()
            self.vis.update_renderer()

        except Exception as e:
            logger.error(f"更新Open3D场景失败: {e}")

    def add_object_visualization(self, obj):
        """添加物体到 Open3D 可视化（点云 + 文本），使用安全的 ID 获取与单位处理。"""
        try:
            obj_id = get_object_id(obj)
        except Exception:
            obj_id = "obj_unknown"

        # 创建点云可视化（期待单位：cm -> 转为 mm）
        try:
            pc = None
            if obj.get('pointcloud') is not None:
                pc = obj.get('pointcloud')
            elif obj.get('points') is not None:
                pc = obj.get('points')

            if pc is not None and len(pc) > 0:
                pts = np.asarray(pc, dtype=np.float32)
                # 防止点太多：下采样到最多 5000 点，避免渲染卡顿/视觉“一坨”
                max_pts = 5000
                if pts.shape[0] > max_pts:
                    idx = np.random.choice(pts.shape[0], max_pts, replace=False)
                    pts = pts[idx, :]

                # cm -> mm, 并调整坐标系（保持与现有可视化一致）
                pts_mm = pts * 10.0
                pts_adj = np.array([[p[0], p[2], -p[1]] for p in pts_mm], dtype=np.float64)

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts_adj)
                # 颜色按类别轻微区分（可扩展）
                color = [0.7, 0.7, 0.7]
                cname = obj.get('class_name', '').lower()
                if 'bottle' in cname:
                    color = [0.0, 0.0, 1.0]
                elif 'cup' in cname or 'glass' in cname:
                    color = [1.0, 0.5, 0.0]
                elif 'person' in cname:
                    color = [0.2, 0.8, 0.2]
                pcd.paint_uniform_color(color)
                self.pointclouds[obj_id] = pcd
                try:
                    self.vis.add_geometry(pcd)
                except Exception:
                    # 如果已经添加则先移除再添加（防止重复）
                    try:
                        self.vis.remove_geometry(self.pointclouds[obj_id], reset_bounding_box=False)
                        self.vis.add_geometry(self.pointclouds[obj_id])
                    except Exception:
                        pass

        except Exception as e:
            logger.debug(f"add_object_visualization: 点云可视化失败 for {obj_id}: {e}")

        # 绘制 3D 文本标签（如果 position 可用）
        try:
            if obj.get('position') is not None:
                pos_cm = np.asarray(obj['position'], dtype=np.float32)
                pos_mm = pos_cm * 10.0
                pos_adjusted = np.array([pos_mm[0], pos_mm[2], -pos_mm[1]])
                text = f"{obj.get('class_name', 'obj')} ID:{obj.get('id', obj_id)}"
                if obj.get('dimensions') is not None:
                    try:
                        d = obj['dimensions'] * 10.0
                        text += f"\n{d[0]:.1f}×{d[1]:.1f}×{d[2]:.1f}mm"
                    except Exception:
                        pass
                try:
                    # 在 Open3D 中创建简单文本 mesh（若失败则忽略，避免崩溃）
                    text_mesh = o3d.geometry.TriangleMesh.create_box(width=1e-3, height=1e-3, depth=1e-3)
                    text_mesh.translate(pos_adjusted)
                    text_mesh.paint_uniform_color([1.0, 1.0, 1.0])
                    self.geometries[f'text_{obj_id}'] = text_mesh
                    try:
                        self.vis.add_geometry(text_mesh)
                    except Exception:
                        pass
                except Exception:
                    # 不要因为文本失败而影响渲染
                    pass
        except Exception as e:
            logger.debug(f"add_object_visualization: 文本绘制失败 for {obj_id}: {e}")

    def clear_objects(self):
        """清除所有物体可视化"""
        for geometry in list(self.pointclouds.values()) + list(self.bboxes.values()):
            try:
                self.vis.remove_geometry(geometry, reset_bounding_box=False)
            except:
                pass

        # 清除文本
        for key in list(self.geometries.keys()):
            if key.startswith('text_'):
                try:
                    self.vis.remove_geometry(self.geometries[key], reset_bounding_box=False)
                except:
                    pass
                del self.geometries[key]

        self.pointclouds.clear()
        self.bboxes.clear()

    def run(self):
        """运行可视化器 - 非阻塞模式"""
        if not self.initialized:
            return

        try:
            while self.running:
                try:
                    self.vis.poll_events()
                    self.vis.update_renderer()
                    time.sleep(0.01)  # 100 FPS
                except:
                    break
        except Exception as e:
            logger.error(f"运行可视化器失败: {e}")

    def destroy(self):
        """销毁可视化器"""
        self.running = False
        if self.vis:
            try:
                self.vis.destroy_window()
            except:
                pass


def create_visualization_interface():
    """创建混合可视化界面"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle('三维物体识别与重建系统', fontsize=16)

    # 使用GridSpec创建复杂的布局
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1],
                           wspace=0.05, hspace=0.1)

    # 3D可视化区域
    ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
    ax_3d.set_xlim(x_range)
    ax_3d.set_ylim(y_range)
    ax_3d.set_zlim(z_range)
    ax_3d.set_xlabel('X轴 (mm)')
    ax_3d.set_ylabel('Y轴 (mm)')
    ax_3d.set_zlabel('Z轴 (mm)')
    ax_3d.grid(True, alpha=0.7)

    # 信息显示区域
    ax_info = fig.add_subplot(gs[1, 0])
    ax_info.axis('off')

    # 检测结果显示区域
    ax_detection = fig.add_subplot(gs[1, 1])
    ax_detection.axis('off')
    ax_detection.set_title('物体检测信息', fontsize=10, pad=5)

    # 绳子长度表格区域（保留原有功能）
    ax_table = fig.add_subplot(gs[0, 1])
    ax_table.axis('tight')
    ax_table.set_axis_off()
    ax_table.set_title('系统状态', fontsize=10, pad=5)

    # 初始化信息文本
    info_text = ax_info.text(0.05, 0.5, '系统初始化中...', fontsize=12,
                             bbox=dict(facecolor='lightblue', alpha=0.5))

    # 检测文本
    detection_text = ax_detection.text(0.05, 0.5, '等待检测数据...', fontsize=10,
                                       bbox=dict(facecolor='lightyellow', alpha=0.5))

    # 系统状态表格改为绳子长度表格
    table_data = [['绳子', '长度 (mm)']]
    for i in range(12):
        table_data.append([rope_names[i], '0.0'])

    table = ax_table.table(cellText=table_data, cellLoc='center', loc='center', bbox=[0, 0.1, 1, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 0.8)

    # 设置表格样式
    for i in range(13):
        if i == 0:
            table[(i, 0)].set_facecolor('#404040')
            table[(i, 0)].set_text_props(color='white', weight='bold')
            table[(i, 1)].set_facecolor('#404040')
            table[(i, 1)].set_text_props(color='white', weight='bold')
        else:
            color_idx = (i - 1) % 4
            table[(i, 0)].set_facecolor(rope_colors[color_idx])
            table[(i, 0)].set_text_props(color='white', weight='bold')
            table[(i, 1)].set_facecolor('#f0f0f0')

    return fig, ax_3d, ax_info, ax_detection, ax_table, table, info_text, detection_text


def simple_point_cloud_visualization(ax_3d, detected_objects, voxel_size_mm=10.0, max_points_per_obj=5000):
    """
    稳健的点云绘制（matplotlib 3D）。
    - detected_objects: list of obj dicts（obj['pointcloud'] 或 obj['points'] 为 cm 单位）
    - voxel_size_mm: 下采样体素大小（mm）
    - 返回：绘制的总点数（用于调试）
    """
    import numpy as np
    total_drawn = 0

    for obj in detected_objects:
        pts_cm = None
        if 'pointcloud' in obj and obj['pointcloud'] is not None:
            pts_cm = obj['pointcloud']
        elif 'points' in obj and obj['points'] is not None:
            pts_cm = obj['points']
        else:
            continue

        pts = np.asarray(pts_cm, dtype=np.float32)
        if pts.size == 0:
            continue

        # 过滤无效点
        valid = ~(np.isnan(pts).any(axis=1) | np.isinf(pts).any(axis=1))
        pts = pts[valid]
        if pts.shape[0] == 0:
            continue

        # cm -> mm
        pts_mm = pts * 10.0

        # 坐标系调整（与你原实现一致： X, Z, -Y）
        pts_adj = np.zeros_like(pts_mm)
        pts_adj[:, 0] = pts_mm[:, 0]
        pts_adj[:, 1] = pts_mm[:, 2]
        pts_adj[:, 2] = -pts_mm[:, 1]

        # voxel 下采样（使用文件中已有 voxel_downsample_numpy，如果没有，使用随机采样）
        try:
            if 'voxel_downsample_numpy' in globals():
                down_pts, down_cols = voxel_downsample_numpy(pts_adj, colors=None, voxel_size=voxel_size_mm)
            else:
                # fallback：随机下采样
                n = pts_adj.shape[0]
                k = min(n, max_points_per_obj)
                if n > k:
                    idx = np.random.choice(n, k, replace=False)
                    down_pts = pts_adj[idx]
                else:
                    down_pts = pts_adj
                down_cols = None
        except Exception:
            # fallback 随机
            n = pts_adj.shape[0]
            k = min(n, max_points_per_obj)
            if n > k:
                idx = np.random.choice(n, k, replace=False)
                down_pts = pts_adj[idx]
            else:
                down_pts = pts_adj
            down_cols = None

        if down_pts is None or down_pts.shape[0] == 0:
            continue

        # 绘制
        try:
            if down_cols is not None:
                cols = down_cols
                if cols.max() > 1.1:
                    cols = cols / 255.0
            else:
                cols = '#87CEEB'
            ax_3d.scatter(down_pts[:, 0], down_pts[:, 1], down_pts[:, 2], c=cols, s=2, alpha=0.85, edgecolors='none')
            total_drawn += down_pts.shape[0]
        except Exception:
            # 绘制失败不阻塞主流程
            try:
                ax_3d.scatter(down_pts[:, 0], down_pts[:, 1], down_pts[:, 2], c='r', s=1)
                total_drawn += down_pts.shape[0]
            except Exception:
                continue

    return total_drawn


def update_visualization(ax_3d, detected_objects, table, info_text, detection_text, fps):
    """
    更稳健的 3D 可视化函数：
    - 使用 get_object_id() 兜底 ID
    - 兼容 'pointcloud' 与 'points' 两种命名
    - 对点云下采样以防“一坨乱”
    - 对缺失字段做保护，避免 KeyError
    """
    try:
        # 清除 3D 轴并设置范围/标签
        ax_3d.clear()
        ax_3d.set_xlim(x_range)
        ax_3d.set_ylim(y_range)
        ax_3d.set_zlim(z_range)
        ax_3d.set_xlabel('X轴 (mm)')
        ax_3d.set_ylabel('Y轴 (mm)')
        ax_3d.set_zlabel('Z轴 (mm)')
        ax_3d.grid(True, alpha=0.7)
        ax_3d.set_title('3D场景可视化')

        # 绘制坐标轴
        axis_length = 200
        ax_3d.quiver(0, 0, 0, axis_length, 0, 0, color='r', linewidth=2, arrow_length_ratio=0.1, label='X')
        ax_3d.quiver(0, 0, 0, 0, axis_length, 0, color='g', linewidth=2, arrow_length_ratio=0.1, label='Y')
        ax_3d.quiver(0, 0, 0, 0, 0, axis_length, color='b', linewidth=2, arrow_length_ratio=0.1, label='Z')
        ax_3d.legend()

        # 绘制检测到的物体
        if detected_objects:
            for obj in detected_objects:
                try:
                    # 统一 ID 并写回 obj['id']（方便历史和其他模块）
                    oid = get_object_id(obj)
                    if 'id' not in obj or obj['id'] is None:
                        obj['id'] = oid

                    # 读取点云：兼容 'pointcloud'（首选）和 'points'
                    pc = obj.get('pointcloud') if obj.get('pointcloud') is not None else obj.get('points')
                    if pc is not None and getattr(pc, '__len__', lambda: 0)() > 0:
                        pts = np.asarray(pc, dtype=np.float32)

                        # === 替换开始：使用改进的点云可视化 ===
                        # 下采样（避免太多点导致一坨）
                        max_pts = 2000  # 进一步减少点数
                        if pts.shape[0] > max_pts:
                            idx = np.random.choice(pts.shape[0], max_pts, replace=False)
                            pts = pts[idx, :]

                        # cm -> mm，然后坐标系调整
                        pts_mm = pts * 10.0
                        pts_adj = np.zeros_like(pts_mm)
                        pts_adj[:, 0] = pts_mm[:, 0]  # X不变
                        pts_adj[:, 1] = pts_mm[:, 2]  # Z -> Y（前）
                        pts_adj[:, 2] = -pts_mm[:, 1]  # -Y -> Z（上）

                        # 根据物体类别选择颜色
                        class_name = obj.get('class_name', '').lower()
                        color = '#87CEEB'  # 默认
                        if 'person' in class_name:
                            color = '#FF6B6B'
                        elif 'cup' in class_name or 'bottle' in class_name:
                            color = '#4ECDC4'
                        elif 'chair' in class_name:
                            color = '#45B7D1'
                        elif 'table' in class_name:
                            color = '#96CEB4'

                        # 绘制点云（使用小点和适当透明度）
                        ax_3d.scatter(pts_adj[:, 0], pts_adj[:, 1], pts_adj[:, 2],
                                      c=color, s=2, alpha=0.6, edgecolors='none')
                        # === 替换结束 ===

                    # 如有 3D bbox（bbox_3d），绘制线框
                    # ---------- 替换为稳健的绘制代码 ----------
                    try:
                        bbox3d = obj.get('bbox_3d', None)
                        if bbox3d is not None and 'corners' in bbox3d:
                            corners = np.asarray(bbox3d['corners'], dtype=np.float32)  # corners in unknown units
                            if corners.size == 0:
                                pass
                            else:
                                # units check: compute median norm to decide if corners are mm or cm
                                med_norm = float(np.median(np.linalg.norm(corners, axis=1)))
                                # If med_norm is very large (>2000) likely mm -> convert to cm
                                if med_norm > 2000:
                                    corners = corners / 10.0  # mm -> cm
                                # If very small (<1) likely meters -> convert to cm
                                if med_norm < 1.5:
                                    corners = corners * 100.0  # m -> cm

                                # Now corners are in cm; convert to mm for plotting/consistency with plotting axes
                                corners_mm = corners * 10.0
                                # coordinate adjustment if your plot expects (X, Z, -Y) or similar:
                                corners_plot = np.array([[c[0], c[2], -c[1]] for c in corners_mm])

                                edges = [
                                    [0, 1], [1, 3], [3, 2], [2, 0],
                                    [4, 5], [5, 7], [7, 6], [6, 4],
                                    [0, 4], [1, 5], [2, 6], [3, 7]
                                ]
                                for e in edges:
                                    p0 = corners_plot[e[0]]
                                    p1 = corners_plot[e[1]]
                                    ax_3d.plot3D([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color='lime',
                                                 linewidth=1.2)
                    except Exception:
                        pass
                    # ---------- 替换结束 ----------

                    # 绘制标签（位置）
                    if obj.get('position') is not None:
                        try:
                            pos_cm = np.asarray(obj['position'], dtype=np.float32)
                            pos_mm = pos_cm * 10.0
                            pos_adjusted = np.array([pos_mm[0], pos_mm[2], -pos_mm[1]])
                            label = f"{obj.get('class_name', 'obj')} ID:{obj.get('id')}"
                            if obj.get('dimensions') is not None:
                                try:
                                    d = obj['dimensions'] * 10.0
                                    label += f"\n尺寸: {d[0]:.1f}×{d[1]:.1f}×{d[2]:.1f}mm"
                                except Exception:
                                    pass
                            ax_3d.text(pos_adjusted[0], pos_adjusted[1], pos_adjusted[2] + 50,
                                       label, fontsize=8, color='white',
                                       bbox=dict(facecolor='green', alpha=0.7, edgecolor='none'))
                        except Exception:
                            pass

                except Exception as e:
                    # 单个对象绘制失败不影响整体
                    logger.debug(f"update_visualization: 绘制单体失败: {e}")
                    continue

        # 更新检测信息文本（右侧）
        detection_info = "检测到的物体:\n"
        if detected_objects:
            for obj in detected_objects:
                try:
                    oid = obj.get('id', get_object_id(obj))
                    cname = obj.get('class_name', 'unknown')
                    detection_info += f"ID:{oid} {cname} "
                    if obj.get('dimensions') is not None:
                        try:
                            dim = obj['dimensions'] * 10.0
                            detection_info += f"({dim[0]:.1f}×{dim[1]:.1f}×{dim[2]:.1f}mm)\n"
                        except Exception:
                            detection_info += "(尺寸未知)\n"
                    else:
                        detection_info += "(尺寸未知)\n"
                except Exception:
                    detection_info += "(解析对象失败)\n"
        else:
            detection_info += "无\n"

        detection_text.set_text(detection_info)

        # 更新系统状态（FPS）
        if fps > 0:
            try:
                table._cells[(5, 1)].get_text().set_text(f"{fps:.1f}")
                color = '#90EE90' if fps > 10 else '#FFCCCB'
                table._cells[(5, 1)].set_facecolor(color)
            except Exception:
                pass

        # 更新信息文本
        if detected_objects:
            info_text.set_text(f"已检测到 {len(detected_objects)} 个物体\n系统运行正常")
            info_text.set_bbox(dict(facecolor='lightgreen', alpha=0.5))
        else:
            info_text.set_text("正在搜索物体...\n请确保场景中有可检测的物体")
            info_text.set_bbox(dict(facecolor='lightyellow', alpha=0.5))

        plt.draw()

    except Exception as e:
        # 整体防护：可视化失败不要抛出致命错误
        logger.error(f"update_visualization: 异常: {e}")
        try:
            plt.draw()
        except Exception:
            pass

def on_click(event, x, y, flags, param):
    """处理鼠标点击事件，选择物体"""
    global selected_object_id, detected_objects

    if event == cv2.EVENT_LBUTTONDOWN:
        with detected_objects_lock:
            current_detected_objects = detected_objects.copy()

        # 简化的物体选择逻辑
        for obj in current_detected_objects:
            # 这里需要根据实际图像坐标计算选择区域
            # 简化实现：通过ID选择
            pass


def onMouse(event, x, y, flags, param):
    """鼠标回调：显示点击点的三维坐标"""
    global xyz_left, xyz_right
    if event == cv2.EVENT_LBUTTONDOWN:
        xyz = None
        try:
            cv2.getWindowImageRect('disparity_left')
            xyz = xyz_left
        except:
            try:
                cv2.getWindowImageRect('disparity_right')
                xyz = xyz_right
            except:
                return
        if xyz is not None and 0 <= x < xyz.shape[1] and 0 <= y < xyz.shape[0]:
            point3 = xyz[y, x]
            if not np.isnan(point3).any():
                point3_cm = point3 / 10.0  # 毫米→厘米
                d = np.linalg.norm(point3_cm)
                logger.info(
                    f"{LABELS['world_coords']}: x={point3_cm[0]:.2f}, y={point3_cm[1]:.2f}, z={point3_cm[2]:.2f}, {LABELS['distance']}={d:.2f} {LABELS['cm']}")
            else:
                logger.info(f"{LABELS['coords']}: {LABELS['invalid']}")

def stereo_vision_thread():
    """双目视觉线程函数 - 添加更多错误处理"""
    global detected_objects, selected_object_id, xyz_left, xyz_right, last_left_frame_bgr

    # 添加初始化代码（按建议新增）
    depth_system = DepthEstimationSystem()
    vis_system = PersistentVisualization()

    camera = None
    verify_calibration()
    stereo_rectify()
    frame_times = deque(maxlen=30)
    frame_count = 0

    try:
        # 尝试打开摄像头
        for camera_index in [0, 1, 2]:
            camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if camera.isOpened():
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                ret, test_frame = camera.read()
                if ret and test_frame is not None:
                    logger.info(f"成功初始化摄像头 {camera_index}")
                    break
                else:
                    camera.release()
                    camera = None
        else:
            raise Exception("无法找到可用的摄像头")

        if camera is None:
            raise Exception("摄像头初始化失败")

        # 创建窗口
        cv2.namedWindow("disparity_left")
        cv2.namedWindow("disparity_right")
        cv2.namedWindow("Detection with Distance")
        cv2.setMouseCallback("disparity_left", onMouse)
        cv2.setMouseCallback("disparity_right", onMouse)
        cv2.setMouseCallback("Detection with Distance", on_click)
        # 确保窗口可显示
        cv2.moveWindow("disparity_left", 100, 100)
        cv2.moveWindow("disparity_right", 400, 100)
        cv2.moveWindow("Detection with Distance", 700, 100)

        retry_count = 0
        max_retries = 3

        while True:
            try:
                start_time = time.time()
                ret, frame = camera.read()
                if not ret:
                    logger.warning("读取帧失败")
                    time.sleep(0.1)
                    continue

                # 检查帧是否有效
                if frame is None or frame.size == 0:
                    logger.warning("接收到空帧或无效帧")
                    continue

                retry_count = 0
                frame_count += 1

                # 分割左右图像
                frameL, frameR = frame[:, :imageWidth], frame[:, imageWidth:]
                if frameL.size == 0 or frameR.size == 0:
                    logger.warning("无效的左右图像分割")
                    continue

                # 显示原始图像
                cv2.imshow("right", frameR)
                cv2.imshow("left", frameL)

                # 存储左目BGR帧用于单目深度估计
                last_left_frame_bgr = frameL.copy()

                # 校正图像
                try:
                    rectifyImageL = cv2.remap(cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY),
                                              mapLx, mapLy, cv2.INTER_LINEAR)
                    rectifyImageR = cv2.remap(cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY),
                                              mapRx, mapRy, cv2.INTER_LINEAR)
                except Exception as e:
                    logger.error(f"图像校正失败: {e}")
                    continue

                # ----------------- 处理立体视觉（替换块开始） -----------------
                try:
                    # 新的 stereo_match 返回：disp_color_left, xyz_left_mm, conf_left, disp_left, disp_right, xyz_right
                    disp_color_left, xyz_left, confidence_left, disp_left, disp_right, xyz_right = stereo_match(rectifyImageL, rectifyImageR)

                    # 显示左视差（伪彩色）
                    try:
                        cv2.imshow("disparity_left", disp_color_left)
                    except Exception:
                        pass

                    # 显示右视差：优先使用 stereo_match 提供的 disp_right，否则用 remap 左视差做退化显示
                    try:
                        if disp_right is not None and np.any(disp_right > 0):
                            disp_vis_r = (disp_right / (np.max(disp_right) + 1e-9) * 255.0).clip(0, 255).astype(np.uint8)
                            disp_color_right = cv2.applyColorMap(disp_vis_r, cv2.COLORMAP_JET)
                        else:
                            # fallback: remap disp_left -> right view
                            if disp_left is None or disp_left.size == 0:
                                disp_color_right = np.zeros((imageHeight, imageWidth, 3), dtype=np.uint8)
                            else:
                                H, W = disp_left.shape
                                xs, ys = np.meshgrid(np.arange(W), np.arange(H))
                                map_x = (xs - disp_left).astype(np.float32)
                                map_y = ys.astype(np.float32)
                                disp_right_est = cv2.remap(disp_left, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
                                if disp_right_est is None or np.max(disp_right_est) <= 0:
                                    disp_color_right = np.zeros((imageHeight, imageWidth, 3), dtype=np.uint8)
                                else:
                                    disp_vis_r = (disp_right_est / (np.max(disp_right_est) + 1e-9) * 255.0).clip(0, 255).astype(np.uint8)
                                    disp_color_right = cv2.applyColorMap(disp_vis_r, cv2.COLORMAP_JET)
                        cv2.imshow("disparity_right", disp_color_right)
                    except Exception as e:
                        logger.debug(f"右视差可视化失败: {e}")
                        cv2.imshow("disparity_right", np.zeros((imageHeight, imageWidth, 3), dtype=np.uint8))

                    # 确保 xyz_right 存在（如果 stereo_match 未返回有效 xyz_right 则 remap 左点云）
                    try:
                        need_remap = False
                        if xyz_right is None or (np.isfinite(xyz_right).sum() == 0):
                            need_remap = True
                        if need_remap:
                            H, W = disp_left.shape
                            xs, ys = np.meshgrid(np.arange(W), np.arange(H))
                            map_x = (xs - disp_left).astype(np.float32)
                            map_y = ys.astype(np.float32)
                            xyz_right = np.full_like(xyz_left, np.nan, dtype=np.float32)
                            for c in range(3):
                                ch = xyz_left[:, :, c].astype(np.float32)
                                xyz_right[:, :, c] = cv2.remap(ch, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                                                              borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
                            invalid_mask = (disp_left <= 0) | (~np.isfinite(xyz_left[:, :, 2]))
                            xyz_right[invalid_mask, :] = np.nan
                    except Exception as e:
                        logger.error(f"构造 xyz_right 失败: {e}")
                        xyz_right = np.full_like(xyz_left, np.nan, dtype=np.float32)

                    # 调用检测与测距（保持接口不变），将左右点云、置信度传入
                    try:
                        frame_with_text = detect_objects_with_distance(frame, xyz_left, xyz_right, confidence_left, None)
                        vis_system.update_detection_history(detected_objects)
                        cv2.imshow("Detection with Distance", frame_with_text)
                    except Exception as e:
                        logger.error(f"物体检测与距离计算失败: {e}")
                        cv2.imshow("Detection with Distance", frame)
                except Exception as e:
                    logger.error(f"立体视觉/检测流程失败: {e}")
                    cv2.imshow("Detection with Distance", frame)
                # ----------------- 处理立体视觉（替换块结束） -----------------

                # 计算FPS
                end_time = time.time()
                frame_times.append(end_time - start_time)
                if len(frame_times) >= 10:
                    avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
                    # 确保在显示前添加FPS文本
                    if 'frame_with_text' in locals():
                        cv2.putText(frame_with_text, f"FPS: {avg_fps:.1f}", (frame_with_text.shape[1] - 150, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 检查退出键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q'键或ESC键
                    logger.info(LABELS['user_quit'])
                    break

            except KeyboardInterrupt:
                logger.info(f"\n{LABELS['user_interrupt']}")
                break
            except Exception as e:
                logger.error(f"主循环错误: {str(e)}")
                retry_count += 1
                if retry_count >= max_retries:
                    break
                time.sleep(0.5)
                continue

    except Exception as e:
        logger.error(f"初始化错误: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        logger.info(LABELS['cleanup'])
        if camera is not None and camera.isOpened():
            camera.release()
        cv2.destroyAllWindows()
        logger.info(LABELS['cleanup_complete'])



def run_animation(fig, ax_3d, ax_info, ax_detection, ax_table, table, info_text, detection_text):
    """修改的动画函数，确保投影持续显示"""
    global animation_running, detected_objects, selected_object_id, target_xyz, planning_result
    global stabilization_complete, stabilization_counter, object_bbox_points
    global calc_three_section_data, create_disc, create_bbox, calculate_rope_lengths
    global end_text, detected_objects_lock

    # 没有规划结果时使用默认规划（按建议修改）
    if planning_result is None:
        logger.warning("没有规划结果，使用默认规划")
        planning_result = {
            'mode': 'single',
            'theta': 0.5,
            'phi': 0.0,
            'error': 100.0,
            'sections': 1
        }

    animation_running = True

    # 存储检测到的物体，确保持续显示（按建议新增）
    persistent_objects = detected_objects.copy() if detected_objects else []

    # 初始化3D场景（保留原逻辑）
    ax_3d.clear()
    ax_3d.set_xlim(x_range)
    ax_3d.set_ylim(y_range)
    ax_3d.set_zlim(z_range)
    ax_3d.set_xlabel('X轴 (mm)')
    ax_3d.set_ylabel('Y轴 (mm)')
    ax_3d.set_zlabel('Z轴 (mm)')
    ax_3d.grid(True, alpha=0.7)

    # 添加坐标轴指示（保留原逻辑）
    axis_length = 200
    ax_3d.quiver(0, 0, 0, axis_length, 0, 0, color='r', linewidth=2, arrow_length_ratio=0.1, label='X')
    ax_3d.quiver(0, 0, 0, 0, axis_length, 0, color='g', linewidth=2, arrow_length_ratio=0.1, label='Y')
    ax_3d.quiver(0, 0, 0, 0, 0, axis_length, color='b', linewidth=2, arrow_length_ratio=0.1, label='Z')
    ax_3d.legend()

    # 动画帧（保留原循环结构）
    for frame in range(total_frames + 1):
        if not animation_running:
            break

        progress = frame / total_frames

        # 清除之前的可视化元素，但保留持久化的物体（按建议修改）
        for line in ax_3d.lines[:]:
            # 跳过标记为持久化的元素
            if hasattr(line, '_persistent') and line._persistent:
                continue
            line.remove()
        for collection in ax_3d.collections[:]:
            # 跳过标记为持久化的元素
            if hasattr(collection, '_persistent') and collection._persistent:
                continue
            collection.remove()

        # 计算当前帧的机械臂数据（保留原逻辑）
        rod_trajs, disc_data, rope_points, end_pos = calc_three_section_data(planning_result, progress)

        # 绘制机械臂杆（保留原逻辑）
        for i, rod_traj in enumerate(rod_trajs):
            color = section_colors[i % len(section_colors)]
            line_width = 8 if len(rod_trajs) == 1 else 6
            ax_3d.plot(rod_traj[:, 0], rod_traj[:, 1], rod_traj[:, 2],
                       color=color, linewidth=line_width, alpha=0.8)

        # 绘制圆盘（保留原逻辑）
        for disc in disc_data:
            section_idx = disc.get('section', 0)
            color = section_colors[section_idx % len(section_colors)]
            create_disc(ax_3d, disc['center'], disc_radius, disc['v1'], disc['v2'],
                        color=color, alpha=0.7)

        # 绘制绳子（保留原逻辑）
        for i in range(4):
            if len(rope_points[i]) > 0:
                rope_array = np.array(rope_points[i])
                ax_3d.plot(rope_array[:, 0], rope_array[:, 1], rope_array[:, 2],
                           color=rope_colors[i], linewidth=rope_linewidth, linestyle='--', alpha=0.6)

        # 更新末端位置（保留原逻辑）
        if end_pos is not None:
            ax_3d.scatter([end_pos[0]], [end_pos[1]], [end_pos[2]], c='blue', s=50, marker='o')
            error = np.linalg.norm(end_pos - target_xyz)
            ax_3d.text(end_pos[0] + 20, end_pos[1], end_pos[2] - 40,
                       f'({end_pos[0]:.1f}, {end_pos[1]:.1f}, {end_pos[2]:.1f}) 误差: {error:.2f}mm',
                       fontsize=10, color='blue')

        # 绘制检测到的物体 - 用持久化列表确保持续显示（按建议修改）
        for obj in persistent_objects:
            # 绘制3D包围盒
            if obj.get('bbox_3d') is not None:
                bbox_3d = obj['bbox_3d']
                corners = bbox_3d['corners']

                # 调整坐标系：厘米转毫米，并调整坐标系（保留原逻辑）
                corners_mm = corners * 10  # 厘米转毫米
                corners_adjusted = []
                for c in corners_mm:
                    # 调整坐标系：X右为正，Y前为正，Z上为正
                    corners_adjusted.append([c[0], c[2], -c[1]])

                corners_adjusted = np.array(corners_adjusted)

                # 定义包围盒的12条边（保留原逻辑）
                edges = [
                    [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
                    [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
                    [0, 4], [1, 5], [2, 6], [3, 7]  # 侧面
                ]

                # 绘制所有边，标记为持久化
                for edge in edges:
                    line = ax_3d.plot3D(
                        [corners_adjusted[edge[0]][0], corners_adjusted[edge[1]][0]],
                        [corners_adjusted[edge[0]][1], corners_adjusted[edge[1]][1]],
                        [corners_adjusted[edge[0]][2], corners_adjusted[edge[1]][2]],
                        color='green', linewidth=2, alpha=0.8
                    )
                    line[0]._persistent = True  # 标记为持久化元素

            # 绘制物体点云（如果存在），标记为持久化
            if obj.get('pointcloud') is not None and len(obj['pointcloud']) > 0:
                points = obj['pointcloud']
                points_mm = points * 10  # 厘米转毫米
                points_adjusted = np.array([[p[0], p[2], -p[1]] for p in points_mm])  # 调整坐标系

                scatter = ax_3d.scatter(
                    points_adjusted[:, 0], points_adjusted[:, 1], points_adjusted[:, 2],
                    c='red', s=2, alpha=0.5, edgecolors='none'
                )
                scatter._persistent = True  # 标记为持久化元素

            # 绘制物体标签，标记为持久化
            if obj.get('position') is not None:
                pos = obj['position'] * 10  # 厘米转毫米
                pos_adjusted = np.array([pos[0], pos[2], -pos[1]])
                label = f"{obj['class_name']} ID:{obj['id']}"
                if obj.get('dimensions') is not None:
                    dim = obj['dimensions'] * 10  # 厘米转毫米
                    label += f"\n尺寸: {dim[0]:.1f}×{dim[1]:.1f}×{dim[2]:.1f}mm"

                text = ax_3d.text(
                    pos_adjusted[0], pos_adjusted[1], pos_adjusted[2] + 50,
                    label, fontsize=8, color='white',
                    bbox=dict(facecolor='green', alpha=0.7, edgecolor='none')
                )
                text._persistent = True  # 标记为持久化元素

        # 更新绳子长度表格（保留原逻辑）
        rope_lengths = calculate_rope_lengths(planning_result, progress)
        for i in range(12):
            length = rope_lengths[i]
            table._cells[(i + 1, 1)]._text.set_text(f"{length:.1f}")

        # 更新信息文本，添加当前误差（按建议修改）
        if end_pos is not None:
            error = np.linalg.norm(end_pos - target_xyz)
            info_text.set_text(f'动画运行中... 进度: {progress * 100:.1f}%, 当前误差: {error:.2f}mm')
        else:
            info_text.set_text(f'动画运行中... 进度: {progress * 100:.1f}%')
        # 移除原有的物体搜索提示，统一显示动画进度
        info_text.set_bbox(None)

        # 重绘（保留原逻辑）
        plt.draw()
        plt.pause(0.05)

    animation_running = False
    # 最终误差检查与显示（保留原逻辑）
    if end_pos is not None:
        final_error = np.linalg.norm(end_pos - target_xyz)
        info_text.set_text(f'动画完成! 最终误差: {final_error:.2f}mm')
    else:
        info_text.set_text('动画完成!')
    info_text.set_bbox(dict(facecolor='lightgreen', alpha=0.5))

def plan_motion(target):
    """确保运动规划总能成功返回结果"""
    print(f"规划运动到目标位置: {target}")

    # 尝试所有规划方法
    methods = [
        ('single', plan_single_section),
        ('two_plus_one', plan_two_plus_one),
        ('three_independent', plan_three_independent)
    ]

    best_result = None
    best_error = float('inf')

    for method_name, planner in methods:
        try:
            result = planner(target)
            if result and result['error'] < best_error:
                best_result = result
                best_error = result['error']
                logger.info(f"方法 {method_name} 找到可行解，误差: {best_error:.2f}mm")
        except Exception as e:
            logger.warning(f"方法 {method_name} 失败: {e}")
            continue

    # 如果所有方法都失败，创建保守的默认规划
    if best_result is None:
        logger.warning("所有规划方法失败，创建保守默认规划")
        best_result = {
            'mode': 'single',
            'theta': 0.3,  # 较小的弯曲角度
            'phi': 0.0,
            'error': 150.0,
            'sections': 1,
            'conservative': True  # 标记为保守规划
        }

    return best_result

def plan_single_section(target):
    """方法1: 三节作为一个整体"""
    L = total_L
    R = np.linalg.norm(target[:2])  # XY平面距离
    Z = target[2]

    # 检查是否在可达范围内
    max_R = L * 2 / np.pi  # 最大弯曲时的半径
    if R > max_R or Z < 0 or Z > L:
        return None

    # 计算弯曲角度和方向
    if R < 1e-6:
        # 直杆情况
        if abs(Z - L) < 1e-6:
            return {
                'mode': 'single',
                'theta': 0.0,
                'phi': 0.0,
                'error': 0.0,
                'sections': 1
            }
        else:
            return None

    # 弯曲情况
    # 使用更精确的公式计算theta
    def objective(theta):
        if abs(theta) < 1e-6:
            return (R ** 2 + (Z - L) ** 2)  # 直杆情况的误差
        else:
            radius = L / theta
            x = radius * (1 - np.cos(theta))
            z = radius * np.sin(theta)
            return (R - x) ** 2 + (Z - z) ** 2

    # 使用优化方法求解
    result = minimize(objective, 0.5, bounds=[(0, np.pi / 2)], method='L-BFGS-B')
    if not result.success:
        return None

    theta = result.x[0]

    # 计算phi
    phi = np.arctan2(target[1], target[0])

    # 验证末端位置
    end_pos = single_section_kinematics(L, theta, phi)
    error = np.linalg.norm(end_pos - target)

    return {
        'mode': 'single',
        'theta': theta,
        'phi': phi,
        'error': error,
        'sections': 1
    }

def single_section_kinematics(L, theta, phi):
    """精确计算单节机械臂的运动学"""
    if abs(theta) < 1e-6:
        # 直杆情况
        return np.array([0, 0, L])
    else:
        # 弯曲情况 - 使用精确公式
        radius = L / theta  # 曲率半径
        x = radius * (1 - np.cos(theta)) * np.cos(phi)
        y = radius * (1 - np.cos(theta)) * np.sin(phi)
        z = radius * np.sin(theta)
        return np.array([x, y, z])


def plan_two_plus_one(target):
    """方法2: 两节+一节 - 改进版"""
    L12 = 2 * L_section
    L3 = L_section

    # 生成前两节的可能配置 - 增加采样密度
    n_theta = 20
    n_phi = 40
    thetas_12 = np.linspace(0, np.pi / 2, n_theta)
    phis_12 = np.linspace(0, 2 * np.pi, n_phi)

    best_error = float('inf')
    best_plan = None

    for theta12 in thetas_12:
        for phi12 in phis_12:
            # 计算前两节的末端位置
            end_12 = single_section_kinematics(L12, theta12, phi12)

            # 计算第三节需要到达的位置
            target_3 = target - end_12

            # 检查第三节是否可达
            R3 = np.linalg.norm(target_3[:2])
            Z3 = target_3[2]

            # 检查第三节是否在可达范围内
            max_R3 = L3 * 2 / np.pi
            if R3 > max_R3 or Z3 < 0 or Z3 > L3:
                continue

            if R3 < 1e-6:
                # 直杆情况
                if abs(Z3 - L3) < 1e-6:
                    error = 0.0
                    theta3 = 0.0
                    phi3 = 0.0
                else:
                    continue
            else:
                # 弯曲情况
                # 使用优化方法求解第三节的参数
                def objective3(theta):
                    if abs(theta) < 1e-6:
                        return (R3 ** 2 + (Z3 - L3) ** 2)
                    else:
                        radius = L3 / theta
                        x = radius * (1 - np.cos(theta))
                        z = radius * np.sin(theta)
                        return (R3 - x) ** 2 + (Z3 - z) ** 2

                result3 = minimize(objective3, 0.5, bounds=[(0, np.pi / 2)], method='L-BFGS-B')
                if not result3.success:
                    continue

                theta3 = result3.x[0]
                phi3 = np.arctan2(target_3[1], target_3[0])

            # 应用前两节的旋转到第三节
            R12 = rotation_matrix(phi12, theta12)
            section3_local = single_section_kinematics(L3, theta3, phi3)
            end_pos3_global = np.dot(R12, section3_local) + end_12

            # 计算总误差
            error = np.linalg.norm(end_pos3_global - target)

            if error < best_error:
                best_error = error
                best_plan = {
                    'mode': 'two_plus_one',
                    'theta12': theta12,
                    'phi12': phi12,
                    'theta3': theta3,
                    'phi3': phi3,
                    'error': error,
                    'sections': 2
                }

    if best_error < 10.0:  # 10mm误差阈值
        return best_plan

    return None


def rotation_matrix(phi, theta):
    """计算旋转矩阵"""
    # 首先绕Z轴旋转phi
    Rz = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi), np.cos(phi), 0],
        [0, 0, 1]
    ])

    # 然后绕Y轴旋转theta
    Ry = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    # 组合旋转
    return np.dot(Rz, Ry)


def plan_three_independent(target):
    """方法3: 三节独立运动"""

    # 使用优化方法
    def objective_function(params, target):
        """目标函数：最小化末端与目标点的距离"""
        theta1, phi1, theta2, phi2, theta3, phi3 = params
        end_pos = three_section_kinematics(theta1, phi1, theta2, phi2, theta3, phi3)
        return np.linalg.norm(end_pos - target)

    # 尝试多个初始猜测
    best_error = float('inf')
    best_params = None

    # 多个初始猜测
    initial_guesses = []

    # 直臂指向目标
    target_dir = target / np.linalg.norm(target)
    phi_guess = np.arctan2(target_dir[1], target_dir[0])
    theta_guess = np.arccos(target_dir[2]) * 0.5
    initial_guesses.append([theta_guess, phi_guess, theta_guess, phi_guess, theta_guess, phi_guess])

    # 添加一些随机初始猜测
    np.random.seed(42)  # 固定随机种子以便重现
    for _ in range(10):
        initial_guesses.append([
            np.random.uniform(0, np.pi / 2),
            np.random.uniform(0, 2 * np.pi),
            np.random.uniform(0, np.pi / 2),
            np.random.uniform(0, 2 * np.pi),
            np.random.uniform(0, np.pi / 2),
            np.random.uniform(0, 2 * np.pi)
        ])

    # 参数边界
    bounds = [
        (0, np.pi / 2), (0, 2 * np.pi),  # 第一节
        (0, np.pi / 2), (0, 2 * np.pi),  # 第二节
        (0, np.pi / 2), (0, 2 * np.pi)  # 第三节
    ]

    # 对每个初始猜测进行优化
    for initial_params in initial_guesses:
        result = minimize(objective_function, initial_params, args=(target,),
                          bounds=bounds, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-8, 'eps': 1e-8})

        if result.success:
            theta1, phi1, theta2, phi2, theta3, phi3 = result.x
            end_pos = three_section_kinematics(theta1, phi1, theta2, phi2, theta3, phi3)
            error = np.linalg.norm(end_pos - target)

            if error < best_error:
                best_error = error
                best_params = [theta1, phi1, theta2, phi2, theta3, phi3]

    if best_error < 10.0:  # 10mm误差阈值
        theta1, phi1, theta2, phi2, theta3, phi3 = best_params
        return {
            'mode': 'three_independent',
            'theta1': theta1,
            'phi1': phi1,
            'theta2': theta2,
            'phi2': phi2,
            'theta3': theta3,
            'phi3': phi3,
            'error': best_error,
            'sections': 3
        }

    return None


def three_section_kinematics(theta1, phi1, theta2, phi2, theta3, phi3):
    """精确计算三节机械臂的运动学"""
    # 计算第一节
    section1 = single_section_kinematics(L_section, theta1, phi1)

    # 计算第二节 (相对于第一节末端)
    section2_local = single_section_kinematics(L_section, theta2, phi2)
    # 应用第一节的旋转
    R1 = rotation_matrix(phi1, theta1)
    section2 = np.dot(R1, section2_local) + section1

    # 计算第三节 (相对于第二节末端)
    section3_local = single_section_kinematics(L_section, theta3, phi3)
    # 应用前两节的旋转
    R2 = rotation_matrix(phi2, theta2)
    R_total = np.dot(R1, R2)
    section3 = np.dot(R_total, section3_local) + section2

    return section3


def main():
    """主程序入口"""
    global last_left_frame_bgr, mono_depth_map_mm, last_mono_frame_idx, planning_result

    logger.info("=== 三维物体识别与重建系统 ===")
    logger.info("集成所有优化改进的完整版本")

    # 初始化全局变量
    last_left_frame_bgr = None
    mono_depth_map_mm = None
    last_mono_frame_idx = 0
    planning_result = None  # 初始化规划结果变量

    # 创建可视化界面
    fig, ax_3d, ax_info, ax_detection, ax_table, table, info_text, detection_text = create_visualization_interface()

    # 启动双目视觉线程
    logger.info("启动双目视觉线程...")
    vision_thread = threading.Thread(target=stereo_vision_thread, daemon=True)
    vision_thread.start()

    # 等待稳定检测完成
    while not stabilization_complete:
        time.sleep(0.1)
        # 更新可视化，显示检测到的物体
        with detected_objects_lock:
            current_objects = detected_objects.copy()
        update_visualization(ax_3d, current_objects, table, info_text, detection_text, 0)
        plt.pause(0.01)

    # 稳定检测完成后，选择一个物体作为目标（应用修改意见：添加规划容错机制）
    with detected_objects_lock:
        if detected_objects:
            # 选择第一个检测到的物体作为目标
            target_obj = detected_objects[0]
            # 获取物体位置（厘米），并转换为毫米，调整坐标系
            target_pos_cm = target_obj['position']
            # 厘米转毫米，并调整坐标系：X右为正，Y前为正，Z上为正
            target_xyz = np.array([target_pos_cm[0] * 10, target_pos_cm[2] * 10, -target_pos_cm[1] * 10])
            logger.info(f"选择目标物体: {target_obj['class_name']}, 位置: {target_xyz} mm")

            # 进行运动规划
            logger.info("开始运动规划...")
            planning_result = plan_motion(target_xyz)

            # 如果规划失败，使用默认规划（修改点1：添加规划失败的容错）
            if planning_result is None:
                logger.warning("无法到达目标位置，使用默认规划")
                # 使用简单的默认规划
                planning_result = {
                    'mode': 'single',
                    'theta': 0.5,
                    'phi': 0.0,
                    'error': 50.0,  # 较大的误差值
                    'sections': 1
                }
        else:
            logger.error("没有检测到物体，无法进行运动规划")
            # 使用默认目标位置和规划（修改点2：添加无检测物体的容错）
            target_xyz = np.array([300, 200, 600])  # 默认目标位置
            logger.info(f"使用默认目标位置: {target_xyz}")
            planning_result = {
                'mode': 'single',
                'theta': 0.5,
                'phi': 0.0,
                'error': 50.0,  # 较大的误差值
                'sections': 1
            }

    # 运行动画（保持原参数传递方式）
    try:
        logger.info("启动主循环...")
        run_animation(fig, ax_3d, ax_info, ax_detection, ax_table, table, info_text, detection_text)
    except KeyboardInterrupt:
        logger.info("用户中断程序")
    except Exception as e:
        logger.error(f"主循环运行错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("程序结束")

    # 显示matplotlib界面
    try:
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"matplotlib显示失败: {e}")


if __name__ == "__main__":
    main()

