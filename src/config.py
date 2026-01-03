# -*- coding: utf-8 -*-
import numpy as np
import torch
from enum import Enum
import os
import importlib.util

# sklearn可用性检查
SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None

# 路径配置
ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets')
MODELS_DIR = os.path.join(ASSETS_DIR, 'models')
CALIBRATION_DIR = os.path.join(ASSETS_DIR, 'calibration')

# YOLO 模型
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, 'yolo11s.pt')

# 自动检测点云单位的阈值
_UNIT_MM_THRESHOLD = 1000.0  # if median z > 1000 -> input likely mm
_UNIT_M_THRESHOLD = 5.0      # if median z < 5 -> input likely meters

# 测量参数
MEASUREMENT_WINDOW_SIZE = 11
DEPTH_CONFIDENCE_THRESHOLD = 0.8
POSITION_SMOOTHING_FACTOR = 0.7
STABILIZATION_THRESHOLD = 10  # 稳定阈值，连续10帧稳定

# 深度校准结构
DEPTH_CALIB = {
    "coeffs": None,
    "degree": None,
    "per_class": {}
}
_SMOOTH_MAX_AGE_S = 2.0  # 深度缓存年龄上限（秒）

# 中文字符串标签
LABELS = {
    'coords': '坐标', 'invalid': '无效', 'distance': '距离',
    'cm': '厘米', 'world_coords': '三维坐标', 'cleanup': '清理资源中...',
    'cleanup_complete': '清理完成', 'user_quit': '用户请求退出',
    'user_interrupt': '用户中断程序', 'error_occurred': '发生错误',
    'frame_error': '无法获取图像帧', 'program_end': '程序结束'
}

# 可视化参数
L_SECTION = 300.0  # 每节长度300mm
TOTAL_ARM_LENGTH = 3 * L_SECTION  # 总长度900mm
ROPE_RADIUS = 16.0  # 绳孔到中心距离
ROPE_OFFSETS = [0, np.pi / 2, np.pi, 3 * np.pi / 2]  # 四线90°间隔
ROPE_COLORS = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00']  # 绳子颜色 (红,绿,蓝,黄)
ROPE_LINEWIDTH = 2  # 绳子粗细
DISC_INTERVAL = 60.0  # 圆盘间隔改为60mm
DISC_DIAMETER = 40.0  # 圆盘直径40mm
DISC_RADIUS = DISC_DIAMETER / 2
X_RANGE = (-1500, 1500)  # X轴范围
Y_RANGE = (-1500, 1500)  # Y轴范围
Z_RANGE = (-1000, 1500)  # Z轴范围扩展为(-1000, 1500)
TOTAL_FRAMES = 100  # 帧数

# 三节圆盘颜色
SECTION_COLORS = ['#A0D2FF', '#90EE90', '#FFB6C1']  # 浅蓝, 浅绿, 浅粉

# 绳子名称
ROPE_NAMES = ["绳1", "绳2", "绳3", "绳4", "绳5", "绳6", "绳7", "绳8", "绳9", "绳10", "绳11", "绳12"]

# 全局变量 (作为配置的一部分，但不作为常量，可在程序运行时修改)
TARGET_XYZ = np.array([300, 200, 600])  # 初始目标位置
MAX_DEPTH = 1500.0  # 最大深度限制（毫米）

# 立体匹配算法类型
class StereoAlgorithm(Enum):
    SGBM = 1
    BM = 2
    ELAS = 3

# 算法配置
STEREO_ALGORITHM = StereoAlgorithm.SGBM
USE_GPU_ACCELERATION = torch.cuda.is_available()
ENABLE_DEPTH_FILTERING = True
ENABLE_POST_PROCESSING = True

# 3D 估算参数
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

# -----------------------------------------------------
# 相机标定参数加载占位符 (这些将从 JSON 文件加载)
# -----------------------------------------------------
class CameraConfig:
    def __init__(self):
        self.image_width = 640
        self.image_height = 480
        self.image_size = (self.image_width, self.image_height)

        self.camera_matrix_l = None
        self.dist_coeff_l = None
        self.camera_matrix_r = None
        self.dist_coeff_r = None
        self.rotation_matrix = None # R
        self.translation_vector = None # T
        self.load_calibration_data()

    def load_calibration_data(self):
        import json
        import numpy as np
        calib_file = os.path.join(CALIBRATION_DIR, 'camera_params.json')
        if not os.path.exists(calib_file):
            print(f"警告: 标定文件未找到: {calib_file}。将使用默认值。")
            self._set_default_calibration()
            return

        try:
            with open(calib_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.image_width = data.get('image_width', 640)
            self.image_height = data.get('image_height', 480)
            self.image_size = (self.image_width, self.image_height)

            self.camera_matrix_l = np.array(data.get('camera_matrix_l')).astype(np.float64)
            self.dist_coeff_l = np.array(data.get('dist_coeff_l')).astype(np.float64)
            self.camera_matrix_r = np.array(data.get('camera_matrix_r')).astype(np.float64)
            self.dist_coeff_r = np.array(data.get('dist_coeff_r')).astype(np.float64)
            self.rotation_matrix = np.array(data.get('rotation_matrix')).astype(np.float64)
            self.translation_vector = np.array(data.get('translation_vector')).astype(np.float64)

            print(f"成功加载标定数据: {calib_file}")

        except Exception as e:
            print(f"加载标定数据失败: {e}。将使用默认值。")
            self._set_default_calibration()

    def _set_default_calibration(self):
        # 从 main.py 中复制的默认值
        self.camera_matrix_l = np.array([[410.3084, -0.2777, 309.3976],
                                          [0, 410.2129, 262.0564],
                                          [0, 0, 1.0000]]).astype(np.float64)
        self.dist_coeff_l = np.array([0.0027, 0.6848, -0.0043, 0.0066, -2.0220], dtype=np.float64)

        self.camera_matrix_r = np.array([[409.2531, -0.3487, 299.1536],
                                          [0, 408.9821, 265.7137],
                                          [0, 0, 1.0000]]).astype(np.float64)
        self.dist_coeff_r = np.array([0.0357, 0.3211, -0.0042, 0.0035, -0.7268], dtype=np.float64)

        self.translation_vector = np.array([-61.114637167122600, -0.044597478744966, 0.583576573837856]).astype(np.float64)
        self.rotation_matrix = np.array([[0.999780788646075, 0.0004806193456666200, 0.020931881407937],
                                          [-0.0004322695666685383, 0.999997228514369, -0.0006092913311984787],
                                          [-0.020932935705560, 0.002304770839986, 0.999778225525102]]).astype(np.float64)

CAMERA_CONFIG = CameraConfig()
