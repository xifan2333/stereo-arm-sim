# -*- coding: utf-8 -*-
import numpy as np
import time
import logging
from collections import deque, defaultdict
import threading # Assuming threading.Lock is used with PositionTracker

from src.utils.helpers import logger, get_object_id
from src.config import CAMERA_CONFIG, POSITION_SMOOTHING_FACTOR, STABILIZATION_THRESHOLD

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

class PositionTracker:
    def __init__(self, history_size=30, yolo_model_names=None):
        self.positions = defaultdict(lambda: {
            'left_positions': deque(maxlen=history_size),
            'right_positions': deque(maxlen=history_size),
            'left_confidences': deque(maxlen=history_size),
            'right_confidences': deque(maxlen=history_size),
            'fused_positions': deque(maxlen=history_size),
            'last_update': 0,
            'class_id': -1
        })
        self.R = CAMERA_CONFIG.rotation_matrix
        self.T = CAMERA_CONFIG.translation_vector
        self.object_counter = 0
        self.tracked_ids = {}
        self.last_print_time = 0
        self.coordinate_system = "X:右为正, Y:下为正, Z:前为正"
        self.depth_filters = {}
        self.yolo_model_names = yolo_model_names if yolo_model_names is not None else {}

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
                    # Use provided model names
                    class_name = self.yolo_model_names.get(obj['class_id'], f"Unknown({obj['class_id']})")
                    logger.info(f"物体ID: {get_object_id(obj)}, 类别: {class_name}")
                    logger.info(f"位置: X={pos[0]:.1f}, Y={pos[1]:.1f}, Z={pos[2]:.1f} cm")
                    if obj['change'] is not None:
                        change = obj['change']
                        logger.info(f"位置变化: ΔX={change[0]:.1f}, ΔY={change[1]:.1f}, ΔZ={change[2]:.1f} cm")
                    logger.info("-" * 40)

# Global instances (to be initialized in main.py once)
# tracker = SimpleTracker(iou_thresh=0.35, max_missed=5)
# stabilizer = PositionStabilizer(window_size=5, ema_alpha=None)
# xyz_lock = threading.Lock()
