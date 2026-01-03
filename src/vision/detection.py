# -*- coding: utf-8 -*-
import torch
from ultralytics import YOLO
import logging
import numpy as np
import cv2

from src.config import YOLO_MODEL_PATH # Import the YOLO model path

logger = logging.getLogger("StereoVision.Detection")

class YOLODetector:
    def __init__(self, model_path=YOLO_MODEL_PATH):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        try:
            self.model = YOLO(model_path)
            # 把模型固定到 device（避免每次推理内部迁移引入延迟）
            self.model.to(self.device)
            logger.info(f"YOLO 模型已加载并移动到设备: {self.device}")
            self._warmup_model()
        except Exception as e:
            logger.error(f"无法加载或初始化 YOLO 模型: {e}")
            self.model = None

    def _warmup_model(self):
        """预热一次模型以减少首帧卡顿"""
        if self.model is None:
            return
        try:
            dummy_img = np.zeros((1, 3, 640, 640), dtype=np.float32)
            with torch.inference_mode():
                _ = self.model.predict(torch.from_numpy(dummy_img).to(self.device), device=self.device, verbose=False, conf=0.1, iou=0.45)
            logger.info("YOLO 模型预热完成")
        except Exception as e:
            logger.debug(f"YOLO 模型预热失败（可忽略）: {e}")

    def detect(self, frame):
        """
        在给定的图像帧上执行物体检测。
        返回结果列表，每个结果包含 bbox, class_name, conf 等。
        """
        if self.model is None:
            return []
        try:
            results = self.model.predict(frame, device=self.device, verbose=False, conf=0.25, iou=0.7)
            detections = []
            for r in results:
                for *xyxy, conf, cls in r.boxes.data.cpu().numpy():
                    class_name = self.model.names[int(cls)]
                    detections.append({
                        'bbox': np.array(xyxy),
                        'conf': float(conf),
                        'class_id': int(cls),
                        'class_name': class_name
                    })
            return detections
        except Exception as e:
            logger.error(f"YOLO 检测失败: {e}")
            return []

    def segment_object_with_yolo(self, frame, bbox_2d):
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


    def enhanced_segmentation(self, frame, bbox_2d, class_name):
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


    def create_precise_mask_from_bbox(self, frame, bbox_2d, class_name):
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
