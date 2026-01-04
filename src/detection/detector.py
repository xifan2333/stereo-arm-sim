"""
YOLO 物体检测模块
支持检测和实例分割

遵循小步骤迭代原则，仅实现基础必要功能
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger()


class DetectionResult:
    """检测结果类"""

    def __init__(
        self,
        bbox: np.ndarray,
        conf: float,
        cls: int,
        cls_name: str,
        mask: Optional[np.ndarray] = None,
    ):
        """
        Args:
            bbox: 边界框 [x1, y1, x2, y2]
            conf: 置信度
            cls: 类别ID
            cls_name: 类别名称
            mask: 分割掩码 (HxW)
        """
        self.bbox = bbox
        self.conf = conf
        self.cls = cls
        self.cls_name = cls_name
        self.mask = mask

    @property
    def x1(self) -> int:
        return int(self.bbox[0])

    @property
    def y1(self) -> int:
        return int(self.bbox[1])

    @property
    def x2(self) -> int:
        return int(self.bbox[2])

    @property
    def y2(self) -> int:
        return int(self.bbox[3])

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x1 + self.width // 2, self.y1 + self.height // 2)


class YOLODetector:
    """YOLO 检测器类"""

    def __init__(self):
        """初始化 YOLO 检测器"""
        self.detection_config = config.get_section("detection")

        # 设备选择
        device_name = self.detection_config.get("device", "cuda:0")
        self.device = device_name if torch.cuda.is_available() else "cpu"

        # 加载模型
        model_path = self.detection_config.get("model_path", "yolo11s-seg.pt")
        logger.info(f"加载 YOLO 模型: {model_path}")

        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            logger.success(f"YOLO 模型已加载并移动到设备: {self.device}")
        except Exception as e:
            logger.error(f"YOLO 模型加载失败: {e}")
            raise

        # 预热模型
        self._warmup_model()

        # 推理参数
        self.conf_threshold = self.detection_config.get("conf_threshold", 0.25)
        self.iou_threshold = self.detection_config.get("iou_threshold", 0.45)
        self.max_det = self.detection_config.get("max_det", 100)
        self.classes = self.detection_config.get("classes", None)

        # 可视化参数
        self.show_boxes = self.detection_config.get("show_boxes", True)
        self.show_masks = self.detection_config.get("show_masks", True)
        self.show_labels = self.detection_config.get("show_labels", True)
        self.show_conf = self.detection_config.get("show_conf", True)
        self.line_width = self.detection_config.get("line_width", 2)
        self.mask_alpha = self.detection_config.get("mask_alpha", 0.3)

        # 窗口名称
        self.window = self.detection_config.get("window", "YOLO Detection")

        logger.success("YOLO 检测器初始化成功")

    def _warmup_model(self) -> None:
        """预热模型"""
        try:
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            with torch.inference_mode():
                _ = self.model.predict(
                    dummy_img, device=self.device, verbose=False, conf=0.1, iou=0.45
                )
            logger.info("YOLO 模型预热完成")
        except Exception as e:
            logger.warning(f"YOLO 模型预热失败（可忽略）: {e}")

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        检测物体

        Args:
            image: 输入图像 (BGR)

        Returns:
            List[DetectionResult]: 检测结果列表
        """
        # 推理
        results = self.model.predict(
            image,
            device=self.device,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            classes=self.classes,
            verbose=False,
        )

        # 解析结果
        detections = []
        if len(results) > 0:
            result = results[0]

            # 获取边界框和类别
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls = int(boxes.cls[i].cpu().numpy())
                    cls_name = result.names[cls]

                    # 获取分割掩码（如果有）
                    mask = None
                    if hasattr(result, "masks") and result.masks is not None:
                        mask = result.masks.data[i].cpu().numpy()

                    detection = DetectionResult(
                        bbox=bbox, conf=conf, cls=cls, cls_name=cls_name, mask=mask
                    )
                    detections.append(detection)

        return detections

    def visualize(
        self, image: np.ndarray, detections: List[DetectionResult]
    ) -> np.ndarray:
        """
        可视化检测结果

        Args:
            image: 原始图像
            detections: 检测结果列表

        Returns:
            np.ndarray: 可视化图像
        """
        vis_image = image.copy()

        # 随机颜色生成器
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)

        for det in detections:
            color = tuple(map(int, colors[det.cls % len(colors)]))

            # 绘制掩码
            if self.show_masks and det.mask is not None:
                # 调整掩码大小到图像尺寸
                mask_resized = cv2.resize(
                    det.mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
                )
                mask_bool = mask_resized > 0.5

                # 创建彩色掩码
                colored_mask = np.zeros_like(image)
                colored_mask[mask_bool] = color

                # 叠加掩码
                vis_image = cv2.addWeighted(
                    vis_image, 1.0, colored_mask, self.mask_alpha, 0
                )

            # 绘制边界框
            if self.show_boxes:
                cv2.rectangle(
                    vis_image,
                    (det.x1, det.y1),
                    (det.x2, det.y2),
                    color,
                    self.line_width,
                )

            # 绘制标签
            if self.show_labels:
                label = det.cls_name
                if self.show_conf:
                    label += f" {det.conf:.2f}"

                # 计算文本尺寸
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )

                # 绘制背景
                cv2.rectangle(
                    vis_image,
                    (det.x1, det.y1 - text_h - baseline - 5),
                    (det.x1 + text_w, det.y1),
                    color,
                    -1,
                )

                # 绘制文本
                cv2.putText(
                    vis_image,
                    label,
                    (det.x1, det.y1 - baseline - 2),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                )

        return vis_image

    def show_detections(self, vis_image: np.ndarray) -> None:
        """
        显示检测结果

        Args:
            vis_image: 可视化图像
        """
        cv2.imshow(self.window, vis_image)
