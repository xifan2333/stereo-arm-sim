"""
摄像头管理模块
负责双目摄像头的初始化、读取和显示

遵循小步骤迭代原则，仅实现基础必要功能
"""

from typing import Optional, Tuple
import cv2
import numpy as np
from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger()


class StereoCamera:
    """双目摄像头管理类"""

    def __init__(self):
        """初始化双目摄像头"""
        # 从配置管理器获取摄像头配置
        camera_config = config.camera

        # 摄像头参数
        self.width = camera_config.get("width", 1280)
        self.height = camera_config.get("height", 480)
        self.single_width = camera_config.get("single_width", 640)
        self.single_height = camera_config.get("single_height", 480)

        # 窗口名称
        windows = camera_config.get("windows", {})
        self.window_left = windows.get("left", "left")
        self.window_right = windows.get("right", "right")

        # 摄像头配置
        self.camera_indices = camera_config.get("camera_indices", [0, 1, 2])
        self.backend_name = camera_config.get("backend", "DSHOW")

        # 摄像头对象
        self.camera: Optional[cv2.VideoCapture] = None
        self.camera_index: Optional[int] = None

    def open(self) -> bool:
        """
        打开摄像头

        Returns:
            bool: 是否成功打开摄像头
        """
        # 获取对应的OpenCV后端
        backend_map = {"DSHOW": cv2.CAP_DSHOW, "V4L2": cv2.CAP_V4L2, "AUTO": cv2.CAP_ANY}
        backend = backend_map.get(self.backend_name, cv2.CAP_DSHOW)

        # 尝试打开摄像头
        for camera_index in self.camera_indices:
            try:
                logger.info(f"尝试打开摄像头 {camera_index} (后端: {self.backend_name})")
                camera = cv2.VideoCapture(camera_index, backend)

                if not camera.isOpened():
                    logger.warning(f"摄像头 {camera_index} 打开失败")
                    continue

                # 设置分辨率
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

                # 测试读取一帧
                ret, test_frame = camera.read()
                if ret and test_frame is not None:
                    logger.success(f"成功初始化摄像头 {camera_index}")
                    self.camera = camera
                    self.camera_index = camera_index
                    return True
                else:
                    logger.warning(f"摄像头 {camera_index} 无法读取帧")
                    camera.release()

            except Exception as e:
                logger.error(f"打开摄像头 {camera_index} 时发生错误: {e}")
                continue

        logger.error("无法找到可用的摄像头")
        return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        读取一帧并分割为左右画面

        Returns:
            Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
                (是否成功, 左画面, 右画面)
        """
        if self.camera is None or not self.camera.isOpened():
            logger.error("摄像头未打开")
            return False, None, None

        ret, frame = self.camera.read()

        if not ret or frame is None:
            logger.warning("读取帧失败")
            return False, None, None

        # 检查帧尺寸
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            logger.warning(
                f"帧尺寸不匹配: 期望 {self.width}x{self.height}, "
                f"实际 {frame.shape[1]}x{frame.shape[0]}"
            )

        # 分割左右画面
        frame_left = frame[:, : self.single_width]
        frame_right = frame[:, self.single_width :]

        return True, frame_left, frame_right

    def show_frames(self, frame_left: np.ndarray, frame_right: np.ndarray) -> None:
        """
        显示左右画面

        Args:
            frame_left: 左画面
            frame_right: 右画面
        """
        if frame_left is not None:
            cv2.imshow(self.window_left, frame_left)

        if frame_right is not None:
            cv2.imshow(self.window_right, frame_right)

    def release(self) -> None:
        """释放摄像头资源"""
        if self.camera is not None and self.camera.isOpened():
            self.camera.release()
            logger.info("摄像头已释放")

        cv2.destroyAllWindows()
        logger.info("所有窗口已关闭")

    def is_opened(self) -> bool:
        """
        检查摄像头是否已打开

        Returns:
            bool: 摄像头是否已打开
        """
        return self.camera is not None and self.camera.isOpened()
