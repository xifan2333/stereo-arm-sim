"""
立体匹配模块
负责双目视差计算和深度估计

遵循小步骤迭代原则，仅实现基础必要功能
"""

from typing import Tuple, Optional
import cv2
import numpy as np
from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger()


class StereoMatcher:
    """立体匹配类 - 基于 SGBM 算法"""

    def __init__(self):
        """初始化立体匹配器"""
        # 从配置获取参数
        self.calib_config = config.get_section("calibration")
        self.stereo_config = config.get_section("stereo_matching")

        # 加载相机标定参数
        self._load_calibration()

        # 初始化 SGBM 匹配器
        self._init_stereo_matcher()

        # 预处理参数
        self.use_clahe = self.stereo_config.get("use_clahe", True)
        if self.use_clahe:
            clip_limit = self.stereo_config.get("clahe_clip_limit", 2.0)
            tile_size = tuple(self.stereo_config.get("clahe_tile_size", [8, 8]))
            self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        else:
            self.clahe = None

        # 窗口名称
        windows = self.stereo_config.get("windows", {})
        self.window_disp_left = windows.get("disparity_left", "Disparity Left")
        self.window_disp_right = windows.get("disparity_right", "Disparity Right")

        logger.success("立体匹配器初始化成功")

    def _load_calibration(self) -> None:
        """加载相机标定参数"""
        # 相机内参
        self.camera_matrix_L = np.array(
            self.calib_config.get("camera_matrix_left"), dtype=np.float64
        )
        self.camera_matrix_R = np.array(
            self.calib_config.get("camera_matrix_right"), dtype=np.float64
        )

        # 畸变系数
        self.dist_coeffs_L = np.array(
            self.calib_config.get("dist_coeffs_left"), dtype=np.float64
        )
        self.dist_coeffs_R = np.array(
            self.calib_config.get("dist_coeffs_right"), dtype=np.float64
        )

        # 旋转和平移
        self.R = np.array(self.calib_config.get("R"), dtype=np.float64)
        self.T = np.array(self.calib_config.get("T"), dtype=np.float64)

        # 图像尺寸
        image_size = self.calib_config.get("image_size", [640, 480])
        self.image_size = tuple(image_size)

        # 计算立体校正参数
        self._compute_rectification()

        logger.info("相机标定参数加载完成")

    def _compute_rectification(self) -> None:
        """计算立体校正参数"""
        # 使用 OpenCV 的 stereoRectify 计算校正参数
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.camera_matrix_L,
            self.dist_coeffs_L,
            self.camera_matrix_R,
            self.dist_coeffs_R,
            self.image_size,
            self.R,
            self.T,
            alpha=0,
        )

        # 计算映射表（用于快速校正图像）
        self.map_Lx, self.map_Ly = cv2.initUndistortRectifyMap(
            self.camera_matrix_L,
            self.dist_coeffs_L,
            self.R1,
            self.P1,
            self.image_size,
            cv2.CV_32FC1,
        )

        self.map_Rx, self.map_Ry = cv2.initUndistortRectifyMap(
            self.camera_matrix_R,
            self.dist_coeffs_R,
            self.R2,
            self.P2,
            self.image_size,
            cv2.CV_32FC1,
        )

        logger.info("立体校正参数计算完成")

    def _init_stereo_matcher(self) -> None:
        """初始化 SGBM 立体匹配器"""
        window_size = self.stereo_config.get("window_size", 5)
        min_disp = self.stereo_config.get("min_disparity", 0)
        num_disp = self.stereo_config.get("num_disparities", 128)

        # 计算 P1 和 P2 参数
        P1 = 8 * 3 * window_size ** 2
        P2 = 32 * 3 * window_size ** 2

        # SGBM 模式映射
        mode_map = {
            "SGBM": cv2.STEREO_SGBM_MODE_SGBM,
            "SGBM_3WAY": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
            "HH": cv2.STEREO_SGBM_MODE_HH,
            "HH4": cv2.STEREO_SGBM_MODE_HH4,
        }
        mode_name = self.stereo_config.get("mode", "SGBM_3WAY")
        mode = mode_map.get(mode_name, cv2.STEREO_SGBM_MODE_SGBM_3WAY)

        # 创建左匹配器
        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=window_size,
            P1=P1,
            P2=P2,
            disp12MaxDiff=self.stereo_config.get("disp12_max_diff", 1),
            uniquenessRatio=self.stereo_config.get("uniqueness_ratio", 15),
            speckleWindowSize=self.stereo_config.get("speckle_window_size", 100),
            speckleRange=self.stereo_config.get("speckle_range", 1),
            preFilterCap=self.stereo_config.get("pre_filter_cap", 63),
            mode=mode,
        )

        # 尝试创建 WLS 滤波器（需要 opencv-contrib）
        self.use_wls = False
        self.wls_filter = None
        postprocess_config = self.stereo_config.get("postprocess", {})

        if postprocess_config.get("use_wls", True):
            try:
                self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
                self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.left_matcher)
                self.wls_filter.setLambda(postprocess_config.get("wls_lambda", 8000.0))
                self.wls_filter.setSigmaColor(postprocess_config.get("wls_sigma", 1.5))
                self.use_wls = True
                logger.info("WLS 滤波器初始化成功")
            except AttributeError:
                logger.warning("opencv-contrib 不可用，使用标准右匹配器")
                self._create_standard_right_matcher(
                    min_disp, num_disp, window_size, P1, P2, mode
                )
        else:
            self._create_standard_right_matcher(
                min_disp, num_disp, window_size, P1, P2, mode
            )

        logger.info(f"SGBM 匹配器初始化完成 (模式: {mode_name}, WLS: {self.use_wls})")

    def _create_standard_right_matcher(
        self, min_disp: int, num_disp: int, window_size: int, P1: int, P2: int, mode: int
    ) -> None:
        """创建标准右匹配器"""
        self.right_matcher = cv2.StereoSGBM_create(
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
            mode=mode,
        )

    def rectify_images(
        self, frame_left: np.ndarray, frame_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        校正左右图像

        Args:
            frame_left: 左图像
            frame_right: 右图像

        Returns:
            Tuple[np.ndarray, np.ndarray]: (校正后的左图像, 校正后的右图像)
        """
        rectified_left = cv2.remap(
            frame_left, self.map_Lx, self.map_Ly, cv2.INTER_LINEAR
        )
        rectified_right = cv2.remap(
            frame_right, self.map_Rx, self.map_Ry, cv2.INTER_LINEAR
        )

        return rectified_left, rectified_right

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像

        Args:
            image: 输入图像

        Returns:
            np.ndarray: 处理后的图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 确保 uint8 类型
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # CLAHE 增强（可选）
        if self.use_clahe and self.clahe is not None:
            gray = self.clahe.apply(gray)

        return gray

    def compute_disparity(
        self, frame_left: np.ndarray, frame_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        计算视差图和3D点云

        Args:
            frame_left: 左图像
            frame_right: 右图像

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                (左视差图, 右视差图, 左视差彩色可视化, 3D点云)
        """
        # 校正图像
        rect_left, rect_right = self.rectify_images(frame_left, frame_right)

        # 预处理
        proc_left = self.preprocess_image(rect_left)
        proc_right = self.preprocess_image(rect_right)

        # 计算左视差
        disp_left_raw = self.left_matcher.compute(proc_left, proc_right).astype(
            np.float32
        ) / 16.0

        # 计算右视差
        disp_right_raw = self.right_matcher.compute(proc_right, proc_left).astype(
            np.float32
        ) / 16.0

        # 应用 WLS 滤波或标准后处理
        if self.use_wls and self.wls_filter is not None:
            try:
                disp_left = self.wls_filter.filter(
                    disp_left_raw, proc_left, None, disp_right_raw
                ).astype(np.float32)
            except Exception as e:
                logger.warning(f"WLS 滤波失败，使用标准后处理: {e}")
                disp_left = self._postprocess_disparity(disp_left_raw)
        else:
            disp_left = self._postprocess_disparity(disp_left_raw)

        # 后处理右视差
        disp_right = self._postprocess_disparity(disp_right_raw)

        # 生成伪彩色可视化
        disp_color = self._visualize_disparity(disp_left)

        # 生成3D点云
        xyz_pointcloud = self._generate_pointcloud(disp_left)

        return disp_left, disp_right, disp_color, xyz_pointcloud

    def _generate_pointcloud(self, disparity: np.ndarray) -> np.ndarray:
        """
        从视差图生成3D点云

        Args:
            disparity: 视差图

        Returns:
            np.ndarray: 3D点云 (HxWx3), 单位：毫米
        """
        # 使用 Q 矩阵进行3D重投影
        xyz = cv2.reprojectImageTo3D(disparity.astype(np.float32), self.Q)

        # 转换为 float32 并确保单位为毫米
        xyz = xyz.astype(np.float32)

        return xyz

    def _postprocess_disparity(self, disparity: np.ndarray) -> np.ndarray:
        """
        后处理视差图

        Args:
            disparity: 原始视差图

        Returns:
            np.ndarray: 处理后的视差图
        """
        disp = disparity.copy()

        # 清理 NaN 和负值
        disp[~np.isfinite(disp)] = 0.0
        disp[disp < 0] = 0.0

        postprocess_config = self.stereo_config.get("postprocess", {})

        # 中值滤波（去除椒盐噪声）
        if postprocess_config.get("median_filter", True):
            kernel_size = postprocess_config.get("median_kernel_size", 5)
            disp = cv2.medianBlur(disp.astype(np.float32), kernel_size)

        # 双边滤波（保边平滑）
        if postprocess_config.get("bilateral_filter", True):
            d = postprocess_config.get("bilateral_d", 5)
            sigma_color = postprocess_config.get("bilateral_sigma_color", 9.0)
            sigma_space = postprocess_config.get("bilateral_sigma_space", 7.0)
            disp = cv2.bilateralFilter(
                disp.astype(np.float32), d, sigma_color, sigma_space
            )

        # 再次清理负值
        disp[disp < 0] = 0.0

        return disp

    def _visualize_disparity(self, disparity: np.ndarray) -> np.ndarray:
        """
        将视差图转换为伪彩色可视化

        Args:
            disparity: 视差图

        Returns:
            np.ndarray: 伪彩色图像 (BGR)
        """
        if np.max(disparity) > 0:
            # 归一化到 0-255
            disp_norm = (
                (disparity / (np.max(disparity) + 1e-9) * 255.0).clip(0, 255).astype(np.uint8)
            )
            # 应用伪彩色映射
            disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
        else:
            # 如果视差全为0，返回黑色图像
            disp_color = np.zeros(
                (disparity.shape[0], disparity.shape[1], 3), dtype=np.uint8
            )

        return disp_color

    def show_disparity(
        self, disp_color_left: np.ndarray, disp_right: Optional[np.ndarray] = None
    ) -> None:
        """
        显示视差图

        Args:
            disp_color_left: 左视差彩色图
            disp_right: 右视差图（可选）
        """
        cv2.imshow(self.window_disp_left, disp_color_left)

        if disp_right is not None:
            disp_color_right = self._visualize_disparity(disp_right)
            cv2.imshow(self.window_disp_right, disp_color_right)
