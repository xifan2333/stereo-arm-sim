# -*- coding: utf-8 -*-
import cv2
import numpy as np
import logging

from src.utils.helpers import ensure_xyz_in_mm
from src.config import CAMERA_CONFIG  # Import the camera config object

logger = logging.getLogger("StereoVision.Stereo")


class StereoProcessor:
    def __init__(self):
        self.camera_config = CAMERA_CONFIG
        # 立体校正结果 (内部变量，不直接作为全局)
        self.Rl = None
        self.Rr = None
        self.Pl = None
        self.Pr = None
        self.Q = None
        self.mapLx = None
        self.mapLy = None
        self.mapRx = None
        self.mapRy = None

        self.verify_calibration()
        self.stereo_rectify()

        # Output variables (for debugging/external access if needed)
        self.last_disp_left = None
        self.last_disp_right = None
        self.last_xyz_left = None
        self.last_xyz_right = None

    def stereo_rectify(self):
        """立体校正，生成校正映射"""
        self.Rl, self.Rr, self.Pl, self.Pr, self.Q, _, _ = cv2.stereoRectify(
            self.camera_config.camera_matrix_l,
            self.camera_config.dist_coeff_l,
            self.camera_config.camera_matrix_r,
            self.camera_config.dist_coeff_r,
            self.camera_config.image_size,
            self.camera_config.rotation_matrix,
            self.camera_config.translation_vector,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=-1,
            newImageSize=self.camera_config.image_size,
        )
        self.mapLx, self.mapLy = cv2.initUndistortRectifyMap(
            self.camera_config.camera_matrix_l,
            self.camera_config.dist_coeff_l,
            self.Rl,
            self.Pl,
            self.camera_config.image_size,
            cv2.CV_32FC1,
        )
        self.mapRx, self.mapRy = cv2.initUndistortRectifyMap(
            self.camera_config.camera_matrix_r,
            self.camera_config.dist_coeff_r,
            self.Rr,
            self.Pr,
            self.camera_config.image_size,
            cv2.CV_32FC1,
        )
        # 调试输出：打印 Q 的关键信息（便于核查 baseline 与单位）
        try:
            if self.Q is not None:
                q_32 = float(self.Q[3, 2]) if self.Q.shape == (4, 4) else None
                baseline_est_mm = (
                    (1.0 / q_32) if (q_32 is not None and abs(q_32) > 1e-9) else None
                )
                logger.info(
                    f"stereo_rectify: Q[3,2]={q_32}, baseline_est (mm) ~ {baseline_est_mm:.3f}, Q[2,3]={float(self.Q[2, 3]):.3f}"
                )
        except Exception:
            pass

    def verify_calibration(self):
        """验证标定参数合理性"""
        R = self.camera_config.rotation_matrix
        det_R = np.linalg.det(R)
        if abs(det_R - 1.0) > 0.01:
            logger.warning(f"警告: 旋转矩阵行列式应为1 (当前: {det_R:.6f})")
        identity_matrix = np.eye(3)
        diff = np.abs(np.dot(R, R.T) - identity_matrix)
        if np.max(diff) > 0.01:
            logger.warning("警告: 旋转矩阵不是正交矩阵")
        logger.info("标定验证完成")

    def postprocess_disparity(self, disp):
        """
        输入 disp (float32, 单位像素)，返回同尺寸的 float32 disparity（空洞和噪声被平滑/填充）。
        """
        if disp is None or disp.size == 0:
            return disp

        d = disp.astype(np.float32).copy()

        # 确保没有NaN或inf
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        d = np.clip(d, 0.0, None)  # 确保非负

        # 1) 中值滤波去孤点
        try:
            # 转换为uint8进行中值滤波以避免float32的问题
            d_max = d.max()
            if d_max > 0:
                d_normalized = (d / d_max * 255.0).astype(np.uint8)
                d_med_norm = cv2.medianBlur(d_normalized, 5)
                d_med = (d_med_norm.astype(np.float32) / 255.0) * d_max
            else:
                d_med = d
        except Exception as e:
            logger.debug(f"medianBlur failed: {e}")
            d_med = d

        # 2) 双边滤波保持边缘
        try:
            d_bi = cv2.bilateralFilter(d_med.astype(np.float32), 9, 75, 75)
        except Exception as e:
            logger.debug(f"bilateralFilter failed: {e}")
            d_bi = d_med

        # 3) 小孔洞填充（局部均值法）
        mask_invalid = (d_bi <= 0) | (~np.isfinite(d_bi))
        if mask_invalid.all():
            return d_bi.astype(np.float32)

        kernel = np.ones((5, 5), dtype=np.float32)
        valid_mask = (~mask_invalid).astype(np.float32)
        sum_local = cv2.filter2D((d_bi * valid_mask).astype(np.float32), -1, kernel)
        count_local = cv2.filter2D(valid_mask, -1, kernel)

        fill_vals = np.zeros_like(d_bi, dtype=np.float32)
        nonzero = count_local > 0
        fill_vals[nonzero] = sum_local[nonzero] / count_local[nonzero]

        fill_mask = mask_invalid & (count_local > 0)
        d_bi[fill_mask] = fill_vals[fill_mask]

        # 4) 最后再一个小的 median 以收尾
        try:
            d_max = d_bi.max()
            if d_max > 0:
                d_normalized = (d_bi / d_max * 255.0).astype(np.uint8)
                d_out_norm = cv2.medianBlur(d_normalized, 3)
                d_out = (d_out_norm.astype(np.float32) / 255.0) * d_max
            else:
                d_out = d_bi
        except Exception as e:
            logger.debug(f"final medianBlur failed: {e}")
            d_out = d_bi

        return d_out.astype(np.float32)

    def stereo_match(self, rectifyImageL, rectifyImageR):
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
            imageWidth, imageHeight = (
                self.camera_config.image_width,
                self.camera_config.image_height,
            )
            if rectifyImageL is None or rectifyImageR is None:
                H = imageHeight
                W = imageWidth
                empty_xyz = np.full((H, W, 3), np.nan, dtype=np.float32)
                return (
                    np.zeros((H, W, 3), dtype=np.uint8),
                    empty_xyz,
                    np.zeros((H, W), dtype=np.float32),
                    np.zeros((H, W), dtype=np.float32),
                    np.zeros((H, W), dtype=np.float32),
                    empty_xyz,
                )

            left = rectifyImageL
            right = rectifyImageR
            # ensure uint8 single channel
            if left.dtype != np.uint8:
                left = cv2.normalize(left, None, 0, 255, cv2.NORM_MINMAX).astype(
                    np.uint8
                )
            if right.dtype != np.uint8:
                right = cv2.normalize(right, None, 0, 255, cv2.NORM_MINMAX).astype(
                    np.uint8
                )

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
            P1 = 8 * 3 * window_size**2
            P2 = 32 * 3 * window_size**2

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
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
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
                    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
                )
                use_wls = False

            # 计算原始视差（SGBM 输出需要 /16）
            disp_left_raw = (
                left_matcher.compute(left_eq, right_eq).astype(np.float32) / 16.0
            )
            disp_right_raw = None
            try:
                # 右 matcher 输入顺序 swapped
                disp_right_raw = (
                    right_matcher.compute(right_eq, left_eq).astype(np.float32) / 16.0
                )
            except Exception:
                # 极端兜底（不应发生，但保证变量存在）
                disp_right_raw = np.zeros_like(disp_left_raw, dtype=np.float32)

            # WLS 或 后处理
            if use_wls and wls_filter is not None:
                try:
                    disp_filtered = wls_filter.filter(
                        disp_left_raw, left_eq, None, disp_right_raw
                    ).astype(np.float32)
                except Exception:
                    disp_filtered = self.postprocess_disparity(disp_left_raw)
            else:
                disp_filtered = self.postprocess_disparity(disp_left_raw)

            # 同步后处理右目视差供显示（后处理保证无负值/NaN）
            disp_right_proc = self.postprocess_disparity(disp_right_raw)

            # 清理 NaN 与负值
            disp_filtered[~np.isfinite(disp_filtered)] = 0.0
            disp_filtered[disp_filtered < 0] = 0.0
            disp_right_proc[~np.isfinite(disp_right_proc)] = 0.0
            disp_right_proc[disp_right_proc < 0] = 0.0

            # 可视化左视差（伪彩色）
            if np.max(disp_filtered) > 0:
                disp_vis = (
                    (disp_filtered / (np.max(disp_filtered) + 1e-9) * 255.0)
                    .clip(0, 255)
                    .astype(np.uint8)
                )
                disp_color_left = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
            else:
                disp_color_left = np.zeros(
                    (left.shape[0], left.shape[1], 3), dtype=np.uint8
                )

            # 简单置信度图（基于视差是否有效与幅值归一）
            conf = np.clip(
                disp_filtered / (np.max(disp_filtered) + 1e-9), 0.0, 1.0
            ).astype(np.float32)

            # 使用 Q 做 3D 反投影（返回 xyz 单位会根据 Q/外参决定，我们后面 normalize）
            H, W = disp_filtered.shape
            if self.Q is None:
                xyz_left = np.full((H, W, 3), np.nan, dtype=np.float32)
            else:
                try:
                    xyz_left = cv2.reprojectImageTo3D(
                        disp_filtered.astype(np.float32), self.Q
                    )
                    xyz_left = np.asarray(xyz_left, dtype=np.float32)
                except Exception:
                    xyz_left = np.full((H, W, 3), np.nan, dtype=np.float32)

            xyz_left = ensure_xyz_in_mm(xyz_left)

            # 构造右目的 xyz：优先使用 remap xyz_left -> right（这个方法和后续点云提取/融合兼容）
            try:
                xs, ys = np.meshgrid(np.arange(W), np.arange(H))
                map_x = (xs - disp_filtered).astype(np.float32)
                map_y = ys.astype(np.float32)
                xyz_right = np.full_like(xyz_left, np.nan, dtype=np.float32)
                for c in range(3):
                    ch = xyz_left[:, :, c].astype(np.float32)
                    # 将NaN替换为0进行remap，然后再恢复NaN
                    ch_valid = np.nan_to_num(ch, nan=0.0)
                    rem = cv2.remap(
                        ch_valid,
                        map_x,
                        map_y,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0.0,
                    )
                    # 恢复原始的NaN位置
                    rem = rem.astype(np.float32)
                    invalid_in_source = ~np.isfinite(ch)
                    if invalid_in_source.any():
                        rem[invalid_in_source] = np.nan
                    xyz_right[:, :, c] = rem
                invalid_mask = (disp_filtered <= 0) | (~np.isfinite(xyz_left[:, :, 2]))
                xyz_right[invalid_mask, :] = np.nan
            except Exception as e:
                logger.debug(f"xyz_right构造失败: {e}")
                xyz_right = np.full_like(xyz_left, np.nan, dtype=np.float32)

            # 把最新的视差与点云导出成内部变量
            self.last_disp_left = disp_filtered.astype(np.float32).copy()
            self.last_disp_right = disp_right_proc.astype(np.float32).copy()
            self.last_xyz_left = xyz_left.copy()
            self.last_xyz_right = xyz_right.copy()

            # 返回（注意：disp 返回 float32 像素单位）
            return (
                disp_color_left,
                xyz_left,
                conf,
                disp_filtered.astype(np.float32),
                disp_right_proc.astype(np.float32),
                xyz_right,
            )

        except Exception as e:
            import traceback
            logger.error(f"stereo_match: 异常: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            H = self.camera_config.image_height
            W = self.camera_config.image_width
            empty_xyz = np.full((H, W, 3), np.nan, dtype=np.float32)
            return (
                np.zeros((H, W, 3), dtype=np.uint8),
                empty_xyz,
                np.zeros((H, W), dtype=np.float32),
                np.zeros((H, W), dtype=np.float32),
                np.zeros((H, W), dtype=np.float32),
                empty_xyz,
            )
