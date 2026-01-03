# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np
import logging
import cv2

from src.config import L_SECTION, ROPE_RADIUS, ROPE_OFFSETS, ROPE_COLORS, ROPE_LINEWIDTH, \
    DISC_INTERVAL, DISC_DIAMETER, DISC_RADIUS, X_RANGE, Y_RANGE, Z_RANGE, \
    TOTAL_FRAMES, SECTION_COLORS, ROPE_NAMES, TARGET_XYZ, MAX_DEPTH
from src.utils.helpers import logger

# Configure matplotlib to support Chinese characters
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False  # Fix minus sign display


class Plotter:
    def __init__(self):
        self.fig = None
        self.ax_3d = None
        self.ax_camera_l = None  # 左摄像头原始图像
        self.ax_camera_r = None  # 右摄像头原始图像
        self.ax_disparity_l = None
        self.ax_disparity_r = None
        self.ax_info = None

        self._setup_plot()

    def _setup_plot(self):
        self.fig = plt.figure(figsize=(24, 10))
        gs = gridspec.GridSpec(2, 5, width_ratios=[2, 1, 1, 1, 1], height_ratios=[3, 1])

        # 3D 场景
        self.ax_3d = self.fig.add_subplot(gs[0, 0], projection='3d')
        self.ax_3d.set_xlabel("X (mm)")
        self.ax_3d.set_ylabel("Y (mm)")
        self.ax_3d.set_zlabel("Z (mm)")
        self.ax_3d.set_title("3D 机械臂运动与障碍物避障")
        self.ax_3d.set_xlim(X_RANGE)
        self.ax_3d.set_ylim(Y_RANGE)
        self.ax_3d.set_zlim(Z_RANGE)
        self.ax_3d.view_init(elev=20, azim=-60)
        self.ax_3d.set_box_aspect([np.ptp(X_RANGE), np.ptp(Y_RANGE), np.ptp(Z_RANGE)])

        # 左摄像头原始图像
        self.ax_camera_l = self.fig.add_subplot(gs[0, 1])
        self.ax_camera_l.set_title("左摄像头")
        self.ax_camera_l.axis('off')

        # 右摄像头原始图像
        self.ax_camera_r = self.fig.add_subplot(gs[0, 2])
        self.ax_camera_r.set_title("右摄像头")
        self.ax_camera_r.axis('off')

        # 左视差图
        self.ax_disparity_l = self.fig.add_subplot(gs[0, 3])
        self.ax_disparity_l.set_title("左视差图 (彩色)")
        self.ax_disparity_l.axis('off')

        # 右视差图
        self.ax_disparity_r = self.fig.add_subplot(gs[0, 4])
        self.ax_disparity_r.set_title("右视差图 (彩色)")
        self.ax_disparity_r.axis('off')

        # 信息显示
        self.ax_info = self.fig.add_subplot(gs[1, :])
        self.ax_info.axis('off')
        self.info_text_obj = self.ax_info.text(0.02, 0.95, "", transform=self.ax_info.transAxes,
                                                verticalalignment='top', fontsize=10,
                                                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="b", lw=1, alpha=0.5))

        plt.tight_layout()
        plt.ion() # Interactive mode
        plt.show(block=False)

    def draw_arm_disc(self, disc_center, disc_idx):
        """绘制机械臂圆盘"""
        angle = np.linspace(0, 2 * np.pi, 50)
        x_disc = disc_center[0] + DISC_RADIUS * np.cos(angle)
        y_disc = disc_center[1] + DISC_RADIUS * np.sin(angle)
        z_disc = disc_center[2] * np.ones_like(angle)

        color = SECTION_COLORS[disc_idx % len(SECTION_COLORS)]
        self.ax_3d.plot(x_disc, y_disc, z_disc, color=color, alpha=0.8)
        self.ax_3d.text(disc_center[0], disc_center[1], disc_center[2], f'Disc {disc_idx+1}', color='black', fontsize=8)


    def draw_rope_segment(self, start_point, end_point, rope_idx):
        """绘制绳索段"""
        color = ROPE_COLORS[rope_idx % len(ROPE_COLORS)]
        self.ax_3d.plot([start_point[0], end_point[0]],
                        [start_point[1], end_point[1]],
                        [start_point[2], end_point[2]],
                        color=color, linewidth=ROPE_LINEWIDTH, alpha=0.7)


    def enhanced_point_cloud_visualization(self, detected_objects, voxel_size_mm=8.0, max_points_per_obj=3000):
        """
        改进的点云可视化，解决"一坨"问题
        - 提供更好的点云分布和可视化效果
        """
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
                            # 从 helpers 导入
                            from src.utils.helpers import voxel_downsample_numpy
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
                self.ax_3d.scatter(
                    pts_adj[:, 0], pts_adj[:, 1], pts_adj[:, 2],
                    c=color, s=1, alpha=0.7, edgecolors='none', depthshade=True
                )
                total_drawn += pts_adj.shape[0]

            except Exception as e:
                logger.debug(f"点云绘制失败: {e}")
                continue

        return total_drawn


    def draw_3d_bounding_box(self, bbox_3d, color='green', linewidth=2):
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
            self.ax_3d.plot3D(
                [corners[edge[0]][0], corners[edge[1]][0]],
                [corners[edge[0]][1], corners[edge[1]][1]],
                [corners[edge[0]][2], corners[edge[1]][2]],
                color=color, linewidth=linewidth
            )


    def draw_object_info(self, obj, color='green'):
        """在3D可视化中绘制物体信息和尺寸"""
        if obj.get('bbox_3d') is None or obj.get('position') is None:
            return

        # 绘制3D包围盒
        self.draw_3d_bounding_box(obj['bbox_3d'], color=color)

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

        # 确保 text_z 在 z_range 范围内，避免文本超出视窗
        safe_z = max(Z_RANGE[0], min(Z_RANGE[1], text_z))

        # 添加文本到3D场景
        self.ax_3d.text(text_x, text_y, safe_z, info_text, color=color, fontsize=8, ha='center', va='bottom')

    def update_display(self, current_frame_idx, max_frames, left_image, right_image, disp_color_left, disp_color_right, detected_objects_list, total_drawn_points):
        self.ax_3d.cla() # Clear current 3D axes
        self.ax_3d.set_xlabel("X (mm)")
        self.ax_3d.set_ylabel("Y (mm)")
        self.ax_3d.set_zlabel("Z (mm)")
        self.ax_3d.set_title(f"3D 机械臂运动与障碍物避障 (Frame: {current_frame_idx}/{max_frames})")
        self.ax_3d.set_xlim(X_RANGE)
        self.ax_3d.set_ylim(Y_RANGE)
        self.ax_3d.set_zlim(Z_RANGE)
        self.ax_3d.view_init(elev=20, azim=-60)
        self.ax_3d.set_box_aspect([np.ptp(X_RANGE), np.ptp(Y_RANGE), np.ptp(Z_RANGE)])

        # Draw arm (placeholder, actual arm calculation needs to be done elsewhere)
        self.draw_arm_disc(TARGET_XYZ, 0) # Example: draw disc at target XYZ

        # Draw detected objects
        self.enhanced_point_cloud_visualization(detected_objects_list)
        for obj in detected_objects_list:
            self.draw_object_info(obj)

        # Update left camera image
        if left_image is not None:
            self.ax_camera_l.clear()
            # Convert BGR to RGB for display
            if len(left_image.shape) == 3 and left_image.shape[2] == 3:
                left_image_rgb = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
            else:
                left_image_rgb = left_image
            self.ax_camera_l.imshow(left_image_rgb)
            self.ax_camera_l.set_title("左摄像头")
            self.ax_camera_l.axis('off')

        # Update right camera image
        if right_image is not None:
            self.ax_camera_r.clear()
            # Convert BGR to RGB for display
            if len(right_image.shape) == 3 and right_image.shape[2] == 3:
                right_image_rgb = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
            else:
                right_image_rgb = right_image
            self.ax_camera_r.imshow(right_image_rgb)
            self.ax_camera_r.set_title("右摄像头")
            self.ax_camera_r.axis('off')

        # Update left disparity image
        if disp_color_left is not None:
            self.ax_disparity_l.clear()
            self.ax_disparity_l.imshow(disp_color_left)
            self.ax_disparity_l.set_title("左视差图 (彩色)")
            self.ax_disparity_l.axis('off')

        # Update right disparity image
        if disp_color_right is not None:
            self.ax_disparity_r.clear()
            self.ax_disparity_r.imshow(disp_color_right)
            self.ax_disparity_r.set_title("右视差图 (彩色)")
            self.ax_disparity_r.axis('off')

        # Update info text
        info = f"Frame: {current_frame_idx}/{max_frames}\n" \
               f"Detected Objects: {len(detected_objects_list)}\n" \
               f"Total Points Drawn: {total_drawn_points}"
        self.info_text_obj.set_text(info)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
