"""
3D可视化器
使用matplotlib 3D显示检测物体的点云和包围盒

坐标系说明：
- 相机坐标系 (OpenCV标准): X右, Y下, Z前(深度)
- 可视化坐标系 (matplotlib): X右, Y前, Z上
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg后端，与OpenCV兼容性更好
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Optional, Tuple
from src.utils.logger import get_logger

logger = get_logger()


def transform_camera_to_viz(points: np.ndarray, unit_mm: bool = True) -> np.ndarray:
    """
    将相机坐标系转换为可视化坐标系

    相机坐标系: X右, Y下, Z前
    可视化坐标系: X右, Y前, Z上

    转换规则:
    - X_viz = X_cam (不变)
    - Y_viz = Z_cam (深度变为前方)
    - Z_viz = -Y_cam (下方变为上方)

    Args:
        points: Nx3 numpy数组，单位可以是cm或mm
        unit_mm: True表示输入是毫米，False表示厘米

    Returns:
        Nx3 numpy数组，转换后的坐标（毫米单位用于可视化）
    """
    if points is None or len(points) == 0:
        return np.array([]).reshape(-1, 3)

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)

    # 如果输入是厘米，转换为毫米
    if not unit_mm:
        pts = pts * 10.0

    # 坐标系转换
    pts_viz = np.zeros_like(pts)
    pts_viz[:, 0] = pts[:, 0]   # X不变
    pts_viz[:, 1] = pts[:, 2]   # Z → Y (前)
    pts_viz[:, 2] = -pts[:, 1]  # -Y → Z (上)

    return pts_viz


class Viewer3D:
    """3D可视化器类"""

    def __init__(self,
                 figsize: Tuple[int, int] = (12, 9),
                 x_range: Tuple[float, float] = (-1500, 1500),
                 y_range: Tuple[float, float] = (-1500, 1500),
                 z_range: Tuple[float, float] = (-1000, 1500)):
        """
        初始化3D可视化器

        Args:
            figsize: 图形尺寸 (宽, 高)
            x_range: X轴范围 (mm)
            y_range: Y轴范围 (mm)
            z_range: Z轴范围 (mm)
        """
        # 启用交互模式
        plt.ion()

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 创建图形和3D轴
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')

        # 设置坐标系范围
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

        # 存储检测到的物体
        self.detected_objects = []

        # 初始化场景
        self._setup_scene()

        logger.info("3D可视化器初始化完成")

    def _setup_scene(self):
        """设置3D场景"""
        self.ax.set_xlim(self.x_range)
        self.ax.set_ylim(self.y_range)
        self.ax.set_zlim(self.z_range)
        self.ax.set_xlabel('X轴 (mm)', fontsize=10)
        self.ax.set_ylabel('Y轴 (mm)', fontsize=10)
        self.ax.set_zlabel('Z轴 (mm)', fontsize=10)
        self.ax.set_title('三维物体重建', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)

        # 绘制坐标轴指示器
        axis_length = 200
        self.ax.quiver(0, 0, 0, axis_length, 0, 0,
                      color='r', linewidth=2, arrow_length_ratio=0.1, label='X')
        self.ax.quiver(0, 0, 0, 0, axis_length, 0,
                      color='g', linewidth=2, arrow_length_ratio=0.1, label='Y')
        self.ax.quiver(0, 0, 0, 0, 0, axis_length,
                      color='b', linewidth=2, arrow_length_ratio=0.1, label='Z')
        self.ax.legend(loc='upper right', fontsize=8)

    def clear(self):
        """清空场景"""
        self.ax.clear()
        self._setup_scene()
        self.detected_objects = []

    def add_object(self, obj_info: Dict):
        """
        添加物体到场景

        Args:
            obj_info: 物体信息字典，包含:
                - 'center_cm': 中心位置 (cm)
                - 'dims_cm': 尺寸 (cm)
                - 'points': 点云 (可选, cm)
                - 'class_name': 类别名称
                - 'confidence': 置信度
        """
        self.detected_objects.append(obj_info)

    def draw_pointcloud(self, points_cm: np.ndarray, color: str = 'skyblue',
                       size: float = 1.0, alpha: float = 0.6):
        """
        绘制点云

        Args:
            points_cm: Nx3点云，单位厘米
            color: 点云颜色
            size: 点大小
            alpha: 透明度
        """
        if points_cm is None or len(points_cm) == 0:
            return

        # 转换坐标系并转为mm
        points_viz = transform_camera_to_viz(points_cm, unit_mm=False)

        # 过滤无效点
        valid_mask = np.isfinite(points_viz).all(axis=1)
        points_viz = points_viz[valid_mask]

        if len(points_viz) == 0:
            return

        # 下采样以提高性能
        max_points = 2000
        if len(points_viz) > max_points:
            indices = np.random.choice(len(points_viz), max_points, replace=False)
            points_viz = points_viz[indices]

        # 绘制
        self.ax.scatter(points_viz[:, 0], points_viz[:, 1], points_viz[:, 2],
                       c=color, s=size, alpha=alpha, edgecolors='none')

        logger.debug(f"绘制点云: {len(points_viz)} 个点")

    def draw_bbox_3d(self, center_cm: np.ndarray, dims_cm: np.ndarray,
                     color: str = 'lime', linewidth: float = 2.0):
        """
        绘制3D包围盒（轴对齐）

        Args:
            center_cm: 中心位置 (cm)
            dims_cm: 尺寸 [长,宽,高] (cm)
            color: 线条颜色
            linewidth: 线条宽度
        """
        if center_cm is None or dims_cm is None:
            return

        # 转换为可视化坐标系 (mm)
        center_mm = center_cm * 10.0
        dims_mm = dims_cm * 10.0

        center_viz = transform_camera_to_viz(center_mm.reshape(1, 3), unit_mm=True)[0]

        # 计算8个顶点（在可视化坐标系中）
        half_dims = dims_mm / 2.0

        # 注意：维度也需要按照坐标系转换进行调整
        # 原始: [X, Y, Z] -> 可视化: [X, Z, -Y]
        # 所以尺寸映射: dims[0]->X, dims[1]->Y, dims[2]->Z
        # 在可视化坐标系中: dims[0]->X, dims[2]->Y, dims[1]->Z
        dx = half_dims[0]
        dy = half_dims[2]  # 原Z
        dz = half_dims[1]  # 原Y

        corners = np.array([
            [-dx, -dy, -dz],
            [dx, -dy, -dz],
            [dx, dy, -dz],
            [-dx, dy, -dz],
            [-dx, -dy, dz],
            [dx, -dy, dz],
            [dx, dy, dz],
            [-dx, dy, dz]
        ]) + center_viz

        # 定义12条边
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
            [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
            [0, 4], [1, 5], [2, 6], [3, 7]   # 侧面
        ]

        # 绘制边
        for edge in edges:
            p0, p1 = corners[edge[0]], corners[edge[1]]
            self.ax.plot3D([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                          color=color, linewidth=linewidth, alpha=0.8)

    def draw_text_3d(self, position_cm: np.ndarray, text: str,
                     color: str = 'white', fontsize: int = 9):
        """
        在3D空间中绘制文本

        Args:
            position_cm: 文本位置 (cm)
            text: 文本内容
            color: 文本颜色
            fontsize: 字体大小
        """
        if position_cm is None:
            return

        # 转换坐标
        pos_viz = transform_camera_to_viz(position_cm.reshape(1, 3) * 10.0, unit_mm=True)[0]

        # 文本偏移（向上50mm）
        pos_viz[2] += 50

        self.ax.text(pos_viz[0], pos_viz[1], pos_viz[2], text,
                    fontsize=fontsize, color=color,
                    bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=2))

    def update(self):
        """更新显示所有检测到的物体"""
        # 清空画布但保留物体列表
        self.ax.clear()
        self._setup_scene()

        for obj in self.detected_objects:
            # 绘制点云
            if 'points' in obj and obj['points'] is not None:
                self.draw_pointcloud(obj['points'], color='cyan', size=2.0, alpha=0.5)

            # 绘制包围盒
            if 'center_cm' in obj and 'dims_cm' in obj:
                self.draw_bbox_3d(obj['center_cm'], obj['dims_cm'],
                                color='lime', linewidth=2.0)

                # 绘制标签
                label = obj.get('class_name', 'Object')
                if 'confidence' in obj:
                    label += f" {obj['confidence']:.2f}"
                if 'dims_cm' in obj:
                    dims = obj['dims_cm']
                    label += f"\n尺寸: {dims[0]:.1f}×{dims[1]:.1f}×{dims[2]:.1f}cm"

                self.draw_text_3d(obj['center_cm'], label, color='white')

        # 使用更安全的更新方式，避免GIL问题
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception as e:
            logger.debug(f"3D更新失败: {e}")
            # 回退到简单的draw
            try:
                plt.draw()
            except Exception:
                pass

        # 清空物体列表，为下一帧做准备
        self.detected_objects = []

    def show(self, block: bool = False):
        """
        显示窗口

        Args:
            block: 是否阻塞。False时立即返回（交互模式）
        """
        if block:
            plt.ioff()  # 关闭交互模式
            plt.show()
        else:
            # 确保窗口显示
            self.fig.show()
            # 立即处理事件
            try:
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
            except Exception:
                pass

    def close(self):
        """关闭窗口"""
        try:
            plt.close(self.fig)
            plt.ioff()  # 关闭交互模式
        except Exception as e:
            logger.debug(f"关闭3D窗口失败: {e}")
