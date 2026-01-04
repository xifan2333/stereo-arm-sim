"""
Stereo Vision Robot Arm - 双目视觉机械臂避障系统
====================================================

A modular system for robot arm obstacle avoidance using stereo vision.

模块说明:
    - vision: 双目视觉和深度感知
    - detection: 目标检测和跟踪
    - perception: 3D感知和点云处理
    - planning: 路径规划和避障
    - control: 机械臂控制
    - visualization: 可视化工具
    - utils: 通用工具函数
"""

__version__ = '0.1.0'
__author__ = 'xifan'

from . import vision
from . import detection
from . import perception
from . import planning
from . import control
from . import visualization
from . import utils

__all__ = [
    'vision',
    'detection',
    'perception',
    'planning',
    'control',
    'visualization',
    'utils',
]
