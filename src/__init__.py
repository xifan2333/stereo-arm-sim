"""
Stereo Vision Robot Arm - 双目视觉机械臂避障系统
====================================================

A modular system for robot arm obstacle avoidance using stereo vision.

模块说明:
    - vision: 双目视觉和深度感知
    - detection: 物体检测和分割
    - utils: 通用工具函数

遵循小步骤迭代原则，其他模块将逐步添加
"""

__version__ = '0.1.0'
__author__ = 'xifan'

from . import vision
from . import detection
from . import utils

__all__ = [
    'vision',
    'detection',
    'utils',
]
