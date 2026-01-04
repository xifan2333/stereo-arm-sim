"""
日志模块 - 基于 loguru 的统一日志系统

遵循小步骤迭代原则，仅提供基础必要功能
"""

import sys
from pathlib import Path
from loguru import logger
from src.utils.config import config

# 全局标志，确保只初始化一次
_logger_initialized = False


def setup_logger() -> None:
    """初始化日志系统"""
    global _logger_initialized

    if _logger_initialized:
        logger.warning("日志系统已经初始化，跳过重复初始化")
        return

    # 从配置管理器获取配置
    log_config = config.logging

    # 移除默认的 handler
    logger.remove()

    # 配置控制台输出
    console_config = log_config.get("console", {})
    if console_config.get("enabled", True):
        logger.add(
            sys.stderr,
            format=log_config.get("format"),
            level=log_config.get("level", "INFO"),
            colorize=console_config.get("colorize", True),
        )

    # 配置文件输出
    file_config = log_config.get("file", {})
    if file_config.get("enabled", True):
        log_path = Path(file_config.get("path", "data/logs/stereo-arm-sim.log"))
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            str(log_path),
            format=log_config.get("format"),
            level=log_config.get("level", "INFO"),
            rotation=file_config.get("rotation", "100 MB"),
            retention=file_config.get("retention", "7 days"),
            compression=file_config.get("compression", "zip"),
            encoding=file_config.get("encoding", "utf-8"),
        )

    # 配置错误日志单独输出
    error_config = log_config.get("error_file", {})
    if error_config.get("enabled", True):
        error_path = Path(error_config.get("path", "data/logs/error.log"))
        error_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            str(error_path),
            format=log_config.get("format"),
            level=error_config.get("level", "ERROR"),
            rotation=error_config.get("rotation", "50 MB"),
            retention=error_config.get("retention", "30 days"),
            compression=error_config.get("compression", "zip"),
            encoding=error_config.get("encoding", "utf-8"),
        )

    _logger_initialized = True
    logger.success("日志系统初始化成功")


def get_logger():
    """
    获取 logger 实例

    Returns:
        loguru.Logger: logger 实例
    """
    if not _logger_initialized:
        logger.warning("日志系统尚未初始化，自动调用 setup_logger()")
        setup_logger()

    return logger
