"""
配置管理模块
统一管理全局配置，避免重复读取配置文件

遵循小步骤迭代原则，仅提供基础必要功能
"""

from pathlib import Path
from typing import Any, Optional
import yaml


class Config:
    """配置管理类（单例模式）"""

    _instance: Optional["Config"] = None
    _config: Optional[dict] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化配置"""
        if self._config is None:
            self._load_config()

    def _load_config(self) -> None:
        """加载配置文件"""
        # 配置文件路径（项目根目录的 config.yaml）
        config_path = Path(__file__).parent.parent.parent / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项

        Args:
            key: 配置键，支持点号分隔的多级键，如 "camera.width"
            default: 默认值

        Returns:
            配置值
        """
        if self._config is None:
            return default

        # 支持多级键访问，如 "camera.width"
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_section(self, section: str) -> dict:
        """
        获取配置节

        Args:
            section: 配置节名称，如 "camera"

        Returns:
            配置节字典
        """
        return self._config.get(section, {})

    @property
    def camera(self) -> dict:
        """获取摄像头配置"""
        return self.get_section("camera")

    @property
    def logging(self) -> dict:
        """获取日志配置"""
        return self.get_section("logging")


# 全局配置实例
config = Config()
