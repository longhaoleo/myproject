"""
人脸检测对比子包。

该子包聚合了：
- 统一检测类型定义
- 各模型检测后端
- 检测器工厂
- 可视化工具
- 批处理入口
"""

from .factory import create_face_detector, supported_detector_names
from .types import FaceDetection, FaceDetector

__all__ = [
    "FaceDetection",
    "FaceDetector",
    "create_face_detector",
    "supported_detector_names",
]

