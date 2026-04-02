"""
人脸检测公共类型定义。

本文件只放“数据结构”和“抽象接口”，不包含具体模型实现。
"""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


# 像素坐标框：(x1, y1, x2, y2)
FaceBox = tuple[int, int, int, int]

# 关键点坐标：(x, y)
LandmarkPoint = tuple[int, int]


@dataclass
class FaceDetection:
    """单张人脸检测结果。"""

    box: FaceBox
    score: float | None = None
    # 统一关键点语义名称：left_eye / right_eye / nose / mouth_left / mouth_right
    landmarks: dict[str, LandmarkPoint] = field(default_factory=dict)


class FaceDetector(Protocol):
    """
    统一检测器接口协议。

    约定：
    - 输入为 OpenCV BGR 图像；
    - 输出为 FaceDetection 列表；
    - close() 可选实现，用于释放模型资源。
    """

    detector_name: str

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        ...

    def close(self) -> None:
        ...

