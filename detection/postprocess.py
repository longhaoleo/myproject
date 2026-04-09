"""
检测结果后处理。

统一在这里做：
- 结果排序；
- 单图单脸约束；
- 多脸时按分数自适应抬高阈值。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .types import FaceDetection


def _box_area(det: FaceDetection) -> int:
    """返回检测框面积，用于同分时稳定排序。"""
    x1, y1, x2, y2 = det.box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _score_value(det: FaceDetection) -> float:
    """把 score 归一成可排序数值。"""
    if det.score is None:
        return float("-inf")
    try:
        return float(det.score)
    except Exception:
        return float("-inf")


def sort_detections(detections: list[FaceDetection]) -> list[FaceDetection]:
    """按分数优先、面积次之排序。"""
    return sorted(
        detections,
        key=lambda det: (_score_value(det), _box_area(det)),
        reverse=True,
    )


def enforce_single_face(
    detections: list[FaceDetection],
    min_confidence: float,
    adaptive_threshold_step: float = 0.02,
) -> list[FaceDetection]:
    """
    限定每张图最多只保留一张脸。

    处理策略：
    - 先按 score/面积排序；
    - 若多脸且带 score，则逐步抬高阈值，尽量只留下最高分人脸；
    - 若仍无法唯一化（例如并列高分或无 score），退回到排序后的首张脸。
    """
    ordered = sort_detections(detections)
    if len(ordered) <= 1:
        return ordered

    step = max(1e-3, float(adaptive_threshold_step))
    scored = [det for det in ordered if det.score is not None]
    if len(scored) >= 2:
        top_score = _score_value(scored[0])
        second_score = _score_value(scored[1])
        adaptive_threshold = max(float(min_confidence), second_score + step)
        adaptive_threshold = min(adaptive_threshold, top_score)

        filtered = [
            det
            for det in ordered
            if det.score is not None and _score_value(det) >= adaptive_threshold
        ]
        while len(filtered) > 1 and adaptive_threshold < top_score:
            adaptive_threshold = min(top_score, adaptive_threshold + step)
            filtered = [
                det
                for det in ordered
                if det.score is not None and _score_value(det) >= adaptive_threshold
            ]

        if len(filtered) == 1:
            return filtered
        if filtered:
            return [filtered[0]]

    return [ordered[0]]


@dataclass
class SingleFaceAdaptiveDetector:
    """给任意 detector 增加单脸约束。"""

    base_detector: Any
    min_confidence: float
    adaptive_threshold_step: float = 0.02
    detector_name: str = field(init=False)

    def __post_init__(self) -> None:
        self.detector_name = str(getattr(self.base_detector, "detector_name", "detector"))

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        """执行底层检测，再统一做单脸后处理。"""
        detections = self.base_detector.detect(image)
        return enforce_single_face(
            detections=detections,
            min_confidence=self.min_confidence,
            adaptive_threshold_step=self.adaptive_threshold_step,
        )

    def close(self) -> None:
        """透传底层 close。"""
        close_fn = getattr(self.base_detector, "close", None)
        if callable(close_fn):
            close_fn()
