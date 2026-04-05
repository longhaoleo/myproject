"""
检测器工厂：通过字符串名称创建对应后端。
"""

from pathlib import Path
from typing import Any

from .backends import (
    BlazeFaceDetector,
    CenterFaceDetector,
    MediaPipeLandmarkerDetector,
    MTCNNDetector,
    RetinaFaceDetector,
    SCRFDDetector,
    YoloV8FaceDetector,
)


def _normalize_name(name: str) -> str:
    """标准化 detector 名称，便于匹配别名。"""
    # 统一命名：下划线与大小写都归一，便于命中别名。
    return name.strip().lower().replace("_", "-")


def supported_detector_names() -> list[str]:
    """返回支持的检测器名称列表（含别名）。"""
    # 对外展示所有支持的 detector 名称（含别名）。
    return [
        "mtcnn",
        "retinaface",
        "scrfd",
        "blazeface",
        "mediapipe-landmarker",
        "face-landmarker",
        "yolov8-face",
        "yolov8face",
        "centerface",
    ]


def create_face_detector(
    detector_name: str,
    model_dir: Path,
    min_confidence: float = 0.5,
    detector_options: dict[str, Any] | None = None,
):
    """创建检测器实例（统一入口）。"""
    # 统一归一化名称后再做分发。
    key = _normalize_name(detector_name)
    options = detector_options or {}

    if key == "mtcnn":
        return MTCNNDetector(model_dir=model_dir, min_confidence=min_confidence)

    if key == "retinaface":
        return RetinaFaceDetector(model_dir=model_dir, min_confidence=min_confidence)

    if key == "scrfd":
        # SCRFD 常用可调项：输入尺寸 det_size。
        det_size = options.get("det_size", (640, 640))
        if (
            not isinstance(det_size, (list, tuple))
            or len(det_size) != 2
        ):
            raise ValueError(f"scrfd.det_size 参数非法: {det_size}")
        return SCRFDDetector(
            model_dir=model_dir,
            min_confidence=min_confidence,
            det_size=(int(det_size[0]), int(det_size[1])),
        )

    if key == "blazeface":
        return BlazeFaceDetector(model_dir=model_dir, min_confidence=min_confidence)

    if key in {"mediapipe-landmarker", "face-landmarker", "landmarker"}:
        return MediaPipeLandmarkerDetector(model_dir=model_dir, min_confidence=min_confidence)

    if key in {"yolov8-face", "yolov8face", "yolov8-face-mod"}:
        # YOLOv8-Face 可调项：模型路径和关键点阈值。
        model_path_opt = options.get("model_path")
        model_path = (
            Path(model_path_opt)
            if model_path_opt is not None
            else model_dir / "yolov8-face" / "yolov8_face.pt"
        )
        keypoint_confidence = float(options.get("keypoint_confidence", 0.2))
        return YoloV8FaceDetector(
            model_path=model_path,
            min_confidence=min_confidence,
            keypoint_confidence=keypoint_confidence,
        )

    if key == "centerface":
        return CenterFaceDetector(model_dir=model_dir, min_confidence=min_confidence)

    raise ValueError(
        f"不支持的检测器: {detector_name}，可选值: {', '.join(supported_detector_names())}"
    )
