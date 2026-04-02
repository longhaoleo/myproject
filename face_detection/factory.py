"""
检测器工厂：通过字符串名称创建对应后端。
"""

from pathlib import Path

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
    return name.strip().lower().replace("_", "-")


def supported_detector_names() -> list[str]:
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
    yolov8_model_path: Path | None = None,
):
    """
    创建检测器实例。

    参数：
    - detector_name: 检测器名称
    - model_dir: 模型目录（缓存和本地权重）
    - min_confidence: 置信度阈值
    - yolov8_model_path: YOLOv8-Face（改造版）权重路径
    """
    key = _normalize_name(detector_name)

    if key == "mtcnn":
        return MTCNNDetector(model_dir=model_dir, min_confidence=min_confidence)

    if key == "retinaface":
        return RetinaFaceDetector(model_dir=model_dir, min_confidence=min_confidence)

    if key == "scrfd":
        return SCRFDDetector(model_dir=model_dir, min_confidence=min_confidence)

    if key == "blazeface":
        return BlazeFaceDetector(model_dir=model_dir, min_confidence=min_confidence)

    if key in {"mediapipe-landmarker", "face-landmarker", "landmarker"}:
        return MediaPipeLandmarkerDetector(model_dir=model_dir, min_confidence=min_confidence)

    if key in {"yolov8-face", "yolov8face", "yolov8-face-mod"}:
        if yolov8_model_path is None:
            yolov8_model_path = model_dir / "yolov8_face.pt"
        return YoloV8FaceDetector(model_path=yolov8_model_path, min_confidence=min_confidence)

    if key == "centerface":
        return CenterFaceDetector(model_dir=model_dir, min_confidence=min_confidence)

    raise ValueError(
        f"不支持的检测器: {detector_name}，可选值: {', '.join(supported_detector_names())}"
    )
