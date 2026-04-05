"""
集中式配置：尽量让三条流程共用一套基础设置。

说明：
- 这里放“跨流程通用”的默认值；
- 各流程仍可在自身文件里按需覆盖。
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectPaths:
    input_root: Path
    model_dir: Path
    output_root_detect: Path
    output_root_mask: Path
    output_root_sam: Path
    preview_root_mask: Path


def default_paths() -> ProjectPaths:
    """
    返回默认路径配置。

    支持环境变量覆盖（便于 notebook 或临时实验）：
    - FD_INPUT_ROOT
    - FD_MODEL_DIR
    - FD_OUTPUT_ROOT_DETECT
    - FD_OUTPUT_ROOT_MASK
    - FD_OUTPUT_ROOT_SAM
    - FD_PREVIEW_ROOT_MASK
    """
    return ProjectPaths(
        input_root=Path(os.getenv("FD_INPUT_ROOT", "~/datasets/deformity")).expanduser(),
        model_dir=Path(os.getenv("FD_MODEL_DIR", "model")).expanduser(),
        output_root_detect=Path(
            os.getenv("FD_OUTPUT_ROOT_DETECT", "~/datasets/deformity_face_detection_preview")
        ).expanduser(),
        output_root_mask=Path(
            os.getenv("FD_OUTPUT_ROOT_MASK", "~/datasets/deformity_masked")
        ).expanduser(),
        output_root_sam=Path(
            os.getenv("FD_OUTPUT_ROOT_SAM", "~/datasets/deformity_sam_mask")
        ).expanduser(),
        preview_root_mask=Path(
            os.getenv("FD_PREVIEW_ROOT_MASK", "~/datasets/deformity_detection_preview")
        ).expanduser(),
    )


def default_min_confidence_map() -> dict[str, float]:
    """不同检测器的默认置信度阈值。"""
    return {
        # 数据已知全是人脸，阈值适当降低以减少漏检
        "mtcnn": 0.6,
        "retinaface": 0.5,
        "scrfd": 0.5,
        "blazeface": 0.4,
        "mediapipe-landmarker": 0.25,
        "yolov8-face": 0.25,
        # CenterFace 容易出大量低质量框，阈值保持稍高
        "centerface": 0.7,
    }


def default_detector_options(model_dir: Path) -> dict[str, dict[str, Any]]:
    """不同检测器的默认参数（通用扩展入口）。"""
    return {
        "scrfd": {"det_size": (640, 640)},
        "yolov8-face": {
            "model_path": str(model_dir / "yolov8-face" / "yolov8_face.pt"),
            "keypoint_confidence": 0.2,
        },
    }


def default_part_scale_by_view() -> dict[str, dict[str, tuple[float, float]]]:
    """按视角缩放部位框（ratio）。"""
    return {
        "1": {"nose": (1.5, 1.0), "mouth": (1.2, 1.0)},
        "2": {"nose": (1.7, 1.0), "mouth": (1.5, 1.2)},
        "3": {"nose": (1.7, 1.0), "mouth": (1.5, 1.2)},
        "4": {"nose": (2.0, 1.0), "mouth": (4.0, 1.2), "left_eye": (0.85, 0.85), "right_eye": (0.85, 0.85)},
        "5": {"nose": (2.0, 1.0), "mouth": (4.0, 1.2), "left_eye": (0.85, 0.85), "right_eye": (0.85, 0.85)},
        "6": {"nose": (1.5, 1.5), "mouth": (1.5, 2.0), "left_eye": (0.85, 0.85), "right_eye": (0.85, 0.85)},
    }


def default_part_offset_by_view() -> dict[str, dict[str, tuple[float, float]]]:
    """按视角偏移部位（ratio，基于人脸框宽高）。"""
    return {
        "1": {},
        "2": {"nose": (0.03, 0.0)},
        "3": {"nose": (-0.03, 0.0)},
        "4": {"nose": (0.03, 0.0)},
        "5": {"nose": (-0.03, 0.0)},
        "6": {},
    }
