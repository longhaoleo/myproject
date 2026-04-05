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


@dataclass(frozen=True)
class ViewTweaks:
    part_scale_by_view: dict[str, dict[str, tuple[float, float]]]
    part_offset_by_view: dict[str, dict[str, tuple[float, float]]]
    part_offset_mode: str = "ratio"

def _normalize_pair(value: object) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"非法的二维配置值: {value!r}")
    return float(value[0]), float(value[1])


def _normalize_part_map(raw: object) -> dict[str, tuple[float, float]]:
    if not isinstance(raw, dict):
        return {}
    return {str(name): _normalize_pair(value) for name, value in raw.items()}


def _normalize_views(raw_views: object) -> tuple[
    dict[str, dict[str, tuple[float, float]]],
    dict[str, dict[str, tuple[float, float]]],
]:
    views = raw_views if isinstance(raw_views, dict) else {}
    part_scale_by_view: dict[str, dict[str, tuple[float, float]]] = {}
    part_offset_by_view: dict[str, dict[str, tuple[float, float]]] = {}
    for view_id, item in views.items():
        if not isinstance(item, dict):
            continue
        part_scale_by_view[str(view_id)] = _normalize_part_map(item.get("scale", {}))
        part_offset_by_view[str(view_id)] = _normalize_part_map(item.get("offset", {}))
    return part_scale_by_view, part_offset_by_view


def _merge_nested_pairs(
    base: dict[str, dict[str, tuple[float, float]]],
    override: dict[str, dict[str, tuple[float, float]]],
) -> dict[str, dict[str, tuple[float, float]]]:
    merged = {str(view_id): dict(part_map) for view_id, part_map in base.items()}
    for view_id, part_map in override.items():
        target = merged.setdefault(str(view_id), {})
        target.update(part_map)
    return merged


def load_view_tweaks(method_name: str = "") -> ViewTweaks:
    """读取按方法叠加后的视角微调配置。"""
    config = VIEW_TWEAKS_CONFIG
    default_cfg = config.get("default", {})
    method_cfg = config.get(str(method_name).strip(), {}) if method_name else {}

    base_scale_by_view, base_offset_by_view = _normalize_views(
        default_cfg.get("views", {}) if isinstance(default_cfg, dict) else {}
    )
    method_scale_by_view, method_offset_by_view = _normalize_views(
        method_cfg.get("views", {}) if isinstance(method_cfg, dict) else {}
    )

    part_offset_mode = "ratio"
    if isinstance(default_cfg, dict) and default_cfg.get("part_offset_mode"):
        part_offset_mode = str(default_cfg.get("part_offset_mode")).strip().lower() or part_offset_mode
    if isinstance(method_cfg, dict) and method_cfg.get("part_offset_mode"):
        part_offset_mode = str(method_cfg.get("part_offset_mode")).strip().lower() or part_offset_mode

    return ViewTweaks(
        part_scale_by_view=_merge_nested_pairs(base_scale_by_view, method_scale_by_view),
        part_offset_by_view=_merge_nested_pairs(base_offset_by_view, method_offset_by_view),
        part_offset_mode=part_offset_mode,
    )


def canonical_part_name(name: str) -> str:
    key = str(name).strip()
    if key in {"mouth_left", "mouth_right"}:
        return "mouth"
    return key


def get_view_scale_map(
    view_id: str | None,
    part_scale_by_view: dict[str, dict[str, tuple[float, float]]] | None,
) -> dict[str, tuple[float, float]]:
    if not view_id or not part_scale_by_view:
        return {}
    return dict(part_scale_by_view.get(str(view_id), {}))


def get_view_offset_map(
    view_id: str | None,
    part_offset_by_view: dict[str, dict[str, tuple[float, float]]] | None,
) -> dict[str, tuple[float, float]]:
    if not view_id or not part_offset_by_view:
        return {}
    return dict(part_offset_by_view.get(str(view_id), {}))


def resolve_part_offset(
    part_name: str,
    offset_map: dict[str, tuple[float, float]] | None,
    face_w: int,
    face_h: int,
    part_offset_mode: str = "ratio",
) -> tuple[int, int]:
    if not offset_map:
        return 0, 0
    ox, oy = offset_map.get(canonical_part_name(part_name), (0.0, 0.0))
    if str(part_offset_mode).strip().lower() == "ratio":
        return int(round(float(ox) * face_w)), int(round(float(oy) * face_h))
    return int(round(float(ox))), int(round(float(oy)))


def adjust_box_for_view(
    box: tuple[int, int, int, int],
    part_name: str,
    image_w: int,
    image_h: int,
    scale_map: dict[str, tuple[float, float]] | None,
    offset_map: dict[str, tuple[float, float]] | None,
    part_offset_mode: str = "ratio",
) -> tuple[int, int, int, int]:
    """按视角规则调整单个框。"""
    x1, y1, x2, y2 = box
    face_w = max(1, x2 - x1)
    face_h = max(1, y2 - y1)
    sx, sy = (scale_map or {}).get(canonical_part_name(part_name), (1.0, 1.0))
    dx, dy = resolve_part_offset(part_name, offset_map, face_w, face_h, part_offset_mode)
    cx = int(round((x1 + x2) / 2 + dx))
    cy = int(round((y1 + y2) / 2 + dy))
    bw = max(1, int(round(face_w * float(sx))))
    bh = max(1, int(round(face_h * float(sy))))
    nx1 = max(0, cx - bw // 2)
    ny1 = max(0, cy - bh // 2)
    nx2 = min(image_w, nx1 + bw)
    ny2 = min(image_h, ny1 + bh)
    nx1 = max(0, nx2 - bw)
    ny1 = max(0, ny2 - bh)
    return int(nx1), int(ny1), int(nx2), int(ny2)


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


def default_part_scale_by_view(method_name: str = "") -> dict[str, dict[str, tuple[float, float]]]:
    """按视角缩放部位框（直接从本文件配置读取，可按方法覆盖）。"""
    return load_view_tweaks(method_name=method_name).part_scale_by_view


def default_part_offset_by_view(method_name: str = "") -> dict[str, dict[str, tuple[float, float]]]:
    """按视角偏移部位（直接从本文件配置读取，可按方法覆盖）。"""
    return load_view_tweaks(method_name=method_name).part_offset_by_view


def default_part_offset_mode(method_name: str = "") -> str:
    """按视角偏移模式（ratio/pixel，可按方法覆盖）。"""
    return load_view_tweaks(method_name=method_name).part_offset_mode


# ---------------------------------------------------------------------------
# View Tweaks Config
# ---------------------------------------------------------------------------
# 这块是“纯配置区”：
# - 顶层按方法分
# - 方法下面按视角分
# - default 是所有方法共享的默认值
# - detect_compare / eye_mask / sam_mask 按需覆盖
VIEW_TWEAKS_CONFIG: dict[str, dict[str, Any]] = {
    "default": {
        "part_offset_mode": "ratio",
        "views": {
            "1": {
                "scale": {
                    "nose": (1.5, 1.0),
                    "mouth": (1.2, 1.0),
                },
            },
            "2": {
                "scale": {
                    "nose": (1.7, 1.0),
                    "mouth": (1.5, 1.2),
                },
                "offset": {
                    "nose": (0.03, 0.0),
                },
            },
            "3": {
                "scale": {
                    "nose": (1.7, 1.0),
                    "mouth": (1.5, 1.2),
                },
                "offset": {
                    "nose": (-0.03, 0.0),
                },
            },
            "4": {
                "scale": {
                    "nose": (2.0, 1.0),
                    "mouth": (4.0, 1.2),
                    "left_eye": (0.85, 0.85),
                    "right_eye": (0.85, 0.85),
                },
                "offset": {
                    "nose": (0.03, 0.0),
                },
            },
            "5": {
                "scale": {
                    "nose": (2.0, 1.0),
                    "mouth": (4.0, 1.2),
                    "left_eye": (0.85, 0.85),
                    "right_eye": (0.85, 0.85),
                },
                "offset": {
                    "nose": (-0.03, 0.0),
                },
            },
            "6": {
                "scale": {
                    "nose": (1.5, 1.5),
                    "mouth": (1.5, 2.0),
                    "left_eye": (0.85, 0.85),
                    "right_eye": (0.85, 0.85),
                },
            },
        },
    },
    "detect_compare": {
        "views": {
        },
    },
    "eye_mask": {
        "views": {
        },
    },
    "sam_mask": {
        "views": {
        },
    },
}
