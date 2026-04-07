"""
集中式配置：尽量让三条流程共用一套基础设置。

说明：
- 这里放“跨流程通用”的默认值；
- 各流程仍可在自身文件里按需覆盖。

配置分层：
- default：所有检测器共享默认参数；
- <detector>：按检测器覆盖；
- scale/offset：先分缩放/偏移，再按视角（1~6 / __all__）展开。
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


def _normalize_view_map(raw_view_map: object) -> dict[str, dict[str, tuple[float, float]]]:
    """把 {view_id: {part: (x,y)}} 规范化。"""
    view_map = raw_view_map if isinstance(raw_view_map, dict) else {}
    normalized: dict[str, dict[str, tuple[float, float]]] = {}
    for view_id, part_map in view_map.items():
        normalized[str(view_id)] = _normalize_part_map(part_map)
    return normalized


def _normalize_profile_tweaks(raw_cfg: object) -> tuple[
    dict[str, dict[str, tuple[float, float]]],
    dict[str, dict[str, tuple[float, float]]],
]:
    """
    解析 profile 的 scale/offset 配置。

    新格式（推荐）：
    - scale: {view_id: {part: (sx, sy)}}
    - offset: {view_id: {part: (dx, dy)}}

    兼容旧格式（仅为平滑迁移）：
    - views: {view_id: {scale: {...}, offset: {...}}}

    返回值：
    - part_scale_by_view: {view_id: {part_name: (sx, sy)}}
    - part_offset_by_view: {view_id: {part_name: (dx, dy)}}
    """
    cfg = raw_cfg if isinstance(raw_cfg, dict) else {}
    if isinstance(cfg.get("scale"), dict) or isinstance(cfg.get("offset"), dict):
        return _normalize_view_map(cfg.get("scale", {})), _normalize_view_map(cfg.get("offset", {}))

    # 旧格式兜底
    views = cfg.get("views", {}) if isinstance(cfg, dict) else {}
    part_scale_by_view: dict[str, dict[str, tuple[float, float]]] = {}
    part_offset_by_view: dict[str, dict[str, tuple[float, float]]] = {}
    if isinstance(views, dict):
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


def normalize_profile_name(profile_name: str) -> str:
    """标准化配置 profile 名称（主要用于检测器别名统一）。"""
    key = str(profile_name).strip().lower().replace("_", "-")
    alias = {
        "face-landmarker": "mediapipe-landmarker",
        "landmarker": "mediapipe-landmarker",
        "yolov8face": "yolov8-face",
    }
    return alias.get(key, key)


def load_view_tweaks(profile_name: str = "") -> ViewTweaks:
    """
    读取按检测器 profile 叠加后的视角微调配置。

    叠加规则：
    1. 先读 default；
    2. 再用 profile 覆盖同名配置；
    3. 最终由 get_view_scale_map/get_view_offset_map 在运行时处理 __all__ 与具体视角。
    """
    config = VIEW_TWEAKS_CONFIG
    default_cfg = config.get("default", {})
    profile_key = normalize_profile_name(profile_name) if profile_name else ""
    profile_cfg = config.get(profile_key, {}) if profile_key else {}

    base_scale_by_view, base_offset_by_view = _normalize_profile_tweaks(default_cfg)
    profile_scale_by_view, profile_offset_by_view = _normalize_profile_tweaks(profile_cfg)

    part_offset_mode = "ratio"
    if isinstance(default_cfg, dict) and default_cfg.get("part_offset_mode"):
        part_offset_mode = str(default_cfg.get("part_offset_mode")).strip().lower() or part_offset_mode
    if isinstance(profile_cfg, dict) and profile_cfg.get("part_offset_mode"):
        part_offset_mode = str(profile_cfg.get("part_offset_mode")).strip().lower() or part_offset_mode

    return ViewTweaks(
        part_scale_by_view=_merge_nested_pairs(base_scale_by_view, profile_scale_by_view),
        part_offset_by_view=_merge_nested_pairs(base_offset_by_view, profile_offset_by_view),
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
    """返回当前视角生效的 scale 配置（__all__ + view 覆盖）。"""
    if not part_scale_by_view:
        return {}
    # __all__ 是全视角默认值，具体视角会覆盖同名配置。
    merged = dict(part_scale_by_view.get("__all__", {}))
    if not view_id:
        return merged
    merged.update(part_scale_by_view.get(str(view_id), {}))
    return merged


def get_view_offset_map(
    view_id: str | None,
    part_offset_by_view: dict[str, dict[str, tuple[float, float]]] | None,
) -> dict[str, tuple[float, float]]:
    """返回当前视角生效的 offset 配置（__all__ + view 覆盖）。"""
    if not part_offset_by_view:
        return {}
    merged = dict(part_offset_by_view.get("__all__", {}))
    if not view_id:
        return merged
    merged.update(part_offset_by_view.get(str(view_id), {}))
    return merged


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


def apply_landmark_offsets(
    detections: list,
    view_id: str | None,
    part_offset_by_view: dict[str, dict[str, tuple[float, float]]] | None,
    part_offset_mode: str = "ratio",
) -> None:
    """
    把视角偏移直接写回检测结果的 landmarks（原地修改）。

    说明：
    - 只在 landmarks 存在时生效；
    - 偏移按人脸框尺寸换算（ratio 模式）；
    - 设计用途：修正关键点定位，使后续部位框/SAM 提示基于修正后的点位。
    """
    if not detections:
        return
    offset_map = get_view_offset_map(view_id, part_offset_by_view)
    if not offset_map:
        return
    offset_mode = str(part_offset_mode).strip().lower()
    for det in detections:
        if not getattr(det, "landmarks", None):
            continue
        x1, y1, x2, y2 = det.box
        face_w = max(1, x2 - x1)
        face_h = max(1, y2 - y1)
        new_landmarks = {}
        for name, (px, py) in det.landmarks.items():
            dx, dy = resolve_part_offset(name, offset_map, face_w, face_h, offset_mode)
            new_landmarks[name] = (int(px + dx), int(py + dy))
        det.landmarks = new_landmarks


def adjust_box_for_view(
    box: tuple[int, int, int, int],
    part_name: str,
    image_w: int,
    image_h: int,
    scale_map: dict[str, tuple[float, float]] | None,
    offset_map: dict[str, tuple[float, float]] | None,
    part_offset_mode: str = "ratio",
) -> tuple[int, int, int, int]:
    """
    按视角规则调整单个框（默认输出方形框）。

    规则：
    - scale/offset 从 view 配置中读取；
    - 输出强制方形，用于统一可视化与后续分割提示。
    """
    x1, y1, x2, y2 = box
    face_w = max(1, x2 - x1)
    face_h = max(1, y2 - y1)
    canonical_name = canonical_part_name(part_name)
    # 缩放参数统一从 VIEW_TWEAKS_CONFIG 读取，这里不做 face 放大硬编码。
    default_scale = (1.0, 1.0)
    sx, sy = (scale_map or {}).get(canonical_name, default_scale)
    dx, dy = resolve_part_offset(part_name, offset_map, face_w, face_h, part_offset_mode)
    cx = int(round((x1 + x2) / 2 + dx))
    cy = int(round((y1 + y2) / 2 + dy))
    side = max(face_w * float(sx), face_h * float(sy))
    bw = max(1, int(round(side)))
    bh = bw
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


def default_part_scale_by_view(profile_name: str = "") -> dict[str, dict[str, tuple[float, float]]]:
    """按视角缩放部位框（按检测器 profile 覆盖）。"""
    return load_view_tweaks(profile_name=profile_name).part_scale_by_view


def default_part_offset_by_view(profile_name: str = "") -> dict[str, dict[str, tuple[float, float]]]:
    """按视角偏移部位（按检测器 profile 覆盖）。"""
    return load_view_tweaks(profile_name=profile_name).part_offset_by_view


def default_part_offset_mode(profile_name: str = "") -> str:
    """按视角偏移模式（按检测器 profile 覆盖）。"""
    return load_view_tweaks(profile_name=profile_name).part_offset_mode


# ---------------------------------------------------------------------------
# View Tweaks Config
# ---------------------------------------------------------------------------
# 这块是“纯配置区”：
# - 顶层按检测器 profile 分
# - profile 内先分 scale/offset，再按视角分
# - default 是所有检测器共享的默认值
# - mtcnn / scrfd / yolov8-face ... 可按需覆盖
VIEW_TWEAKS_CONFIG: dict[str, dict[str, Any]] = {
    # 默认参数根据yolo调整的
    "default": {
        "part_offset_mode": "ratio",
        "scale": {
            "__all__": {
                # 全局人脸框缩放（人脸框默认就是方形）
                "face": (1.2, 1.2),
            },
            "1": {
                "nose": (1.5, 1.0),
                "mouth": (1.2, 1.0),
            },
            "2": {
                "nose": (1.7, 1.0),
                "mouth": (2.2, 1.2),
            },
            "3": {
                "nose": (1.7, 1.0),
                "mouth": (2.2, 1.2),
            },
            "4": {
                "nose": (2.0, 1.0),
                "mouth": (4.0, 1.2),
                "left_eye": (0.85, 0.85),
                "right_eye": (0.85, 0.85),
            },
            "5": {
                "nose": (2.0, 1.0),
                "mouth": (4.0, 1.2),
                "left_eye": (0.85, 0.85),
                "right_eye": (0.85, 0.85),
            },
            "6": {
                "nose": (1.5, 1.5),
                "mouth": (1.5, 2.0),
                "left_eye": (0.85, 0.85),
                "right_eye": (0.85, 0.85),
            },
        },
        "offset": {
            # offset 里 y 方向向下是正值，向上是负值。x 方向同理：向右正，向左负。
            "2": {
                "nose": (0.10, 0.0),
            },
            "3": {
                "nose": (-0.10, 0.0),
            },
            "4": {
                "nose": (0.10, 0.0),
            },
            "5": {
                "nose": (-0.10, 0.0),
            },
            "6": {
                "nose": (0.0, 0.05),
            }
        },
    },
    "mtcnn": {
        "scale": {},
        "offset": {},
    },
    "retinaface": {
        "scale": {},
        "offset": {},
    },
    "scrfd": {
        "scale": {},
        "offset": {},
    },
    "blazeface": {
        "scale": {},
        "offset": {},
    },
    "yolov8-face": {
        "scale": {},
        "offset": {},
    },
    "centerface": {
        "scale": {},
        "offset": {},
    },
}
