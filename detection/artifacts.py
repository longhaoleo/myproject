"""
检测侧共享的 mask 产物构建工具。

用于把分部位 mask 直接整理成 generation 侧需要的标准化产物：
- 方形 face mask
- 连续鼻嘴 inpaint mask
- feather mask
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ArtifactConfig:
    use_box_mask: bool = True
    part_box_expand_px: int = 18
    face_box_expand_px: int = 48
    face_box_force_square: bool = True
    face_clip_expand_px: int = 0
    feather_blur_px: int = 31


def mask_to_u8(mask: np.ndarray) -> np.ndarray:
    return mask.astype(np.uint8) * 255


def blur_mask_u8(mask: np.ndarray, blur_px: int) -> np.ndarray:
    if blur_px <= 0:
        return mask_to_u8(mask)
    kernel = max(1, int(blur_px))
    if kernel % 2 == 0:
        kernel += 1
    return cv2.GaussianBlur(mask_to_u8(mask), (kernel, kernel), 0)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def bbox_from_bool_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    return bbox_from_mask(mask.astype(np.uint8))


def clip_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def expand_box(
    box: tuple[int, int, int, int] | None,
    width: int,
    height: int,
    expand_px: int,
) -> tuple[int, int, int, int] | None:
    if box is None:
        return None
    return clip_box(
        box[0] - expand_px,
        box[1] - expand_px,
        box[2] + expand_px,
        box[3] + expand_px,
        width,
        height,
    )


def square_expand_box(
    box: tuple[int, int, int, int] | None,
    width: int,
    height: int,
    expand_px: int,
) -> tuple[int, int, int, int] | None:
    if box is None:
        return None
    x1, y1, x2, y2 = box
    side = max(1, x2 - x1, y2 - y1) + max(0, expand_px) * 2
    side = min(side, width, height)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    x1_new = int(round(cx - side / 2.0))
    y1_new = int(round(cy - side / 2.0))
    x1_new = max(0, min(width - side, x1_new))
    y1_new = max(0, min(height - side, y1_new))
    return int(x1_new), int(y1_new), int(x1_new + side), int(y1_new + side)


def box_mask(box: tuple[int, int, int, int] | None, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if box is None:
        return mask
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = True
    return mask


def build_continuous_edit_mask(
    nose_mask: np.ndarray,
    mouth_mask: np.ndarray,
    face_mask: np.ndarray,
) -> np.ndarray:
    height, width = nose_mask.shape[:2]
    edit_mask = nose_mask.astype(bool) | mouth_mask.astype(bool)
    nose_box = bbox_from_bool_mask(nose_mask)
    mouth_box = bbox_from_bool_mask(mouth_mask)

    if nose_box is not None and mouth_box is not None:
        nose_cy = (nose_box[1] + nose_box[3]) / 2.0
        mouth_cy = (mouth_box[1] + mouth_box[3]) / 2.0
        upper_box, lower_box = (nose_box, mouth_box) if nose_cy <= mouth_cy else (mouth_box, nose_box)
        bridge = clip_box(
            min(upper_box[0], lower_box[0]),
            upper_box[3],
            max(upper_box[2], lower_box[2]),
            lower_box[1],
            width,
            height,
        )
        if bridge is not None:
            edit_mask |= box_mask(bridge, height, width)

    return edit_mask & face_mask.astype(bool)


def standardize_part_masks(
    part_masks: dict[str, np.ndarray],
    height: int,
    width: int,
    config: ArtifactConfig,
) -> dict[str, np.ndarray]:
    standardized: dict[str, np.ndarray] = {}
    for part_name, raw_mask in part_masks.items():
        raw_box = bbox_from_mask(raw_mask)
        if part_name == "face" and config.use_box_mask:
            if config.face_box_force_square:
                box = square_expand_box(raw_box, width, height, config.face_box_expand_px)
            else:
                box = expand_box(raw_box, width, height, config.face_box_expand_px)
        else:
            expand_px = config.part_box_expand_px if config.use_box_mask else 0
            box = expand_box(raw_box, width, height, expand_px)
        standardized[part_name] = box_mask(box, height, width) if config.use_box_mask else raw_mask.astype(bool)

    face_mask = standardized.get("face")
    if face_mask is not None and config.face_clip_expand_px > 0:
        face_box = bbox_from_bool_mask(face_mask)
        if config.face_box_force_square:
            clipped = square_expand_box(face_box, width, height, config.face_clip_expand_px)
        else:
            clipped = expand_box(face_box, width, height, config.face_clip_expand_px)
        standardized["face"] = box_mask(clipped, height, width)
    return standardized

