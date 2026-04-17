"""
部位提示框的统一生成逻辑。

这块负责把检测框 + 关键点，转换成给 SAM / 可视化共用的部位提示框。
"""

from __future__ import annotations

from .settings import get_view_offset_map, get_view_scale_map, resolve_part_offset
from .types import FaceDetection


PART_NAMES = ("face", "left_eye", "right_eye", "nose", "mouth")

# 这些阈值只保留一份，避免分割和可视化各写各的默认值。
EYE_MIN_W = 12
EYE_MIN_H = 10
EYE_FACE_W_RATIO = 0.22
EYE_FACE_H_RATIO = 0.14

NOSE_MIN_W = 16
NOSE_MIN_H = 12
NOSE_FACE_W_RATIO = 0.26
NOSE_FACE_H_RATIO = 0.22

MOUTH_MIN_W = 14
MOUTH_MIN_H = 10
MOUTH_FACE_W_RATIO = 0.30
MOUTH_FACE_H_RATIO = 0.16
MOUTH_CORNER_PAD_RATIO = 0.05
MOUTH_CORNER_HALF_H_RATIO = 0.08

LANDMARK_ALIASES = {
    "left_eye": "left_eye",
    "lefteye": "left_eye",
    "right_eye": "right_eye",
    "righteye": "right_eye",
    "nose": "nose",
    "nose_tip": "nose",
    "nosetip": "nose",
    "mouth": "mouth",
    "mouth_center": "mouth",
    "mouthcenter": "mouth",
    "mouth_left": "mouth_left",
    "mouthleft": "mouth_left",
    "left_mouth": "mouth_left",
    "leftmouth": "mouth_left",
    "mouth_right": "mouth_right",
    "mouthright": "mouth_right",
    "right_mouth": "mouth_right",
    "rightmouth": "mouth_right",
}


def normalize_landmarks(landmarks: dict[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
    """把不同后端关键点名统一到同一语义。"""
    normalized: dict[str, tuple[int, int]] = {}
    for raw_name, point in landmarks.items():
        key = str(raw_name).strip().lower().replace("-", "_").replace(" ", "_")
        alias = LANDMARK_ALIASES.get(key)
        if alias is None:
            alias = LANDMARK_ALIASES.get(key.replace("_", ""))
        if alias is not None:
            normalized[alias] = point
    return normalized


def clip_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """裁剪框到图像边界，非法则返回 None。"""
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(0, min(width - 1, int(round(x2))))
    y2 = max(0, min(height - 1, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def box_from_center(
    cx: int,
    cy: int,
    bw: int,
    bh: int,
    image_w: int,
    image_h: int,
) -> tuple[int, int, int, int] | None:
    """由中心点和尺寸构造框，并裁剪到图像边界。"""
    half_w = max(1, bw // 2)
    half_h = max(1, bh // 2)
    return clip_box(
        cx - half_w,
        cy - half_h,
        cx + half_w,
        cy + half_h,
        image_w,
        image_h,
    )


def build_part_prompt_boxes(
    det: FaceDetection,
    image_w: int,
    image_h: int,
    part_names: tuple[str, ...] = PART_NAMES,
    view_id: str | None = None,
    part_scale_by_view: dict[str, dict[str, tuple[float, float]]] | None = None,
    part_offset_by_view: dict[str, dict[str, tuple[float, float]]] | None = None,
    part_offset_mode: str = "ratio",
) -> dict[str, tuple[int, int, int, int]]:
    """
    基于检测结果生成 SAM / 可视化共用的部位提示框。

    规则：
    - `face` 用检测框本身；
    - 期望用不到 `left_eye/right_eye/nose/mouth` 优先用关键点 + 人脸比例估计；
    - 若缺少关键点，则只返回能构造出来的部位框。
    """
    if view_id is None:
        view_id = getattr(det, "view_id", None)

    # 先从人脸框出发，后续部位框都围绕它构造。
    boxes: dict[str, tuple[int, int, int, int]] = {}
    face_box = clip_box(*det.box, image_w, image_h)
    if face_box is None:
        return boxes

    face_x1, face_y1, face_x2, face_y2 = face_box
    face_w = max(1, face_x2 - face_x1)
    face_h = max(1, face_y2 - face_y1)
    scale_map = get_view_scale_map(view_id, part_scale_by_view)
    offset_map = get_view_offset_map(view_id, part_offset_by_view)

    # face 直接沿用检测框，只做视角缩放/偏移修正。
    if "face" in part_names:
        sx, sy = scale_map.get("face", (1.0, 1.0))
        ox, oy = resolve_part_offset("face", offset_map, face_w, face_h, part_offset_mode)
        if abs(sx - 1.0) < 1e-6 and abs(sy - 1.0) < 1e-6:
            fx1, fy1, fx2, fy2 = face_box
            shifted = clip_box(fx1 + ox, fy1 + oy, fx2 + ox, fy2 + oy, image_w, image_h)
            if shifted is not None:
                boxes["face"] = shifted
        else:
            cx = int(round((face_x1 + face_x2) / 2 + ox))
            cy = int(round((face_y1 + face_y2) / 2 + oy))
            bw = max(1, int(round(face_w * float(sx))))
            bh = max(1, int(round(face_h * float(sy))))
            scaled = box_from_center(cx, cy, bw, bh, image_w, image_h)
            if scaled is not None:
                boxes["face"] = scaled

    if not det.landmarks:
        return boxes

    lm = normalize_landmarks(det.landmarks)

    # 眼睛框围绕关键点生成，默认框不要太大，避免吃进鼻梁。
    if "left_eye" in part_names and "left_eye" in lm:
        eye_w = max(EYE_MIN_W, int(face_w * EYE_FACE_W_RATIO))
        eye_h = max(EYE_MIN_H, int(face_h * EYE_FACE_H_RATIO))
        eye_sx, eye_sy = scale_map.get("left_eye", (1.0, 1.0))
        eye_ox, eye_oy = resolve_part_offset("left_eye", offset_map, face_w, face_h, part_offset_mode)
        box = box_from_center(
            lm["left_eye"][0] + int(round(eye_ox)),
            lm["left_eye"][1] + int(round(eye_oy)),
            max(1, int(round(eye_w * float(eye_sx)))),
            max(1, int(round(eye_h * float(eye_sy)))),
            image_w,
            image_h,
        )
        if box is not None:
            boxes["left_eye"] = box

    if "right_eye" in part_names and "right_eye" in lm:
        eye_w = max(EYE_MIN_W, int(face_w * EYE_FACE_W_RATIO))
        eye_h = max(EYE_MIN_H, int(face_h * EYE_FACE_H_RATIO))
        eye_sx, eye_sy = scale_map.get("right_eye", (1.0, 1.0))
        eye_ox, eye_oy = resolve_part_offset("right_eye", offset_map, face_w, face_h, part_offset_mode)
        box = box_from_center(
            lm["right_eye"][0] + int(round(eye_ox)),
            lm["right_eye"][1] + int(round(eye_oy)),
            max(1, int(round(eye_w * float(eye_sx)))),
            max(1, int(round(eye_h * float(eye_sy)))),
            image_w,
            image_h,
        )
        if box is not None:
            boxes["right_eye"] = box

    # 鼻子框默认水平更宽，减少鼻翼和鼻尖的漏包。
    if "nose" in part_names and "nose" in lm:
        nose_w = max(NOSE_MIN_W, int(face_w * NOSE_FACE_W_RATIO))
        nose_h = max(NOSE_MIN_H, int(face_h * NOSE_FACE_H_RATIO))
        nose_sx, nose_sy = scale_map.get("nose", (1.0, 1.0))
        nose_ox, nose_oy = resolve_part_offset("nose", offset_map, face_w, face_h, part_offset_mode)
        box = box_from_center(
            lm["nose"][0] + int(round(nose_ox)),
            lm["nose"][1] + int(round(nose_oy)),
            max(1, int(round(nose_w * float(nose_sx)))),
            max(1, int(round(nose_h * float(nose_sy)))),
            image_w,
            image_h,
        )
        if box is not None:
            boxes["nose"] = box

    # 嘴巴优先用左右嘴角合成；如果没有嘴角，再退回 mouth 中心点。
    if "mouth" in part_names:
        mouth_sx, mouth_sy = scale_map.get("mouth", (1.0, 1.0))
        mouth_ox, mouth_oy = resolve_part_offset("mouth", offset_map, face_w, face_h, part_offset_mode)
        if "mouth_left" in lm and "mouth_right" in lm:
            mlx, mly = lm["mouth_left"]
            mrx, mry = lm["mouth_right"]
            pad_x = max(MOUTH_MIN_W, int(face_w * MOUTH_CORNER_PAD_RATIO))
            half_h = max(MOUTH_MIN_H, int(face_h * MOUTH_CORNER_HALF_H_RATIO))
            pad_x = int(round(pad_x * float(mouth_sx)))
            half_h = int(round(half_h * float(mouth_sy)))
            box = clip_box(
                min(mlx, mrx) - pad_x + mouth_ox,
                int((mly + mry) / 2) - half_h + mouth_oy,
                max(mlx, mrx) + pad_x + mouth_ox,
                int((mly + mry) / 2) + half_h + mouth_oy,
                image_w,
                image_h,
            )
            if box is not None:
                boxes["mouth"] = box
        elif "mouth" in lm:
            mouth_w = max(MOUTH_MIN_W, int(face_w * MOUTH_FACE_W_RATIO))
            mouth_h = max(MOUTH_MIN_H, int(face_h * MOUTH_FACE_H_RATIO))
            box = box_from_center(
                lm["mouth"][0] + int(round(mouth_ox)),
                lm["mouth"][1] + int(round(mouth_oy)),
                max(1, int(round(mouth_w * float(mouth_sx)))),
                max(1, int(round(mouth_h * float(mouth_sy)))),
                image_w,
                image_h,
            )
            if box is not None:
                boxes["mouth"] = box

    return boxes
