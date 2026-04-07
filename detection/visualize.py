"""
人脸检测结果可视化模块。

功能：
- 绘制人脸框 + 置信度；
- 绘制关键点（按统一语义名）；
- 绘制部位框（眼/鼻/嘴）；
- 可选叠加分割 mask。
"""

import cv2
import numpy as np

from .settings import adjust_box_for_view, get_view_offset_map, get_view_scale_map, resolve_part_offset
from .types import FaceDetection


POINT_COLORS = {
    "left_eye": (0, 255, 0),
    "right_eye": (0, 220, 0),
    "nose": (0, 160, 255),
    "mouth": (255, 0, 180),
    "mouth_left": (255, 0, 180),
    "mouth_right": (255, 0, 180),
}

PART_BOX_COLORS = {
    "left_eye": (0, 255, 0),
    "right_eye": (0, 220, 0),
    "nose": (0, 160, 255),
    "mouth": (255, 0, 180),
}

# 分割 mask 的默认配色（用于可视化叠加，不影响实际 mask 输出）。
SEG_MASK_COLORS = {
    "face": (30, 200, 30),
    "left_eye": (0, 255, 0),
    "right_eye": (0, 220, 0),
    "nose": (0, 160, 255),
    "mouth": (255, 0, 180),
}

CANONICAL_LANDMARKS = {
    "left_eye",
    "right_eye",
    "nose",
    "mouth",
    "mouth_left",
    "mouth_right",
}


def _normalize_landmarks(landmarks: dict[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
    """把不同后端关键点命名映射到统一语义。"""
    # 把不同后端的关键点命名归一化为统一语义。
    normalized: dict[str, tuple[int, int]] = {}
    for raw_name, point in landmarks.items():
        key = str(raw_name).strip().lower().replace("-", "_").replace(" ", "_")
        if key in CANONICAL_LANDMARKS:
            normalized[key] = point
    return normalized


def _clip_rect(x1: int, y1: int, x2: int, y2: int, w: int, h: int):
    """裁剪矩形到图像边界，非法则返回 None。"""
    # 约束框在图像边界内。
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _box_from_center(
    center_x: int,
    center_y: int,
    box_w: int,
    box_h: int,
    image_w: int,
    image_h: int,
):
    """根据中心点与尺寸构造框，并裁剪到图像边界。"""
    half_w = max(1, box_w // 2)
    half_h = max(1, box_h // 2)
    return _clip_rect(
        center_x - half_w,
        center_y - half_h,
        center_x + half_w,
        center_y + half_h,
        image_w,
        image_h,
    )


def _draw_part_box(canvas, box: tuple[int, int, int, int], part_name: str):
    """绘制单个部位框与标签。"""
    # 画部位框和标签。
    color = PART_BOX_COLORS.get(part_name, (200, 200, 200))
    x1, y1, x2, y2 = box
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    text_y = y1 - 6 if y1 > 16 else y1 + 14
    cv2.putText(
        canvas,
        part_name,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_semantic_part_boxes(
    canvas,
    det: FaceDetection,
    view_id: str | None = None,
    part_scale_by_view: dict[str, dict[str, tuple[float, float]]] | None = None,
    part_offset_by_view: dict[str, dict[str, tuple[float, float]]] | None = None,
    part_offset_mode: str = "ratio",
):
    """依据关键点估计并绘制眼/鼻/嘴部位框。"""
    # 基于关键点估计眼睛/鼻子/嘴巴的部位框。
    if not det.landmarks:
        return

    h, w = canvas.shape[:2]
    face_x1, face_y1, face_x2, face_y2 = det.box
    face_w = max(1, face_x2 - face_x1)
    face_h = max(1, face_y2 - face_y1)

    lm = _normalize_landmarks(det.landmarks)

    # 眼睛框：由眼睛关键点中心点 + 人脸比例得到。
    eye_w = max(12, int(face_w * 0.22))
    eye_h = max(10, int(face_h * 0.14))
    scale_map = get_view_scale_map(view_id, part_scale_by_view)
    eye_sx, eye_sy = scale_map.get("left_eye", (1.0, 1.0))
    if "left_eye" in lm:
        ex, ey = lm["left_eye"]
        eye_box = _box_from_center(
            ex,
            ey,
            max(1, int(round(eye_w * float(eye_sx)))),
            max(1, int(round(eye_h * float(eye_sy)))),
            w,
            h,
        )
        if eye_box is not None:
            _draw_part_box(canvas, eye_box, "left_eye")
    eye_sx, eye_sy = scale_map.get("right_eye", (1.0, 1.0))
    if "right_eye" in lm:
        ex, ey = lm["right_eye"]
        eye_box = _box_from_center(
            ex,
            ey,
            max(1, int(round(eye_w * float(eye_sx)))),
            max(1, int(round(eye_h * float(eye_sy)))),
            w,
            h,
        )
        if eye_box is not None:
            _draw_part_box(canvas, eye_box, "right_eye")

    # 鼻子框：由鼻尖关键点 + 人脸比例得到。
    if "nose" in lm:
        nx, ny = lm["nose"]
        # 鼻子框水平向放宽，便于观察覆盖范围。
        nose_w = max(16, int(face_w * 0.26))
        nose_h = max(12, int(face_h * 0.22))
        nose_sx, nose_sy = scale_map.get("nose", (1.0, 1.0))
        nose_box = _box_from_center(
            nx,
            ny,
            max(1, int(round(nose_w * float(nose_sx)))),
            max(1, int(round(nose_h * float(nose_sy)))),
            w,
            h,
        )
        if nose_box is not None:
            _draw_part_box(canvas, nose_box, "nose")

    # 嘴巴框：优先用左右嘴角合成更稳定的框；没有嘴角时退化到 mouth 中心点。
    mouth_sx, mouth_sy = scale_map.get("mouth", (1.0, 1.0))
    if "mouth_left" in lm and "mouth_right" in lm:
        mlx, mly = lm["mouth_left"]
        mrx, mry = lm["mouth_right"]
        pad_x = max(6, int(face_w * 0.05))
        half_h = max(6, int(face_h * 0.08))
        pad_x = int(round(pad_x * float(mouth_sx)))
        half_h = int(round(half_h * float(mouth_sy)))
        mouth_box = _clip_rect(
            min(mlx, mrx) - pad_x,
            int((mly + mry) / 2) - half_h,
            max(mlx, mrx) + pad_x,
            int((mly + mry) / 2) + half_h,
            w,
            h,
        )
        if mouth_box is not None:
            _draw_part_box(canvas, mouth_box, "mouth")
    elif "mouth" in lm:
        mx, my = lm["mouth"]
        mouth_w = max(14, int(face_w * 0.30))
        mouth_h = max(10, int(face_h * 0.16))
        mouth_box = _box_from_center(
            mx,
            my,
            max(1, int(round(mouth_w * float(mouth_sx)))),
            max(1, int(round(mouth_h * float(mouth_sy)))),
            w,
            h,
        )
        if mouth_box is not None:
            _draw_part_box(canvas, mouth_box, "mouth")


def _draw_segmentation_masks(canvas, segmentation_masks, alpha: float = 0.22):
    """
    叠加分割 mask 可视化。

    约定：
    - mask 尺寸应与图像同高同宽；
    - mask 非零像素表示该位置属于目标区域。
    """
    if not segmentation_masks:
        return

    h, w = canvas.shape[:2]
    overlay = canvas.copy()
    mask_color = (30, 200, 30)

    for item in segmentation_masks:
        if item is None:
            continue

        # 兼容两种格式：
        # 1) 单张合并 mask（ndarray）
        # 2) 分部位 mask 字典（part_name -> ndarray）
        if isinstance(item, dict):
            for part_name, part_mask in item.items():
                if part_mask is None or part_mask.shape[:2] != (h, w):
                    continue
                mask_bool = part_mask.astype(bool)
                color = SEG_MASK_COLORS.get(str(part_name), mask_color)
                overlay[mask_bool] = color
            continue

        if isinstance(item, np.ndarray):
            if item.shape[:2] != (h, w):
                continue
            mask_bool = item.astype(bool)
            overlay[mask_bool] = mask_color

    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, dst=canvas)


def draw_face_detections(
    image,
    detections: list[FaceDetection],
    segmentation_masks=None,
    view_id: str | None = None,
    part_scale_by_view: dict[str, dict[str, tuple[float, float]]] | None = None,
    part_offset_by_view: dict[str, dict[str, tuple[float, float]]] | None = None,
    part_offset_mode: str = "ratio",
):
    """
    综合绘制检测框、关键点、部位框与分割叠加。

    注意：
    - 人脸框经过 settings.adjust_box_for_view 处理（方形化 + 放大）。
    - 关键点偏移已在上游写回检测结果，这里只负责绘制。
    """
    # 画框和关键点，返回可视化图像。
    canvas = image.copy()

    # 先叠加分割区域，再画框和关键点，视觉上更清晰。
    if segmentation_masks:
        _draw_segmentation_masks(canvas, segmentation_masks)

    for det in detections:
        active_view = getattr(det, "view_id", None) or view_id
        scale_map = get_view_scale_map(active_view, part_scale_by_view)
        offset_map = get_view_offset_map(active_view, part_offset_by_view)
        x1, y1, x2, y2 = adjust_box_for_view(
            box=det.box,
            part_name="face",
            image_w=canvas.shape[1],
            image_h=canvas.shape[0],
            scale_map=scale_map,
            offset_map=offset_map,
            part_offset_mode=part_offset_mode,
        )
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 0), 2)

        score_text = "score: n/a"
        if det.score is not None:
            score_text = f"score: {det.score:.3f}"
        text_y = y1 - 8 if y1 > 20 else y1 + 20
        cv2.putText(
            canvas,
            score_text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        _draw_semantic_part_boxes(
            canvas,
            det,
            view_id=getattr(det, "view_id", None) or view_id,
            part_scale_by_view=part_scale_by_view,
            part_offset_by_view=part_offset_by_view,
            part_offset_mode=part_offset_mode,
        )

        if det.landmarks:
            # 关键点绘制按图像尺寸自适应，避免高分辨率图上点太小看不见。
            h, w = canvas.shape[:2]
            base = max(h, w)
            point_radius = max(2, int(round(base / 900)))
            line_thickness = max(1, int(round(base / 1800)))
            font_scale = max(0.4, base / 3600.0)

            normalized = _normalize_landmarks(det.landmarks)
            fx1, fy1, fx2, fy2 = det.box
            face_w = max(1, fx2 - fx1)
            face_h = max(1, fy2 - fy1)

            for name, (px, py) in normalized.items():
                color = POINT_COLORS.get(name, (200, 200, 200))
                cv2.circle(canvas, (px, py), point_radius, color, -1)
                cv2.putText(
                    canvas,
                    name,
                    (px + point_radius + 1, py - point_radius - 1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    line_thickness,
                    cv2.LINE_AA,
                )

    return canvas
