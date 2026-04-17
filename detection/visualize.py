"""
人脸检测结果可视化模块。
"""

import cv2
import numpy as np

from .part_boxes import build_part_prompt_boxes, normalize_landmarks
from .settings import get_view_offset_map, resolve_part_offset
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
    if not det.landmarks:
        return

    part_boxes = build_part_prompt_boxes(
        det,
        image_w=canvas.shape[1],
        image_h=canvas.shape[0],
        part_names=("left_eye", "right_eye", "nose", "mouth"),
        view_id=view_id,
        part_scale_by_view=part_scale_by_view,
        part_offset_by_view=part_offset_by_view,
        part_offset_mode=part_offset_mode,
    )
    for part_name in ("left_eye", "right_eye", "nose", "mouth"):
        box = part_boxes.get(part_name)
        if box is not None:
            _draw_part_box(canvas, box, part_name)


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
    detector_name: str,
    status_text: str = "",
    draw_landmarks: bool = True,
    draw_part_boxes: bool = True,
    segmentation_masks=None,
    view_id: str | None = None,
    part_scale_by_view: dict[str, dict[str, tuple[float, float]]] | None = None,
    part_offset_by_view: dict[str, dict[str, tuple[float, float]]] | None = None,
    part_offset_mode: str = "ratio",
):
    """综合绘制检测框、关键点、部位框与分割叠加。"""
    # 画框和关键点，返回可视化图像。
    canvas = image.copy()

    title = f"detector: {detector_name}"
    if status_text:
        title = f"{title} | {status_text}"

    cv2.putText(
        canvas,
        title,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if not detections:
        cv2.putText(
            canvas,
            "no face detected",
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas

    # 先叠加分割区域，再画框和关键点，视觉上更清晰。
    if segmentation_masks:
        _draw_segmentation_masks(canvas, segmentation_masks)

    for det in detections:
        x1, y1, x2, y2 = det.box
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

        if draw_part_boxes:
            _draw_semantic_part_boxes(
                canvas,
                det,
                view_id=getattr(det, "view_id", None) or view_id,
                part_scale_by_view=part_scale_by_view,
                part_offset_by_view=part_offset_by_view,
                part_offset_mode=part_offset_mode,
            )

        if draw_landmarks and det.landmarks:
            # 关键点绘制按图像尺寸自适应，避免高分辨率图上点太小看不见。
            h, w = canvas.shape[:2]
            base = max(h, w)
            point_radius = max(2, int(round(base / 900)))
            line_thickness = max(1, int(round(base / 1800)))
            font_scale = max(0.4, base / 3600.0)

            normalized = _normalize_landmarks(det.landmarks)
            # 视角偏移：让关键点与部位框保持一致（仅影响可视化）
            active_view = getattr(det, "view_id", None) or view_id
            offset_map = get_view_offset_map(active_view, part_offset_by_view)
            fx1, fy1, fx2, fy2 = det.box
            face_w = max(1, fx2 - fx1)
            face_h = max(1, fy2 - fy1)

            for name, (px, py) in normalized.items():
                dx, dy = resolve_part_offset(name, offset_map, face_w, face_h, part_offset_mode)
                color = POINT_COLORS.get(name, (200, 200, 200))
                cv2.circle(canvas, (px + dx, py + dy), point_radius, color, -1)
                cv2.putText(
                    canvas,
                    name,
                    (px + dx + point_radius + 1, py + dy - point_radius - 1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    line_thickness,
                    cv2.LINE_AA,
                )

    return canvas
