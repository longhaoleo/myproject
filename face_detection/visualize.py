"""
人脸检测结果可视化模块。
"""

import cv2

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


def _normalize_landmarks(landmarks: dict[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
    # 把不同后端的关键点命名归一化为统一语义。
    normalized: dict[str, tuple[int, int]] = {}
    for raw_name, point in landmarks.items():
        key = str(raw_name).strip().lower().replace("-", "_").replace(" ", "_")
        alias = LANDMARK_ALIASES.get(key)
        if alias is None:
            alias = LANDMARK_ALIASES.get(key.replace("_", ""))
        if alias is not None:
            normalized[alias] = point
    return normalized


def _clip_rect(x1: int, y1: int, x2: int, y2: int, w: int, h: int):
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


def _draw_semantic_part_boxes(canvas, det: FaceDetection):
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
    if "left_eye" in lm:
        ex, ey = lm["left_eye"]
        eye_box = _box_from_center(ex, ey, eye_w, eye_h, w, h)
        if eye_box is not None:
            _draw_part_box(canvas, eye_box, "left_eye")
    if "right_eye" in lm:
        ex, ey = lm["right_eye"]
        eye_box = _box_from_center(ex, ey, eye_w, eye_h, w, h)
        if eye_box is not None:
            _draw_part_box(canvas, eye_box, "right_eye")

    # 鼻子框：由鼻尖关键点 + 人脸比例得到。
    if "nose" in lm:
        nx, ny = lm["nose"]
        nose_w = max(12, int(face_w * 0.18))
        nose_h = max(12, int(face_h * 0.22))
        nose_box = _box_from_center(nx, ny, nose_w, nose_h, w, h)
        if nose_box is not None:
            _draw_part_box(canvas, nose_box, "nose")

    # 嘴巴框：优先用左右嘴角合成更稳定的框；没有嘴角时退化到 mouth 中心点。
    if "mouth_left" in lm and "mouth_right" in lm:
        mlx, mly = lm["mouth_left"]
        mrx, mry = lm["mouth_right"]
        pad_x = max(6, int(face_w * 0.05))
        half_h = max(6, int(face_h * 0.08))
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
        mouth_box = _box_from_center(mx, my, mouth_w, mouth_h, w, h)
        if mouth_box is not None:
            _draw_part_box(canvas, mouth_box, "mouth")


def draw_face_detections(
    image,
    detections: list[FaceDetection],
    detector_name: str,
    status_text: str = "",
    draw_landmarks: bool = True,
    draw_part_boxes: bool = True,
):
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
            _draw_semantic_part_boxes(canvas, det)

        if draw_landmarks and det.landmarks:
            for name, (px, py) in det.landmarks.items():
                color = POINT_COLORS.get(name, (200, 200, 200))
                cv2.circle(canvas, (px, py), 2, color, -1)
                cv2.putText(
                    canvas,
                    name,
                    (px + 3, py - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )

    return canvas
