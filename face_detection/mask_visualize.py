"""
可视化模块：把语义检测框（眼睛/鼻子/嘴巴）和标签画到图片上。
用于评估检测器定位能力，不参与最终打码逻辑。
"""

import cv2


LABEL_COLORS = {
    "left_eye": (0, 200, 0),
    "right_eye": (0, 160, 0),
    "nose": (255, 120, 0),
    "mouth": (220, 0, 180),
}


def draw_detection_preview(image, detections, status_text: str = ""):
    # 在图片上画语义框和标签，输出一张调试预览图。
    canvas = image.copy()

    if status_text:
        cv2.putText(
            canvas,
            status_text,
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
            "no face/feature detected",
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas

    for det in detections:
        label = str(det.get("label", "obj"))
        source = str(det.get("source", "model"))
        x1, y1, x2, y2 = det["box"]
        color = LABEL_COLORS.get(label, (255, 255, 0))

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text = f"{label} ({source})"
        text_y = y1 - 8 if y1 > 20 else y1 + 20
        cv2.putText(
            canvas,
            text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return canvas
