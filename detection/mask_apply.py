"""
绘制模块：根据检测框在图像上执行黑色眼罩遮挡。
"""

import cv2


def safe_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int):
    """裁剪框到图像边界，非法则返回 None。"""
    # 裁剪到图像边界内，非法框返回 None。
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w - 1, x2)
    y2 = min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def keep_gap_between_two_boxes(box_a, box_b, min_gap: int = 6):
    """调整两个框的相邻边界，确保中间留缝。"""
    # 调整左右边界，确保中间留白，不遮鼻梁。
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    if ax1 > bx1:
        ax1, ay1, ax2, ay2, bx1, by1, bx2, by2 = bx1, by1, bx2, by2, ax1, ay1, ax2, ay2

    if ax2 + min_gap <= bx1:
        return (ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2)

    mid = (ax2 + bx1) // 2
    new_ax2 = max(ax1, mid - (min_gap // 2))
    new_bx1 = min(bx2, mid + (min_gap - min_gap // 2))
    return (ax1, ay1, new_ax2, ay2), (new_bx1, by1, bx2, by2)


def apply_padding_and_keep_gap(boxes, w: int, h: int, pad_x: int, pad_y: int, min_gap: int = 8):
    """先扩边再留缝，返回最终可绘制框。"""
    # 先扩边，再保证左右眼之间留缝。
    # 这里不关心框来源（MediaPipe/MTCNN），只做几何后处理。
    padded = []
    for x1, y1, x2, y2 in boxes:
        box = safe_box(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, w, h)
        if box is not None:
            padded.append(box)

    if len(padded) < 2:
        return padded

    padded = sorted(padded, key=lambda box: (box[0] + box[2]) / 2.0)
    adjusted = padded[:]
    for i in range(len(adjusted) - 1):
        adjusted[i], adjusted[i + 1] = keep_gap_between_two_boxes(
            adjusted[i], adjusted[i + 1], min_gap=min_gap
        )
    return adjusted


def draw_black_eye_masks(image, boxes, pad_x: int, pad_y: int, min_gap: int = 8):
    """在图像上画黑色遮挡并返回最终框。"""
    # 在图像上画黑色眼罩，并返回最终落地的遮挡框。
    h, w = image.shape[:2]
    final_boxes = apply_padding_and_keep_gap(
        boxes=boxes,
        w=w,
        h=h,
        pad_x=pad_x,
        pad_y=pad_y,
        min_gap=min_gap,
    )
    for x1, y1, x2, y2 in final_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
    return final_boxes


def draw_black_boxes(image, boxes):
    """直接把给定框涂黑（不做扩边与留缝）。"""
    h, w = image.shape[:2]
    final_boxes = []
    for x1, y1, x2, y2 in boxes:
        box = safe_box(x1, y1, x2, y2, w, h)
        if box is None:
            continue
        final_boxes.append(box)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 0), -1)
    return final_boxes
