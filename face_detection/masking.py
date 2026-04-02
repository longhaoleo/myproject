"""
批量眼部遮挡主入口（单阶段直出）：
1. 逐张图片检测眼部框；
2. 直接在原图上执行黑色遮挡；
3. 立即输出处理后的图片。

同时会输出分割点信息（遮挡框坐标）用于检查定位效果。
"""

from pathlib import Path

import cv2

from .dataset_utils import iter_images, sort_key
from .mask_apply import draw_black_eye_masks
from .mask_check import (
    MTCNN_PAD_H_RATIO,
    MTCNN_PAD_W_RATIO,
    create_face_landmarker,
    create_mtcnn_detector,
    detect_eye_mask_boxes,
    detect_semantic_boxes,
)
from .mask_visualize import draw_detection_preview

# 模型目录
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

# 跳过原因
SKIP_REASON = {
    "read_failed": "读取失败",
    "no_face": "未检测到人脸",
}


def format_boxes(boxes: list[tuple[int, int, int, int]]) -> str:
    # 用于日志输出：把遮挡框坐标格式化为可读字符串。
    if not boxes:
        return "[]"
    return "[" + ", ".join(f"({x1},{y1},{x2},{y2})" for x1, y1, x2, y2 in boxes) + "]"


def process_one(
    image_path: Path,
    output_path: Path | None,
    preview_output_path: Path | None,
    landmarker,
    mtcnn_detector,
    pad_x: int,
    pad_y: int,
):
    """
    单图处理：
    - 检测眼部框（MediaPipe 优先，MTCNN 兜底）；
    - 在图像上直接打码；
    - 返回状态和最终实际遮挡框。

    返回值：
    - status: ok_mp / ok_mtcnn / no_face / read_failed
    - final_boxes: 最终用于遮挡的框（已考虑 padding 和保留鼻梁间隙）
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return "read_failed", []

    # 可选输出检测预览图（眼睛/鼻子/嘴巴语义框），用于调试定位能力。
    if preview_output_path is not None:
        detections, det_source = detect_semantic_boxes(
            image=image,
            landmarker=landmarker,
            mtcnn_detector=mtcnn_detector,
        )
        preview = draw_detection_preview(
            image=image,
            detections=detections,
            status_text=f"semantic source: {det_source}",
        )
        preview_output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview_output_path), preview)

    boxes, source = detect_eye_mask_boxes(
        image=image,
        image_path=image_path,
        landmarker=landmarker,
        mtcnn_detector=mtcnn_detector,
    )
    if not boxes:
        return "no_face", []

    draw_pad_x = pad_x
    draw_pad_y = pad_y

    # MTCNN 框按自身大小动态扩边。
    if source == "ok_mtcnn":
        avg_w = sum((x2 - x1) for x1, _, x2, _ in boxes) / max(1, len(boxes))
        avg_h = sum((y2 - y1) for _, y1, _, y2 in boxes) / max(1, len(boxes))
        draw_pad_x = int(avg_w * MTCNN_PAD_W_RATIO)
        draw_pad_y = int(avg_h * MTCNN_PAD_H_RATIO)

    final_boxes = draw_black_eye_masks(
        image=image,
        boxes=boxes,
        pad_x=draw_pad_x,
        pad_y=draw_pad_y,
        min_gap=8,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
    return source, final_boxes


def main():
    # 输入目录
    input_root = Path("~/datasets/deformity").expanduser().resolve()
    # 打码图输出目录（仅在 enable_mask_output=True 时生效）
    output_root = Path("~/datasets/deformity_masked").expanduser().resolve()

    # 是否输出打码结果图
    enable_mask_output = False
    # 是否输出检测可视化图（用于看分割点定位）
    enable_detection_preview = True
    preview_root = Path("~/datasets/deformity_detection_preview").expanduser().resolve()

    # 是否在终端打印每张图的分割点信息（遮挡框坐标）
    print_box_info = False

    # MediaPipe 基础扩边像素（MTCNN 会按框大小动态扩边）
    padding_x_pixels = 18
    padding_y_pixels = 12

    image_paths = sorted(iter_images(input_root), key=lambda p: sort_key(p, input_root))
    if not image_paths:
        print(f"未找到图片: {input_root}")
        return

    mtcnn_detector = create_mtcnn_detector(MODEL_DIR)

    ok_count = 0
    ok_mp_count = 0
    ok_mtcnn_count = 0
    skip_count = 0

    with create_face_landmarker(MODEL_DIR) as landmarker:
        for image_path in image_paths:
            output_path = None
            if enable_mask_output:
                output_path = output_root / image_path.relative_to(input_root)

            preview_output_path = None
            if enable_detection_preview:
                preview_output_path = preview_root / image_path.relative_to(input_root)

            status, final_boxes = process_one(
                image_path=image_path,
                output_path=output_path,
                preview_output_path=preview_output_path,
                landmarker=landmarker,
                mtcnn_detector=mtcnn_detector,
                pad_x=padding_x_pixels,
                pad_y=padding_y_pixels,
            )

            if status in {"ok_mp", "ok_mtcnn"}:
                ok_count += 1
                if status == "ok_mp":
                    ok_mp_count += 1
                    if output_path is not None:
                        print(f"已处理(MediaPipe): {output_path}")
                    else:
                        print("已处理(MediaPipe)")
                else:
                    ok_mtcnn_count += 1
                    if output_path is not None:
                        print(f"已处理(MTCNN): {output_path}")
                    else:
                        print("已处理(MTCNN)")

                if print_box_info:
                    print(f"分割点(遮挡框): {format_boxes(final_boxes)}")
            else:
                skip_count += 1
                print(f"已跳过({SKIP_REASON.get(status, status)}): {image_path}")

    print(
        f"处理完成，总成功 {ok_count} 张 (MediaPipe {ok_mp_count} / MTCNN {ok_mtcnn_count})，"
        f"跳过 {skip_count} 张。"
    )


if __name__ == "__main__":
    main()
