"""
批量眼部遮挡主入口（单阶段直出）：
1. 逐张图片检测眼部框；
2. 直接在原图上执行黑色遮挡；
3. 立即输出处理后的图片。

同时会输出分割点信息（遮挡框坐标）用于检查定位效果。
"""

from pathlib import Path
import time

import cv2

from .factory import create_face_detector
from .mask_apply import draw_black_boxes
from .settings import default_detector_options, default_min_confidence_map, default_paths
from project_utils.dataset import iter_images, sort_key


# 跳过原因
SKIP_REASON = {
    "read_failed": "读取失败",
    "no_face": "未检测到人脸",
}


def _write_lines(file_path: Path, lines: list[str]) -> None:
    """把文本行写入文件（用于统计落盘）。"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def format_boxes(boxes: list[tuple[int, int, int, int]]) -> str:
    """把遮挡框列表格式化成日志字符串。"""
    # 用于日志输出：把遮挡框坐标格式化为可读字符串。
    if not boxes:
        return "[]"
    return "[" + ", ".join(f"({x1},{y1},{x2},{y2})" for x1, y1, x2, y2 in boxes) + "]"


def process_one(
    image_path: Path,
    output_path: Path | None,
    detectors,
):
    """单图处理：按优先级尝试检测器 -> 直接置黑 -> 返回状态与最终框。"""
    image = cv2.imread(str(image_path))
    if image is None:
        return "read_failed", []

    detections = []
    for detector in detectors:
        try:
            detections = detector.detect(image)
        except Exception:
            detections = []
        if detections:
            break

    boxes = [det.box for det in detections]
    if not boxes:
        return "no_face", []

    final_boxes = draw_black_boxes(image=image, boxes=boxes)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
    return "ok", final_boxes


def main():
    """打码批处理入口：直接把检测框置黑。"""
    # 输入/输出/模型路径（集中管理）
    paths = default_paths()
    input_root = paths.input_root
    output_root = paths.output_root_mask

    # 选择用于打框的检测器
    detector_name = "yolov8-face"
    # 兜底检测器（按顺序尝试）
    fallback_detectors = [
        "mediapipe-landmarker",
        "retinaface",
        "scrfd",
        "mtcnn",
        "blazeface",
        "centerface",
    ]
    min_confidence_map = default_min_confidence_map()
    detector_options_map = default_detector_options(paths.model_dir)

    # 是否输出打码结果图
    enable_mask_output = True
    # 是否在终端打印每张图的框坐标
    print_box_info = False

    image_paths = sorted(iter_images(input_root), key=lambda p: sort_key(p, input_root))
    if not image_paths:
        print(f"未找到图片: {input_root}")
        return

    start = time.perf_counter()
    detectors = []
    all_detector_names = [detector_name] + fallback_detectors
    for name in all_detector_names:
        detectors.append(
            create_face_detector(
                detector_name=name,
                model_dir=paths.model_dir,
                min_confidence=min_confidence_map.get(name, 0.5),
                detector_options=detector_options_map.get(name),
            )
        )

    ok_count = 0
    skip_count = 0
    read_failed_count = 0
    no_face_count = 0
    try:
        for image_path in image_paths:
            output_path = None
            if enable_mask_output:
                output_path = output_root / image_path.relative_to(input_root)

            status, final_boxes = process_one(
                image_path=image_path,
                output_path=output_path,
                detectors=detectors,
            )

            if status == "ok":
                ok_count += 1
                if output_path is not None:
                    print(f"已处理({detector_name}+fallback): {output_path}")
                else:
                    print(f"已处理({detector_name}+fallback)")
                if print_box_info:
                    print(f"遮挡框: {format_boxes(final_boxes)}")
            else:
                skip_count += 1
                if status == "read_failed":
                    read_failed_count += 1
                elif status == "no_face":
                    no_face_count += 1
                print(f"已跳过({SKIP_REASON.get(status, status)}): {image_path}")
    finally:
        for detector in detectors:
            detector.close()

    elapsed = time.perf_counter() - start
    print("\n=== eye-mask 汇总 ===")
    print(f"处理成功: {ok_count}")
    print(f"跳过总数: {skip_count}")
    print(f"未检测到人脸: {no_face_count}")
    print(f"读取失败: {read_failed_count}")
    print(f"总耗时: {elapsed:.2f}s")

    _write_lines(
        output_root / "_run_stats" / "eye_mask_summary.txt",
        [
            f"ok_count={ok_count}",
            f"skip_count={skip_count}",
            f"no_face_count={no_face_count}",
            f"read_failed_count={read_failed_count}",
            f"elapsed_seconds={elapsed:.6f}",
            f"primary_detector={detector_name}",
            f"fallback_detectors={','.join(fallback_detectors)}",
        ],
    )


if __name__ == "__main__":
    main()
