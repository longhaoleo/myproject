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
from .mask_apply import draw_black_eye_masks
from .part_boxes import build_part_prompt_boxes
from .settings import (
    default_detector_options,
    default_min_confidence_map,
    default_part_offset_mode,
    default_part_offset_by_view,
    default_part_scale_by_view,
    default_paths,
)
from project_utils.dataset import iter_images, sort_key


EYE_PART_NAMES = ("left_eye", "right_eye")
EYE_MASK_PAD_X = 4
EYE_MASK_PAD_Y = 3
EYE_MASK_MIN_GAP = 8

# 跳过原因
SKIP_REASON = {
    "read_failed": "读取失败",
    "no_face": "未检测到人脸",
    "no_eye": "未定位到眼睛",
}


def _write_lines(file_path: Path, lines: list[str]) -> None:
    """把文本行写入文件（用于统计落盘）。"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def format_boxes(boxes: list[tuple[int, int, int, int]]) -> str:
    """把遮挡框列表格式化成日志字符串。"""
    if not boxes:
        return "[]"
    return "[" + ", ".join(f"({x1},{y1},{x2},{y2})" for x1, y1, x2, y2 in boxes) + "]"


def _eye_boxes_from_detections(
    detections,
    image_w: int,
    image_h: int,
    view_id: str,
    part_scale_by_view: dict[str, dict[str, tuple[float, float]]],
    part_offset_by_view: dict[str, dict[str, tuple[float, float]]],
    part_offset_mode: str,
) -> list[tuple[int, int, int, int]]:
    """从检测结果中提取左右眼遮挡框。"""
    boxes: list[tuple[int, int, int, int]] = []
    for det in detections:
        part_boxes = build_part_prompt_boxes(
            det=det,
            image_w=image_w,
            image_h=image_h,
            part_names=EYE_PART_NAMES,
            view_id=view_id,
            part_scale_by_view=part_scale_by_view,
            part_offset_by_view=part_offset_by_view,
            part_offset_mode=part_offset_mode,
        )
        for part_name in EYE_PART_NAMES:
            box = part_boxes.get(part_name)
            if box is not None:
                boxes.append(box)
    return boxes


def _detect_eye_boxes(
    image,
    view_id: str,
    detectors,
    part_scale_by_view: dict[str, dict[str, tuple[float, float]]],
    part_offset_by_view: dict[str, dict[str, tuple[float, float]]],
    part_offset_mode: str,
) -> tuple[str, list[tuple[int, int, int, int]]]:
    """按优先级尝试检测器，返回首个能定位眼睛的结果。"""
    h, w = image.shape[:2]
    saw_face = False
    for detector in detectors:
        try:
            detections = detector.detect(image)
        except Exception:
            continue
        if not detections:
            continue

        saw_face = True
        eye_boxes = _eye_boxes_from_detections(
            detections=detections,
            image_w=w,
            image_h=h,
            view_id=view_id,
            part_scale_by_view=part_scale_by_view,
            part_offset_by_view=part_offset_by_view,
            part_offset_mode=part_offset_mode,
        )
        if eye_boxes:
            return "ok", eye_boxes

    return ("no_eye" if saw_face else "no_face"), []


def _create_detectors(
    detector_names: list[str],
    model_dir: Path,
    min_confidence_map: dict[str, float],
    detector_options_map: dict[str, dict],
):
    """按顺序创建检测器，单个失败时跳过，避免 fallback 缺权重导致整批退出。"""
    detectors = []
    for name in detector_names:
        try:
            detectors.append(
                create_face_detector(
                    detector_name=name,
                    model_dir=model_dir,
                    min_confidence=min_confidence_map.get(name, 0.5),
                    detector_options=detector_options_map.get(name),
                )
            )
        except Exception as exc:
            print(f"[masking] 检测器不可用，已跳过: {name} | {exc}")
    return detectors


def process_one(
    image_path: Path,
    output_path: Path | None,
    detectors,
    part_scale_by_view: dict[str, dict[str, tuple[float, float]]],
    part_offset_by_view: dict[str, dict[str, tuple[float, float]]],
    part_offset_mode: str,
):
    """单图处理：按优先级尝试检测器 -> 定位眼睛 -> 直接置黑。"""
    image = cv2.imread(str(image_path))
    if image is None:
        return "read_failed", []

    view_id = image_path.stem
    status, eye_boxes = _detect_eye_boxes(
        image=image,
        view_id=view_id,
        detectors=detectors,
        part_scale_by_view=part_scale_by_view,
        part_offset_by_view=part_offset_by_view,
        part_offset_mode=part_offset_mode,
    )
    if status != "ok":
        return status, []

    final_boxes = draw_black_eye_masks(
        image=image,
        boxes=eye_boxes,
        pad_x=EYE_MASK_PAD_X,
        pad_y=EYE_MASK_PAD_Y,
        min_gap=EYE_MASK_MIN_GAP,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
    return "ok", final_boxes


def main():
    """打码批处理入口：定位左右眼并置黑。"""
    # 输入/输出/模型路径（集中管理）
    paths = default_paths()
    input_root = paths.input_root
    output_root = paths.output_root_mask

    # 主线检测器：优先稳定性，后续 fallback 只保留当前仍考虑的路线。
    detector_name = "scrfd"
    tweak_method_name = "eye_mask"
    # 兜底检测器（按顺序尝试）
    fallback_detectors = [
        "yolov8-face",
        "blazeface",
    ]
    min_confidence_map = default_min_confidence_map()
    detector_options_map = default_detector_options(paths.model_dir)
    part_scale_by_view = default_part_scale_by_view(method_name=tweak_method_name)
    part_offset_by_view = default_part_offset_by_view(method_name=tweak_method_name)
    part_offset_mode = default_part_offset_mode(method_name=tweak_method_name)

    # 是否输出打码结果图
    enable_mask_output = True
    # 是否在终端打印每张图的框坐标
    print_box_info = False

    image_paths = sorted(iter_images(input_root), key=lambda p: sort_key(p, input_root))
    if not image_paths:
        print(f"未找到图片: {input_root}")
        return

    start = time.perf_counter()
    all_detector_names = [detector_name] + fallback_detectors
    detectors = _create_detectors(
        detector_names=all_detector_names,
        model_dir=paths.model_dir,
        min_confidence_map=min_confidence_map,
        detector_options_map=detector_options_map,
    )
    if not detectors:
        print("没有可用检测器，请先检查 model/ 权重和 Python 环境。")
        return

    ok_count = 0
    skip_count = 0
    read_failed_count = 0
    no_face_count = 0
    no_eye_count = 0
    try:
        for image_path in image_paths:
            output_path = None
            if enable_mask_output:
                output_path = output_root / image_path.relative_to(input_root)

            status, final_boxes = process_one(
                image_path=image_path,
                output_path=output_path,
                detectors=detectors,
                part_scale_by_view=part_scale_by_view,
                part_offset_by_view=part_offset_by_view,
                part_offset_mode=part_offset_mode,
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
                elif status == "no_eye":
                    no_eye_count += 1
                print(f"已跳过({SKIP_REASON.get(status, status)}): {image_path}")
    finally:
        for detector in detectors:
            detector.close()

    elapsed = time.perf_counter() - start
    print("\n=== eye-mask 汇总 ===")
    print(f"处理成功: {ok_count}")
    print(f"跳过总数: {skip_count}")
    print(f"未检测到人脸: {no_face_count}")
    print(f"未定位到眼睛: {no_eye_count}")
    print(f"读取失败: {read_failed_count}")
    print(f"总耗时: {elapsed:.2f}s")

    _write_lines(
        output_root / "_run_stats" / "eye_mask_summary.txt",
        [
            f"ok_count={ok_count}",
            f"skip_count={skip_count}",
            f"no_face_count={no_face_count}",
            f"no_eye_count={no_eye_count}",
            f"read_failed_count={read_failed_count}",
            f"elapsed_seconds={elapsed:.6f}",
            f"primary_detector={detector_name}",
            f"fallback_detectors={','.join(fallback_detectors)}",
            f"tweak_method_name={tweak_method_name}",
        ],
    )
if __name__ == "__main__":
    main()
