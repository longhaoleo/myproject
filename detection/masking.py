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
from .segmentation import build_part_boxes_for_detections
from .settings import (
    apply_landmark_offsets,
    default_detector_options,
    default_min_confidence_map,
    default_part_offset_mode,
    default_part_offset_by_view,
    default_part_scale_by_view,
    default_paths,
    normalize_profile_name,
)
from project_utils.dataset import iter_images, sort_key


# 跳过原因
SKIP_REASON = {
    "read_failed": "读取失败",
    "no_face": "未检测到人脸",
}


def _clip_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    image_w: int,
    image_h: int,
) -> tuple[int, int, int, int] | None:
    """把框裁剪到图像边界，非法返回 None。"""
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_w - 1, x2)
    y2 = min(image_h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _draw_black_boxes(
    image,
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """把给定框直接涂黑，并返回实际生效的框。"""
    h, w = image.shape[:2]
    final_boxes: list[tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in boxes:
        clipped = _clip_box(x1, y1, x2, y2, w, h)
        if clipped is None:
            continue
        final_boxes.append(clipped)
        cv2.rectangle(image, (clipped[0], clipped[1]), (clipped[2], clipped[3]), (0, 0, 0), -1)
    return final_boxes


def _trim_eye_box_against_nose(
    eye_box: tuple[int, int, int, int],
    nose_box: tuple[int, int, int, int] | None,
    side: str,
) -> tuple[int, int, int, int] | None:
    """若眼框与鼻框重合，则把眼框朝外侧裁剪，避免遮到鼻子。"""
    if nose_box is None:
        return eye_box
    ex1, ey1, ex2, ey2 = eye_box
    nx1, ny1, nx2, ny2 = nose_box
    overlap_x = min(ex2, nx2) - max(ex1, nx1)
    overlap_y = min(ey2, ny2) - max(ey1, ny1)
    if overlap_x <= 0 or overlap_y <= 0:
        return eye_box
    if side == "left":
        ex2 = min(ex2, nx1 - 1)
    else:
        ex1 = max(ex1, nx2 + 1)
    if ex2 <= ex1 or ey2 <= ey1:
        return None
    return ex1, ey1, ex2, ey2


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
    detector_tweaks: dict[str, tuple[
        dict[str, dict[str, tuple[float, float]]],
        dict[str, dict[str, tuple[float, float]]],
        str,
    ]],
):
    """
    单图处理：按优先级尝试检测器 -> 直接置黑 -> 返回状态与最终框。

    detectors 结构：
    - [(profile_name, detector_instance), ...]
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return "read_failed", [], ""

    detections = []
    used_profile = ""
    # 依次尝试检测器，命中即停止（用于兜底策略）。
    for profile_name, detector in detectors:
        try:
            detections = detector.detect(image)
        except Exception:
            detections = []
        if detections:
            used_profile = profile_name
            break

    view_id = image_path.stem
    part_scale_by_view, part_offset_by_view, part_offset_mode = detector_tweaks.get(
        used_profile,
        ({}, {}, "ratio"),
    )
    # 先把视角偏移写回关键点，再基于修正后的关键点生成左右眼框。
    apply_landmark_offsets(
        detections=detections,
        view_id=view_id,
        part_offset_by_view=part_offset_by_view,
        part_offset_mode=part_offset_mode,
    )
    part_boxes_list = build_part_boxes_for_detections(
        detections=detections,
        image_w=image.shape[1],
        image_h=image.shape[0],
        part_names=("left_eye", "right_eye", "nose"),
        part_scale_by_view=part_scale_by_view,
        part_offset_by_view=part_offset_by_view,
        part_offset_mode=part_offset_mode,
    )
    boxes: list[tuple[int, int, int, int]] = []
    for det, part_boxes in zip(detections, part_boxes_list):
        nose_box = part_boxes.get("nose")
        eye_boxes: list[tuple[int, int, int, int]] = []
        if "left_eye" in part_boxes:
            left_box = _trim_eye_box_against_nose(part_boxes["left_eye"], nose_box, "left")
            if left_box is not None:
                eye_boxes.append(left_box)
        if "right_eye" in part_boxes:
            right_box = _trim_eye_box_against_nose(part_boxes["right_eye"], nose_box, "right")
            if right_box is not None:
                eye_boxes.append(right_box)
        if eye_boxes:
            boxes.extend(eye_boxes)
        else:
            # 某些检测器只给人脸框不给关键点，此时退回原始检测框。
            boxes.append(tuple(map(int, det.box)))
    if not boxes:
        return "no_face", [], used_profile

    final_boxes = _draw_black_boxes(image=image, boxes=boxes)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
    return "ok", final_boxes, used_profile


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
    detector_tweaks: dict[str, tuple[
        dict[str, dict[str, tuple[float, float]]],
        dict[str, dict[str, tuple[float, float]]],
        str,
    ]] = {}
    all_detector_names = [detector_name] + fallback_detectors
    for name in all_detector_names:
        profile_name = normalize_profile_name(name)
        detectors.append(
            (
                profile_name,
                create_face_detector(
                    detector_name=name,
                    model_dir=paths.model_dir,
                    min_confidence=min_confidence_map[name],
                    detector_options=detector_options_map.get(name),
                ),
            )
        )
        detector_tweaks[profile_name] = (
            default_part_scale_by_view(profile_name=profile_name),
            default_part_offset_by_view(profile_name=profile_name),
            default_part_offset_mode(profile_name=profile_name),
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

            status, final_boxes, used_profile = process_one(
                image_path=image_path,
                output_path=output_path,
                detectors=detectors,
                detector_tweaks=detector_tweaks,
            )

            if status == "ok":
                ok_count += 1
                used_detector = used_profile or detector_name
                if output_path is not None:
                    print(f"已处理({used_detector}): {output_path}")
                else:
                    print(f"已处理({used_detector})")
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
        for _, detector in detectors:
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
            "mask_box_source=eye_boxes_or_detector.box",
        ],
    )


if __name__ == "__main__":
    main()
