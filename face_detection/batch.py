"""
多模型人脸检测效果批量对比入口。

输出：
- 按“检测器名称/原始目录树”保存可视化图片；
- 每张图画人脸框、置信度和关键点（若模型提供）。
"""

from pathlib import Path
import time
from typing import Any

import cv2

from .dataset_utils import iter_images, pick_random_groups, sort_key
from .factory import create_face_detector, supported_detector_names
from .segmentation import FaceSegmenter, create_face_segmenter
from .settings import default_detector_options, default_min_confidence_map, default_paths
from .visualize import draw_face_detections


def _format_faces(detections) -> str:
    """把检测框列表格式化成易读字符串。"""
    # 日志用：输出简短框坐标。
    if not detections:
        return "[]"
    boxes = [f"({d.box[0]},{d.box[1]},{d.box[2]},{d.box[3]})" for d in detections]
    return "[" + ", ".join(boxes) + "]"


def _write_lines(file_path: Path, lines: list[str]) -> None:
    """把文本行写入文件。"""
    # 把文本行写入文件。
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def run_one_detector(
    detector_name: str,
    image_paths: list[Path],
    input_root: Path,
    output_root: Path,
    model_dir: Path,
    min_confidence: float,
    detector_options: dict[str, Any] | None,
    segmenter: FaceSegmenter,
    draw_landmarks: bool,
    draw_part_boxes: bool,
    draw_segmentation_masks: bool,
    draw_segmentation_parts: bool,
    print_box_info: bool,
):
    """运行单个检测器并输出可视化结果。"""
    # 每次仅实例化一个检测器，避免多模型同时占用内存。
    detector = create_face_detector(
        detector_name=detector_name,
        model_dir=model_dir,
        min_confidence=min_confidence,
        detector_options=detector_options,
    )

    ok_count = 0
    no_face_count = 0
    read_failed_count = 0
    infer_failed_count = 0

    detector_key = detector_name.lower().replace("_", "-")
    detector_output_root = output_root / detector_key

    detector_start = time.perf_counter()
    try:
        for image_path in image_paths:
            rel_path = image_path.relative_to(input_root)
            image = cv2.imread(str(image_path))
            if image is None:
                read_failed_count += 1
                print(f"[{detector_name}] 跳过(读取失败): {image_path}")
                continue

            try:
                detections = detector.detect(image)
            except Exception as exc:
                infer_failed_count += 1
                print(f"[{detector_name}] 跳过(推理失败): {image_path} | {exc}")
                continue

            # 分割接口：开启后使用 SAM 对每张脸生成 mask（关闭则完全不影响检测流程）。
            segmentation_masks = [None for _ in detections]
            if draw_segmentation_masks and detections:
                try:
                    segmentation_masks = segmenter.segment(
                        image=image,
                        detections=detections,
                    )
                    # 如果需要查看“按部位分割”效果，优先使用分割器产出的部位 mask。
                    if draw_segmentation_parts and hasattr(segmenter, "last_part_masks"):
                        part_masks = getattr(segmenter, "last_part_masks", None)
                        if isinstance(part_masks, list) and len(part_masks) == len(detections):
                            segmentation_masks = part_masks
                except Exception as exc:
                    print(f"[{detector_name}] 分割失败，已退化为仅画框: {image_path} | {exc}")
                    segmentation_masks = [None for _ in detections]

            if detections:
                ok_count += 1
            else:
                no_face_count += 1
                print(f"[{detector_name}] 未检测到人脸: {image_path}")

            # 可视化层：框 + 关键点 +（可选）部位框 +（可选）分割叠加。
            vis = draw_face_detections(
                image=image,
                detections=detections,
                detector_name=detector.detector_name,
                status_text=f"faces={len(detections)}",
                draw_landmarks=draw_landmarks,
                draw_part_boxes=draw_part_boxes,
                segmentation_masks=segmentation_masks,
            )

            output_path = detector_output_root / rel_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), vis)

            print(f"[{detector_name}] 已输出: {output_path}")
            if print_box_info:
                print(f"[{detector_name}] 人脸框: {_format_faces(detections)}")
    finally:
        detector.close()

    elapsed = time.perf_counter() - detector_start
    print(f"[{detector_name}] 完成：检测到人脸 {ok_count} 张，耗时 {elapsed:.2f}s。")
    print(f"[{detector_name}] 未检测到人脸 {no_face_count} 张。")
    print(f"[{detector_name}] 读取失败 {read_failed_count} 张。")
    print(f"[{detector_name}] 推理失败 {infer_failed_count} 张。")
    return {
        "detector": detector_name,
        "elapsed_seconds": elapsed,
        "ok_count": ok_count,
        "no_face_count": no_face_count,
        "read_failed_count": read_failed_count,
        "infer_failed_count": infer_failed_count,
    }


def main():
    """批量运行多个检测器并输出对比结果。"""
    # 输入/输出/模型路径（集中管理）
    paths = default_paths()
    input_root = paths.input_root
    output_root = paths.output_root_detect
    model_dir = paths.model_dir

    # 要运行的检测器列表（按顺序执行，便于横向比较）
    detectors_to_run = [
        # "retinaface",
        # "mtcnn",
        # "scrfd",
        # "blazeface",
        # "mediapipe-landmarker",
        # "yolov8-face",
        # "centerface",
    ]

    # 每个检测器置信度阈值
    default_min_confidence = 0.5
    min_confidence_map = default_min_confidence_map()

    # 检测器专属参数（通用扩展入口）
    # 后续新增检测器时，优先在这里加参数，不要再添加“单独函数参数”。
    detector_options_map: dict[str, dict[str, Any]] = default_detector_options(model_dir)

    # 分割接口开关（默认关闭，不影响现有检测流程）
    enable_segmentation_refine = False
    segmenter_name = "mobile-sam"
    segmenter_options: dict[str, Any] = {
        # 轻量 SAM 权重默认位置
        "model_path": str(model_dir / "sam" / "mobile_sam.pt"),
        # MobileSAM 一般使用 vit_t
        "model_type": "vit_t",
        # part_names 决定“按部位分割”时会提示哪些区域
        "part_names": ("face", "left_eye", "right_eye", "nose", "mouth"),
    }

    # 是否绘制关键点（模型不提供时会自动跳过）
    draw_landmarks = True
    # 是否绘制眼睛/鼻子/嘴巴部位框（依赖关键点）
    draw_part_boxes = True
    # 是否绘制分割 mask（需要 enable_segmentation_refine=True）
    draw_segmentation_masks = False
    # 是否按部位显示分割（face/eyes/nose/mouth 分别着色）
    draw_segmentation_parts = False
    # 是否打印每张图的人脸框坐标
    print_box_info = False

    # 随机抽样组数（按 input_root 第一层子目录计组）：
    # 0 表示不抽样，处理全部组。
    random_group_count = 20
    # 随机种子：保证每次抽样可复现；想每次不一样就改成当前时间戳。
    random_seed = 42

    image_paths = sorted(iter_images(input_root), key=lambda p: sort_key(p, input_root))
    if not image_paths:
        print(f"未找到图片: {input_root}")
        return

    image_paths, selected_groups, total_groups = pick_random_groups(
        image_paths=image_paths,
        root=input_root,
        random_group_count=random_group_count,
        random_seed=random_seed,
    )

    if random_group_count > 0:
        print(
            f"随机抽样组数: {len(selected_groups)} / {total_groups} "
            f"(seed={random_seed})"
        )
        print("抽中组:", ", ".join(selected_groups))
    else:
        print(f"组数: {total_groups}（未启用随机抽样）")

    print("支持的检测器:", ", ".join(supported_detector_names()))
    print("本次运行检测器:", ", ".join(detectors_to_run))

    # 创建全局分割器：如果关闭分割，就走 Noop，不会改变现有行为。
    segmenter = create_face_segmenter(
        segmenter_name=segmenter_name if enable_segmentation_refine else "none",
        model_dir=model_dir,
        segmenter_options=segmenter_options,
    )

    total_start = time.perf_counter()
    stats_by_detector: list[dict[str, float | int | str]] = []

    try:
        for detector_name in detectors_to_run:
            min_conf = min_confidence_map.get(detector_name, default_min_confidence)
            detector_options = detector_options_map.get(detector_name)
            try:
                stats = run_one_detector(
                    detector_name=detector_name,
                    image_paths=image_paths,
                    input_root=input_root,
                    output_root=output_root,
                    model_dir=model_dir,
                    min_confidence=min_conf,
                    detector_options=detector_options,
                    segmenter=segmenter,
                    draw_landmarks=draw_landmarks,
                    draw_part_boxes=draw_part_boxes,
                    draw_segmentation_masks=draw_segmentation_masks,
                    draw_segmentation_parts=draw_segmentation_parts,
                    print_box_info=print_box_info,
                )
                stats_by_detector.append(stats)
            except Exception as exc:
                print(f"[{detector_name}] 初始化或运行失败: {exc}")
                print(f"[{detector_name}] 已跳过，继续下一个检测器。")
    finally:
        segmenter.close()

    total_elapsed = time.perf_counter() - total_start
    print("\n=== 运行耗时统计 ===")
    print(f"图片总数: {len(image_paths)}")
    print(f"检测器数量: {len(detectors_to_run)}")
    print(f"总耗时: {total_elapsed:.2f}s")
    if image_paths:
        print(f"平均每张图片耗时(总耗时/图片数): {total_elapsed / len(image_paths):.4f}s")
    for item in stats_by_detector:
        name = str(item["detector"])
        print(f"- {name}: {float(item['elapsed_seconds']):.2f}s")
    print("="*12)

    run_stats_root = output_root / "_run_stats"
    # 每个模型的最终时间与漏检/失败统计
    summary_lines = []
    for item in stats_by_detector:
        summary_lines.append(
            f"{item['detector']}\t"
            f"elapsed={float(item['elapsed_seconds']):.6f}\t"
            f"ok={int(item['ok_count'])}\t"
            f"no_face={int(item['no_face_count'])}\t"
            f"read_failed={int(item['read_failed_count'])}\t"
            f"infer_failed={int(item['infer_failed_count'])}"
        )
    _write_lines(run_stats_root / "detect_compare_summary.txt", summary_lines)


if __name__ == "__main__":
    main()
