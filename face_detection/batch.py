"""
多模型人脸检测效果批量对比入口。

输出：
- 按“检测器名称/原始目录树”保存可视化图片；
- 每张图画人脸框、置信度和关键点（若模型提供）。
"""

import random
from pathlib import Path

import cv2

from .dataset_utils import iter_images, sort_key
from .factory import create_face_detector, supported_detector_names
from .visualize import draw_face_detections


def _format_faces(detections) -> str:
    # 日志用：输出简短框坐标。
    if not detections:
        return "[]"
    boxes = [f"({d.box[0]},{d.box[1]},{d.box[2]},{d.box[3]})" for d in detections]
    return "[" + ", ".join(boxes) + "]"


def _person_group_name(path: Path, root: Path) -> str:
    # 约定第一层子目录为“组”（例如 deformity/68/... -> 组名 68）。
    rel = path.relative_to(root)
    return rel.parts[0] if rel.parts else ""


def _group_sort_key(name: str):
    # 组名自然排序：纯数字按数值，其余按文本。
    return (0, int(name)) if name.isdigit() else (1, name.lower())


def pick_random_groups(
    image_paths: list[Path],
    input_root: Path,
    random_group_count: int,
    random_seed: int,
) -> tuple[list[Path], list[str], int]:
    """
    从所有组中随机抽样若干组，只返回这些组内的图片。

    返回：
    - sampled_paths: 抽样后的图片列表（保留原有排序）
    - selected_groups: 抽中的组名（已排序，便于日志查看）
    - total_groups: 输入目录中可用组总数
    """
    groups: set[str] = set()
    for path in image_paths:
        group = _person_group_name(path, input_root)
        if group:
            groups.add(group)
    total_groups = len(groups)
    if random_group_count <= 0 or random_group_count >= total_groups:
        all_groups = sorted(groups, key=_group_sort_key)
        return image_paths, all_groups, total_groups

    rng = random.Random(random_seed)
    selected_groups = rng.sample(list(groups), k=random_group_count)
    selected_set = set(selected_groups)

    sampled_paths = [
        path for path in image_paths if _person_group_name(path, input_root) in selected_set
    ]
    selected_groups = sorted(selected_groups, key=_group_sort_key)
    return sampled_paths, selected_groups, total_groups


def run_one_detector(
    detector_name: str,
    image_paths: list[Path],
    input_root: Path,
    output_root: Path,
    model_dir: Path,
    min_confidence: float,
    yolov8_model_path: Path | None,
    draw_landmarks: bool,
    draw_part_boxes: bool,
    print_box_info: bool,
):
    # 运行单个检测器并输出可视化结果。
    detector = create_face_detector(
        detector_name=detector_name,
        model_dir=model_dir,
        min_confidence=min_confidence,
        yolov8_model_path=yolov8_model_path,
    )

    ok_count = 0
    no_face_count = 0
    read_failed_count = 0
    infer_failed_count = 0

    detector_key = detector_name.lower().replace("_", "-")
    detector_output_root = output_root / detector_key

    try:
        for image_path in image_paths:
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

            if detections:
                ok_count += 1
            else:
                no_face_count += 1

            vis = draw_face_detections(
                image=image,
                detections=detections,
                detector_name=detector.detector_name,
                status_text=f"faces={len(detections)}",
                draw_landmarks=draw_landmarks,
                draw_part_boxes=draw_part_boxes,
            )

            output_path = detector_output_root / image_path.relative_to(input_root)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), vis)

            print(f"[{detector_name}] 已输出: {output_path}")
            if print_box_info:
                print(f"[{detector_name}] 人脸框: {_format_faces(detections)}")
    finally:
        detector.close()

    print(
        f"[{detector_name}] 完成：检测到人脸 {ok_count} 张，"
        f"未检测到人脸 {no_face_count} 张，读取失败 {read_failed_count} 张，"
        f"推理失败 {infer_failed_count} 张。"
    )


def main():
    # 输入图片根目录
    input_root = Path("~/datasets/deformity").expanduser().resolve()

    # 可视化输出根目录（每个检测器一个子目录）
    output_root = Path("~/datasets/deformity_face_detection_preview").expanduser().resolve()

    # 模型目录（缓存 + 本地权重）
    model_dir = Path(__file__).resolve().parent.parent / "model"

    # 要运行的检测器列表（按顺序执行）
    detectors_to_run = [
        # "mtcnn",
        # "retinaface",
        # "scrfd",
        # "blazeface",
        # "mediapipe-landmarker",
        # "yolov8-face",
        "centerface",
    ]

    # 每个检测器置信度阈值（未列出的用默认值）
    default_min_confidence = 0.5
    min_confidence_map = {
        "mtcnn": 0.7,
        "retinaface": 0.6,
        "scrfd": 0.6,
        "blazeface": 0.5,
        "mediapipe-landmarker": 0.3,
        "yolov8-face": 0.4,
        "centerface": 0.5,
    }

    # YOLOv8-Face（改造版）权重路径
    yolov8_model_path = model_dir / "yolov8_face.pt"

    # 是否绘制关键点（模型不提供时会自动跳过）
    draw_landmarks = True
    # 是否绘制眼睛/鼻子/嘴巴部位框（依赖关键点）
    draw_part_boxes = True
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
        input_root=input_root,
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

    for detector_name in detectors_to_run:
        min_conf = min_confidence_map.get(detector_name, default_min_confidence)
        try:
            run_one_detector(
                detector_name=detector_name,
                image_paths=image_paths,
                input_root=input_root,
                output_root=output_root,
                model_dir=model_dir,
                min_confidence=min_conf,
                yolov8_model_path=yolov8_model_path,
                draw_landmarks=draw_landmarks,
                draw_part_boxes=draw_part_boxes,
                print_box_info=print_box_info,
            )
        except Exception as exc:
            print(f"[{detector_name}] 初始化或运行失败: {exc}")
            print(f"[{detector_name}] 已跳过，继续下一个检测器。")


if __name__ == "__main__":
    main()
