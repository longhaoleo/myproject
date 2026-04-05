"""
SAM 分割批处理入口（与 detect_compare / eye_mask 并列）。

流程：
1. 先用人脸检测器得到人脸框与关键点；
2. 再用 SAM（MobileSAM）按部位提示框做分割；
3. 输出：
   - 预览图（可叠加分割）；
   - 合并 mask（二值图）；
   - 分部位 mask（二值图，按 part 子目录保存）。
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from .artifacts import (
    ArtifactConfig,
    blur_mask_u8,
    build_continuous_edit_mask,
    mask_to_u8,
    standardize_part_masks,
)
from .factory import create_face_detector, supported_detector_names
from .segmentation import PART_NAMES, create_face_segmenter
from .settings import (
    default_detector_options,
    default_min_confidence_map,
    default_part_offset_mode,
    default_part_offset_by_view,
    default_part_scale_by_view,
    default_paths,
    get_view_offset_map,
    resolve_part_offset,
)
from .types import FaceDetection
from .visualize import draw_face_detections
from project_utils.dataset import iter_images, pick_random_groups, sort_key


def _write_lines(file_path: Path, lines: list[str]) -> None:
    """把文本行写入文件（用于统计落盘）。"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def _bool_mask_to_u8(mask: np.ndarray) -> np.ndarray:
    """把 bool/0-1 mask 转成 0/255 灰度图。"""
    # 把 bool/0-1 mask 转成 0/255，便于可视化与后续处理。
    return (mask.astype(np.uint8) * 255)


def _make_preview_strip(
    base_preview: np.ndarray,
    inpaint_mask: np.ndarray,
    feather_mask_u8: np.ndarray,
) -> np.ndarray:
    """把检测预览、二值 inpaint、羽化 mask 拼成一张预览图。"""
    h, w = base_preview.shape[:2]
    binary_bgr = cv2.cvtColor(mask_to_u8(inpaint_mask), cv2.COLOR_GRAY2BGR)
    feather_bgr = cv2.cvtColor(feather_mask_u8, cv2.COLOR_GRAY2BGR)
    tiles = [
        base_preview,
        cv2.resize(binary_bgr, (w, h), interpolation=cv2.INTER_NEAREST),
        cv2.resize(feather_bgr, (w, h), interpolation=cv2.INTER_NEAREST),
    ]
    return np.concatenate(tiles, axis=1)


def _cache_path(cache_root: Path, rel_path: Path) -> Path:
    """按原目录树生成检测缓存 JSON 路径。"""
    # 缓存文件路径：按原目录树保存为 json。
    return (cache_root / rel_path).with_suffix(".json")


def _save_detections_cache(
    cache_root: Path,
    rel_path: Path,
    detector_name: str,
    detections: list[FaceDetection],
) -> None:
    """把检测结果写入缓存 JSON。"""
    cache_file = _cache_path(cache_root, rel_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "detector": detector_name,
        "detections": [],
    }
    for det in detections:
        payload["detections"].append(
            {
                "box": [int(v) for v in det.box],
                "score": None if det.score is None else float(det.score),
                "landmarks": {k: [int(p[0]), int(p[1])] for k, p in det.landmarks.items()},
            }
        )

    cache_file.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def _load_detections_cache(
    cache_root: Path,
    rel_path: Path,
) -> list[FaceDetection] | None:
    """读取检测缓存 JSON，失败返回 None。"""
    cache_file = _cache_path(cache_root, rel_path)
    if not cache_file.exists():
        return None

    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        raw_dets = data.get("detections", [])
        detections: list[FaceDetection] = []
        for item in raw_dets:
            box = tuple(int(v) for v in item["box"])
            score = item.get("score", None)
            landmarks = {
                str(k): (int(v[0]), int(v[1]))
                for k, v in item.get("landmarks", {}).items()
            }
            detections.append(
                FaceDetection(
                    box=box,
                    score=None if score is None else float(score),
                    landmarks=landmarks,
                )
            )
        return detections
    except Exception:
        return None


def _merge_masks(masks: list[np.ndarray | None], h: int, w: int) -> np.ndarray:
    """把多张人脸 mask 合并成一张全图 mask。"""
    merged = np.zeros((h, w), dtype=bool)
    for m in masks:
        if m is None:
            continue
        if m.shape[:2] != (h, w):
            continue
        merged |= m.astype(bool)
    return merged


def _merge_part_masks(
    part_masks_list: list[dict[str, np.ndarray]],
    h: int,
    w: int,
    part_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """把多张人脸的“同名部位”合并成全图部位 mask。"""
    # 把多张脸的“同名部位”合并成一张全图 mask。
    merged = {name: np.zeros((h, w), dtype=bool) for name in part_names}
    for part_masks in part_masks_list:
        for part_name, mask in part_masks.items():
            if part_name not in merged:
                continue
            if mask.shape[:2] != (h, w):
                continue
            merged[part_name] |= mask.astype(bool)
    return merged


def main():
    """SAM 分割批处理入口。"""
    # 数据与输出目录（集中管理）
    paths = default_paths()
    input_root = paths.input_root
    output_root = paths.output_root_sam
    model_dir = paths.model_dir

    # 选择用于提供 SAM 提示的人脸检测器（改这一行即可切换）
    # 可选值：
    # - "mediapipe-landmarker"：关键点细，SAM 部位提示更稳
    # - "scrfd"：速度与精度均衡（5点关键点）
    # - "yolov8-face"：依赖你本地权重质量（5点关键点）
    # - "retinaface"：检测较稳，但关键点命名来自第三方实现
    # - "mtcnn"：侧脸兜底可用，但框稳定性一般
    # - "blazeface"：速度快，关键点少，部位框粗
    # - "centerface"：轻量 ONNX，关键点能力一般
    detector_name = "yolov8-face"
    # 提示来源开关：
    # - "detector"（默认）：每次直接跑检测器，再喂给 SAM
    # - "cache"：优先读取历史检测缓存（适合“前一步检测已完成，再跑 SAM”）
    prompt_source = "detector"
    # 缓存缺失时是否回退到检测器（仅 prompt_source="cache" 时生效）
    cache_miss_fallback_to_detector = True
    # 是否把本次检测结果写入缓存（建议开启）
    save_detection_cache = True
    default_min_confidence = 0.5
    min_confidence_map = default_min_confidence_map()
    detector_options_map: dict[str, dict[str, Any]] = default_detector_options(model_dir)

    # SAM 配置（轻量版）
    segmenter_name = "mobile-sam"
    tweak_method_name = "sam_mask"
    # 视角缩放/偏移：默认走 settings.py，你可以在这里做“仅本流程”的覆盖
    part_scale_by_view = default_part_scale_by_view(method_name=tweak_method_name)
    part_offset_by_view = default_part_offset_by_view(method_name=tweak_method_name)
    part_offset_mode = default_part_offset_mode(method_name=tweak_method_name)
    segmenter_options: dict[str, Any] = {
        "model_path": str(model_dir / "sam" / "mobile_sam.pt"),
        "model_type": "vit_t",
        "part_names": PART_NAMES,
        "multimask_output": True,
        # 视角独立缩放（按文件名 1~6）
        "part_scale_by_view": part_scale_by_view,
        # 视角独立偏移（按脸的比例算 dx, dy）：正值向右/向下，负值向左/向上
        "part_offset_by_view": part_offset_by_view,
        "part_offset_mode": part_offset_mode,
    }

    # 输出控制
    draw_landmarks = True
    draw_part_boxes = True
    draw_segmentation_masks = True
    draw_segmentation_parts = True
    export_part_masks = True
    export_generation_artifacts = True
    artifact_config = ArtifactConfig(
        use_box_mask=True,
        part_box_expand_px=18,
        face_box_expand_px=48,
        face_box_force_square=True,
        face_clip_expand_px=0,
        feather_blur_px=31,
    )

    # 随机抽样组数：0 表示全量
    random_group_count = 20
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
        print(f"随机抽样组数: {len(selected_groups)} / {total_groups} (seed={random_seed})")
        print("抽中组:", ", ".join(selected_groups))
    else:
        print(f"组数: {total_groups}（未启用随机抽样）")

    print("支持的检测器:", ", ".join(supported_detector_names()))
    print("SAM 提示检测器:", detector_name)
    print("SAM 提示来源:", prompt_source)

    min_conf = min_confidence_map.get(detector_name, default_min_confidence)
    detector_options = detector_options_map.get(detector_name)

    detection_cache_root = output_root / "_detection_cache" / detector_name

    need_detector = (
        prompt_source == "detector"
        or (prompt_source == "cache" and cache_miss_fallback_to_detector)
    )
    detector = None
    detector_label = detector_name
    if need_detector:
        detector = create_face_detector(
            detector_name=detector_name,
            model_dir=model_dir,
            min_confidence=min_conf,
            detector_options=detector_options,
        )
        detector_label = detector.detector_name
    segmenter = create_face_segmenter(
        segmenter_name=segmenter_name,
        model_dir=model_dir,
        segmenter_options=segmenter_options,
    )

    ok_count = 0
    no_face_count = 0
    read_failed_count = 0
    infer_failed_count = 0
    seg_failed_count = 0
    cache_hit_count = 0
    cache_miss_count = 0

    start = time.perf_counter()
    try:
        for image_path in image_paths:
            rel_path = image_path.relative_to(input_root)
            image = cv2.imread(str(image_path))
            if image is None:
                read_failed_count += 1
                print(f"[sam-mask] 跳过(读取失败): {image_path}")
                continue

            h, w = image.shape[:2]

            detections = None
            # 提示来源分支：
            # - cache：优先读历史检测结果；缺失时可回退到检测器
            # - detector：每次直接跑检测器
            if prompt_source == "cache":
                detections = _load_detections_cache(
                    cache_root=detection_cache_root,
                    rel_path=rel_path,
                )
                if detections is not None:
                    cache_hit_count += 1
                else:
                    cache_miss_count += 1
                    if cache_miss_fallback_to_detector and detector is not None:
                        try:
                            detections = detector.detect(image)
                            if save_detection_cache:
                                _save_detections_cache(
                                    cache_root=detection_cache_root,
                                    rel_path=rel_path,
                                    detector_name=detector_name,
                                    detections=detections,
                                )
                        except Exception as exc:
                            infer_failed_count += 1
                            print(f"[sam-mask] 跳过(检测失败): {image_path} | {exc}")
                            continue
                    else:
                        print(f"[sam-mask] 跳过(缓存缺失): {image_path}")
                        continue
            else:
                try:
                    detections = detector.detect(image) if detector is not None else []
                    if save_detection_cache:
                        _save_detections_cache(
                            cache_root=detection_cache_root,
                            rel_path=rel_path,
                            detector_name=detector_name,
                            detections=detections,
                        )
                except Exception as exc:
                    infer_failed_count += 1
                    print(f"[sam-mask] 跳过(检测失败): {image_path} | {exc}")
                    continue

            if not detections:
                no_face_count += 1
                print(f"[sam-mask] 未检测到人脸: {image_path}")

                preview = draw_face_detections(
                    image=image,
                    detections=[],
                    detector_name=f"{detector_label}+{segmenter_name}",
                    status_text="faces=0",
                    draw_landmarks=draw_landmarks,
                    draw_part_boxes=draw_part_boxes,
                    segmentation_masks=None,
                )
                preview_path = output_root / "preview" / rel_path
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(preview_path), preview)
                continue

            # 写入视角编号（按文件名 1~6）
            view_id = rel_path.stem
            for det in detections:
                det.view_id = view_id

            # 按视角对关键点做“根本偏移”，影响后续所有框与分割提示。
            offset_mode = str(segmenter_options.get("part_offset_mode", part_offset_mode)).strip().lower()
            offset_map = get_view_offset_map(view_id, segmenter_options.get("part_offset_by_view"))

            if offset_map:
                for det in detections:
                    x1, y1, x2, y2 = det.box
                    face_w = max(1, x2 - x1)
                    face_h = max(1, y2 - y1)

                    if det.landmarks:
                        new_landmarks = {}
                        for name, (px, py) in det.landmarks.items():
                            dx, dy = resolve_part_offset(name, offset_map, face_w, face_h, offset_mode)
                            new_landmarks[name] = (int(px + dx), int(py + dy))
                        det.landmarks = new_landmarks

            try:
                merged_masks = segmenter.segment(image=image, detections=detections)
            except Exception as exc:
                seg_failed_count += 1
                print(f"[sam-mask] 跳过(分割失败): {image_path} | {exc}")
                continue

            # 读取分部位 mask（如果分割器支持）
            part_masks_list = None
            if hasattr(segmenter, "last_part_masks"):
                cand = getattr(segmenter, "last_part_masks", None)
                if isinstance(cand, list) and len(cand) == len(detections):
                    part_masks_list = cand

            vis_masks = merged_masks
            if draw_segmentation_parts and part_masks_list is not None:
                vis_masks = part_masks_list

            preview = draw_face_detections(
                image=image,
                detections=detections,
                detector_name=f"{detector_label}+{segmenter_name}",
                status_text=f"faces={len(detections)}",
                draw_landmarks=draw_landmarks,
                draw_part_boxes=draw_part_boxes,
                segmentation_masks=vis_masks if draw_segmentation_masks else None,
                view_id=view_id,
                part_scale_by_view=segmenter_options.get("part_scale_by_view"),
                part_offset_by_view=segmenter_options.get("part_offset_by_view"),
                part_offset_mode=str(segmenter_options.get("part_offset_mode", "pixel")).strip().lower(),
            )

            # 导出全图合并 mask
            merged_all = _merge_masks(merged_masks, h=h, w=w)
            merged_path = (output_root / "merged" / rel_path).with_suffix(".png")
            merged_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(merged_path), _bool_mask_to_u8(merged_all))

            # 导出分部位 mask（可选）
            preview_to_write = preview
            if export_part_masks and part_masks_list is not None:
                part_union = _merge_part_masks(
                    part_masks_list=part_masks_list,
                    h=h,
                    w=w,
                    part_names=PART_NAMES,
                )
                if export_generation_artifacts:
                    standardized = standardize_part_masks(
                        part_masks=part_union,
                        height=h,
                        width=w,
                        config=artifact_config,
                    )
                    part_union.update(standardized)
                    face_mask = part_union.get("face", np.ones((h, w), dtype=bool))
                    nose_mask = part_union.get("nose", np.zeros((h, w), dtype=bool))
                    mouth_mask = part_union.get("mouth", np.zeros((h, w), dtype=bool))
                    inpaint_mask = build_continuous_edit_mask(
                        nose_mask=nose_mask,
                        mouth_mask=mouth_mask,
                        face_mask=face_mask,
                    )
                    feather_mask_u8 = blur_mask_u8(inpaint_mask, artifact_config.feather_blur_px)
                    inpaint_path = (output_root / "inpaint_mask" / rel_path).with_suffix(".png")
                    feather_path = (output_root / "feather_mask" / rel_path).with_suffix(".png")
                    inpaint_path.parent.mkdir(parents=True, exist_ok=True)
                    feather_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(inpaint_path), mask_to_u8(inpaint_mask))
                    cv2.imwrite(str(feather_path), feather_mask_u8)
                    preview_to_write = _make_preview_strip(
                        base_preview=preview,
                        inpaint_mask=inpaint_mask,
                        feather_mask_u8=feather_mask_u8,
                    )
                for part_name, part_mask in part_union.items():
                    part_path = (output_root / "parts" / part_name / rel_path).with_suffix(".png")
                    part_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(part_path), _bool_mask_to_u8(part_mask))

            preview_path = output_root / "preview" / rel_path
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(preview_path), preview_to_write)

            ok_count += 1
            print(f"[sam-mask] 已输出: {preview_path}")

    finally:
        if detector is not None:
            detector.close()
        segmenter.close()

    elapsed = time.perf_counter() - start
    print("\n=== sam-mask 汇总 ===")
    print(f"处理成功: {ok_count}")
    print(f"未检测到人脸: {no_face_count}")
    print(f"读取失败: {read_failed_count}")
    print(f"检测失败: {infer_failed_count}")
    print(f"分割失败: {seg_failed_count}")
    print(f"缓存命中: {cache_hit_count}")
    print(f"缓存缺失: {cache_miss_count}")
    print(f"总耗时: {elapsed:.2f}s")

    _write_lines(
        output_root / "_run_stats" / "sam_mask_summary.txt",
        [
            f"ok_count={ok_count}",
            f"no_face_count={no_face_count}",
            f"read_failed_count={read_failed_count}",
            f"infer_failed_count={infer_failed_count}",
            f"seg_failed_count={seg_failed_count}",
            f"cache_hit_count={cache_hit_count}",
            f"cache_miss_count={cache_miss_count}",
            f"elapsed_seconds={elapsed:.6f}",
            f"detector={detector_name}",
            f"segmenter={segmenter_name}",
            f"prompt_source={prompt_source}",
        ],
    )


if __name__ == "__main__":
    main()
