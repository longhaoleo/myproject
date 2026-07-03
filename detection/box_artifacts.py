"""
基于检测框/关键点直接导出 generation 需要的矩形 mask 产物。

用途：
- 不跑 SAM；
- 直接用 `build_part_prompt_boxes` 生成 face / nose / mouth 等部位框；
- 落盘到与 `sam_mask.py` 相同的目录结构，供 generation 侧复用。
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from .artifacts import (
    ArtifactConfig,
    blur_mask_u8,
    box_mask,
    build_continuous_edit_mask,
    mask_to_u8,
    standardize_part_masks,
)
from .factory import create_face_detector
from .part_boxes import PART_NAMES, build_part_prompt_boxes
from .settings import (
    default_detector_options,
    default_min_confidence_map,
    default_part_offset_mode,
    default_part_offset_by_view,
    default_part_scale_by_view,
    default_paths,
)
from project_utils.dataset import iter_images, sort_key


def _write_lines(file_path: Path, lines: list[str]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_part_masks_from_boxes(
    image_shape: tuple[int, int],
    detections,
    view_id: str,
    part_scale_by_view: dict[str, dict[str, tuple[float, float]]],
    part_offset_by_view: dict[str, dict[str, tuple[float, float]]],
    part_offset_mode: str,
) -> dict[str, np.ndarray]:
    """把所有人脸的部位框合并成 part -> bool mask。"""
    h, w = image_shape
    part_union = {part_name: np.zeros((h, w), dtype=bool) for part_name in PART_NAMES}
    for det in detections:
        det.view_id = view_id
        part_boxes = build_part_prompt_boxes(
            det=det,
            image_w=w,
            image_h=h,
            part_names=PART_NAMES,
            view_id=view_id,
            part_scale_by_view=part_scale_by_view,
            part_offset_by_view=part_offset_by_view,
            part_offset_mode=part_offset_mode,
        )
        for part_name, box in part_boxes.items():
            part_union[part_name] |= box_mask(box, h, w)
    return {name: mask for name, mask in part_union.items() if np.any(mask)}


def _save_generation_artifacts(
    output_root: Path,
    rel_path: Path,
    part_masks: dict[str, np.ndarray],
    image_shape: tuple[int, int],
    artifact_config: ArtifactConfig,
) -> None:
    h, w = image_shape
    standardized = standardize_part_masks(
        part_masks=part_masks,
        height=h,
        width=w,
        config=artifact_config,
    )
    part_masks = {**part_masks, **standardized}

    face_mask = part_masks.get("face", np.ones((h, w), dtype=bool))
    nose_mask = part_masks.get("nose", np.zeros((h, w), dtype=bool))
    mouth_mask = part_masks.get("mouth", np.zeros((h, w), dtype=bool))
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

    for part_name, part_mask in part_masks.items():
        part_path = (output_root / "parts" / part_name / rel_path).with_suffix(".png")
        part_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(part_path), mask_to_u8(part_mask))


def main() -> None:
    """批量生成矩形 mask 产物。"""
    paths = default_paths()
    input_root = paths.input_root
    output_root = paths.output_root_sam
    model_dir = paths.model_dir

    detector_name = "scrfd"
    fallback_detectors = [
        "yolov8-face",
        "blazeface",
    ]
    tweak_method_name = "sam_mask"
    min_confidence_map = default_min_confidence_map()
    detector_options_map: dict[str, dict[str, Any]] = default_detector_options(model_dir)
    part_scale_by_view = default_part_scale_by_view(method_name=tweak_method_name)
    part_offset_by_view = default_part_offset_by_view(method_name=tweak_method_name)
    part_offset_mode = default_part_offset_mode(method_name=tweak_method_name)
    artifact_config = ArtifactConfig(
        use_box_mask=True,
        part_box_expand_px=18,
        face_box_expand_px=48,
        face_box_force_square=True,
        face_clip_expand_px=0,
        feather_blur_px=31,
    )

    image_paths = sorted(iter_images(input_root), key=lambda p: sort_key(p, input_root))
    if not image_paths:
        print(f"未找到图片: {input_root}")
        return

    detectors = []
    for name in [detector_name] + fallback_detectors:
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
            print(f"[box-artifacts] 检测器不可用，已跳过: {name} | {exc}")
    if not detectors:
        print("没有可用检测器，请先检查 model/ 权重和 Python 环境。")
        return

    ok_count = 0
    no_face_count = 0
    no_part_count = 0
    read_failed_count = 0
    infer_failed_count = 0
    start = time.perf_counter()
    try:
        for image_path in image_paths:
            rel_path = image_path.relative_to(input_root)
            image = cv2.imread(str(image_path))
            if image is None:
                read_failed_count += 1
                print(f"[box-artifacts] 跳过(读取失败): {image_path}")
                continue

            detections = []
            for detector in detectors:
                try:
                    detections = detector.detect(image)
                except Exception:
                    continue
                if detections:
                    break
            if not detections:
                no_face_count += 1
                print(f"[box-artifacts] 跳过(未检测到人脸): {image_path}")
                continue

            h, w = image.shape[:2]
            view_id = rel_path.stem
            try:
                part_masks = _make_part_masks_from_boxes(
                    image_shape=(h, w),
                    detections=detections,
                    view_id=view_id,
                    part_scale_by_view=part_scale_by_view,
                    part_offset_by_view=part_offset_by_view,
                    part_offset_mode=part_offset_mode,
                )
            except Exception as exc:
                infer_failed_count += 1
                print(f"[box-artifacts] 跳过(部位框失败): {image_path} | {exc}")
                continue
            if not {"face", "nose", "mouth"}.issubset(part_masks):
                no_part_count += 1
                print(f"[box-artifacts] 跳过(缺少 face/nose/mouth): {image_path}")
                continue

            _save_generation_artifacts(
                output_root=output_root,
                rel_path=rel_path,
                part_masks=part_masks,
                image_shape=(h, w),
                artifact_config=artifact_config,
            )
            ok_count += 1
            print(f"[box-artifacts] 已输出: {output_root / 'parts' / 'face' / rel_path.with_suffix('.png')}")
    finally:
        for detector in detectors:
            detector.close()

    elapsed = time.perf_counter() - start
    print("\n=== box-artifacts 汇总 ===")
    print(f"处理成功: {ok_count}")
    print(f"未检测到人脸: {no_face_count}")
    print(f"缺少部位框: {no_part_count}")
    print(f"读取失败: {read_failed_count}")
    print(f"部位框失败: {infer_failed_count}")
    print(f"总耗时: {elapsed:.2f}s")

    _write_lines(
        output_root / "_run_stats" / "box_artifacts_summary.txt",
        [
            f"ok_count={ok_count}",
            f"no_face_count={no_face_count}",
            f"no_part_count={no_part_count}",
            f"read_failed_count={read_failed_count}",
            f"infer_failed_count={infer_failed_count}",
            f"elapsed_seconds={elapsed:.6f}",
            f"primary_detector={detector_name}",
            f"fallback_detectors={','.join(fallback_detectors)}",
            f"tweak_method_name={tweak_method_name}",
        ],
    )


if __name__ == "__main__":
    main()
