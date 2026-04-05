"""
检测侧的数据准备模块。

职责：
1. 消费 detection/sam_mask 的 part mask 输出；
2. 规范化鼻子、嘴巴、眼睛和 face 区域；
3. 生成生成侧需要的 inpaint / privacy / manifest 产物；
4. depth 条件图目前保留为可选扩展，不默认启用。

虽然最终服务于生成器，但它本质上仍属于“检测与条件构建”阶段，
所以放在 detection 包里作为正式归属。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from generation.settings import (
    DatasetBuildConfig,
    GenerationPaths,
    dataclass_to_dict,
    default_dataset_build_config,
    default_generation_paths,
    dump_config_snapshot,
)
from project_utils.dataset import iter_images, sort_key
from project_utils.io import write_json, write_jsonl


VALID_STAGES = {"术前", "术后"}
MANIFEST_FIELDS = (
    "case_id",
    "view_id",
    "stage",
    "image_path",
    "original_image_path",
    "sanitized_image_path",
    "nose_mask_path",
    "face_mask_path",
    "inpaint_mask_path",
    "feather_mask_path",
    "eye_privacy_mask_path",
    "paired_pre_path",
    "paired_post_path",
    "depth_path",
    "doctor_token",
    "view_token",
    "split",
)


@dataclass
class _Issue:
    kind: str
    rel_path: str
    message: str
    details: dict[str, Any]


class _DepthEstimator:
    """离线 depth 图生成器。"""

    def __init__(self, config: DatasetBuildConfig):
        self.config = config
        self._backend = str(config.depth_backend).strip().lower()
        self._pipeline = None

    def estimate(self, image_bgr: np.ndarray) -> tuple[np.ndarray, str]:
        if self._backend == "opencv-luma":
            return self._estimate_opencv_luma(image_bgr), "opencv-luma"

        if self._backend != "transformers":
            raise ValueError(f"不支持的 depth backend: {self._backend}")

        try:
            return self._estimate_transformers(image_bgr), "transformers"
        except Exception as exc:
            if not self.config.allow_depth_fallback:
                raise
            return self._estimate_opencv_luma(image_bgr), f"opencv-luma-fallback:{exc}"

    def _estimate_transformers(self, image_bgr: np.ndarray) -> np.ndarray:
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(
                task="depth-estimation",
                model=self.config.depth_model_id,
            )
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self._pipeline(Image.fromarray(image_rgb))
        depth = result.get("depth")
        if depth is None:
            raise RuntimeError("depth-estimation 未返回 depth 字段")
        return _normalize_depth_to_rgb(np.array(depth))

    @staticmethod
    def _estimate_opencv_luma(image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (0, 0), 3.0)
        grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        pseudo_depth = cv2.normalize(
            255.0 - magnitude,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
        )
        return _normalize_depth_to_rgb(pseudo_depth)


def _normalize_depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    depth_f = depth.astype(np.float32)
    if depth_f.ndim == 3:
        depth_f = depth_f[..., 0]
    min_v = float(depth_f.min()) if depth_f.size else 0.0
    max_v = float(depth_f.max()) if depth_f.size else 0.0
    if max_v - min_v < 1e-6:
        depth_u8 = np.zeros_like(depth_f, dtype=np.uint8)
    else:
        depth_u8 = ((depth_f - min_v) / (max_v - min_v) * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)


def _is_valid_sample_path(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    return len(rel.parts) >= 3 and rel.parts[1] in VALID_STAGES


def _sample_key(path: Path, root: Path) -> tuple[str, str, str]:
    rel = path.relative_to(root)
    return rel.parts[0], rel.parts[1], rel.stem


def _relative_from_input(path: Path, root: Path) -> Path:
    return path.relative_to(root)


def _prepared_part_path(paths: GenerationPaths, part_name: str, rel_path: Path) -> Path:
    return (paths.prepared_root / "parts" / part_name / rel_path).with_suffix(".png")


def _sam_part_path(paths: GenerationPaths, part_name: str, rel_path: Path) -> Path:
    return (paths.sam_root / "parts" / part_name / rel_path).with_suffix(".png")


def _mask_output_path(root: Path, rel_path: Path) -> Path:
    return (root / rel_path).with_suffix(".png")


def _load_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return image > 0


def _mask_to_u8(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(np.uint8) * 255)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _bbox_from_bool_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """从布尔 mask 里提取最小包围框。"""
    return _bbox_from_mask(mask.astype(np.uint8))


def _clip_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int] | None:
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _expand_box(
    box: tuple[int, int, int, int] | None,
    width: int,
    height: int,
    expand_px: int,
) -> tuple[int, int, int, int] | None:
    if box is None:
        return None
    return _clip_box(box[0] - expand_px, box[1] - expand_px, box[2] + expand_px, box[3] + expand_px, width, height)


def _square_expand_box(
    box: tuple[int, int, int, int] | None,
    width: int,
    height: int,
    expand_px: int,
) -> tuple[int, int, int, int] | None:
    """把包围框扩成更大的方形头框。"""
    if box is None:
        return None
    x1, y1, x2, y2 = box
    side = max(1, x2 - x1, y2 - y1) + max(0, expand_px) * 2
    side = min(side, width, height)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    x1_new = int(round(cx - side / 2.0))
    y1_new = int(round(cy - side / 2.0))
    x1_new = max(0, min(width - side, x1_new))
    y1_new = max(0, min(height - side, y1_new))
    return int(x1_new), int(y1_new), int(x1_new + side), int(y1_new + side)


def _box_mask(box: tuple[int, int, int, int] | None, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if box is None:
        return mask
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = True
    return mask


def _build_continuous_edit_mask(
    nose_mask: np.ndarray,
    mouth_mask: np.ndarray,
    face_mask: np.ndarray,
) -> np.ndarray:
    """
    生成连续的鼻嘴编辑区域。

    规则：
    - 鼻子和嘴巴各自保留原 box mask；
    - 若两者同时存在，则在两者之间补一个桥接矩形；
    - 最终整体裁回 face 区域内。
    """
    height, width = nose_mask.shape[:2]
    edit_mask = nose_mask.astype(bool) | mouth_mask.astype(bool)
    nose_box = _bbox_from_bool_mask(nose_mask)
    mouth_box = _bbox_from_bool_mask(mouth_mask)

    if nose_box is not None and mouth_box is not None:
        nose_cy = (nose_box[1] + nose_box[3]) / 2.0
        mouth_cy = (mouth_box[1] + mouth_box[3]) / 2.0
        upper_box, lower_box = (nose_box, mouth_box) if nose_cy <= mouth_cy else (mouth_box, nose_box)

        # 桥接框直接覆盖两者之间的垂直空隙，并使用两框的水平并集，
        # 让 inpaint 时鼻子到嘴巴成为一整块连续区域。
        bridge = _clip_box(
            min(upper_box[0], lower_box[0]),
            upper_box[3],
            max(upper_box[2], lower_box[2]),
            lower_box[1],
            width,
            height,
        )
        if bridge is not None:
            edit_mask |= _box_mask(bridge, height, width)

    return edit_mask & face_mask.astype(bool)


def _dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return mask.astype(bool)
    kernel_size = max(1, radius_px * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _blur_mask_u8(mask: np.ndarray, blur_px: int) -> np.ndarray:
    if blur_px <= 0:
        return _mask_to_u8(mask)
    kernel = max(1, int(blur_px))
    if kernel % 2 == 0:
        kernel += 1
    return cv2.GaussianBlur(_mask_to_u8(mask), (kernel, kernel), 0)


def _apply_privacy_fill(image_bgr: np.ndarray, mask: np.ndarray, fill_mode: str) -> np.ndarray:
    if not mask.any():
        return image_bgr.copy()
    output = image_bgr.copy()
    if str(fill_mode).strip().lower() == "blur":
        blurred = cv2.GaussianBlur(image_bgr, (31, 31), 0)
        output[mask] = blurred[mask]
        return output
    output[mask] = 0
    return output


def _make_preview(
    original_bgr: np.ndarray,
    sanitized_bgr: np.ndarray,
    edit_mask: np.ndarray,
    privacy_mask: np.ndarray,
    depth_bgr: np.ndarray,
) -> np.ndarray:
    height, width = original_bgr.shape[:2]
    edit_vis = cv2.cvtColor(_mask_to_u8(edit_mask), cv2.COLOR_GRAY2BGR)
    privacy_vis = cv2.cvtColor(_mask_to_u8(privacy_mask), cv2.COLOR_GRAY2BGR)
    tiles = [original_bgr, sanitized_bgr, edit_vis, privacy_vis, depth_bgr]
    resized = [cv2.resize(tile, (width, height), interpolation=cv2.INTER_NEAREST) for tile in tiles]
    return np.concatenate(resized, axis=1)


def _build_case_split(case_ids: list[str], train_ratio: float, val_ratio: float, seed: int) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    ordered = list(case_ids)
    rng.shuffle(ordered)
    total = len(ordered)
    train_count = int(round(total * train_ratio))
    val_count = int(round(total * val_ratio))
    if train_count + val_count > total:
        val_count = max(0, total - train_count)
    split_map: dict[str, str] = {}
    for idx, case_id in enumerate(ordered):
        if idx < train_count:
            split_map[case_id] = "train"
        elif idx < train_count + val_count:
            split_map[case_id] = "val"
        else:
            split_map[case_id] = "test"
    return split_map


def prepare_dataset(
    paths: GenerationPaths | None = None,
    config: DatasetBuildConfig | None = None,
) -> dict[str, Any]:
    """生成标准化 manifest、mask，以及可选的 depth 条件图。"""
    paths = paths or default_generation_paths()
    config = config or default_dataset_build_config()

    images = sorted(iter_images(paths.input_root), key=lambda p: sort_key(p, paths.input_root))
    issues: list[_Issue] = []
    valid_images = [path for path in images if _is_valid_sample_path(path, paths.input_root)]
    samples_by_key = {_sample_key(path, paths.input_root): path for path in valid_images}
    case_ids = sorted({key[0] for key in samples_by_key})
    split_map = _build_case_split(case_ids, config.train_ratio, config.val_ratio, config.split_seed)

    estimator = _DepthEstimator(config) if config.enable_depth_condition else None
    manifest_rows: list[dict[str, Any]] = []
    fallback_depth_count = 0
    depth_failure_count = 0
    missing_pair_count = 0
    missing_mask_count = 0
    privacy_conflict_count = 0

    for key in sorted(samples_by_key):
        case_id, stage, view_id = key
        image_path = samples_by_key[key]
        rel_path = _relative_from_input(image_path, paths.input_root)
        paired_pre = samples_by_key.get((case_id, "术前", view_id))
        paired_post = samples_by_key.get((case_id, "术后", view_id))

        if paired_pre is None or paired_post is None:
            missing_pair_count += 1
            issues.append(_Issue("missing_pair", str(rel_path), "同 case_id/view_id 的术前或术后图缺失", {
                "case_id": case_id, "view_id": view_id, "paired_pre_exists": paired_pre is not None, "paired_post_exists": paired_post is not None,
            }))

        image = cv2.imread(str(image_path))
        if image is None:
            issues.append(_Issue("read_failed", str(rel_path), "原图读取失败", {"image_path": str(image_path)}))
            continue

        height, width = image.shape[:2]
        part_masks: dict[str, np.ndarray] = {}
        for part_name in {*config.edit_parts, *config.eye_parts, "face"}:
            upstream_path = _sam_part_path(paths, part_name, rel_path)
            raw_mask = _load_mask(upstream_path)
            if raw_mask is None:
                if part_name in {"face", *config.edit_parts}:
                    missing_mask_count += 1
                    issues.append(_Issue("missing_mask", str(rel_path), f"缺少 part mask: {part_name}", {"part": part_name, "source_path": str(upstream_path)}))
                continue
            raw_box = _bbox_from_mask(raw_mask)
            if part_name == "face" and config.use_box_mask:
                if config.face_box_force_square:
                    box = _square_expand_box(raw_box, width, height, config.face_box_expand_px)
                else:
                    box = _expand_box(raw_box, width, height, config.face_box_expand_px)
            else:
                box = _expand_box(raw_box, width, height, config.part_box_expand_px if config.use_box_mask else 0)
            standardized = _box_mask(box, height, width) if config.use_box_mask else raw_mask.astype(bool)
            part_masks[part_name] = standardized
            out_path = _prepared_part_path(paths, part_name, rel_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), _mask_to_u8(standardized))

        face_mask = part_masks.get("face", np.ones((height, width), dtype=bool))
        if config.face_clip_expand_px > 0 and part_masks.get("face") is not None:
            face_box = _bbox_from_bool_mask(face_mask)
            if config.face_box_force_square:
                expanded_face_box = _square_expand_box(face_box, width, height, config.face_clip_expand_px)
            else:
                expanded_face_box = _expand_box(face_box, width, height, config.face_clip_expand_px)
            face_mask = _box_mask(expanded_face_box, height, width)

        nose_edit_mask = part_masks.get("nose", np.zeros((height, width), dtype=bool))
        mouth_edit_mask = part_masks.get("mouth", np.zeros((height, width), dtype=bool))
        edit_mask = _build_continuous_edit_mask(
            nose_mask=nose_edit_mask,
            mouth_mask=mouth_edit_mask,
            face_mask=face_mask,
        )

        privacy_mask = np.zeros((height, width), dtype=bool)
        for part_name in config.eye_parts:
            privacy_mask |= part_masks.get(part_name, np.zeros((height, width), dtype=bool))

        if config.apply_eye_privacy_mask:
            # 眼部隐私遮挡整体避开连续的鼻嘴编辑区，
            # 防止黑块压进最终要重绘的区域。
            protected_zone = _dilate_mask(edit_mask, config.edit_safety_margin_px)
            original_privacy_pixels = int(privacy_mask.sum())
            privacy_mask &= ~protected_zone
            remaining_privacy_pixels = int(privacy_mask.sum())
            overlap_pixels = max(0, original_privacy_pixels - remaining_privacy_pixels)
            if overlap_pixels > 0:
                privacy_conflict_count += 1
                issues.append(_Issue("eye_edit_overlap", str(rel_path), "眼部隐私 mask 与鼻嘴重绘区重叠，已自动裁切", {
                    "overlap_pixels": overlap_pixels, "original_privacy_pixels": original_privacy_pixels, "remaining_privacy_pixels": remaining_privacy_pixels,
                }))

        sanitized = _apply_privacy_fill(image, privacy_mask, config.privacy_fill) if config.apply_eye_privacy_mask else image.copy()

        inpaint_mask_path = _mask_output_path(paths.inpaint_mask_root, rel_path)
        inpaint_mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(inpaint_mask_path), _mask_to_u8(edit_mask))

        feather_mask_path = _mask_output_path(paths.feather_mask_root, rel_path)
        feather_mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(feather_mask_path), _blur_mask_u8(edit_mask, config.feather_blur_px))

        privacy_mask_path = _mask_output_path(paths.privacy_mask_root, rel_path)
        privacy_mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(privacy_mask_path), _mask_to_u8(privacy_mask))

        sanitized_path = _mask_output_path(paths.sanitized_root, rel_path)
        sanitized_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sanitized_path), sanitized)

        depth_path = Path("")
        depth_image = np.zeros_like(image)
        if config.enable_depth_condition and estimator is not None:
            depth_path = _mask_output_path(paths.depth_root, rel_path)
            depth_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                depth_image, depth_backend = estimator.estimate(sanitized)
                if depth_backend.startswith("opencv-luma-fallback"):
                    fallback_depth_count += 1
                    issues.append(_Issue("depth_fallback", str(rel_path), "depth 生成使用了 fallback backend", {"backend": depth_backend}))
                cv2.imwrite(str(depth_path), depth_image)
            except Exception as exc:
                depth_failure_count += 1
                depth_path = Path("")
                depth_image = np.zeros_like(image)
                issues.append(_Issue("depth_failed", str(rel_path), "depth 图生成失败", {"error": str(exc)}))

        if config.write_preview:
            preview_path = (paths.prepared_root / "preview" / rel_path).with_suffix(".png")
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(preview_path), _make_preview(image, sanitized, edit_mask, privacy_mask, depth_image))

        row = {
            "case_id": case_id,
            "view_id": view_id,
            "stage": stage,
            "image_path": str(sanitized_path if config.apply_eye_privacy_mask else image_path),
            "original_image_path": str(image_path),
            "sanitized_image_path": str(sanitized_path),
            "nose_mask_path": str(_prepared_part_path(paths, "nose", rel_path)),
            "face_mask_path": str(_prepared_part_path(paths, "face", rel_path)),
            "inpaint_mask_path": str(inpaint_mask_path),
            "feather_mask_path": str(feather_mask_path),
            "eye_privacy_mask_path": str(privacy_mask_path),
            "paired_pre_path": None if paired_pre is None else str(_mask_output_path(paths.sanitized_root, paired_pre.relative_to(paths.input_root)) if config.apply_eye_privacy_mask else paired_pre),
            "paired_post_path": None if paired_post is None else str(_mask_output_path(paths.sanitized_root, paired_post.relative_to(paths.input_root)) if config.apply_eye_privacy_mask else paired_post),
            "depth_path": None if not str(depth_path) else str(depth_path),
            "doctor_token": config.doctor_token,
            "view_token": f"{config.view_token_prefix}{view_id}",
            "split": split_map.get(case_id, "train"),
        }
        manifest_rows.append({name: row.get(name) for name in MANIFEST_FIELDS})

    write_jsonl(paths.manifest_path, manifest_rows)
    write_jsonl(paths.issues_path, [{"kind": i.kind, "rel_path": i.rel_path, "message": i.message, "details": i.details} for i in issues])

    summary = {
        "total_images": len(valid_images),
        "manifest_rows": len(manifest_rows),
        "total_cases": len(case_ids),
        "missing_pair_count": missing_pair_count,
        "missing_mask_count": missing_mask_count,
        "privacy_conflict_count": privacy_conflict_count,
        "fallback_depth_count": fallback_depth_count,
        "depth_failure_count": depth_failure_count,
        "input_root": str(paths.input_root),
        "sam_root": str(paths.sam_root),
        "manifest_path": str(paths.manifest_path),
        "issues_path": str(paths.issues_path),
        "config": dataclass_to_dict(config),
    }
    write_json(paths.summaries_root / "prepare_dataset_summary.json", summary)
    dump_config_snapshot(paths.summaries_root / "prepare_dataset_config.json", paths=paths, dataset=config)
    return summary


def main() -> None:
    import json

    print(json.dumps(prepare_dataset(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
