"""
生成侧运行时样本索引。

不再依赖单独的 prepare_dataset 步骤，
而是直接基于原始输入目录和 detection 已落盘的产物构建 manifest。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from project_utils.dataset import iter_images, sort_key
from project_utils.io import write_jsonl
from .settings import GenerationPaths, default_generation_paths


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
    "available_pre_views",
    "available_post_views",
    "paired_views",
    "paired_view_count",
    "is_paired_view",
    "split",
)


def _is_valid_sample_path(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    return len(rel.parts) >= 3 and rel.parts[1] in VALID_STAGES


def _sample_key(path: Path, root: Path) -> tuple[str, str, str]:
    rel = path.relative_to(root)
    return rel.parts[0], rel.parts[1], rel.stem


def _png_path(root: Path, rel_path: Path) -> Path:
    return (root / rel_path).with_suffix(".png")


def part_mask_path(paths: GenerationPaths, part_name: str, rel_path: Path) -> Path:
    return _png_path(paths.sam_root / "parts" / part_name, rel_path)


def _image_output_path(root: Path, rel_path: Path) -> Path:
    return root / rel_path


def _existing_path(path: Path) -> str | None:
    return str(path) if path.exists() else None


def _view_sort_key(view_id: str) -> tuple[int, int | str]:
    text = str(view_id)
    if text.isdigit():
        return 0, int(text)
    return 1, text


def _sorted_views(views: set[str]) -> list[str]:
    return sorted((str(view_id) for view_id in views), key=_view_sort_key)


def _preferred_image_path(paths: GenerationPaths, rel_path: Path) -> tuple[Path, str | None]:
    """优先返回 eye_mask 产出的打码图；没有则回退原图。"""
    original = paths.input_root / rel_path
    sanitized = _image_output_path(paths.sanitized_root, rel_path)
    if sanitized.exists():
        return sanitized, str(sanitized)
    return original, None


def _build_case_split(case_ids: list[str], train_ratio: float, val_ratio: float, seed: int) -> dict[str, str]:
    """按病例随机切分 train / val / test，避免同一病例泄漏到不同集合。"""
    import numpy as np

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


def build_generation_manifest(
    paths: GenerationPaths | None = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    split_seed: int = 42,
    doctor_token: str = "dr_style",
    view_token_prefix: str = "view_",
) -> list[dict[str, Any]]:
    """
    基于 detection 已落盘产物构建 generation manifest。

    当前约定：
    - 训练/推理输入图优先使用 eye_mask 后的图片；
    - head 参考区域使用标准化后的 face mask；
    - 重绘目标区域使用连续的 nose + mouth inpaint_mask；
    - feather_mask 仅用于最终回贴时的羽化混合。
    """
    paths = paths or default_generation_paths()
    images = sorted(iter_images(paths.input_root), key=lambda p: sort_key(p, paths.input_root))
    valid_images = [path for path in images if _is_valid_sample_path(path, paths.input_root)]
    samples_by_key = {_sample_key(path, paths.input_root): path for path in valid_images}
    case_ids = sorted({key[0] for key in samples_by_key})
    split_map = _build_case_split(case_ids, train_ratio, val_ratio, split_seed)
    pre_views_by_case: dict[str, set[str]] = {}
    post_views_by_case: dict[str, set[str]] = {}
    for case_id, stage, view_id in samples_by_key:
        target = pre_views_by_case if stage == "术前" else post_views_by_case
        target.setdefault(case_id, set()).add(str(view_id))

    manifest_rows: list[dict[str, Any]] = []
    for key in sorted(samples_by_key):
        case_id, stage, view_id = key
        image_path = samples_by_key[key]
        rel_path = image_path.relative_to(paths.input_root)
        paired_pre = samples_by_key.get((case_id, "术前", view_id))
        paired_post = samples_by_key.get((case_id, "术后", view_id))
        resolved_image_path, sanitized_image_path = _preferred_image_path(paths, rel_path)

        paired_pre_path = None
        if paired_pre is not None:
            paired_pre_rel = paired_pre.relative_to(paths.input_root)
            paired_pre_path = str(_preferred_image_path(paths, paired_pre_rel)[0])

        paired_post_path = None
        if paired_post is not None:
            paired_post_rel = paired_post.relative_to(paths.input_root)
            paired_post_path = str(_preferred_image_path(paths, paired_post_rel)[0])

        available_pre_views = _sorted_views(pre_views_by_case.get(case_id, set()))
        available_post_views = _sorted_views(post_views_by_case.get(case_id, set()))
        # manifest 保持逐图输出，但同时附带病例级视角信息，
        # 方便 generation 侧直接处理缺视角和 paired/unpaired 逻辑。
        paired_views = [view for view in available_pre_views if view in set(available_post_views)]

        row = {
            "case_id": case_id,
            "view_id": view_id,
            "stage": stage,
            # image_path 是 generation 的默认输入图，优先取打码图。
            "image_path": str(resolved_image_path),
            "original_image_path": str(image_path),
            "sanitized_image_path": sanitized_image_path,
            "nose_mask_path": _existing_path(part_mask_path(paths, "nose", rel_path)),
            "face_mask_path": _existing_path(part_mask_path(paths, "face", rel_path)),
            "inpaint_mask_path": _existing_path(_png_path(paths.inpaint_mask_root, rel_path)),
            "feather_mask_path": _existing_path(_png_path(paths.feather_mask_root, rel_path)),
            "eye_privacy_mask_path": _existing_path(_png_path(paths.privacy_mask_root, rel_path)),
            "paired_pre_path": paired_pre_path,
            "paired_post_path": paired_post_path,
            "depth_path": _existing_path(_png_path(paths.depth_root, rel_path)),
            "doctor_token": doctor_token,
            "view_token": f"{view_token_prefix}{view_id}",
            "available_pre_views": available_pre_views,
            "available_post_views": available_post_views,
            "paired_views": paired_views,
            "paired_view_count": len(paired_views),
            "is_paired_view": str(view_id) in set(paired_views),
            "split": split_map.get(case_id, "train"),
        }
        manifest_rows.append({name: row.get(name) for name in MANIFEST_FIELDS})

    write_jsonl(paths.manifest_path, manifest_rows)
    return manifest_rows
