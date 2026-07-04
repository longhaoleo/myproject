"""
Build a unified train/test/feature-test dataset from masked deformity images and masks.

Default output:
  ~/deformity_dataset/
    Train/
      images/<case>/<stage>/<view>.<ext>
      masks/inpaint_mask/<case>/<stage>/<view>.png
      masks/feather_mask/<case>/<stage>/<view>.png
      masks/parts/{face,nose,mouth}/<case>/<stage>/<view>.png
    Test/
    Feature_test/

The split follows the before-dataset report. The script links files by default.
Use --copy if you need physical copies.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STAGES = ("术前", "术后")
STANDARD_VIEWS = tuple(str(i) for i in range(1, 7))
SPLIT_DIRS = {
    "train": "Train",
    "test": "Test",
    "feature_test": "Feature_test",
}
MASK_GROUPS = (
    ("inpaint_mask", ("inpaint_mask",)),
    ("feather_mask", ("feather_mask",)),
    ("parts/face", ("parts", "face")),
    ("parts/nose", ("parts", "nose")),
    ("parts/mouth", ("parts", "mouth")),
)
EXCLUDED_CASES = {"165"}
MODEL_TEST_CASES = {"14", "16", "79", "148", "150", "161"}
MISSING_VIEW_FEATURE_CASES = {"50", "97", "150", "184", "185"}
EXTRA_VIEW_FEATURE_CASES = {"21", "35", "36", "43", "105", "161"}


def _sort_key(text: str) -> tuple[int, int | str]:
    return (0, int(text)) if text.isdigit() else (1, text.lower())


def _iter_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        (
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: _sort_key(path.stem),
    )


def _stage_files(case_dir: Path, stage: str) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in _iter_images(case_dir / stage):
        files[path.stem] = path
    return files


def _case_info(masked_root: Path) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for case_dir in sorted((p for p in masked_root.iterdir() if p.is_dir()), key=lambda p: _sort_key(p.name)):
        pre = _stage_files(case_dir, "术前")
        post = _stage_files(case_dir, "术后")
        all_views = set(pre) | set(post)
        extra_views = sorted(all_views - set(STANDARD_VIEWS), key=_sort_key)
        missing_pre = [view for view in STANDARD_VIEWS if view not in pre]
        missing_post = [view for view in STANDARD_VIEWS if view not in post]
        feature_test = bool(extra_views or missing_pre or missing_post)
        cases.append(
            {
                "case_id": case_dir.name,
                "pre": pre,
                "post": post,
                "extra_views": extra_views,
                "missing_pre": missing_pre,
                "missing_post": missing_post,
                "feature_test": feature_test,
            }
        )
    return cases


def _target_image_path(output_root: Path, split: str, rel_path: Path) -> Path:
    return output_root / SPLIT_DIRS[split] / "images" / rel_path


def _target_mask_path(output_root: Path, split: str, mask_group: str, rel_png: Path) -> Path:
    return output_root / SPLIT_DIRS[split] / "masks" / mask_group / rel_png


def _replace_file(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def _clear_split_dirs(output_root: Path) -> None:
    for dirname in SPLIT_DIRS.values():
        path = output_root / dirname
        if path.exists():
            shutil.rmtree(path)


def _mask_path(mask_root: Path, rel_path: Path, parts: tuple[str, ...]) -> Path:
    return (mask_root.joinpath(*parts) / rel_path).with_suffix(".png")


def _link_sample(
    image_path: Path,
    masked_root: Path,
    mask_root: Path,
    output_root: Path,
    split: str,
    copy: bool,
) -> bool:
    rel_path = image_path.relative_to(masked_root)
    rel_png = rel_path.with_suffix(".png")
    mask_paths = [
        (mask_group, _mask_path(mask_root, rel_path, parts))
        for mask_group, parts in MASK_GROUPS
    ]
    if any(not path.exists() for _, path in mask_paths):
        return False

    _replace_file(image_path, _target_image_path(output_root, split, rel_path), copy=copy)
    for mask_group, src in mask_paths:
        _replace_file(src, _target_mask_path(output_root, split, mask_group, rel_png), copy=copy)
    return True


def build_dataset(
    masked_root: Path,
    mask_root: Path,
    output_root: Path,
    copy: bool,
) -> dict[str, dict[str, int]]:
    masked_root = masked_root.expanduser().resolve()
    mask_root = mask_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    cases = _case_info(masked_root)

    _clear_split_dirs(output_root)
    counts = {
        "Train": {"cases": 0, "samples": 0, "skipped_missing_masks": 0},
        "Test": {"cases": 0, "samples": 0, "skipped_missing_masks": 0},
        "Feature_test": {"cases": 0, "samples": 0, "skipped_missing_masks": 0},
    }
    touched_cases = {split_dir: set() for split_dir in counts}

    for case in cases:
        case_id = str(case["case_id"])
        if case_id in EXCLUDED_CASES:
            continue

        split_views: dict[str, set[str]] = {}
        if case_id in MODEL_TEST_CASES:
            split_views["test"] = set(STANDARD_VIEWS)
        elif case_id not in MISSING_VIEW_FEATURE_CASES:
            split_views["train"] = set(STANDARD_VIEWS)

        if case_id in MISSING_VIEW_FEATURE_CASES:
            split_views.setdefault("feature_test", set()).update(STANDARD_VIEWS)
        if case_id in EXTRA_VIEW_FEATURE_CASES:
            extra_views = case["extra_views"]
            assert isinstance(extra_views, list)
            split_views.setdefault("feature_test", set()).update(str(view) for view in extra_views)

        for split, views in split_views.items():
            split_dir = SPLIT_DIRS[split]
            case_has_sample = False
            for stage, files in (("术前", case["pre"]), ("术后", case["post"])):
                stage_files = files
                assert isinstance(stage_files, dict)
                for view_id in sorted(views, key=_sort_key):
                    image_path = stage_files.get(view_id)
                    if image_path is None:
                        continue
                    if _link_sample(
                        image_path=image_path,
                        masked_root=masked_root,
                        mask_root=mask_root,
                        output_root=output_root,
                        split=split,
                        copy=copy,
                    ):
                        counts[split_dir]["samples"] += 1
                        case_has_sample = True
                    else:
                        counts[split_dir]["skipped_missing_masks"] += 1
            if case_has_sample:
                touched_cases[split_dir].add(case_id)

    for split_dir, case_ids in touched_cases.items():
        counts[split_dir]["cases"] = len(case_ids)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified Train/Test/Feature_test deformity dataset.")
    parser.add_argument("--masked-root", type=Path, default=Path("~/deformity_masked"))
    parser.add_argument("--mask-root", type=Path, default=Path("~/deformity_sam_mask"))
    parser.add_argument("--output-root", type=Path, default=Path("~/deformity_dataset"))
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating symlinks.")
    args = parser.parse_args()

    counts = build_dataset(
        masked_root=args.masked_root,
        mask_root=args.mask_root,
        output_root=args.output_root,
        copy=args.copy,
    )
    for split_name, item in counts.items():
        print(
            f"{split_name}: cases={item['cases']} "
            f"samples={item['samples']} "
            f"skipped_missing_masks={item['skipped_missing_masks']}"
        )


if __name__ == "__main__":
    main()
