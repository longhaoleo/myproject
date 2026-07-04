"""
Create case-level train/val/future-test splits for the deformity dataset.

The script only writes split manifests; it does not copy or move images.
Default policy follows the existing data-cleaning report:
- split by case_id, never by individual image;
- complete paired six-view cases are eligible for train/val;
- cases missing pre/post or standard views are reserved for future_test;
- non-standard extra views are recorded, but excluded from the main manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STAGES = ("术前", "术后")
STANDARD_VIEWS = tuple(str(i) for i in range(1, 7))


def _case_sort_key(case_id: str) -> tuple[int, int | str]:
    return (0, int(case_id)) if case_id.isdigit() else (1, case_id.lower())


def _iter_image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        (
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: _case_sort_key(path.stem),
    )


def _view_id(path: Path) -> str:
    return path.stem


def _stage_summary(case_dir: Path, stage: str) -> dict[str, Any]:
    files = _iter_image_files(case_dir / stage)
    standard_views = sorted(
        {_view_id(path) for path in files if _view_id(path) in STANDARD_VIEWS},
        key=lambda item: int(item),
    )
    extra_files = [path for path in files if _view_id(path) not in STANDARD_VIEWS]
    missing_views = [view for view in STANDARD_VIEWS if view not in set(standard_views)]
    return {
        "files": files,
        "standard_views": standard_views,
        "missing_views": missing_views,
        "extra_files": extra_files,
    }


def _analyze_cases(input_root: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for case_dir in sorted((p for p in input_root.iterdir() if p.is_dir()), key=lambda p: _case_sort_key(p.name)):
        pre = _stage_summary(case_dir, "术前")
        post = _stage_summary(case_dir, "术后")
        reasons: list[str] = []
        if not pre["files"]:
            reasons.append("missing_pre")
        if not post["files"]:
            reasons.append("missing_post")
        if pre["files"] and pre["missing_views"]:
            reasons.append("missing_pre_views")
        if post["files"] and post["missing_views"]:
            reasons.append("missing_post_views")
        if pre["extra_files"] or post["extra_files"]:
            reasons.append("extra_nonstandard_views")

        has_complete_pair = (
            not pre["missing_views"]
            and not post["missing_views"]
            and bool(pre["files"])
            and bool(post["files"])
        )
        cases.append(
            {
                "case_id": case_dir.name,
                "case_dir": case_dir,
                "pre": pre,
                "post": post,
                "has_complete_pair": has_complete_pair,
                "reasons": reasons,
            }
        )
    return cases


def _assign_splits(
    cases: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
    include_missing_views_in_main: bool,
) -> dict[str, str]:
    eligible: list[str] = []
    split_by_case: dict[str, str] = {}
    for item in cases:
        case_id = str(item["case_id"])
        has_pre = bool(item["pre"]["files"])
        has_post = bool(item["post"]["files"])
        has_missing_views = bool(item["pre"]["missing_views"] or item["post"]["missing_views"])
        is_main_eligible = has_pre and has_post and (include_missing_views_in_main or not has_missing_views)
        if is_main_eligible:
            eligible.append(case_id)
        else:
            split_by_case[case_id] = "future_test"

    rng = random.Random(seed)
    shuffled = list(eligible)
    rng.shuffle(shuffled)
    val_count = int(round(len(shuffled) * val_ratio))
    if len(shuffled) > 1:
        val_count = max(1, min(len(shuffled) - 1, val_count))
    val_cases = set(shuffled[:val_count])
    for case_id in eligible:
        split_by_case[case_id] = "val" if case_id in val_cases else "train"
    return split_by_case


def _case_rows(cases: list[dict[str, Any]], split_by_case: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in cases:
        pre = item["pre"]
        post = item["post"]
        rows.append(
            {
                "case_id": item["case_id"],
                "split": split_by_case[item["case_id"]],
                "has_complete_pair": item["has_complete_pair"],
                "reasons": ";".join(item["reasons"]),
                "pre_image_count": len(pre["files"]),
                "post_image_count": len(post["files"]),
                "pre_standard_views": ",".join(pre["standard_views"]),
                "post_standard_views": ",".join(post["standard_views"]),
                "missing_pre_views": ",".join(pre["missing_views"]),
                "missing_post_views": ",".join(post["missing_views"]),
                "extra_pre_count": len(pre["extra_files"]),
                "extra_post_count": len(post["extra_files"]),
            }
        )
    return rows


def _image_rows(cases: list[dict[str, Any]], input_root: Path, split_by_case: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in cases:
        case_id = str(item["case_id"])
        split = split_by_case[case_id]
        for stage in STAGES:
            for path in item["pre" if stage == "术前" else "post"]["files"]:
                view_id = _view_id(path)
                is_standard_view = view_id in STANDARD_VIEWS
                include_in_main = is_standard_view and split in {"train", "val"}
                rows.append(
                    {
                        "case_id": case_id,
                        "split": split,
                        "stage": stage,
                        "view_id": view_id,
                        "is_standard_view": is_standard_view,
                        "include_in_main": include_in_main,
                        "rel_path": str(path.relative_to(input_root)),
                        "path": str(path),
                    }
                )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_case_list(path: Path, rows: list[dict[str, Any]], split: str) -> None:
    case_ids = [str(row["case_id"]) for row in rows if row["split"] == split]
    path.write_text("\n".join(case_ids) + ("\n" if case_ids else ""), encoding="utf-8")


def build_splits(
    input_root: Path,
    output_root: Path,
    val_ratio: float,
    seed: int,
    include_missing_views_in_main: bool,
) -> dict[str, Any]:
    input_root = input_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    cases = _analyze_cases(input_root)
    split_by_case = _assign_splits(
        cases=cases,
        val_ratio=val_ratio,
        seed=seed,
        include_missing_views_in_main=include_missing_views_in_main,
    )
    case_rows = _case_rows(cases, split_by_case)
    image_rows = _image_rows(cases, input_root, split_by_case)

    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "cases.csv", case_rows)
    _write_jsonl(output_root / "images.jsonl", image_rows)
    for split in ("train", "val", "future_test"):
        _write_case_list(output_root / f"{split}_cases.txt", case_rows, split)

    split_counts: dict[str, int] = {}
    image_counts: dict[str, int] = {}
    for row in case_rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    for row in image_rows:
        image_counts[row["split"]] = image_counts.get(row["split"], 0) + 1

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "case_count": len(case_rows),
        "image_count": len(image_rows),
        "split_case_counts": split_counts,
        "split_image_counts": image_counts,
        "val_ratio": val_ratio,
        "seed": seed,
        "include_missing_views_in_main": include_missing_views_in_main,
        "policy": {
            "unit": "case_id",
            "train_val_source": "cases with both pre and post; standard views complete unless include_missing_views_in_main is set",
            "future_test_source": "cases missing pre/post or reserved missing standard views",
            "extra_views": "recorded in images.jsonl but excluded from main train/val rows",
        },
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Split deformity cases into train/val/future_test.")
    parser.add_argument("--input-root", type=Path, default=Path("~/deformity"))
    parser.add_argument("--output-root", type=Path, default=Path("~/deformity_splits"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-missing-views-in-main",
        action="store_true",
        help="Keep paired cases with missing standard views in train/val instead of future_test.",
    )
    args = parser.parse_args()

    summary = build_splits(
        input_root=args.input_root,
        output_root=args.output_root,
        val_ratio=args.val_ratio,
        seed=args.seed,
        include_missing_views_in_main=args.include_missing_views_in_main,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
