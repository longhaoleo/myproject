"""
生成侧评估与验收。

当前聚焦工程可用性：
- 配对是否完整
- 核心条件是否齐全
- 推理输出是否落盘
- 三联预览是否可复查
- v1 轻量一致性指标是否稳定
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from project_utils.io import write_json
from .index import build_generation_manifest
from .settings import GenerationPaths, default_generation_paths, dump_config_snapshot


def _resolve_rel_path(paths: GenerationPaths, sample_path: str | Path) -> Path:
    path = Path(sample_path)
    for root in (paths.sanitized_root, paths.input_root, paths.inference_root):
        try:
            return path.relative_to(root)
        except ValueError:
            continue
    raise ValueError(f"无法解析相对路径: {path}")


def _png_path(root: Path, rel_path: Path) -> Path:
    return (root / rel_path).with_suffix(".png")


def _concat_triptych(left: np.ndarray, middle: np.ndarray, right: np.ndarray) -> np.ndarray:
    height, width = left.shape[:2]
    tiles = [left, middle, right]
    resized = [cv2.resize(tile, (width, height), interpolation=cv2.INTER_AREA) for tile in tiles]
    return np.concatenate(resized, axis=1)


def _load_mask(path: str | Path) -> np.ndarray | None:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return mask > 0


def _dilate_mask(mask: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0


def _region_embedding(image_bgr: np.ndarray, region_mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(region_mask)
    if xs.size == 0 or ys.size == 0:
        return None
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    crop = image_bgr[y1:y2, x1:x2]
    crop_mask = region_mask[y1:y2, x1:x2].astype(np.float32)
    if crop.size == 0 or crop_mask.sum() <= 0:
        return None
    resized_image = cv2.resize(crop, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    resized_mask = cv2.resize(crop_mask, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
    resized_mask = np.clip(resized_mask, 0.0, 1.0)[..., None]
    masked = resized_image * resized_mask
    denom = max(float(resized_mask.sum()), 1e-6)
    mean = masked.sum(axis=(0, 1)) / denom
    centered = (masked - mean) * resized_mask
    std = np.sqrt((centered * centered).sum(axis=(0, 1)) / denom + 1e-6)
    vector = np.concatenate([masked.reshape(-1), resized_mask[..., 0].reshape(-1), mean, std]).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return None
    return vector / norm


def _cosine_similarity(lhs: np.ndarray | None, rhs: np.ndarray | None) -> float | None:
    if lhs is None or rhs is None:
        return None
    return float(np.clip(np.dot(lhs, rhs), -1.0, 1.0))


def _region_similarities(
    pre_bgr: np.ndarray,
    pred_bgr: np.ndarray,
    face_mask: np.ndarray | None,
    inpaint_mask: np.ndarray | None,
) -> tuple[float | None, float | None]:
    if face_mask is None:
        return None, None
    soft_region = face_mask
    hard_region = face_mask.copy()
    if inpaint_mask is not None:
        hard_region = np.logical_and(face_mask, np.logical_not(_dilate_mask(inpaint_mask)))

    # 这里是 v1 的轻量一致性指标：
    # hard_region 看非编辑区是否稳定，soft_region 把鼻嘴也保留为弱参考。
    hard_similarity = _cosine_similarity(
        _region_embedding(pre_bgr, hard_region),
        _region_embedding(pred_bgr, hard_region),
    )
    soft_similarity = _cosine_similarity(
        _region_embedding(pre_bgr, soft_region),
        _region_embedding(pred_bgr, soft_region),
    )
    return hard_similarity, soft_similarity


def _placeholder_tile(reference: np.ndarray, label: str) -> np.ndarray:
    tile = np.full_like(reference, 24)
    cv2.putText(tile, label, (12, max(24, tile.shape[0] // 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
    return tile


def _build_case_sheet(case_rows: dict[str, dict[str, dict[str, Any]]], case_id: str) -> np.ndarray | None:
    case_map = case_rows.get(case_id, {})
    if not case_map:
        return None

    view_ids = sorted(case_map, key=lambda item: (0, int(item)) if str(item).isdigit() else (1, str(item)))
    rows: list[np.ndarray] = []
    for view_id in view_ids:
        bundle = case_map[view_id]
        pre_bgr = bundle.get("pre")
        post_bgr = bundle.get("post")
        pred_bgr = bundle.get("pred")
        reference = pre_bgr if pre_bgr is not None else post_bgr if post_bgr is not None else pred_bgr
        if reference is None:
            continue
        pre_tile = pre_bgr if pre_bgr is not None else _placeholder_tile(reference, f"view {view_id} pre missing")
        post_tile = post_bgr if post_bgr is not None else _placeholder_tile(reference, f"view {view_id} no gt")
        pred_tile = pred_bgr if pred_bgr is not None else _placeholder_tile(reference, f"view {view_id} no pred")
        triptych = _concat_triptych(pre_tile, post_tile, pred_tile)
        cv2.putText(triptych, f"case {case_id} view {view_id}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        rows.append(triptych)
    if not rows:
        return None
    return np.concatenate(rows, axis=0)


def run_evaluation(
    paths: GenerationPaths | None = None,
    max_triptychs: int = 64,
) -> dict[str, Any]:
    """统计数据准备与推理结果，并导出三联预览。"""
    paths = paths or default_generation_paths()
    manifest_rows = build_generation_manifest(paths)

    total_rows = len(manifest_rows)
    paired_ok = 0
    nose_mask_ok = 0
    face_mask_ok = 0
    depth_ok = 0
    inference_ok = 0
    triptych_count = 0
    complete_six_view_cases: set[str] = set()
    partial_view_cases: set[str] = set()
    unpaired_pre_view_count = 0
    unpaired_post_view_count = 0
    hard_scores: list[float] = []
    soft_scores: list[float] = []

    pre_rows = [row for row in manifest_rows if row.get("stage") == "术前"]
    outputs_by_case: dict[str, set[str]] = {}
    case_rows: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for row in manifest_rows:
        paired_pre = Path(str(row.get("paired_pre_path") or "")) if row.get("paired_pre_path") else None
        paired_post = Path(str(row.get("paired_post_path") or "")) if row.get("paired_post_path") else None
        if paired_pre is not None and paired_post is not None and paired_pre.exists() and paired_post.exists():
            paired_ok += 1
        elif row.get("stage") == "术前":
            unpaired_pre_view_count += 1
        elif row.get("stage") == "术后":
            unpaired_post_view_count += 1
        if Path(str(row.get("nose_mask_path") or "")).exists():
            nose_mask_ok += 1
        if Path(str(row.get("face_mask_path") or "")).exists():
            face_mask_ok += 1
        if row.get("depth_path") and Path(str(row.get("depth_path"))).exists():
            depth_ok += 1

    for row in pre_rows:
        image_path = Path(str(row.get("image_path") or ""))
        if not image_path.exists():
            continue

        rel_path = _resolve_rel_path(paths, image_path)
        composite_path = _png_path(paths.inference_root / "composited", rel_path)
        if not composite_path.exists():
            continue

        inference_ok += 1
        case_id = str(row.get("case_id") or "")
        view_id = str(row.get("view_id") or "")
        outputs_by_case.setdefault(case_id, set()).add(view_id)

        paired_post_path = Path(str(row.get("paired_post_path") or ""))
        pre_bgr = cv2.imread(str(image_path))
        pred_bgr = cv2.imread(str(composite_path))
        post_bgr = cv2.imread(str(paired_post_path)) if paired_post_path.exists() else None
        if pre_bgr is None or pred_bgr is None:
            continue

        face_mask = _load_mask(str(row.get("face_mask_path") or "")) if row.get("face_mask_path") else None
        inpaint_mask = _load_mask(str(row.get("inpaint_mask_path") or "")) if row.get("inpaint_mask_path") else None
        hard_similarity, soft_similarity = _region_similarities(pre_bgr, pred_bgr, face_mask, inpaint_mask)
        if hard_similarity is not None:
            hard_scores.append(hard_similarity)
        if soft_similarity is not None:
            soft_scores.append(soft_similarity)

        case_rows[case_id].setdefault(view_id, {})
        case_rows[case_id][view_id]["pre"] = pre_bgr
        case_rows[case_id][view_id]["pred"] = pred_bgr
        if post_bgr is not None:
            case_rows[case_id][view_id]["post"] = post_bgr

        if triptych_count >= max_triptychs or post_bgr is None:
            continue

        triptych = _concat_triptych(pre_bgr, post_bgr, pred_bgr)
        out_path = _png_path(paths.eval_root / "triptychs", rel_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), triptych)
        triptych_count += 1

    for case_id, views in outputs_by_case.items():
        if all(str(v) in views for v in range(1, 7)):
            complete_six_view_cases.add(case_id)
        else:
            partial_view_cases.add(case_id)

        case_sheet = _build_case_sheet(case_rows, case_id)
        if case_sheet is not None:
            sheet_path = paths.eval_root / "case_sheets" / f"{case_id}.png"
            sheet_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(sheet_path), case_sheet)

    summary = {
        "total_rows": total_rows,
        "paired_ok": paired_ok,
        "nose_mask_ok": nose_mask_ok,
        "face_mask_ok": face_mask_ok,
        "depth_ok": depth_ok,
        "depth_optional": True,
        "pre_rows": len(pre_rows),
        "inference_ok": inference_ok,
        "paired_view_count": sum(1 for row in pre_rows if bool(row.get("is_paired_view"))),
        "unpaired_pre_view_count": unpaired_pre_view_count,
        "unpaired_post_view_count": unpaired_post_view_count,
        "predicted_view_count": sum(len(views) for views in outputs_by_case.values()),
        "triptych_count": triptych_count,
        "complete_six_view_cases": sorted(complete_six_view_cases),
        "partial_view_cases": sorted(partial_view_cases),
        "hard_identity_similarity": float(np.mean(hard_scores)) if hard_scores else None,
        "soft_face_similarity": float(np.mean(soft_scores)) if soft_scores else None,
        "eval_root": str(paths.eval_root),
    }

    write_json(paths.summaries_root / "evaluate_summary.json", summary)
    dump_config_snapshot(paths.summaries_root / "evaluate_config.json", paths=paths)
    return summary


def main() -> None:
    summary = run_evaluation()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
