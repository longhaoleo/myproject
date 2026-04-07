"""
生成侧评估与验收。

当前聚焦工程可用性：
- 配对是否完整
- 核心条件是否齐全
- 推理输出是否落盘
- 三联预览是否可复查
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from project_utils.io import write_json
from .index import build_generation_manifest
from .settings import GenerationPaths, default_generation_paths, dump_config_snapshot


def _resolve_rel_path(paths: GenerationPaths, sample_path: str | Path) -> Path:
    """把样本绝对路径映射为相对路径，保持评估输出目录树一致。"""
    path = Path(sample_path)
    for root in (paths.sanitized_root, paths.input_root, paths.inference_root):
        try:
            return path.relative_to(root)
        except ValueError:
            continue
    raise ValueError(f"无法解析相对路径: {path}")


def _png_path(root: Path, rel_path: Path) -> Path:
    """把样本相对路径映射到 PNG 文件。"""
    return (root / rel_path).with_suffix(".png")


def _concat_triptych(left: np.ndarray, middle: np.ndarray, right: np.ndarray) -> np.ndarray:
    """拼接三联图：术前 | 术后GT | 预测结果。"""
    height, width = left.shape[:2]
    tiles = [left, middle, right]
    resized = [cv2.resize(tile, (width, height), interpolation=cv2.INTER_AREA) for tile in tiles]
    return np.concatenate(resized, axis=1)


def run_evaluation(
    paths: GenerationPaths | None = None,
    max_triptychs: int = 64,
) -> dict[str, Any]:
    """统计数据准备与推理结果，并导出三联预览。"""
    paths = paths or default_generation_paths()
    manifest_rows = build_generation_manifest(paths)

    # 统计项分两类：
    # 1) 条件齐备度（pair、mask、depth）
    # 2) 推理落盘覆盖率（术前样本是否有 composited 结果）
    total_rows = len(manifest_rows)
    paired_ok = 0
    nose_mask_ok = 0
    face_mask_ok = 0
    depth_ok = 0
    inference_ok = 0
    triptych_count = 0
    complete_six_view_cases: set[str] = set()

    pre_rows = [row for row in manifest_rows if row.get("stage") == "术前"]
    outputs_by_case: dict[str, set[str]] = {}

    # 第一遍：扫描全量 manifest，统计前置条件完整度。
    for row in manifest_rows:
        paired_pre = Path(str(row.get("paired_pre_path") or "")) if row.get("paired_pre_path") else None
        paired_post = Path(str(row.get("paired_post_path") or "")) if row.get("paired_post_path") else None
        if paired_pre is not None and paired_post is not None and paired_pre.exists() and paired_post.exists():
            paired_ok += 1
        if Path(str(row.get("nose_mask_path") or "")).exists():
            nose_mask_ok += 1
        if Path(str(row.get("face_mask_path") or "")).exists():
            face_mask_ok += 1
        if row.get("depth_path") and Path(str(row.get("depth_path"))).exists():
            depth_ok += 1

    # 第二遍：只看术前样本，检查是否已有推理结果，并按需导出三联图。
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

        # 三联图用于人工验收，数量过大时按 max_triptychs 截断。
        if triptych_count >= max_triptychs:
            continue

        paired_post_path = Path(str(row.get("paired_post_path") or ""))
        if not paired_post_path.exists():
            continue

        pre_bgr = cv2.imread(str(image_path))
        post_bgr = cv2.imread(str(paired_post_path))
        pred_bgr = cv2.imread(str(composite_path))
        if pre_bgr is None or post_bgr is None or pred_bgr is None:
            continue

        triptych = _concat_triptych(pre_bgr, post_bgr, pred_bgr)
        out_path = _png_path(paths.eval_root / "triptychs", rel_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), triptych)
        triptych_count += 1

    # 统计“六视角齐全”的 case，便于后续挑选完整样本做定量分析。
    for case_id, views in outputs_by_case.items():
        if all(str(v) in views for v in range(1, 7)):
            complete_six_view_cases.add(case_id)

    summary = {
        "total_rows": total_rows,
        "paired_ok": paired_ok,
        "nose_mask_ok": nose_mask_ok,
        "face_mask_ok": face_mask_ok,
        "depth_ok": depth_ok,
        "depth_optional": True,
        "pre_rows": len(pre_rows),
        "inference_ok": inference_ok,
        "triptych_count": triptych_count,
        "complete_six_view_cases": sorted(complete_six_view_cases),
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
