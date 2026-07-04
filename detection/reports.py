"""Small helpers for writing failed-image detection run reports."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping


REPORT_COLUMNS = [
    "case_id",
    "stage",
    "view_id",
    "rel_path",
    "status",
    "reason",
    "output_path",
    "detail",
]


def make_failure_row(
    image_path: Path,
    input_root: Path,
    status: str,
    reason: str = "",
    output_path: Path | None = None,
    detail: str = "",
) -> dict[str, str]:
    rel_path = image_path.relative_to(input_root)
    case_id = rel_path.parts[0] if len(rel_path.parts) > 0 else ""
    stage = rel_path.parts[1] if len(rel_path.parts) > 1 else ""
    return {
        "case_id": case_id,
        "stage": stage,
        "view_id": rel_path.stem,
        "rel_path": str(rel_path),
        "status": status,
        "reason": reason,
        "output_path": "" if output_path is None else str(output_path),
        "detail": detail,
    }


def _write_lines(file_path: Path, lines: list[str]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(file_path: Path, rows: list[dict[str, str]]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(REPORT_COLUMNS)
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    with file_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_run_report(
    run_stats_root: Path,
    report_name: str,
    summary: Mapping[str, object],
    failures: list[dict[str, str]],
) -> None:
    summary_lines = [f"{key}={value}" for key, value in summary.items()]
    summary_lines.append(f"failures_report={report_name}_failures.csv")

    _write_lines(run_stats_root / f"{report_name}_summary.txt", summary_lines)
    _write_csv(run_stats_root / f"{report_name}_failures.csv", failures)
