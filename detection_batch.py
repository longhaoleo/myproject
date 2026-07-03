"""
检测侧统一入口。

运行方式：
    python detection_batch.py
"""

import os


RUN_MODE = "box_artifacts"


def main() -> None:
    """按 RUN_MODE 调度 detection 主流程。"""
    mode = os.getenv("FD_RUN_MODE", RUN_MODE).strip().lower()
    if mode == "detect_compare":
        from detection.batch import main as run_detect_compare

        run_detect_compare()
        return
    if mode == "eye_mask":
        from detection.masking import main as run_eye_mask

        run_eye_mask()
        return
    if mode == "box_artifacts":
        from detection.box_artifacts import main as run_box_artifacts

        run_box_artifacts()
        return
    if mode == "sam_mask":
        from detection.sam_mask import main as run_sam_mask

        run_sam_mask()
        return

    raise ValueError(
        f"不支持的 RUN_MODE: {mode}。可选值：detect_compare / eye_mask / box_artifacts / sam_mask"
    )


if __name__ == "__main__":
    main()
