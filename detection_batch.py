"""
检测侧统一入口。

运行方式：
    python detection_batch.py
"""

RUN_MODE = "detect_compare"


def main() -> None:
    # 入口只做模式分发；具体配置在各子模块内部维护。
    mode = RUN_MODE.strip().lower()
    if mode == "detect_compare":
        from detection.batch import main as run_detect_compare

        run_detect_compare()
        return
    if mode == "eye_mask":
        from detection.masking import main as run_eye_mask

        run_eye_mask()
        return
    if mode == "sam_mask":
        from detection.sam_mask import main as run_sam_mask

        run_sam_mask()
        return

    raise ValueError(
        f"不支持的 RUN_MODE: {RUN_MODE}。可选值：detect_compare / eye_mask / sam_mask"
    )


if __name__ == "__main__":
    main()
