"""
生成侧统一入口。

运行方式：
    python generation_pipeline.py

通过修改 RUN_MODE 或环境变量 GEN_RUN_MODE 选择流程：
- prepare_dataset
- train_lora
- infer
- evaluate
"""

from __future__ import annotations

import os


RUN_MODE = "prepare_dataset"


def main() -> None:
    mode = os.getenv("GEN_RUN_MODE", RUN_MODE).strip().lower()

    if mode == "prepare_dataset":
        from detection.prepare_dataset import main as run_prepare_dataset

        run_prepare_dataset()
        return
    if mode == "train_lora":
        from generation.train import main as run_train_lora

        run_train_lora()
        return
    if mode == "infer":
        from generation.infer import main as run_infer

        run_infer()
        return
    if mode == "evaluate":
        from generation.eval import main as run_eval

        run_eval()
        return

    raise ValueError(
        f"不支持的 RUN_MODE: {mode}。可选值：prepare_dataset / train_lora / infer / evaluate"
    )


if __name__ == "__main__":
    main()
