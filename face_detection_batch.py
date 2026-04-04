"""
项目唯一入口。

运行方式：
    python face_detection_batch.py

通过修改 RUN_MODE 选择要执行的流程：
- detect_compare: 多模型人脸检测对比
- eye_mask: 眼部打码批处理
- sam_mask: SAM 分割批处理（基于检测框+关键点提示）
"""


# 单一入口开关：只改这里，不用改命令行参数。
# 说明：
# - detect_compare：多模型检测对比
# - eye_mask：眼部打码
# - sam_mask：检测框+关键点提示 + SAM 分割
RUN_MODE = "detect_compare"

def main():
    mode = RUN_MODE.strip().lower()
    # 统一入口分发，避免命令行传参混乱。
    if mode == "detect_compare":
        from face_detection.batch import main as run_detect_compare

        run_detect_compare()
        return
    if mode == "eye_mask":
        from face_detection.masking import main as run_eye_mask

        run_eye_mask()
        return
    if mode == "sam_mask":
        from face_detection.sam_mask import main as run_sam_mask

        run_sam_mask()
        return

    raise ValueError(
        f"不支持的 RUN_MODE: {RUN_MODE}。可选值：detect_compare / eye_mask / sam_mask"
    )


if __name__ == "__main__":
    main()
