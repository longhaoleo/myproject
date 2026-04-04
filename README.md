# 眼部遮挡与多模型人脸检测

项目统一入口是 `face_detection_batch.py`，支持三种模式：

1. `detect_compare`：多检测器可视化对比
2. `eye_mask`：根据检测框直接眼部打码
3. `sam_mask`：检测框/关键点 + SAM 分割

更详细的模型下载、放置路径、各检测器加载方式见 [face_detection/README.md](face_detection/README.md)。

## 快速开始

```bash
cd ~/myproject
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python face_detection_batch.py
```

建议使用 `Python 3.10`，并保证“运行脚本的 python”和“安装依赖的 python”是同一个环境。

在 `face_detection_batch.py` 中修改：

- `RUN_MODE = "detect_compare"`
- `RUN_MODE = "eye_mask"`
- `RUN_MODE = "sam_mask"`

在 notebook 里不需要专门接口，直接调用现有模块：

```python
from face_detection.batch import main as run_detect_compare
from face_detection.masking import main as run_eye_mask
from face_detection.sam_mask import main as run_sam_mask

run_detect_compare()  # 或 run_eye_mask() / run_sam_mask()
```

## 说明

- 当前 `requirements.txt` 已按单环境可安装整理（MediaPipe + TensorFlow 2.15 兼容）。
- 模型文件默认放在项目根目录 `model/`，并按检测器分子目录。
