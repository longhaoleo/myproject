# 眼部遮挡、多模型检测与 SDXL 生成 Pipeline

项目统一入口是 `detection_batch.py`，支持三种模式：

1. `detect_compare`：多检测器可视化对比
2. `eye_mask`：根据检测框直接眼部打码
3. `sam_mask`：检测框/关键点 + SAM 分割

更详细的模型下载、放置路径、各检测器加载方式见 [detection/README.md](detection/README.md)。
生成侧依赖、权重下载、输出目录和运行顺序见 [generation/README.md](generation/README.md)。

生成侧统一入口是 `generation_pipeline.py`，支持三种模式：

1. `train_lora`：训练医生风格 SDXL inpainting LoRA
2. `infer`：运行 `SDXL Inpainting + LoRA`，并可选接入 `ControlNet-Depth`
3. `evaluate`：统计验收并导出三联预览

## 快速开始

```bash
cd ~/myproject
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python detection_batch.py
python generation_pipeline.py
```

建议使用 `Python 3.10`，并保证“运行脚本的 python”和“安装依赖的 python”是同一个环境。

在 `detection_batch.py` 中修改：

- `RUN_MODE = "detect_compare"`
- `RUN_MODE = "eye_mask"`
- `RUN_MODE = "sam_mask"`

在 notebook 里不需要专门接口，直接调用现有模块：

```python
from detection.batch import main as run_detect_compare
from detection.masking import main as run_eye_mask
from detection.sam_mask import main as run_sam_mask

run_detect_compare()  # 或 run_eye_mask() / run_sam_mask()
```

生成侧可以直接 import：

```python
from generation.train import train_lora
from generation.infer import run_inference
from generation.eval import run_evaluation

train_lora()
```

## 说明

- 当前 `requirements.txt` 已按单环境可安装整理（MediaPipe + TensorFlow 2.15 兼容）。
- 模型文件默认放在项目根目录 `model/`，并按检测器分子目录。
- `RetinaFace` 仅建议放在独立环境中做对照实验；不建议作为主工程默认检测器长期维护。需要单独测试时，优先使用 [environment-retinaface.yml](environment-retinaface.yml)。
- 当前主逻辑是：先得到 `eye_mask` 后的图片，再在 `face` 框内取头部参考区域训练，同时用连续的 `inpaint_mask` 指定鼻嘴重绘目标区域。
- 连续鼻嘴 `inpaint_mask`、羽化版 `feather_mask` 和标准化 `face_mask` 由 detection 侧直接生成，generation 只消费这些产物。
- 若已运行 `eye_mask`，generation 会优先读取打码后的图片；否则回退到原图。
