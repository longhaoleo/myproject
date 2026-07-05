# 眼部遮挡、多模型检测与 SDXL 生成 Pipeline

项目统一入口是 `detection_batch.py`，支持三种模式：

1. `detect_compare`：多检测器可视化对比
2. `eye_mask`：根据检测框直接眼部打码
3. `box_artifacts`：检测框/关键点直接生成 generation 所需矩形 mask
4. `sam_mask`：检测框/关键点 + SAM 分割

更详细的模型下载、放置路径、各检测器加载方式见 [detection/README.md](detection/README.md)。
生成侧依赖、权重下载、输出目录和运行顺序见 [generation/README.md](generation/README.md)。

生成侧统一入口是 `generation_pipeline.py`，支持三种模式：

1. `train_lora`：训练医生风格 SDXL inpainting LoRA
2. `infer`：运行 `SDXL Inpainting + LoRA`，并可选接入 `ControlNet-Depth`
3. `evaluate`：统计验收并导出三联预览

## 快速开始

```bash
cd ~/myproject
conda create -n venv python=3.10 -y
conda activate vnev
python -m pip install --upgrade pip
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python detection_batch.py
python generation_pipeline.py
```

建议使用 `Python 3.10`，并保证“运行脚本的 python”和“安装依赖的 python”是同一个环境。

在 `detection_batch.py` 中修改：

- `RUN_MODE = "detect_compare"`
- `RUN_MODE = "eye_mask"`
- `RUN_MODE = "box_artifacts"`
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

- 当前 `requirements.txt` 已按主线检测器整理：`MTCNN / SCRFD / BlazeFace / YOLOv8-Face`。
- 模型文件默认放在项目根目录 `model/`，并按检测器分子目录。
- 当前训练 baseline 是：使用同病例同视角配对样本，以术前 `face_mask` 区域作为条件 crop，术前 `inpaint_mask` 区域挖空，目标为同视角术后 face crop，让 SDXL inpainting LoRA 学习在术前人脸上下文里重绘术后鼻嘴。
- 连续鼻嘴 `inpaint_mask`、羽化版 `feather_mask` 和标准化 `face_mask` 由 detection 侧直接生成，既可以来自 `box_artifacts` 的矩形框，也可以来自 `sam_mask` 的分割结果；generation 只消费这些产物。
- 多视角参考和缺视角利用先保留为后续工作；当前先跑通最小术前到术后 paired inpainting 闭环。
- 评估会同时输出三联图、病例级 contact sheet，以及 `hard_identity_similarity / soft_face_similarity` 两类轻量一致性指标。
