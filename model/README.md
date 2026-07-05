# 模型权重下载与统一存放

本目录统一存放本项目用到的本地模型权重。权重文件体积较大，默认被 Git 忽略；仓库只跟踪本说明文件和 `.gitkeep`。

所有命令都在项目根目录执行：

```bash
cd /Users/leo/myproject
source .venv/bin/activate
which python
python -V
```

如果 `which python` 不是 `/Users/leo/myproject/.venv/bin/python`，说明虚拟环境没有激活。下面的命令会尽量显式使用 `.venv/bin/...`，避免误用 conda base 的 Python。

## 目录约定

```text
model/
├── blazeface/
│   └── blaze_face_short_range.tflite
├── centerface/
│   └── centerface.onnx
├── generation/
│   ├── controlnet-depth-sdxl/
│   ├── dpt-hybrid-midas/
│   ├── ip-adapter/
│   └── sdxl-inpaint/
├── mediapipe-landmarker/
│   └── face_landmarker.task
├── mtcnn/
│   ├── onet.pt
│   ├── pnet.pt
│   └── rnet.pt
├── retinaface/
│   └── retinaface.h5
├── sam/
│   └── mobile_sam.pt
├── scrfd/
│   └── insightface/
└── yolov8-face/
    └── yolov8_face.pt
```

当前默认检测主线优先使用 `scrfd`，然后 fallback 到 `yolov8-face` 和 `blazeface`。`sam_mask` 需要 `mobile_sam.pt`。

## 一次性下载命令

这段会把常用和可选权重都放到 `model/` 下。Hugging Face 模型可能需要提前登录：

```bash
.venv/bin/huggingface-cli login
```

完整命令：

```bash
mkdir -p \
  model/blazeface \
  model/centerface \
  model/generation/controlnet-depth-sdxl \
  model/generation/dpt-hybrid-midas \
  model/generation/ip-adapter \
  model/generation/sdxl-inpaint \
  model/mediapipe-landmarker \
  model/retinaface \
  model/sam \
  model/scrfd/insightface \
  model/yolov8-face

# SCRFD / InsightFace buffalo_l
# 说明：insightface 会自动下载到 model/scrfd/insightface。
python - <<'PY'
from insightface.app import FaceAnalysis

app = FaceAnalysis(
    name="buffalo_l",
    root="model/scrfd/insightface",
    providers=["CPUExecutionProvider"],
)
app.prepare(ctx_id=-1, det_size=(640, 640))
print("SCRFD ready: model/scrfd/insightface")
PY

# YOLOv8-Face
.venv/bin/python -m gdown 'https://drive.google.com/uc?id=1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb' \
  -O model/yolov8-face/yolov8_face.pt

# MobileSAM
curl -L \
  https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt \
  -o model/sam/mobile_sam.pt

# MediaPipe Face Landmarker
curl -L \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task \
  -o model/mediapipe-landmarker/face_landmarker.task

# BlazeFace task model
# 默认 blazeface 通常走 mediapipe legacy API，不一定需要这个文件；
# 但放好它可以兼容 tasks fallback。
curl -L \
  https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite \
  -o model/blazeface/blaze_face_short_range.tflite

# CenterFace
#
curl -L \
  https://github.com/Star-Clouds/CenterFace/raw/master/models/onnx/centerface.onnx \
  -o model/centerface/centerface.onnx

# RetinaFace
# 当前默认主线不启用 RetinaFace；需要时还要额外安装 retina-face / tensorflow。
curl -L \
  https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5 \
  -o model/retinaface/retinaface.h5

# SDXL Inpainting
.venv/bin/huggingface-cli download diffusers/stable-diffusion-xl-1.0-inpainting-0.1 \
  --local-dir model/generation/sdxl-inpaint

# ControlNet Depth SDXL
.venv/bin/huggingface-cli download diffusers/controlnet-depth-sdxl-1.0 \
  --local-dir model/generation/controlnet-depth-sdxl

# IP-Adapter for SDXL Inpainting reference images
# 只下载当前推理需要的 SDXL face adapter 和 image encoder。
.venv/bin/hf download h94/IP-Adapter \
  sdxl_models/image_encoder/config.json \
  sdxl_models/image_encoder/model.safetensors \
  sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors \
  --local-dir model/generation/ip-adapter \
  --max-workers 1

# Depth estimator, optional
.venv/bin/huggingface-cli download Intel/dpt-hybrid-midas \
  --local-dir model/generation/dpt-hybrid-midas
```

## 分项说明

### SCRFD

- 默认检测器。
- 通常会由 `insightface` 自动下载。
- 本项目统一让它落在 `model/scrfd/insightface/`。

### YOLOv8-Face

代码期望路径：

```text
model/yolov8-face/yolov8_face.pt
```

如果 Google Drive 下载失败，可以先确认 `.venv` 里有 `gdown`：

```bash
.venv/bin/python -m pip show gdown
```

### MobileSAM

`sam_mask` 流程期望路径：

```text
model/sam/mobile_sam.pt
```

Python 包 `mobile_sam` 和 `timm` 已写入项目依赖，权重仍需要单独下载。

### 生成侧模型

生成侧默认配置仍然使用 Hugging Face repo id，例如：

```text
diffusers/stable-diffusion-xl-1.0-inpainting-0.1
diffusers/controlnet-depth-sdxl-1.0
```

如果你想强制使用 `model/generation/` 下的本地目录，需要在 `generation/settings.py` 或调用配置里把：

```python
base_model_id = "model/generation/sdxl-inpaint"
controlnet_model_id = "model/generation/controlnet-depth-sdxl"
ip_adapter_model_id = "model/generation/ip-adapter"
```

#### IP-Adapter

推理侧默认从本地目录读取：

```text
model/generation/ip-adapter
```

需要的文件是：

```text
model/generation/ip-adapter/sdxl_models/image_encoder/config.json
model/generation/ip-adapter/sdxl_models/image_encoder/model.safetensors
model/generation/ip-adapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors
```

如果下载中断，可以单独重跑：

```bash
.venv/bin/hf download h94/IP-Adapter \
  sdxl_models/image_encoder/config.json \
  sdxl_models/image_encoder/model.safetensors \
  sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors \
  --local-dir model/generation/ip-adapter \
  --max-workers 1
```

### MTCNN

当前默认环境不安装 `facenet-pytorch`，因为它和生成侧依赖冲突。若需要复测 `mtcnn`，建议另建 detection-only 环境后执行：

```bash
cd /Users/leo/myproject
python3.11 -m venv .venv-mtcnn
source .venv-mtcnn/bin/activate
python -m pip install --upgrade pip
python -m pip install "numpy==1.26.4" "opencv-python==4.10.0.84" "torch>=2.2,<2.3" "torchvision>=0.17,<0.18" "facenet-pytorch==2.6.0"

python - <<'PY'
from pathlib import Path
import torch
from facenet_pytorch import MTCNN

out_dir = Path("model/mtcnn")
out_dir.mkdir(parents=True, exist_ok=True)
torch.hub.set_dir(str(out_dir))

det = MTCNN(keep_all=True, device="cpu")
torch.save(det.pnet.state_dict(), out_dir / "pnet.pt")
torch.save(det.rnet.state_dict(), out_dir / "rnet.pt")
torch.save(det.onet.state_dict(), out_dir / "onet.pt")
print("MTCNN weights exported to:", out_dir)
PY
```

## 检查下载结果

```bash
python - <<'PY'
from pathlib import Path

checks = [
    ("SCRFD dir", Path("model/scrfd/insightface")),
    ("YOLOv8-Face", Path("model/yolov8-face/yolov8_face.pt")),
    ("MobileSAM", Path("model/sam/mobile_sam.pt")),
    ("MediaPipe Landmarker", Path("model/mediapipe-landmarker/face_landmarker.task")),
    ("BlazeFace task model", Path("model/blazeface/blaze_face_short_range.tflite")),
    ("CenterFace", Path("model/centerface/centerface.onnx")),
    ("MTCNN pnet", Path("model/mtcnn/pnet.pt")),
    ("MTCNN rnet", Path("model/mtcnn/rnet.pt")),
    ("MTCNN onet", Path("model/mtcnn/onet.pt")),
    ("RetinaFace", Path("model/retinaface/retinaface.h5")),
    ("SDXL Inpaint", Path("model/generation/sdxl-inpaint/model_index.json")),
    ("ControlNet Depth", Path("model/generation/controlnet-depth-sdxl/config.json")),
    ("IP-Adapter image encoder", Path("model/generation/ip-adapter/sdxl_models/image_encoder/model.safetensors")),
    ("IP-Adapter face SDXL", Path("model/generation/ip-adapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors")),
    ("DPT Hybrid MiDaS", Path("model/generation/dpt-hybrid-midas/config.json")),
]

for name, path in checks:
    print(f"{name:24} {'OK' if path.exists() else 'missing'}  {path}")
PY
```

## 注意

- 权重文件不要提交到 Git。
- `mtcnn` 目前不是默认环境的一部分，因为 `facenet-pytorch==2.6.0` 会把 `torch` 锁到 `<2.3`，和生成侧量化依赖冲突。如果要复测 MTCNN，建议另建 detection-only 环境。
- 如果 Hugging Face 下载失败，先运行 `.venv/bin/huggingface-cli login`，并确认目标模型仓库许可已经接受。
