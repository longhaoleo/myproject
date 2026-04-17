# detection 模块说明

该目录包含三个核心流程：

1. 人脸检测可视化对比：`batch.py`
2. 眼部打码：`masking.py`
3. SAM 分割批处理：`sam_mask.py`
4. 分割模块：`segmentation.py`（供 `sam_mask.py` 调用）
5. 部位框共享逻辑：`part_boxes.py`（`segmentation.py` / `visualize.py` 共用）

统一入口：`python detection_batch.py`

配置集中管理：`detection/settings.py`
- 三个流程共用的默认路径、检测器阈值、视角缩放/偏移都在这里；
- 各流程仍可在自己的文件里覆盖（例如只改 `sam_mask.py` 的视角参数）。
- `sam_mask.py` 会直接落盘标准化后的 `face_mask`、连续鼻嘴 `inpaint_mask` 和 `feather_mask`；
- `eye_mask.py` 会直接输出眼部打码后的图片；
- generation 侧不负责定义 mask 区域，只直接消费 detection 侧落盘结果。

---

## 1. 运行前准备（环境版本）

建议统一使用 `Python 3.10` 的同一个环境（例如 `myenv`），不要混用 `/opt/micromamba/bin/python` 和 conda 环境。
当前依赖已按“单环境可安装”整理：`mediapipe + tensorflow==2.15.1`，不需要单独安装 `tf-keras`。

### 1.1 CPU 环境（默认）

```bash
cd ~/myproject
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 1.2 GPU 环境（可选）

如果你希望 `MTCNN/YOLOv8/SCRFD` 尽量走 GPU，建议按下面顺序安装：

```bash
cd ~/myproject

# 先安装 CUDA 版 torch（按你的 CUDA 版本选择官方 index）
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  "torch>=2.1,<2.11" "torchvision>=0.16,<0.21"

# 再装项目依赖
python -m pip install -r requirements.txt

# SCRFD 若要 GPU，把 onnxruntime 换成 onnxruntime-gpu
python -m pip uninstall -y onnxruntime
python -m pip install onnxruntime-gpu==1.19.2
```

安装完成后可快速检查：

```bash
python -c "import torch; print('torch cuda:', torch.cuda.is_available())"
python -c "import onnxruntime as ort; print('ort providers:', ort.get_available_providers())"
```

依赖完整性快速检查（可选）：

```bash
python - <<'PY'
mods = [
    "cv2", "numpy", "mediapipe", "torch", "facenet_pytorch",
    "retinaface", "insightface", "onnxruntime", "ultralytics", "mobile_sam"
]
for m in mods:
    try:
        __import__(m)
        print("OK ", m)
    except Exception as e:
        print("FAIL", m, "|", e)
PY
```

---

## 2. 模型文件放置

模型统一放在项目根目录 `model/`，并按检测器拆分子目录。

推荐结构：

```text
model/
├── blazeface/
├── centerface/
├── mediapipe-landmarker/
├── mtcnn/
├── retinaface/
├── sam/
├── scrfd/
└── yolov8-face/
```

### 2.1 MediaPipe FaceLandmarker（细粒度关键点，468点）

- 文件：`model/mediapipe-landmarker/face_landmarker.task`

```bash
mkdir -p model/mediapipe-landmarker
wget -O model/mediapipe-landmarker/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

### 2.2 BlazeFace（MediaPipe FaceDetection）

```bash
mkdir -p model/blazeface
wget -O model/blazeface/blaze_face_short_range.tflite \
  https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite
```

### 2.3 CenterFace（本地 ONNX 版）

- 文件：`model/centerface/centerface.onnx`
- 说明：不需要 `pip install centerface`

```bash
mkdir -p model/centerface
wget -O model/centerface/centerface.onnx \
  https://github.com/Star-Clouds/CenterFace/raw/master/models/onnx/centerface.onnx
```

### 2.4 其他模型下载（全部）

#### 2.4.1 RetinaFace

```bash
mkdir -p model/retinaface
wget -O model/retinaface/retinaface.h5 \
  https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5
```

#### 2.4.2 SCRFD（buffalo_l）

```bash
python - <<'PY'
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_l", root="model/scrfd/insightface", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))
print("SCRFD 模型已就绪: model/scrfd/insightface")
PY
```

#### 2.4.3 MTCNN（pnet/rnet/onet）

```bash
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
print("MTCNN 权重已导出到:", out_dir)
PY
```

#### 2.4.4 YOLOv8-Face（5 点关键点版）

```bash
# 若缺少 gdown：python -m pip install gdown
mkdir -p model/yolov8-face
python -m gdown 'https://drive.google.com/uc?id=1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb' -O model/yolov8-face/yolov8_face.pt
```

### 2.5 SAM（轻量版，推荐 MobileSAM）

- `model/sam/mobile_sam.pt`

```bash
mkdir -p model/sam
wget -O model/sam/mobile_sam.pt \
  https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
```

可选安装 ：

```bash
# 方式1（直接安装源码，已写入 requirements.txt）
python -m pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

如果上面网络慢，改用 zip 本地安装：

```bash
wget -O /tmp/MobileSAM.zip \
  https://codeload.github.com/ChaoningZhang/MobileSAM/zip/refs/heads/master
python -m pip install /tmp/MobileSAM.zip
```

---

## 3. 运行方式

在 `detection_batch.py` 里设置：

- `RUN_MODE = "detect_compare"`：检测可视化
- `RUN_MODE = "eye_mask"`：眼部打码
- `RUN_MODE = "sam_mask"`：SAM 分割批处理

然后执行：

```bash
python detection_batch.py
```

如果你要在 `test.ipynb` 里直接跑，不需要额外接口，直接 import 现有模块：

```python
from detection.batch import main as run_detect_compare
from detection.masking import main as run_eye_mask
from detection.sam_mask import main as run_sam_mask

run_detect_compare()  # 或 run_eye_mask() / run_sam_mask()
```

如需给 generation 准备输入条件，直接运行：

1. `sam_mask`：生成 `parts/`、`inpaint_mask/`、`feather_mask/`
2. `eye_mask`：按需生成打码图

generation 后续会在运行时直接扫描这些 detection 产物并构建 manifest，不再需要单独的准备步骤。

---

## 3.1 统一配置位置

`detection/settings.py` 里集中维护：
- `default_paths()`：输入/输出/模型目录
- `default_min_confidence_map()`：检测器阈值
- `default_detector_options()`：检测器专属参数
- `default_part_scale_by_view()`：按视角缩放部位框（ratio）
- `default_part_offset_by_view()`：按视角偏移部位（ratio）
- `detection/part_boxes.py`：统一维护部位框的基础定义与生成逻辑

说明：
- `default_paths()` 支持环境变量覆盖（`FD_INPUT_ROOT` 等）；
- 视角微调直接写在 [detection/settings.py](settings.py) 里的 `_view_tweaks_config()`；
- 视角用文件名 `1~6` 匹配；
- 偏移使用 **ratio**（相对人脸框宽高），并且会影响关键点、部位框和 SAM 提示。
- `sam_mask`、`segmentation`、`visualize` 都共用这份配置，改一处即可全局生效。
- 部位框的基础定义现在统一在 `detection/part_boxes.py`，`segmentation` 和 `visualize` 不再各写一份。
- 配置结构是“顶层方法，下一层视角”：
  - `methods.default.views."<id>"`：所有方法默认值
  - `methods.<method_name>.views."<id>"`：某个方法的覆盖值
- 当前已接入的方法名：
  - `detect_compare`
  - `eye_mask`
  - `sam_mask`

---

## 4. 检测对比配置

配置文件：`detection/batch.py`

- `detectors_to_run`：本次运行的检测器列表
- `min_confidence_map`：各检测器阈值
- `detector_options_map`：各检测器专属参数（通用扩展入口）
- `enable_segmentation_refine`：是否启用分割接口
- `segmenter_name`：分割器名称（支持 `none` / `mobile-sam`）
- `segmenter_options`：分割器参数（例如 `model_path`）
- `draw_segmentation_masks`：是否在可视化图中叠加分割 mask
- `draw_segmentation_parts`：是否按部位分色显示（face/eyes/nose/mouth）
- `random_group_count`：随机抽样组数
- `draw_landmarks` / `draw_part_boxes`：可视化开关

注意：当前仓库里 `detectors_to_run` 是示例配置，你可以按实验需要随时取消注释或增删检测器。
说明：默认关闭分割。你开启 `enable_segmentation_refine=True` 后，系统会用人脸框与关键点生成部位提示框，并调用 MobileSAM 做 face/eyes/nose/mouth 分割。

最小开启示例（`detection/batch.py`）：

```python
enable_segmentation_refine = True
segmenter_name = "mobile-sam"
segmenter_options = {
    "model_path": "model/sam/mobile_sam.pt",
    "model_type": "vit_t",
    "part_names": ("face", "left_eye", "right_eye", "nose", "mouth"),
}
draw_segmentation_masks = True
draw_segmentation_parts = True
```

已支持检测器：

- `mtcnn`
- `retinaface`
- `scrfd`
- `blazeface`
- `mediapipe-landmarker`
- `yolov8-face`
- `centerface`

---

## 5. SAM 分割流程配置

配置文件：`detection/sam_mask.py`

- `detector_name`：用哪个检测器给 SAM 提示框（建议 `mediapipe-landmarker`）
- `prompt_source`：提示来源（`detector` 直接检测 / `cache` 复用历史检测结果）
- `cache_miss_fallback_to_detector`：缓存缺失时是否回退检测器
- `save_detection_cache`：是否保存检测结果到 `_detection_cache`
- `segmenter_name`：分割器名称（`mobile-sam`）
- `segmenter_options`：分割参数（`model_path`、`model_type`、`part_names`）
- `draw_segmentation_masks`：预览图是否叠加 mask
- `draw_segmentation_parts`：预览图是否按部位分色显示
- `export_part_masks`：是否导出 `parts/<part_name>/xxx.png`

---

## 6. 所有检测器的加载方式（逐个说明）

统一入口在 `detection/factory.py` 的 `create_face_detector(...)`，按名称分发到各后端类。

### 6.1 `mtcnn`

- 后端类：`MTCNNDetector`
- 依赖导入：`torch`、`facenet_pytorch.MTCNN`
- 加载方式：
  - 初始化时执行 `torch.hub.set_dir(model/mtcnn)`，把缓存固定到 `model/mtcnn/`
  - 用 `MTCNN(keep_all=True, device=...)` 创建检测器
- 权重来源：
  - 默认由 `facenet-pytorch` 自动管理缓存（在你设置的 hub 目录下）
  - 这个检测后端不强制要求你手动放 `pnet/rnet/onet`（打码流程另说）

### 6.2 `retinaface`

- 后端类：`RetinaFaceDetector`
- 依赖导入：`retinaface`（pip 包名 `retina-face`）
- 加载方式：
  - 设置 `DEEPFACE_HOME = model/retinaface`
  - RetinaFace 期望权重路径是 `model/retinaface/.deepface/weights/retinaface.h5`
  - 若你提供了 `model/retinaface/retinaface.h5`，代码会自动软链接/复制到上述期望路径
- 权重来源：
  - 推荐手动放置 `model/retinaface/retinaface.h5`
  - 也可由第三方库自身机制下载（取决于环境）

### 6.3 `scrfd`

- 后端类：`SCRFDDetector`
- 依赖导入：`insightface.app.FaceAnalysis`、`onnxruntime`、`torch`
- 加载方式：
  - `FaceAnalysis(name="buffalo_l", root="model/scrfd/insightface", providers=...)`
  - `prepare(det_size=(640,640))`
  - GPU 可用时优先 `CUDAExecutionProvider`，否则 CPU
- 权重来源：
  - `insightface` 会在 `model/scrfd/insightface` 下自动下载/加载
  - 你也可以提前手动放模型文件

### 6.4 `blazeface`

- 后端类：`BlazeFaceDetector`
- 依赖导入：`mediapipe`
- 加载方式分两条：
  - 若 `mediapipe` 含 `mp.solutions.face_detection`：
    - 直接走 `mp.solutions.face_detection.FaceDetection(...)`
    - 这一分支不需要本地 `.tflite`
  - 否则走 `mediapipe.tasks.vision.FaceDetector`：
    - 从 `model/blazeface/` 查找：
      - `blaze_face_short_range.tflite`
      - `blaze_face_full_range_sparse.tflite`
    - 创建后会先做一次空图 probe，提前验证模型兼容性
- 说明：
  - 旧版 `face_detection_*.tflite` 不再用于此后端，避免 `RET_CHECK ... dims[1]==num_boxes_`

### 6.5 `mediapipe-landmarker`（别名：`face-landmarker` / `landmarker`）

- 后端类：`MediaPipeLandmarkerDetector`
- 依赖导入：`mediapipe.tasks.python.vision.FaceLandmarker`
- 加载方式：
  - 强依赖本地模型：`model/mediapipe-landmarker/face_landmarker.task`
  - `FaceLandmarkerOptions(..., num_faces=5, min_face_detection_confidence=...)`
  - 输出来自 468 点，再映射为 `left_eye/right_eye/nose/mouth/mouth_left/mouth_right`
- 适用：
  - 想看更细粒度关键点时用它

### 6.6 `yolov8-face`

- 后端类：`YoloV8FaceDetector`
- 依赖导入：`ultralytics.YOLO`
- 加载方式：
  - 默认读取 `model/yolov8-face/yolov8_face.pt`（可在 `batch.py` 的 `detector_options_map["yolov8-face"]["model_path"]` 修改）
  - 推理后读取 bbox；若模型包含关键点输出则映射到眼鼻嘴语义点
- 权重来源：
  - 必须有本地 `.pt` 权重文件
  - 推荐 `derronqi/yolov8-face` 的 `yolov8n-face.pt`（5点关键点）
- 快速验证是否带关键点：

```bash
python - <<'PY'
from pathlib import Path
from ultralytics import YOLO
import cv2
m = YOLO('model/yolov8-face/yolov8_face.pt')
img = cv2.imread(str(Path("~/datasets/deformity/1/术前/1.JPG").expanduser()))
r = m(img, verbose=False)[0]
print('has_keypoints:', getattr(r, 'keypoints', None) is not None and getattr(r.keypoints, 'xy', None) is not None)
PY
```

### 6.7 `centerface`

- 后端类：`CenterFaceDetector`
- 依赖导入：`cv2.dnn`（不依赖 `pip install centerface`）
- 加载方式：
  - 强依赖本地 ONNX：`model/centerface/centerface.onnx`
  - `cv2.dnn.readNetFromONNX(...)` 直接加载
  - 前处理/解码/NMS 在后端代码内实现（输出框 + 5 点）

---

## 7. 代码定位（便于你后续改）

- 工厂分发：`detection/factory.py`
- 各检测器实现：`detection/backends.py`
- SAM 批处理：`detection/sam_mask.py`
- 分割模块（SAM）：`detection/segmentation.py`
- 运行列表与阈值：`detection/batch.py` 里的 `detectors_to_run`、`min_confidence_map`
