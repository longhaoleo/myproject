# face_detection 模块说明

该目录包含两个流程：

1. 人脸检测可视化对比：`batch.py`
2. 眼部打码：`masking.py`

统一入口：`python face_detection_batch.py`

---

## 1. 运行前准备

```bash
cd /home/leo/myproject
pip install -r requirements.txt
pip install -r requirements-detectors.txt
```

---

## 2. 模型文件放置

模型统一放在项目根目录 `model/`。

### 2.1 MediaPipe FaceLandmarker（细粒度关键点，468点）

- 文件：`model/face_landmarker.task`

```bash
wget -O model/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

### 2.2 BlazeFace（MediaPipe FaceDetection）

- 文件：`model/blazeface/blaze_face_short_range.tflite`

```bash
mkdir -p model/blazeface
wget -O model/blazeface/blaze_face_short_range.tflite \
  https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite
```

旧版 `face_detection_*.tflite` 不建议使用，容易触发：
`RET_CHECK ... dims[1]==num_boxes_`

```bash
rm -f model/blazeface/face_detection_short_range.tflite
rm -f model/blazeface/face_detection_full_range_sparse.tflite
```

### 2.3 CenterFace（本地 ONNX 版）

- 文件：`model/centerface/centerface.onnx`
- 说明：不需要 `pip install centerface`

```bash
mkdir -p model/centerface
wget -O model/centerface/centerface.onnx \
  https://github.com/Star-Clouds/CenterFace/raw/master/models/onnx/centerface.onnx
```

### 2.4 其他模型（按需）

- RetinaFace：`model/retinaface.h5`
- SCRFD：`model/insightface/`
- YOLOv8-Face（带 5 点关键点）：`model/yolov8_face.pt`
- MTCNN：`model/pnet.pt`、`model/rnet.pt`、`model/onet.pt`

推荐使用带关键点版本（已验证可输出 `pred.keypoints`）：

```bash
# 若缺少 gdown：pip install gdown

# 备份旧权重（可选）
cp model/yolov8_face.pt model/yolov8_face.bak.pt

# 下载并覆盖为关键点版
python -m gdown 'https://drive.google.com/uc?id=1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb' -O model/yolov8_face.pt
```

---

## 3. 运行方式

在 `face_detection_batch.py` 里设置：

- `RUN_MODE = "detect_compare"`：检测可视化
- `RUN_MODE = "eye_mask"`：眼部打码

然后执行：

```bash
python face_detection_batch.py
```

---

## 4. 检测对比配置

配置文件：`face_detection/batch.py`

- `detectors_to_run`：本次运行的检测器列表
- `min_confidence_map`：各检测器阈值
- `random_group_count`：随机抽样组数
- `draw_landmarks` / `draw_part_boxes`：可视化开关

注意：当前仓库里 `detectors_to_run` 是示例配置，你可以按实验需要随时取消注释或增删检测器。

已支持检测器：

- `mtcnn`
- `retinaface`
- `scrfd`
- `blazeface`
- `mediapipe-landmarker`
- `yolov8-face`
- `centerface`

---

## 5. 所有检测器的加载方式（逐个说明）

统一入口在 `face_detection/factory.py` 的 `create_face_detector(...)`，按名称分发到各后端类。

### 5.1 `mtcnn`

- 后端类：`MTCNNDetector`
- 依赖导入：`torch`、`facenet_pytorch.MTCNN`
- 加载方式：
  - 初始化时执行 `torch.hub.set_dir(model_dir)`，把 torch hub 缓存目录指向项目 `model/`
  - 用 `MTCNN(keep_all=True, device=...)` 创建检测器
- 权重来源：
  - 默认由 `facenet-pytorch` 自动管理缓存（在你设置的 hub 目录下）
  - 这个检测后端不强制要求你手动放 `pnet/rnet/onet`（打码流程另说）

### 5.2 `retinaface`

- 后端类：`RetinaFaceDetector`
- 依赖导入：`retinaface`（pip 包名 `retina-face`）
- 加载方式：
  - 设置 `DEEPFACE_HOME = model_dir`
  - RetinaFace 期望权重路径是 `model/.deepface/weights/retinaface.h5`
  - 若你提供了 `model/retinaface.h5`，代码会自动软链接/复制到上述期望路径
- 权重来源：
  - 推荐手动放置 `model/retinaface.h5`
  - 也可由第三方库自身机制下载（取决于环境）

### 5.3 `scrfd`

- 后端类：`SCRFDDetector`
- 依赖导入：`insightface.app.FaceAnalysis`、`onnxruntime`、`torch`
- 加载方式：
  - `FaceAnalysis(name="buffalo_l", root="model/insightface", providers=...)`
  - `prepare(det_size=(640,640))`
  - GPU 可用时优先 `CUDAExecutionProvider`，否则 CPU
- 权重来源：
  - `insightface` 会在 `model/insightface` 下自动下载/加载
  - 你也可以提前手动放模型文件

### 5.4 `blazeface`

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

### 5.5 `mediapipe-landmarker`（别名：`face-landmarker` / `landmarker`）

- 后端类：`MediaPipeLandmarkerDetector`
- 依赖导入：`mediapipe.tasks.python.vision.FaceLandmarker`
- 加载方式：
  - 强依赖本地模型：`model/face_landmarker.task`
  - `FaceLandmarkerOptions(..., num_faces=5, min_face_detection_confidence=...)`
  - 输出来自 468 点，再映射为 `left_eye/right_eye/nose/mouth/mouth_left/mouth_right`
- 适用：
  - 想看更细粒度关键点时用它

### 5.6 `yolov8-face`

- 后端类：`YoloV8FaceDetector`
- 依赖导入：`ultralytics.YOLO`
- 加载方式：
  - 默认读取 `model/yolov8_face.pt`（可在 `batch.py` 改 `yolov8_model_path`）
  - 推理后读取 bbox；若模型包含关键点输出则映射到眼鼻嘴语义点
- 权重来源：
  - 必须有本地 `.pt` 权重文件
  - 推荐 `derronqi/yolov8-face` 的 `yolov8n-face.pt`（5点关键点）
- 快速验证是否带关键点：

```bash
python - <<'PY'
from ultralytics import YOLO
import cv2
m = YOLO('model/yolov8_face.pt')
img = cv2.imread('/home/leo/datasets/deformity/1/术前/1.JPG')
r = m(img, verbose=False)[0]
print('has_keypoints:', getattr(r, 'keypoints', None) is not None and getattr(r.keypoints, 'xy', None) is not None)
PY
```

### 5.7 `centerface`

- 后端类：`CenterFaceDetector`
- 依赖导入：`cv2.dnn`（不依赖 `pip install centerface`）
- 加载方式：
  - 强依赖本地 ONNX：`model/centerface/centerface.onnx`
  - `cv2.dnn.readNetFromONNX(...)` 直接加载
  - 前处理/解码/NMS 在后端代码内实现（输出框 + 5 点）

---

## 6. 代码定位（便于你后续改）

- 工厂分发：`face_detection/factory.py`
- 各检测器实现：`face_detection/backends.py`
- 运行列表与阈值：`face_detection/batch.py` 里的 `detectors_to_run`、`min_confidence_map`
