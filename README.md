# 眼部遮挡与多模型人脸检测

本项目有两个主流程，统一入口是 `face_detection_batch.py`：

1. `detect_compare`：人脸检测可视化对比（多检测器）
2. `eye_mask`：眼部黑条打码

详细模型下载与放置路径见：

- `face_detection/README.md`
- 其中“所有检测器的加载方式（逐个说明）”章节包含每个检测器的导入、权重路径和加载逻辑

---

## 1. 快速开始

```bash
cd /home/leo/myproject
pip install -r requirements.txt
pip install -r requirements-detectors.txt
```

运行：

```bash
python face_detection_batch.py
```

在 `face_detection_batch.py` 中改 `RUN_MODE`：

- `RUN_MODE = "detect_compare"`
- `RUN_MODE = "eye_mask"`

---

## 2. 检测对比说明

检测流程主文件：`face_detection/batch.py`

关键配置：

- `detectors_to_run`：本次要跑哪些检测器
- `min_confidence_map`：每个检测器阈值
- `random_group_count`：随机抽样组数（0 表示全量）
- `draw_landmarks` / `draw_part_boxes`：可视化开关

当前支持检测器名：

- `mtcnn`
- `retinaface`
- `scrfd`
- `blazeface`
- `mediapipe-landmarker`
- `yolov8-face`
- `centerface`

输出目录按检测器分子目录，且保留原始数据树。

---

## 3. 关键模型说明

- `blazeface`：MediaPipe FaceDetection（速度快，关键点少）
- `mediapipe-landmarker`：MediaPipe FaceLandmarker（468 点，更细粒度）
- `yolov8-face`：当前建议使用带 5 点关键点的权重（详见 `face_detection/README.md` 的下载与验证命令）
- `centerface`：已改为直接加载本地 ONNX（`model/centerface/centerface.onnx`），不需要 `pip install centerface`

---

## 4. 常见问题

1. `ModuleNotFoundError`：确认运行脚本的 Python 和安装依赖的 Python 是同一个环境。
2. BlazeFace 报 `RET_CHECK ... dims[1]==num_boxes_`：请使用 `blaze_face_short_range.tflite`，不要用旧的 `face_detection_*.tflite`。
3. 单个检测器初始化失败：脚本会跳过并继续跑下一个，终端会打印具体异常。
