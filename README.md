# 眼部遮挡批处理

这个项目用于批量给人脸图片做眼部黑色遮挡，并保留原始目录树输出。

当前流程：

- `MediaPipe` 作为主检测
- `MTCNN` 作为兜底检测
- 左右眼分别遮挡，中间尽量留出鼻梁

## 目录结构

```text
myproject/
├── mask_eyes_batch.py   # 批处理入口
├── mask_check.py        # 检测该遮哪里
├── mask_apply.py        # 真正画黑块
├── requirements.txt
├── .gitignore
└── model/
    ├── face_landmarker.task
    ├── pnet.pt
    ├── rnet.pt
    └── onet.pt
```

## 环境配置

### venv

```bash
cd /home/leo/myproject
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### conda

```bash
cd /home/leo/myproject
conda create -n eye-mask python=3.10 -y
conda activate eye-mask
pip install -U pip
pip install -r requirements.txt
```

说明：

- `torch` 如果要用 GPU，建议按 PyTorch 官方页面选对应 CUDA 版本。
- `facenet-pytorch` 需要 `requests`，这里已经写进 `requirements.txt`。

## 模型下载

模型统一放到 `model/` 目录。

### 1. MediaPipe 模型

下载 `face_landmarker.task` 到：

```text
model/face_landmarker.task
```

官方地址：

```text
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

下载示例：

```bash
mkdir -p model
wget -O model/face_landmarker.task   https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

### 2. MTCNN 模型

需要这 3 个文件：

```text
model/pnet.pt
model/rnet.pt
model/onet.pt
```

最简单的方法是先让 `facenet-pytorch` 自动下载，再复制到 `model/`：

```bash
python -c "from facenet_pytorch import MTCNN; MTCNN()"
find ~/.cache -name pnet.pt
find ~/.cache -name rnet.pt
find ~/.cache -name onet.pt
```

找到后复制：

```bash
cp <pnet.pt路径> model/pnet.pt
cp <rnet.pt路径> model/rnet.pt
cp <onet.pt路径> model/onet.pt
```

## 使用方法

1. 打开 `mask_eyes_batch.py`
2. 修改 `main()` 里的输入和输出目录
3. 运行：

```bash
python mask_eyes_batch.py
```

## 模块说明

### `mask_check.py`

负责“检查该图应该遮哪里”：

- `create_face_landmarker()`：创建 MediaPipe 检测器
- `create_mtcnn_detector()`：创建 MTCNN 检测器
- `detect_eye_mask_boxes()`：统一返回眼罩框

### `mask_apply.py`

负责“真正打码”：

- `apply_padding_and_keep_gap()`：扩边并尽量保留鼻梁间隙
- `draw_black_eye_masks()`：在图像上画黑色遮挡块

### `mask_eyes_batch.py`

负责批处理：

- 遍历输入目录
- 保留原始目录树输出
- 打印每张图的处理状态

## 当前规则

- `MediaPipe` 优先，失败后再走 `MTCNN`
- 文件名是 `4` 或 `5` 时，`MTCNN` 强制按侧脸单眼框处理
- `MTCNN` 的框大小按人脸比例动态计算，不是固定像素
