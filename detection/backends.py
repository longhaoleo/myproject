"""
多模型人脸检测后端实现。

支持：
- MTCNN
- RetinaFace
- SCRFD
- BlazeFace（MediaPipe FaceDetection）
- YOLOv8-Face（改造版，支持关键点）
- CenterFace

约定：
- 所有 detect() 接收 OpenCV BGR 图像；
- 返回 FaceDetection 列表（box/score/landmarks 坐标均为像素）。
"""

from dataclasses import dataclass
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from .types import FaceDetection


def _clip_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int):
    """裁剪框到图像边界并返回 int 坐标。"""
    # 把框裁剪到图像边界内，并转为 int。
    ix1 = max(0, min(w - 1, int(round(x1))))
    iy1 = max(0, min(h - 1, int(round(y1))))
    ix2 = max(0, min(w - 1, int(round(x2))))
    iy2 = max(0, min(h - 1, int(round(y2))))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1, iy1, ix2, iy2


def _clip_point(x: float, y: float, w: int, h: int):
    """裁剪关键点到图像边界。"""
    # 把关键点裁剪到图像边界内。
    ix = max(0, min(w - 1, int(round(x))))
    iy = max(0, min(h - 1, int(round(y))))
    return ix, iy


def _as_float(value, default: float = 0.0) -> float:
    """安全转 float，失败返回默认值。"""
    try:
        return float(value)
    except Exception:
        return default


def _yolo_class_is_face(names, cls_arr, i: int) -> bool:
    """YOLO 多类别时仅保留 face 类别。"""
    if not isinstance(names, dict) or len(names) == 0:
        return True
    class_id = int(cls_arr[i]) if i < len(cls_arr) else 0
    label = str(names.get(class_id, "")).lower()
    return (not label) or ("face" in label)


# FaceLandmarker 的语义关键点索引（从 468 点提取左右眼/鼻子/嘴）
LEFT_EYE_INDICES = [
    33, 246, 161, 160, 159, 158, 157, 173,
    133, 155, 154, 153, 145, 144, 163, 7,
]
RIGHT_EYE_INDICES = [
    362, 398, 384, 385, 386, 387, 388, 466,
    263, 249, 390, 373, 374, 380, 381, 382,
]
NOSE_CENTER_INDICES = [1, 2, 4, 5, 6, 168, 195, 197]
MOUTH_CENTER_INDICES = [13, 14, 17, 78, 308, 81, 311]
MOUTH_LEFT_INDEX = 61
MOUTH_RIGHT_INDEX = 291


@dataclass
class BaseDetector:
    detector_name: str

    def close(self) -> None:
        """释放资源（多数后端无需显式释放）。"""
        # 多数 Python 后端无需显式释放，保留统一接口。
        return None


class MTCNNDetector(BaseDetector):
    """facenet-pytorch MTCNN。"""

    def __init__(self, model_dir: Path, min_confidence: float = 0.6):
        """初始化 MTCNN，并设置置信度阈值。"""
        super().__init__(detector_name="MTCNN")
        try:
            import torch
            from facenet_pytorch import MTCNN
        except ImportError as exc:
            raise ImportError("MTCNN 需要安装 torch 与 facenet-pytorch。") from exc

        self.min_confidence = float(min_confidence)
        mtcnn_dir = model_dir / "mtcnn"
        mtcnn_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(mtcnn_dir))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = MTCNN(keep_all=True, device=device)

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        """返回检测到的人脸框与 5 点关键点。"""
        # MTCNN 自带 5 点关键点：left_eye/right_eye/nose/mouth_left/mouth_right
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, probs, landmarks = self._model.detect(rgb, landmarks=True)
        if boxes is None:
            return []

        results: list[FaceDetection] = []
        for i, box in enumerate(boxes):
            score = _as_float(probs[i]) if probs is not None else 0.0
            if score < self.min_confidence:
                continue

            clipped = _clip_box(box[0], box[1], box[2], box[3], w, h)
            if clipped is None:
                continue

            lm_dict: dict[str, tuple[int, int]] = {}
            if landmarks is not None and i < len(landmarks):
                points = landmarks[i]
                names = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
                for name, p in zip(names, points):
                    lm_dict[name] = _clip_point(p[0], p[1], w, h)

            results.append(FaceDetection(box=clipped, score=score, landmarks=lm_dict))
        return results


class RetinaFaceDetector(BaseDetector):
    """RetinaFace（pip 包名：retina-face；导入名：retinaface）。"""

    def __init__(self, model_dir: Path, min_confidence: float = 0.6):
        """初始化 RetinaFace，并设置权重目录映射。"""
        super().__init__(detector_name="RetinaFace")
        try:
            # 先显式导入 tensorflow。
            # 某些 retina-face 版本在未先导入 tensorflow 时，
            # 会错误地报 `No module named tensorflow.keras`。
            import tensorflow as tf  # noqa: F401
            from retinaface import RetinaFace
        except ImportError as exc:
            raise ImportError("RetinaFace 需要安装 retina-face（导入名为 retinaface）。") from exc

        self.min_confidence = float(min_confidence)
        self.model_dir = model_dir / "retinaface"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # RetinaFace 读取路径固定为: <DEEPFACE_HOME>/.deepface/weights/retinaface.h5
        # 这里把 DEEPFACE_HOME 指向 model/retinaface，避免散落到其他目录。
        os.environ["DEEPFACE_HOME"] = str(self.model_dir.resolve())

        # 权重放在 model/retinaface/retinaface.h5。
        local_weight = self.model_dir / "retinaface.h5"
        expected_weight = self.model_dir / ".deepface" / "weights" / "retinaface.h5"
        expected_weight.parent.mkdir(parents=True, exist_ok=True)
        if local_weight.exists() and not expected_weight.exists():
            try:
                expected_weight.symlink_to(local_weight.resolve())
            except OSError:
                shutil.copy2(local_weight, expected_weight)

        self._retina = RetinaFace

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        """返回检测到的人脸框与关键点。"""
        # RetinaFace 的 keypoint 命名依赖其内部实现，统一在 visualize 里做别名归一化。
        h, w = image.shape[:2]
        raw = self._retina.detect_faces(image)
        if raw is None:
            return []

        if isinstance(raw, dict):
            items = raw.values()
        elif isinstance(raw, list):
            items = raw
        else:
            return []

        results: list[FaceDetection] = []
        for item in items:
            if not isinstance(item, dict):
                continue

            score = _as_float(item.get("score", 0.0))
            if score < self.min_confidence:
                continue

            area = item.get("facial_area")
            if not area or len(area) < 4:
                continue
            clipped = _clip_box(area[0], area[1], area[2], area[3], w, h)
            if clipped is None:
                continue

            lm_dict: dict[str, tuple[int, int]] = {}
            landmarks = item.get("landmarks")
            if isinstance(landmarks, dict):
                for key, value in landmarks.items():
                    if not isinstance(value, (list, tuple)) or len(value) < 2:
                        continue
                    lm_dict[str(key)] = _clip_point(value[0], value[1], w, h)

            results.append(FaceDetection(box=clipped, score=score, landmarks=lm_dict))
        return results


class SCRFDDetector(BaseDetector):
    """SCRFD（insightface FaceAnalysis）。"""

    def __init__(self, model_dir: Path, min_confidence: float = 0.6, det_size: tuple[int, int] = (640, 640)):
        """初始化 SCRFD（FaceAnalysis），可指定输入尺寸。"""
        super().__init__(detector_name="SCRFD")
        try:
            import torch
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ImportError("SCRFD 需要安装 insightface（以及 onnxruntime）。") from exc

        self.min_confidence = float(min_confidence)
        providers = ["CPUExecutionProvider"]
        if torch.cuda.is_available():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        root = model_dir / "scrfd" / "insightface"
        root.mkdir(parents=True, exist_ok=True)
        self._app = FaceAnalysis(name="buffalo_l", root=str(root), providers=providers)
        self._app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=det_size)

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        """返回检测到的人脸框与 5 点关键点。"""
        h, w = image.shape[:2]
        faces = self._app.get(image)
        results: list[FaceDetection] = []

        for face in faces:
            score = _as_float(getattr(face, "det_score", 0.0))
            if score < self.min_confidence:
                continue

            bbox = getattr(face, "bbox", None)
            if bbox is None or len(bbox) < 4:
                continue
            clipped = _clip_box(bbox[0], bbox[1], bbox[2], bbox[3], w, h)
            if clipped is None:
                continue

            lm_dict: dict[str, tuple[int, int]] = {}
            kps = getattr(face, "kps", None)
            if kps is not None and len(kps) >= 5:
                names = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
                for name, p in zip(names, kps):
                    lm_dict[name] = _clip_point(p[0], p[1], w, h)

            results.append(FaceDetection(box=clipped, score=score, landmarks=lm_dict))
        return results


class BlazeFaceDetector(BaseDetector):
    """
    BlazeFace（MediaPipe FaceDetection）。

    兼容两类 mediapipe：
    - 旧接口：mp.solutions.face_detection
    - 新接口：mediapipe.tasks.python.vision.FaceDetector
    """

    def __init__(self, model_dir: Path, min_confidence: float = 0.5):
        """初始化 BlazeFace，优先使用 mediapipe solutions。"""
        super().__init__(detector_name="BlazeFace")
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError("BlazeFace 需要安装 mediapipe。") from exc

        self.min_confidence = float(min_confidence)
        self._mp = mp
        self._legacy_detector = None
        self._task_detector = None
        self._backend = ""

        # 1) 旧版 mediapipe：直接走 solutions。
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
            self._legacy_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=self.min_confidence,
            )
            self._backend = "solutions"
            return

        # 2) 新版 mediapipe：走 tasks FaceDetector（需要本地 tflite）。
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except Exception as exc:
            raise ImportError(
                "当前 mediapipe 不含 solutions，且无法导入 tasks FaceDetector。"
            ) from exc

        model_candidates = [
            # tasks 推荐模型（带 metadata）
            model_dir / "blazeface" / "blaze_face_short_range.tflite",
            model_dir / "blazeface" / "blaze_face_full_range_sparse.tflite",
        ]
        existing_models = [p for p in model_candidates if p.exists()]
        if not existing_models:
            raise FileNotFoundError(
                "BlazeFace(tasks) 未找到模型。请下载到: "
                f"{model_candidates[0]}"
            )

        last_error = None
        for model_path in existing_models:
            try:
                options = mp_vision.FaceDetectorOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                    running_mode=mp_vision.RunningMode.IMAGE,
                    min_detection_confidence=self.min_confidence,
                )
                self._task_detector = mp_vision.FaceDetector.create_from_options(options)

                # 用空图做一次探测，提前过滤与 tasks 不兼容的旧模型文件。
                probe = np.zeros((256, 256, 3), dtype=np.uint8)
                probe_image = self._mp.Image(
                    image_format=self._mp.ImageFormat.SRGB,
                    data=probe,
                )
                self._task_detector.detect(probe_image)

                self._backend = "tasks"
                return
            except Exception as exc:
                if self._task_detector is not None:
                    try:
                        self._task_detector.close()
                    except Exception:
                        pass
                    self._task_detector = None
                last_error = exc

        raise RuntimeError(
            "BlazeFace(tasks) 模型初始化失败。"
            "请使用 mediapipe-models 版本的 blaze_face_short_range.tflite。"
        ) from last_error

    def close(self) -> None:
        """关闭 mediapipe detector 资源。"""
        if self._legacy_detector is not None:
            self._legacy_detector.close()
        if self._task_detector is not None:
            self._task_detector.close()

    def _parse_tasks_keypoints(self, det, w: int, h: int) -> dict[str, tuple[int, int]]:
        """解析 tasks 版关键点为统一语义名称。"""
        # tasks keypoints 可能包含 label，也可能只有顺序位置。
        lm_dict: dict[str, tuple[int, int]] = {}
        keypoints = getattr(det, "keypoints", None) or []
        if not keypoints:
            return lm_dict

        label_map = {
            "left_eye": "left_eye",
            "right_eye": "right_eye",
            "nose": "nose",
            "nose_tip": "nose",
            "mouth": "mouth",
            "mouth_center": "mouth",
            "mouth_left": "mouth_left",
            "mouth_right": "mouth_right",
            "left_mouth": "mouth_left",
            "right_mouth": "mouth_right",
        }
        index_map = {
            0: "right_eye",
            1: "left_eye",
            2: "nose",
            3: "mouth",
        }

        for idx, kp in enumerate(keypoints):
            x = getattr(kp, "x", None)
            y = getattr(kp, "y", None)
            if x is None or y is None:
                continue

            # tasks keypoint 是归一化坐标；保险起见也兼容像素坐标。
            px = int(round(x * w)) if 0.0 <= float(x) <= 1.0 else int(round(x))
            py = int(round(y * h)) if 0.0 <= float(y) <= 1.0 else int(round(y))

            raw_label = str(getattr(kp, "label", "") or "").strip().lower().replace(" ", "_")
            name = label_map.get(raw_label, index_map.get(idx))
            if name is not None:
                lm_dict[name] = _clip_point(px, py, w, h)
        return lm_dict

    def _detect_by_tasks(self, image: np.ndarray) -> list[FaceDetection]:
        """使用 mediapipe tasks 版模型执行检测。"""
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._task_detector.detect(mp_image)
        detections = getattr(result, "detections", None) or []
        if not detections:
            return []

        results: list[FaceDetection] = []
        for det in detections:
            bbox = getattr(det, "bounding_box", None)
            if bbox is None:
                continue

            x1 = _as_float(getattr(bbox, "origin_x", 0.0))
            y1 = _as_float(getattr(bbox, "origin_y", 0.0))
            bw = _as_float(getattr(bbox, "width", 0.0))
            bh = _as_float(getattr(bbox, "height", 0.0))
            clipped = _clip_box(x1, y1, x1 + bw, y1 + bh, w, h)
            if clipped is None:
                continue

            score = 0.0
            categories = getattr(det, "categories", None) or []
            if categories:
                score = _as_float(getattr(categories[0], "score", 0.0), 0.0)

            lm_dict = self._parse_tasks_keypoints(det, w, h)
            results.append(FaceDetection(box=clipped, score=score, landmarks=lm_dict))

        return results

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        """检测入口：按当前后端分支执行。"""
        h, w = image.shape[:2]
        if self._backend == "tasks":
            return self._detect_by_tasks(image)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self._legacy_detector.process(rgb)
        if not result.detections:
            return []

        results: list[FaceDetection] = []
        for det in result.detections:
            rel = det.location_data.relative_bounding_box
            clipped = _clip_box(
                rel.xmin * w,
                rel.ymin * h,
                (rel.xmin + rel.width) * w,
                (rel.ymin + rel.height) * h,
                w,
                h,
            )
            if clipped is None:
                continue

            score = _as_float(det.score[0], 0.0) if det.score else 0.0
            lm_dict: dict[str, tuple[int, int]] = {}
            kps = det.location_data.relative_keypoints
            if len(kps) >= 4:
                lm_dict["right_eye"] = _clip_point(kps[0].x * w, kps[0].y * h, w, h)
                lm_dict["left_eye"] = _clip_point(kps[1].x * w, kps[1].y * h, w, h)
                lm_dict["nose"] = _clip_point(kps[2].x * w, kps[2].y * h, w, h)
                lm_dict["mouth"] = _clip_point(kps[3].x * w, kps[3].y * h, w, h)

            results.append(FaceDetection(box=clipped, score=score, landmarks=lm_dict))
        return results


class MediaPipeLandmarkerDetector(BaseDetector):
    """
    MediaPipe FaceLandmarker（468 点）。

    输出：
    - 人脸框：由 468 点外接矩形生成；
    - 语义点：left_eye/right_eye/nose/mouth/mouth_left/mouth_right。
    """

    def __init__(self, model_dir: Path, min_confidence: float = 0.3):
        """初始化 FaceLandmarker（468 点）。"""
        super().__init__(detector_name="MediaPipe-Landmarker")
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as exc:
            raise ImportError("FaceLandmarker 需要安装 mediapipe。") from exc

        self._mp = mp
        self._landmarker = None
        self._max_sem_index = max(
            max(LEFT_EYE_INDICES),
            max(RIGHT_EYE_INDICES),
            max(NOSE_CENTER_INDICES),
            max(MOUTH_CENTER_INDICES),
            MOUTH_LEFT_INDEX,
            MOUTH_RIGHT_INDEX,
        )

        model_path = model_dir / "mediapipe-landmarker" / "face_landmarker.task"
        if not model_path.exists():
            raise FileNotFoundError(f"未找到 FaceLandmarker 模型: {model_path}")

        conf = float(min_confidence)
        conf = max(0.0, min(1.0, conf))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=5,
            min_face_detection_confidence=conf,
            min_face_presence_confidence=max(0.1, conf),
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        """关闭 landmarker 资源。"""
        if self._landmarker is not None:
            self._landmarker.close()

    @staticmethod
    def _point_from_index(landmarks, idx: int, w: int, h: int):
        """把单个关键点索引映射到像素坐标。"""
        if idx >= len(landmarks):
            return None
        px = int(round(float(landmarks[idx].x) * w))
        py = int(round(float(landmarks[idx].y) * h))
        return _clip_point(px, py, w, h)

    @staticmethod
    def _center_from_indices(landmarks, indices: list[int], w: int, h: int):
        """把多关键点取均值作为中心点。"""
        points: list[tuple[int, int]] = []
        for idx in indices:
            p = MediaPipeLandmarkerDetector._point_from_index(landmarks, idx, w, h)
            if p is not None:
                points.append(p)
        if not points:
            return None
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        return _clip_point(cx, cy, w, h)

    @staticmethod
    def _face_box_from_landmarks(landmarks, w: int, h: int):
        """根据 468 点外接矩形生成人脸框。"""
        xs = [int(round(float(p.x) * w)) for p in landmarks]
        ys = [int(round(float(p.y) * h)) for p in landmarks]
        if not xs or not ys:
            return None
        return _clip_box(min(xs), min(ys), max(xs), max(ys), w, h)

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        """返回检测到的人脸框与语义关键点。"""
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        faces = getattr(result, "face_landmarks", None) or []
        if not faces:
            return []

        outputs: list[FaceDetection] = []
        for landmarks in faces:
            if len(landmarks) <= self._max_sem_index:
                continue

            box = self._face_box_from_landmarks(landmarks, w, h)
            if box is None:
                continue

            lm_dict: dict[str, tuple[int, int]] = {}
            left_eye = self._center_from_indices(landmarks, LEFT_EYE_INDICES, w, h)
            right_eye = self._center_from_indices(landmarks, RIGHT_EYE_INDICES, w, h)
            nose = self._center_from_indices(landmarks, NOSE_CENTER_INDICES, w, h)
            mouth = self._center_from_indices(landmarks, MOUTH_CENTER_INDICES, w, h)
            mouth_left = self._point_from_index(landmarks, MOUTH_LEFT_INDEX, w, h)
            mouth_right = self._point_from_index(landmarks, MOUTH_RIGHT_INDEX, w, h)

            if left_eye is not None:
                lm_dict["left_eye"] = left_eye
            if right_eye is not None:
                lm_dict["right_eye"] = right_eye
            if nose is not None:
                lm_dict["nose"] = nose
            if mouth is not None:
                lm_dict["mouth"] = mouth
            if mouth_left is not None:
                lm_dict["mouth_left"] = mouth_left
            if mouth_right is not None:
                lm_dict["mouth_right"] = mouth_right

            outputs.append(FaceDetection(box=box, score=None, landmarks=lm_dict))
        return outputs


class YoloV8FaceDetector(BaseDetector):
    """
    YOLOv8-Face（改造版）。

    适配思路：
    - 使用 ultralytics YOLO 推理；
    - 读取 bbox；
    - 若模型输出 keypoints，则映射为 left_eye/right_eye/nose/mouth_left/mouth_right。
    """

    def __init__(
        self,
        model_path: Path,
        min_confidence: float = 0.4,
        keypoint_confidence: float = 0.2,
    ):
        """初始化 YOLOv8-Face 模型与阈值。"""
        super().__init__(detector_name="YOLOv8-Face")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("YOLOv8-Face 需要安装 ultralytics。") from exc

        if not model_path.exists():
            raise FileNotFoundError(f"未找到 YOLOv8-Face 模型文件: {model_path}")

        self.min_confidence = float(min_confidence)
        self.keypoint_confidence = float(keypoint_confidence)
        self._model = YOLO(str(model_path))

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        """返回检测到的人脸框与（可选）关键点。"""
        h, w = image.shape[:2]
        pred = self._model(image, verbose=False)[0]
        if pred.boxes is None or len(pred.boxes) == 0:
            return []

        xyxy = pred.boxes.xyxy.cpu().numpy()
        conf = pred.boxes.conf.cpu().numpy() if pred.boxes.conf is not None else np.zeros((len(xyxy),))
        cls = pred.boxes.cls.cpu().numpy() if pred.boxes.cls is not None else np.zeros((len(xyxy),))
        names = getattr(pred, "names", {})

        # 关键点是可选输出；无关键点时只返回框。
        kp_xy = None
        kp_conf = None
        keypoints_obj = getattr(pred, "keypoints", None)
        if keypoints_obj is not None and getattr(keypoints_obj, "xy", None) is not None:
            try:
                kp_xy = keypoints_obj.xy.cpu().numpy()
                if getattr(keypoints_obj, "conf", None) is not None:
                    kp_conf = keypoints_obj.conf.cpu().numpy()
            except Exception:
                kp_xy = None
                kp_conf = None

        results: list[FaceDetection] = []
        for i, box in enumerate(xyxy):
            score = _as_float(conf[i], 0.0)
            if score < self.min_confidence:
                continue
            if not _yolo_class_is_face(names, cls, i):
                continue

            clipped = _clip_box(box[0], box[1], box[2], box[3], w, h)
            if clipped is None:
                continue

            lm_dict: dict[str, tuple[int, int]] = {}
            if kp_xy is not None and i < len(kp_xy):
                points = kp_xy[i]
                confs = kp_conf[i] if kp_conf is not None and i < len(kp_conf) else None

                # 约定顺序：left_eye, right_eye, nose, mouth_left, mouth_right
                names5 = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
                max_k = min(5, len(points))
                for k in range(max_k):
                    if confs is not None and k < len(confs):
                        if _as_float(confs[k], 0.0) < self.keypoint_confidence:
                            continue
                    lm_dict[names5[k]] = _clip_point(points[k][0], points[k][1], w, h)

            results.append(FaceDetection(box=clipped, score=score, landmarks=lm_dict))

        return results


class CenterFaceDetector(BaseDetector):
    """CenterFace（本地 ONNX 版本，无需安装 centerface 包）。"""

    def __init__(self, model_dir: Path, min_confidence: float = 0.4):
        """初始化 CenterFace ONNX 并设置阈值。"""
        super().__init__(detector_name="CenterFace")
        self.min_confidence = float(min_confidence)
        self.model_path = model_dir / "centerface" / "centerface.onnx"
        if not self.model_path.exists():
            raise FileNotFoundError(f"未找到 CenterFace ONNX 模型: {self.model_path}")
        self._net = cv2.dnn.readNetFromONNX(str(self.model_path))

    @staticmethod
    def _transform(h: int, w: int):
        """计算符合 32 对齐的输入尺寸与缩放比例。"""
        img_h_new = int(np.ceil(h / 32) * 32)
        img_w_new = int(np.ceil(w / 32) * 32)
        scale_h = img_h_new / h
        scale_w = img_w_new / w
        return img_h_new, img_w_new, scale_h, scale_w

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, nms_thresh: float):
        """简单 NMS 过滤重叠框。"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(scores)[::-1]
        num_detections = boxes.shape[0]
        suppressed = np.zeros((num_detections,), dtype=np.bool_)

        keep = []
        for _i in range(num_detections):
            i = order[_i]
            if suppressed[i]:
                continue
            keep.append(i)

            ix1, iy1, ix2, iy2 = x1[i], y1[i], x2[i], y2[i]
            iarea = areas[i]

            for _j in range(_i + 1, num_detections):
                j = order[_j]
                if suppressed[j]:
                    continue
                xx1 = max(ix1, x1[j])
                yy1 = max(iy1, y1[j])
                xx2 = min(ix2, x2[j])
                yy2 = min(iy2, y2[j])
                ww = max(0, xx2 - xx1 + 1)
                hh = max(0, yy2 - yy1 + 1)
                inter = ww * hh
                ovr = inter / (iarea + areas[j] - inter)
                if ovr >= nms_thresh:
                    suppressed[j] = True
        return keep

    @staticmethod
    def _decode(heatmap, scale, offset, landmark, size, threshold=0.1):
        """解码 CenterFace 输出为框与关键点。"""
        heatmap = np.squeeze(heatmap)
        scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
        offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
        c0, c1 = np.where(heatmap > threshold)
        boxes, lms = [], []

        for i in range(len(c0)):
            s0 = np.exp(scale0[c0[i], c1[i]]) * 4
            s1 = np.exp(scale1[c0[i], c1[i]]) * 4
            o0 = offset0[c0[i], c1[i]]
            o1 = offset1[c0[i], c1[i]]
            score = heatmap[c0[i], c1[i]]

            x1 = max(0, (c1[i] + o1 + 0.5) * 4 - s1 / 2)
            y1 = max(0, (c0[i] + o0 + 0.5) * 4 - s0 / 2)
            x1 = min(x1, size[1])
            y1 = min(y1, size[0])
            boxes.append([x1, y1, min(x1 + s1, size[1]), min(y1 + s0, size[0]), score])

            lm = []
            for j in range(5):
                lm.append(landmark[0, j * 2 + 1, c0[i], c1[i]] * s1 + x1)
                lm.append(landmark[0, j * 2, c0[i], c1[i]] * s0 + y1)
            lms.append(lm)

        if not boxes:
            return (
                np.empty(shape=[0, 5], dtype=np.float32),
                np.empty(shape=[0, 10], dtype=np.float32),
            )

        boxes_arr = np.asarray(boxes, dtype=np.float32)
        lms_arr = np.asarray(lms, dtype=np.float32)
        keep = CenterFaceDetector._nms(boxes_arr[:, :4], boxes_arr[:, 4], 0.3)
        return boxes_arr[keep, :], lms_arr[keep, :]

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        """执行 CenterFace 检测并输出人脸框与 5 点关键点。"""
        h, w = image.shape[:2]
        img_h_new, img_w_new, scale_h, scale_w = self._transform(h, w)

        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0,
            size=(img_w_new, img_h_new),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        self._net.setInput(blob)
        out_names = self._net.getUnconnectedOutLayersNames()
        outs = self._net.forward(out_names)
        if len(outs) < 4:
            return []
        heatmap, scale, offset, lms = outs[0], outs[1], outs[2], outs[3]

        dets, lms = self._decode(
            heatmap=heatmap,
            scale=scale,
            offset=offset,
            landmark=lms,
            size=(img_h_new, img_w_new),
            threshold=self.min_confidence,
        )
        if dets is None or len(dets) == 0:
            return []

        # 映射回原图尺寸
        dets[:, 0:4:2] = dets[:, 0:4:2] / scale_w
        dets[:, 1:4:2] = dets[:, 1:4:2] / scale_h
        if lms is not None and len(lms) > 0:
            lms[:, 0:10:2] = lms[:, 0:10:2] / scale_w
            lms[:, 1:10:2] = lms[:, 1:10:2] / scale_h

        results: list[FaceDetection] = []
        for i, det in enumerate(dets):
            if len(det) < 5:
                continue
            score = _as_float(det[4], 0.0)
            if score < self.min_confidence:
                continue

            clipped = _clip_box(det[0], det[1], det[2], det[3], w, h)
            if clipped is None:
                continue

            lm_dict: dict[str, tuple[int, int]] = {}
            if lms is not None and i < len(lms):
                points = lms[i]
                if len(points) >= 10:
                    names = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
                    for j, name in enumerate(names):
                        lm_dict[name] = _clip_point(points[2 * j], points[2 * j + 1], w, h)

            results.append(FaceDetection(box=clipped, score=score, landmarks=lm_dict))
        return results
