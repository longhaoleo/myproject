"""
人脸分割接口层（SAM 版）。

设计目标：
1. 输入：检测器输出的人脸框 + 关键点；
2. 输出：每张脸一张合并后的二值 mask（用于叠加可视化）；
3. 同时保留每个部位（face / left_eye / right_eye / nose / mouth）的单独 mask，
   便于后续接更精细的业务逻辑。

说明：
- 本模块只负责“把检测框/关键点转成 SAM 提示框并分割”；
- 分割结果的标准化与落盘在 sam_mask.py 里处理。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from .settings import adjust_box_for_view, get_view_offset_map, get_view_scale_map, resolve_part_offset
from .types import FaceDetection


# 单张二值 mask：H x W，元素为 bool/0/1
BinaryMask = np.ndarray

# 部位名 -> 二值 mask
PartMaskMap = dict[str, BinaryMask]


PART_NAMES = ("face", "left_eye", "right_eye", "nose", "mouth")

CANONICAL_LANDMARKS = {
    "left_eye",
    "right_eye",
    "nose",
    "mouth",
    "mouth_left",
    "mouth_right",
}


class FaceSegmenter(Protocol):
    """统一分割器接口。"""

    segmenter_name: str

    def segment(
        self,
        image: np.ndarray,
        detections: list[FaceDetection],
    ) -> list[BinaryMask | None]:
        """
        返回每张脸的一张“合并 mask”。
        合并规则由各具体实现决定（例如 face + eyes + nose + mouth 的并集）。
        """
        ...

    def close(self) -> None:
        ...


def build_part_boxes_for_detections(
    detections: list[FaceDetection],
    image_w: int,
    image_h: int,
    part_names: tuple[str, ...] = PART_NAMES,
    part_scale_by_view: dict[str, dict[str, tuple[float, float]]] | None = None,
    part_offset_by_view: dict[str, dict[str, tuple[float, float]]] | None = None,
    part_offset_mode: str = "ratio",
) -> list[dict[str, tuple[int, int, int, int]]]:
    """
    根据检测结果生成每张脸的部位提示框（不执行 SAM）。

    返回：
    - list[dict]，每个元素对应一张脸的 {part_name: box}
    """
    boxes_list: list[dict[str, tuple[int, int, int, int]]] = []
    for det in detections:
        boxes_list.append(
            _part_prompt_boxes(
                det,
                image_w=image_w,
                image_h=image_h,
                part_names=part_names,
                view_id=getattr(det, "view_id", None),
                part_scale_by_view=part_scale_by_view,
                part_offset_by_view=part_offset_by_view,
                part_offset_mode=part_offset_mode,
            )
        )
    return boxes_list


def _normalize_landmarks(landmarks: dict[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
    """把不同后端关键点名归一化到统一语义。"""
    # 把不同后端关键点名归一化到统一语义。
    normalized: dict[str, tuple[int, int]] = {}
    for raw_name, point in landmarks.items():
        key = str(raw_name).strip().lower().replace("-", "_").replace(" ", "_")
        if key in CANONICAL_LANDMARKS:
            normalized[key] = point
    return normalized


def _clip_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int] | None:
    """裁剪框到图像边界，非法则返回 None。"""
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(0, min(w - 1, int(round(x2))))
    y2 = max(0, min(h - 1, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _box_from_center(
    cx: int,
    cy: int,
    bw: int,
    bh: int,
    image_w: int,
    image_h: int,
) -> tuple[int, int, int, int] | None:
    """由中心点和尺寸构造框，并裁剪到图像边界。"""
    half_w = max(1, bw // 2)
    half_h = max(1, bh // 2)
    return _clip_box(
        cx - half_w,
        cy - half_h,
        cx + half_w,
        cy + half_h,
        image_w,
        image_h,
    )


def _part_prompt_boxes(
    det: FaceDetection,
    image_w: int,
    image_h: int,
    part_names: tuple[str, ...],
    view_id: str | None = None,
    part_scale_by_view: dict[str, dict[str, tuple[float, float]]] | None = None,
    part_offset_by_view: dict[str, dict[str, tuple[float, float]]] | None = None,
    part_offset_mode: str = "ratio",
) -> dict[str, tuple[int, int, int, int]]:
    """
    基于检测结果生成 SAM 的部位提示框。

    规则：
    - `face` 用检测框（视角缩放 + 方形化）；
    - `left_eye/right_eye/nose/mouth` 优先用关键点 + 人脸比例估计；
    - 若缺少关键点，则只返回能构造出来的部位框。
    """
    boxes: dict[str, tuple[int, int, int, int]] = {}
    face_box = _clip_box(*det.box, image_w, image_h)
    if face_box is None:
        return boxes

    # 人脸框尺寸用于估计眼/鼻/口的“提示框”大小。
    face_x1, face_y1, face_x2, face_y2 = face_box
    face_w = max(1, face_x2 - face_x1)
    face_h = max(1, face_y2 - face_y1)

    if "face" in part_names:
        scale_map = get_view_scale_map(view_id, part_scale_by_view)
        offset_map = get_view_offset_map(view_id, part_offset_by_view)
        boxes["face"] = adjust_box_for_view(
            box=face_box,
            part_name="face",
            image_w=image_w,
            image_h=image_h,
            scale_map=scale_map,
            offset_map=offset_map,
            part_offset_mode=part_offset_mode,
        )

    if not det.landmarks:
        return boxes

    lm = _normalize_landmarks(det.landmarks)

    # 经验比例：让部位框覆盖关键点周围区域，尽量少吃到鼻梁。
    eye_w = max(12, int(face_w * 0.22))
    eye_h = max(10, int(face_h * 0.14))
    scale_map = get_view_scale_map(view_id, part_scale_by_view)
    eye_sx, eye_sy = scale_map.get("left_eye", (1.0, 1.0))
    if "left_eye" in part_names and "left_eye" in lm:
        box = _box_from_center(
            lm["left_eye"][0],
            lm["left_eye"][1],
            max(1, int(round(eye_w * float(eye_sx)))),
            max(1, int(round(eye_h * float(eye_sy)))),
            image_w,
            image_h,
        )
        if box is not None:
            boxes["left_eye"] = box
    eye_sx, eye_sy = scale_map.get("right_eye", (1.0, 1.0))
    if "right_eye" in part_names and "right_eye" in lm:
        box = _box_from_center(
            lm["right_eye"][0],
            lm["right_eye"][1],
            max(1, int(round(eye_w * float(eye_sx)))),
            max(1, int(round(eye_h * float(eye_sy)))),
            image_w,
            image_h,
        )
        if box is not None:
            boxes["right_eye"] = box

    # 鼻子框：用鼻尖为中心的小框。
    if "nose" in part_names and "nose" in lm:
        # 鼻子框略宽一些，尤其水平向，避免漏到鼻翼。
        nose_w = max(16, int(face_w * 0.26))
        nose_h = max(12, int(face_h * 0.22))
        nose_sx, nose_sy = scale_map.get("nose", (1.0, 1.0))
        box = _box_from_center(
            lm["nose"][0],
            lm["nose"][1],
            max(1, int(round(nose_w * float(nose_sx)))),
            max(1, int(round(nose_h * float(nose_sy)))),
            image_w,
            image_h,
        )
        if box is not None:
            boxes["nose"] = box

    # 嘴巴框：优先由左右嘴角合成，侧脸情况下更稳。
    if "mouth" in part_names:
        mouth_sx, mouth_sy = scale_map.get("mouth", (1.0, 1.0))
        if "mouth_left" in lm and "mouth_right" in lm:
            mlx, mly = lm["mouth_left"]
            mrx, mry = lm["mouth_right"]
            pad_x = max(6, int(face_w * 0.05))
            half_h = max(6, int(face_h * 0.08))
            pad_x = int(round(pad_x * float(mouth_sx)))
            half_h = int(round(half_h * float(mouth_sy)))
            box = _clip_box(
                min(mlx, mrx) - pad_x,
                int((mly + mry) / 2) - half_h,
                max(mlx, mrx) + pad_x,
                int((mly + mry) / 2) + half_h,
                image_w,
                image_h,
            )
            if box is not None:
                boxes["mouth"] = box
        elif "mouth" in lm:
            mouth_w = max(14, int(face_w * 0.30))
            mouth_h = max(10, int(face_h * 0.16))
            box = _box_from_center(
                lm["mouth"][0],
                lm["mouth"][1],
                max(1, int(round(mouth_w * float(mouth_sx)))),
                max(1, int(round(mouth_h * float(mouth_sy)))),
                image_w,
                image_h,
            )
            if box is not None:
                boxes["mouth"] = box

    return boxes


def _best_sam_mask(
    masks: np.ndarray,
    scores: np.ndarray | None,
) -> BinaryMask | None:
    """多候选 mask 中选最佳（按最高分）。"""
    if masks is None or len(masks) == 0:
        return None
    if scores is None or len(scores) == 0:
        return masks[0].astype(bool)
    best_idx = int(np.argmax(scores))
    return masks[best_idx].astype(bool)


@dataclass
class NoopSegmenter:
    """空实现：保持现有检测流程不变。"""

    segmenter_name: str = "none"

    def segment(
        self,
        image: np.ndarray,
        detections: list[FaceDetection],
    ) -> list[BinaryMask | None]:
        """空实现：保持与检测流程兼容。"""
        return [None for _ in detections]

    def close(self) -> None:
        """空实现：保持统一接口。"""
        return None


@dataclass
class MobileSamSegmenter:
    """
    MobileSAM 分割实现。

    输出：
    - segment() 返回每张脸合并后的 mask；
    - last_part_masks 保存最近一次推理的“每张脸-每个部位”mask 明细。
    """

    model_path: Path
    model_type: str = "vit_t"
    device: str | None = None
    multimask_output: bool = True
    part_names: tuple[str, ...] = PART_NAMES
    # 每个视角对每个部位的缩放系数（sx, sy）
    part_scale_by_view: dict[str, dict[str, tuple[float, float]]] = field(default_factory=dict)
    # 每个视角对每个部位的偏移（dx, dy）
    part_offset_by_view: dict[str, dict[str, tuple[float, float]]] = field(default_factory=dict)
    # 偏移模式保留字段（固定为 ratio）
    part_offset_mode: str = "ratio"
    segmenter_name: str = "mobile-sam"
    last_part_masks: list[PartMaskMap] = field(default_factory=list)

    def __post_init__(self) -> None:
        """加载 MobileSAM 模型并创建预测器。"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"未找到 MobileSAM 权重: {self.model_path}")

        try:
            import torch
        except ImportError as exc:
            raise ImportError("MobileSAM 需要 torch，请先安装 torch。") from exc

        predictor_cls = None
        sam_registry = None
        from mobile_sam import SamPredictor, sam_model_registry
        predictor_cls = SamPredictor
        sam_registry = sam_model_registry

        use_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        self._device = use_device

        if self.model_type not in sam_registry:
            raise ValueError(
                f"model_type={self.model_type} 不在 sam_model_registry 中，"
                f"可选: {list(sam_registry.keys())}"
            )

        sam = sam_registry[self.model_type](checkpoint=str(self.model_path))
        sam.to(device=use_device)
        self._predictor = predictor_cls(sam)

    def _predict_with_box(
        self,
        box: tuple[int, int, int, int],
    ) -> BinaryMask | None:
        """对单个提示框执行 SAM 推理。"""
        box_arr = np.array([box[0], box[1], box[2], box[3]], dtype=np.float32)
        masks, scores, _ = self._predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_arr,
            multimask_output=self.multimask_output,
        )
        return _best_sam_mask(masks, scores)

    def segment(
        self,
        image: np.ndarray,
        detections: list[FaceDetection],
    ) -> list[BinaryMask | None]:
        """按人脸检测结果生成部位提示框并执行 SAM 分割。"""
        if len(detections) == 0:
            self.last_part_masks = []
            return []

        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self._predictor.set_image(rgb)

        merged_masks: list[BinaryMask | None] = []
        all_part_masks: list[PartMaskMap] = []

        for det in detections:
            part_boxes = _part_prompt_boxes(
                det,
                image_w=w,
                image_h=h,
                part_names=self.part_names,
                view_id=getattr(det, "view_id", None),
                part_scale_by_view=self.part_scale_by_view,
                part_offset_by_view=self.part_offset_by_view,
                part_offset_mode=self.part_offset_mode,
            )
            part_masks: PartMaskMap = {}

            for part_name, box in part_boxes.items():
                mask = self._predict_with_box(box)
                if mask is not None:
                    part_masks[part_name] = mask

            all_part_masks.append(part_masks)

            if not part_masks:
                merged_masks.append(None)
                continue

            merged = np.zeros((h, w), dtype=bool)
            for m in part_masks.values():
                merged |= m.astype(bool)
            merged_masks.append(merged)

        self.last_part_masks = all_part_masks
        return merged_masks

    def close(self) -> None:
        """预留 close 接口（当前模型无需释放）。"""
        # 目前 predictor/sam 无显式 close，预留接口保持统一。
        return None


def create_face_segmenter(
    segmenter_name: str,
    model_dir: Path,
    segmenter_options: dict[str, Any] | None = None,
) -> FaceSegmenter:
    """创建分割器实例。"""
    key = segmenter_name.strip().lower().replace("_", "-")
    options = segmenter_options or {}

    if key in {"", "none", "off", "disable"}:
        return NoopSegmenter()

    if key in {"mobile-sam", "mobilesam", "sam-mobile", "sam"}:
        model_path = Path(options.get("model_path", model_dir / "sam" / "mobile_sam.pt"))
        model_type = str(options.get("model_type", "vit_t"))
        device = options.get("device")
        multimask_output = bool(options.get("multimask_output", True))
        part_names = tuple(options.get("part_names", PART_NAMES))
        part_scale_by_view = dict(options.get("part_scale_by_view", {}))
        part_offset_by_view = dict(options.get("part_offset_by_view", {}))
        part_offset_mode = "ratio"
        return MobileSamSegmenter(
            model_path=model_path,
            model_type=model_type,
            device=device,
            multimask_output=multimask_output,
            part_names=part_names,
            part_scale_by_view=part_scale_by_view,
            part_offset_by_view=part_offset_by_view,
            part_offset_mode=part_offset_mode,
        )

    raise ValueError(f"不支持的分割器: {segmenter_name}")
