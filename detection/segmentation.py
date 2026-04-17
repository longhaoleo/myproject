"""
人脸分割接口层（SAM 版）。

设计目标：
1. 输入：检测器输出的人脸框 + 关键点；
2. 输出：每张脸一张合并后的二值 mask（用于叠加可视化）；
3. 同时保留每个部位（face / left_eye / right_eye / nose / mouth）的单独 mask，
   便于后续接更精细的业务逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from .part_boxes import PART_NAMES, build_part_prompt_boxes
from .types import FaceDetection


# 单张二值 mask：H x W，元素为 bool/0/1
BinaryMask = np.ndarray

# 部位名 -> 二值 mask
PartMaskMap = dict[str, BinaryMask]


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
            part_boxes = build_part_prompt_boxes(
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
