"""
生成侧集中配置。

说明：
- 这里定义生成 pipeline 的公开 dataclass；
- 默认路径会复用 detection 的输入、SAM 输出目录和眼部打码输出目录；
- 具体运行阶段的业务逻辑分别放在 train/infer/eval 模块。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import os
from pathlib import Path
from typing import Any, TypeVar

from detection.settings import default_paths as default_face_paths


T = TypeVar("T")


@dataclass(frozen=True)
class GenerationPaths:
    """生成侧路径配置。"""

    input_root: Path
    sam_root: Path
    model_dir: Path
    generation_root: Path
    prepared_root: Path
    manifest_path: Path
    issues_path: Path
    summaries_root: Path
    depth_root: Path
    inpaint_mask_root: Path
    feather_mask_root: Path
    privacy_mask_root: Path
    sanitized_root: Path
    lora_root: Path
    inference_root: Path
    eval_root: Path


@dataclass(frozen=True)
class LoRATrainConfig:
    """LoRA 训练配置。"""

    base_model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    resolution: int = 1024
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    num_train_epochs: int = 1
    max_train_steps: int = 200
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    rank: int = 16
    alpha: int = 16
    dropout: float = 0.0
    mixed_precision: str = "fp16"
    train_text_encoder: bool = False
    seed: int = 42
    save_every_steps: int = 100
    num_workers: int = 0
    paired_head_weight: float = 1.0
    paired_edit_weight: float = 1.35
    self_identity_weight: float = 0.45
    crop_context_ratio: float = 0.35
    caption_suffix: str = "rhinoplasty postoperative portrait"
    apply_eye_privacy_mask: bool = True
    use_case_grouped_sampling: bool = True
    views_per_case_step: int = 2
    allow_unpaired_self_reconstruction: bool = True


@dataclass(frozen=True)
class InferenceConfig:
    """推理配置。"""

    base_model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    enable_depth_condition: bool = False
    controlnet_model_id: str = "diffusers/controlnet-depth-sdxl-1.0"
    lora_weights_path: str = ""
    quantization_backend: str = "quanto"
    quantization_dtype: str = "float8"
    quantize_components: tuple[str, ...] = ("unet", "controlnet")
    torchao_quant_type: str = "float8wo"
    pipeline_device_map: str = ""
    prompt_template: str = "{doctor_token}, {view_token}, rhinoplasty postoperative portrait"
    negative_prompt: str = "deformed face, extra nose, broken mouth, duplicated features, low quality"
    num_inference_steps: int = 30
    guidance_scale: float = 7.0
    strength: float = 0.88
    controlnet_conditioning_scale: float = 0.8
    control_guidance_start: float = 0.0
    control_guidance_end: float = 0.65
    num_images_per_prompt: int = 1
    seed: int = 42
    torch_dtype: str = "float16"
    lora_scale: float = 1.0
    max_samples: int = 0
    apply_eye_privacy_mask: bool = True
    privacy_fill: str = "black"
    composite_blur_px: int = 31
    shared_case_seed: bool = True
    case_seed_stride: int = 1000


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser()


def default_generation_paths() -> GenerationPaths:
    """返回生成侧默认路径。"""
    face_paths = default_face_paths()
    generation_root = _path_from_env(
        "GEN_OUTPUT_ROOT",
        Path("~/datasets/deformity_generation"),
    )
    prepared_root = _path_from_env("GEN_PREPARED_ROOT", generation_root / "prepared")
    summaries_root = _path_from_env("GEN_SUMMARIES_ROOT", generation_root / "summaries")
    manifest_path = _path_from_env("GEN_MANIFEST_PATH", summaries_root / "manifest.jsonl")
    issues_path = _path_from_env("GEN_ISSUES_PATH", summaries_root / "issues.jsonl")
    return GenerationPaths(
        input_root=_path_from_env("GEN_INPUT_ROOT", face_paths.input_root),
        sam_root=_path_from_env("GEN_SAM_ROOT", face_paths.output_root_sam),
        model_dir=_path_from_env("GEN_MODEL_DIR", face_paths.model_dir),
        generation_root=generation_root,
        prepared_root=prepared_root,
        manifest_path=manifest_path,
        issues_path=issues_path,
        summaries_root=summaries_root,
        depth_root=_path_from_env("GEN_DEPTH_ROOT", prepared_root / "depth"),
        inpaint_mask_root=_path_from_env("GEN_INPAINT_MASK_ROOT", face_paths.output_root_sam / "inpaint_mask"),
        feather_mask_root=_path_from_env("GEN_FEATHER_MASK_ROOT", face_paths.output_root_sam / "feather_mask"),
        privacy_mask_root=_path_from_env("GEN_PRIVACY_MASK_ROOT", prepared_root / "eye_privacy_mask"),
        sanitized_root=_path_from_env("GEN_SANITIZED_ROOT", face_paths.output_root_mask),
        lora_root=_path_from_env("GEN_LORA_ROOT", generation_root / "lora"),
        inference_root=_path_from_env("GEN_INFERENCE_ROOT", generation_root / "infer"),
        eval_root=_path_from_env("GEN_EVAL_ROOT", generation_root / "eval"),
    )


def default_lora_train_config() -> LoRATrainConfig:
    """返回 LoRA 训练默认配置。"""
    return LoRATrainConfig()


def default_inference_config() -> InferenceConfig:
    """返回推理默认配置。"""
    return InferenceConfig()


def dataclass_to_dict(config: Any) -> dict[str, Any]:
    """把 dataclass 转成可 JSON 序列化的 dict。"""
    if hasattr(config, "__dataclass_fields__"):
        data = asdict(config)
    else:
        raise TypeError(f"不支持的配置类型: {type(config)!r}")
    return _normalize_for_json(data)


def _normalize_for_json(value: Any) -> Any:
    """递归把 Path / tuple / dict 等配置值转成 JSON 可写形式。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _normalize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(v) for v in value]
    return value


def dump_config_snapshot(path: Path, **configs: Any) -> None:
    """把一组配置快照写到 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: dataclass_to_dict(cfg) for name, cfg in configs.items()}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def update_dataclass(config: T, **kwargs: Any) -> T:
    """基于 dataclass 生成带覆盖值的新实例。"""
    if not hasattr(config, "__dataclass_fields__"):
        raise TypeError(f"不支持的配置类型: {type(config)!r}")
    allowed = {f.name for f in fields(config)}
    unknown = sorted(set(kwargs) - allowed)
    if unknown:
        raise ValueError(f"未知配置字段: {', '.join(unknown)}")
    values = asdict(config)
    values.update(kwargs)
    return type(config)(**values)
