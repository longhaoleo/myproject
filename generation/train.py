"""
SDXL Inpainting LoRA 训练。

默认策略：
- 只训练 UNet LoRA；
- 使用同病例同视角 paired 样本：术前 face crop 作为条件图，术后 face crop 作为目标图；
- 使用术前侧鼻嘴 inpaint mask 指示重绘区域，mask 外术前上下文用于稳定身份；
- 不训练 ControlNet。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
from PIL import Image

from project_utils.io import write_json
from .index import build_generation_manifest
from .settings import (
    GenerationPaths,
    LoRATrainConfig,
    dataclass_to_dict,
    default_generation_paths,
    default_lora_train_config,
    dump_config_snapshot,
)


def _row_image_path(row: dict[str, Any]) -> str:
    return str(row.get("sanitized_image_path") or row.get("image_path") or "").strip()


def _view_sort_key(view_id: str) -> tuple[int, int | str]:
    text = str(view_id)
    if text.isdigit():
        return 0, int(text)
    return 1, text


def _load_mask_bbox(path: Path) -> tuple[int, int, int, int] | None:
    """从 mask 图中提取外接框，给 crop 逻辑复用。"""
    if not path.exists():
        return None
    try:
        with Image.open(path) as mask:
            bbox = mask.convert("L").getbbox()
    except OSError:
        return None
    if bbox is None:
        return None
    return tuple(int(value) for value in bbox)


def _image_to_tensor(image: Image.Image, resolution: int):
    import torch

    resized = image.convert("RGB").resize((resolution, resolution), Image.Resampling.BICUBIC)
    arr = np.asarray(resized).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _mask_to_tensor(mask: Image.Image, resolution: int):
    import torch

    resized = mask.convert("L").resize((resolution, resolution), Image.Resampling.NEAREST)
    arr = np.asarray(resized).astype(np.float32) / 255.0
    arr = (arr > 0.5).astype(np.float32)
    return torch.from_numpy(arr[None, ...]).contiguous()


@dataclass
class _TrainSample:
    case_id: str
    view_id: str
    condition_image_path: str
    target_image_path: str
    inpaint_mask_path: str
    prompt: str
    sample_type: str
    weight: float
    crop_box: tuple[int, int, int, int] | None = None
    ip_adapter_reference_image_paths: tuple[str, ...] = ()
    ip_adapter_reference_crop_boxes: tuple[tuple[int, int, int, int] | None, ...] = ()


class _SdxlLoRADataset:
    """训练数据集：用统一 face crop 坐标系做鼻嘴 inpaint。"""

    def __init__(self, samples: list[_TrainSample], resolution: int):
        self.samples = samples
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        condition_image = Image.open(sample.condition_image_path).convert("RGB")
        target_image = Image.open(sample.target_image_path).convert("RGB")
        mask = Image.open(sample.inpaint_mask_path).convert("L")
        if sample.crop_box is not None:
            condition_image = condition_image.crop(sample.crop_box)
            target_image = target_image.crop(sample.crop_box)
            mask = mask.crop(sample.crop_box)
        condition_tensor = _image_to_tensor(condition_image, self.resolution)
        raw_target_tensor = _image_to_tensor(target_image, self.resolution)
        mask_tensor = _mask_to_tensor(mask, self.resolution)
        target_tensor = condition_tensor * (1.0 - mask_tensor) + raw_target_tensor * mask_tensor
        # 标准 inpainting 条件：mask 内挖空，mask 外保留术前上下文以稳定身份。
        masked_tensor = condition_tensor * (mask_tensor < 0.5)
        return {
            "pixel_values": target_tensor,
            "mask_values": mask_tensor,
            "masked_pixel_values": masked_tensor,
            "ip_adapter_images": self._load_ip_adapter_images(sample),
            "prompt": sample.prompt,
            "weight": float(sample.weight),
            "sample_type": sample.sample_type,
        }

    def _load_ip_adapter_images(self, sample: _TrainSample) -> list[Image.Image]:
        images: list[Image.Image] = []
        for image_path, crop_box in zip(
            sample.ip_adapter_reference_image_paths,
            sample.ip_adapter_reference_crop_boxes,
        ):
            image = Image.open(image_path).convert("RGB")
            if crop_box is not None:
                image = image.crop(crop_box)
            images.append(image)
        return images


def _collate_fn(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """把单样本字典拼成 batch 张量。"""
    import torch

    return {
        "pixel_values": torch.stack([ex["pixel_values"] for ex in examples]),
        "mask_values": torch.stack([ex["mask_values"] for ex in examples]),
        "masked_pixel_values": torch.stack([ex["masked_pixel_values"] for ex in examples]),
        "ip_adapter_images": [ex["ip_adapter_images"] for ex in examples],
        "prompts": [str(ex["prompt"]) for ex in examples],
        "weights": torch.tensor([float(ex["weight"]) for ex in examples], dtype=torch.float32),
        "sample_types": [str(ex["sample_type"]) for ex in examples],
    }


def _build_training_samples(
    manifest_rows: list[dict[str, Any]],
    paths: GenerationPaths,
    config: LoRATrainConfig,
) -> list[_TrainSample]:
    """
    从 manifest 构造训练样本。

    当前使用同病例同视角配对样本：
    - pre_to_post_inpaint：术前 face crop 作为条件图，目标为同视角术后 face crop。
    """
    samples: list[_TrainSample] = []
    rows_by_stage: dict[tuple[str, str, str], dict[str, Any]] = {}
    pre_rows_by_case: dict[str, list[dict[str, Any]]] = {}
    bbox_cache_path = paths.summaries_root / "mask_bbox_cache.json"
    bbox_cache: dict[str, tuple[int, int, int, int] | None] = {}
    bbox_cache_dirty = False
    if bbox_cache_path.exists():
        try:
            raw_cache = json.loads(bbox_cache_path.read_text(encoding="utf-8"))
            raw_items = raw_cache.get("items", raw_cache) if isinstance(raw_cache, dict) else {}
            for key, value in raw_items.items():
                if isinstance(value, list) and len(value) == 4:
                    bbox_cache[str(key)] = tuple(int(item) for item in value)
                else:
                    bbox_cache[str(key)] = None
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            bbox_cache = {}

    def cached_mask_bbox(path_text: str) -> tuple[int, int, int, int] | None:
        nonlocal bbox_cache_dirty
        path_text = str(path_text or "").strip()
        if not path_text:
            return None
        if path_text not in bbox_cache:
            bbox_cache[path_text] = _load_mask_bbox(Path(path_text))
            bbox_cache_dirty = True
        return bbox_cache[path_text]

    for row in manifest_rows:
        case_id = str(row.get("case_id") or "").strip()
        view_id = str(row.get("view_id") or "").strip()
        stage = str(row.get("stage") or "").strip()
        if case_id and view_id and stage:
            rows_by_stage[(case_id, view_id, stage)] = row
        if case_id and stage == "术前":
            pre_rows_by_case.setdefault(case_id, []).append(row)
    for pre_rows in pre_rows_by_case.values():
        pre_rows.sort(key=lambda item: _view_sort_key(str(item.get("view_id") or "")))

    for row in manifest_rows:
        if row.get("split") != "train":
            continue
        if str(row.get("stage") or "") != "术后":
            continue
        case_id = str(row.get("case_id") or "").strip()
        view_id = str(row.get("view_id") or "").strip()
        pre_row = rows_by_stage.get((case_id, view_id, "术前"))
        if pre_row is None:
            continue
        target_image_path = _row_image_path(row)
        condition_image_path = _row_image_path(pre_row)
        inpaint_mask_path = str(pre_row.get("inpaint_mask_path") or "").strip()
        face_mask_path = str(pre_row.get("face_mask_path") or "").strip()
        if not condition_image_path or not target_image_path or not inpaint_mask_path or not face_mask_path:
            continue
        prompt = ", ".join(
            part
            for part in (
                str(row.get("doctor_token") or "").strip(),
                str(row.get("view_token") or "").strip(),
                str(config.caption_suffix).strip(),
            )
            if part
        )
        face_box = cached_mask_bbox(face_mask_path)
        if face_box is None:
            continue
        ip_adapter_reference_image_paths: list[str] = []
        ip_adapter_reference_crop_boxes: list[tuple[int, int, int, int] | None] = []
        if config.enable_ip_adapter_condition:
            for ref_row in pre_rows_by_case.get(case_id, [])[: max(0, config.ip_adapter_max_reference_images)]:
                ref_image_path = _row_image_path(ref_row)
                if not ref_image_path:
                    continue
                ref_crop_box = None
                if config.ip_adapter_use_face_crop:
                    ref_face_mask_path = str(ref_row.get("face_mask_path") or "").strip()
                    if ref_face_mask_path:
                        ref_crop_box = cached_mask_bbox(ref_face_mask_path)
                ip_adapter_reference_image_paths.append(ref_image_path)
                ip_adapter_reference_crop_boxes.append(ref_crop_box)

        samples.append(
            _TrainSample(
                case_id=case_id,
                view_id=view_id,
                condition_image_path=condition_image_path,
                target_image_path=target_image_path,
                inpaint_mask_path=inpaint_mask_path,
                prompt=prompt,
                sample_type="pre_to_post_inpaint",
                weight=1.0,
                crop_box=face_box,
                ip_adapter_reference_image_paths=tuple(ip_adapter_reference_image_paths),
                ip_adapter_reference_crop_boxes=tuple(ip_adapter_reference_crop_boxes),
            )
        )
    if bbox_cache_dirty:
        write_json(
            bbox_cache_path,
            {
                "version": 1,
                "items": {
                    key: list(value) if value is not None else None
                    for key, value in sorted(bbox_cache.items())
                },
            },
        )
    return samples


def _order_train_samples_for_cases(
    samples: list[_TrainSample],
    config: LoRATrainConfig,
) -> list[_TrainSample]:
    """按病例和视角重排训练样本，让一次梯度累积更容易看到同一个人。"""
    if not config.use_case_grouped_sampling or len(samples) <= 1:
        return samples

    rng = random.Random(config.seed)
    samples_by_case: dict[str, list[_TrainSample]] = {}
    for sample in samples:
        samples_by_case.setdefault(sample.case_id, []).append(sample)
    for case_samples in samples_by_case.values():
        case_samples.sort(key=lambda item: _view_sort_key(item.view_id))

    ordered: list[_TrainSample] = []
    while True:
        active_cases = [case_id for case_id, case_samples in samples_by_case.items() if case_samples]
        if not active_cases:
            break
        rng.shuffle(active_cases)
        for case_id in active_cases:
            for _ in range(max(1, config.views_per_case_step)):
                if not samples_by_case.get(case_id):
                    break
                ordered.append(samples_by_case[case_id].pop(0))
    return ordered


def _torch_dtype(name: str, device: torch.device) -> torch.dtype:
    """把配置里的 dtype 名称转换成 torch.dtype。"""
    import torch

    key = str(name).strip().lower()
    if device.type != "cuda":
        return torch.float32
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def _autocast_context(device: torch.device, dtype: torch.dtype):
    import torch

    enabled = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _encode_prompts(
    prompts: list[str],
    tokenizers,
    text_encoders,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把文本 prompt 编成 SDXL 需要的 prompt embeddings。"""
    import torch

    prompt_embeds_list = []
    pooled_prompt_embeds = None

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_inputs = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        encoder_output = text_encoder(text_input_ids, output_hidden_states=True)
        hidden_states = encoder_output.hidden_states[-2]
        prompt_embeds_list.append(hidden_states)
        pooled_prompt_embeds = encoder_output[0]

    prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
    if pooled_prompt_embeds is None:
        raise RuntimeError("未生成 pooled prompt embeddings")
    return prompt_embeds, pooled_prompt_embeds


def _time_ids(batch_size: int, resolution: int, device: torch.device) -> torch.Tensor:
    import torch

    values = torch.tensor(
        [[resolution, resolution, 0, 0, resolution, resolution]],
        dtype=torch.float32,
        device=device,
    )
    return values.repeat(batch_size, 1)


def _validate_local_ip_adapter_files(config: LoRATrainConfig) -> None:
    model_path = Path(str(config.ip_adapter_model_id))
    if not model_path.exists():
        return

    required = [
        model_path / str(config.ip_adapter_subfolder).strip() / str(config.ip_adapter_weight_name).strip(),
    ]
    image_encoder_folder = str(config.ip_adapter_image_encoder_folder).strip()
    if image_encoder_folder:
        image_encoder_path = model_path / image_encoder_folder
        required.extend(
            [
                image_encoder_path / "config.json",
                image_encoder_path / "model.safetensors",
            ]
        )

    missing = [path for path in required if not path.exists()]
    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "本地 IP-Adapter 目录缺少必要文件：\n"
            f"{missing_text}\n"
            "可重新下载：\n"
            "hf download h94/IP-Adapter "
            "sdxl_models/image_encoder/config.json "
            "sdxl_models/image_encoder/model.safetensors "
            "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors "
            f"--local-dir {model_path} --max-workers 1"
        )


def _attach_ip_adapter_for_training(
    *,
    config: LoRATrainConfig,
    vae,
    text_encoder,
    text_encoder_2,
    tokenizer,
    tokenizer_2,
    unet,
    noise_scheduler,
):
    from diffusers import StableDiffusionXLInpaintPipeline

    _validate_local_ip_adapter_files(config)
    pipe = StableDiffusionXLInpaintPipeline(
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=unet,
        scheduler=noise_scheduler,
        add_watermarker=False,
    )
    pipe.load_ip_adapter(
        config.ip_adapter_model_id,
        subfolder=str(config.ip_adapter_subfolder).strip(),
        weight_name=str(config.ip_adapter_weight_name).strip(),
        image_encoder_folder=str(config.ip_adapter_image_encoder_folder).strip(),
    )
    if hasattr(pipe, "set_ip_adapter_scale"):
        pipe.set_ip_adapter_scale(float(config.ip_adapter_scale))
    return pipe.unet, pipe.image_encoder, pipe.feature_extractor


def _encode_ip_adapter_image_embeds(
    *,
    image_batches: list[list[Image.Image]],
    image_encoder,
    feature_extractor,
    unet,
    device: torch.device,
) -> list[torch.Tensor] | None:
    import torch
    from diffusers.models.embeddings import ImageProjection

    if image_encoder is None or feature_extractor is None:
        return None
    if len(image_batches) != 1:
        raise ValueError("当前 IP-Adapter 条件训练只支持 train_batch_size=1；请用梯度累积扩大有效 batch。")
    images = image_batches[0]
    if not images:
        return None

    dtype = next(image_encoder.parameters()).dtype
    pixel_values = feature_extractor(images, return_tensors="pt").pixel_values.to(device=device, dtype=dtype)
    projection_layer = unet.encoder_hid_proj.image_projection_layers[0]
    with torch.no_grad():
        encoder_output = image_encoder(pixel_values, output_hidden_states=True)

    expected_dim = None
    if isinstance(projection_layer, ImageProjection):
        expected_dim = int(projection_layer.image_embeds.in_features)
    elif hasattr(projection_layer, "proj_in"):
        expected_dim = int(projection_layer.proj_in.in_features)

    pooled_embeds = getattr(encoder_output, "image_embeds", None)
    hidden_embeds = None
    if getattr(encoder_output, "hidden_states", None):
        hidden_embeds = encoder_output.hidden_states[-2]

    candidates = (
        (pooled_embeds, "image_embeds"),
        (hidden_embeds, "hidden_states[-2]"),
    )
    if not isinstance(projection_layer, ImageProjection):
        candidates = (
            (hidden_embeds, "hidden_states[-2]"),
            (pooled_embeds, "image_embeds"),
        )

    image_embeds = None
    source_name = ""
    for candidate, name in candidates:
        if candidate is None:
            continue
        if expected_dim is not None and int(candidate.shape[-1]) != expected_dim:
            continue
        image_embeds = candidate
        source_name = name
        break

    if image_embeds is None:
        pooled_shape = tuple(pooled_embeds.shape) if pooled_embeds is not None else None
        hidden_shape = tuple(hidden_embeds.shape) if hidden_embeds is not None else None
        raise RuntimeError(
            "IP-Adapter image embedding 维度不匹配："
            f"projection_expected_dim={expected_dim}, "
            f"image_embeds_shape={pooled_shape}, hidden_states_shape={hidden_shape}"
        )

    if not isinstance(projection_layer, ImageProjection) and image_embeds.ndim == 2:
        image_embeds = image_embeds[:, None, :]
    if isinstance(projection_layer, ImageProjection) and image_embeds.ndim != 2:
        raise RuntimeError(f"IP-Adapter ImageProjection 需要 2D image_embeds，实际来自 {source_name}: {tuple(image_embeds.shape)}")
    return [image_embeds[None, :]]


def _save_lora_weights(save_dir: Path, unet) -> None:
    """只保存 UNet LoRA 权重，避免把无关模块一起落盘。"""
    from diffusers import StableDiffusionXLInpaintPipeline
    from peft.utils import get_peft_model_state_dict
    import torch

    raw_state_dict = {
        key: value
        for key, value in get_peft_model_state_dict(unet).items()
        if "encoder_hid_proj" not in key
    }
    bad_tensors: list[str] = []
    for key, value in raw_state_dict.items():
        if not torch.isfinite(value.detach()).all():
            bad_tensors.append(key)
    if bad_tensors:
        preview = ", ".join(bad_tensors[:5])
        raise FloatingPointError(
            f"LoRA 权重包含 NaN/Inf，拒绝保存: bad_tensor_count={len(bad_tensors)} bad_tensors={preview}"
        )

    state_dict = {
        key: value.detach().to(device="cpu", dtype=torch.float16)
        for key, value in raw_state_dict.items()
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    StableDiffusionXLInpaintPipeline.save_lora_weights(
        save_directory=str(save_dir),
        unet_lora_layers=state_dict,
        safe_serialization=True,
    )


def train_lora(
    paths: GenerationPaths | None = None,
    config: LoRATrainConfig | None = None,
) -> dict[str, Any]:
    """训练 SDXL inpainting LoRA。"""
    paths = paths or default_generation_paths()
    config = config or default_lora_train_config()
    if config.enable_ip_adapter_condition and config.train_batch_size != 1:
        raise ValueError("启用 IP-Adapter 条件训练时 train_batch_size 必须为 1；请用 gradient_accumulation_steps 扩大有效 batch。")
    print("[train_lora] building manifest...", flush=True)
    manifest_rows = build_generation_manifest(paths)
    print(f"[train_lora] manifest_rows={len(manifest_rows)} building train samples...", flush=True)
    train_samples = _order_train_samples_for_cases(_build_training_samples(manifest_rows, paths, config), config)
    if not train_samples:
        raise RuntimeError("没有可用的训练样本，请先完成 detection 侧产物生成并检查 manifest。")
    print(f"[train_lora] train_samples={len(train_samples)} loading models...", flush=True)

    import torch
    import torch.nn.functional as F
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from peft import LoraConfig
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = _torch_dtype(config.mixed_precision, device)
    model_load_kwargs = {"variant": config.base_model_variant} if config.base_model_variant else {}

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_id,
        subfolder="tokenizer",
        use_fast=False,
    )
    tokenizer_2 = AutoTokenizer.from_pretrained(
        config.base_model_id,
        subfolder="tokenizer_2",
        use_fast=False,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        config.base_model_id,
        subfolder="text_encoder",
        **model_load_kwargs,
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        config.base_model_id,
        subfolder="text_encoder_2",
        **model_load_kwargs,
    )
    vae = AutoencoderKL.from_pretrained(config.base_model_id, subfolder="vae", **model_load_kwargs)
    unet = UNet2DConditionModel.from_pretrained(config.base_model_id, subfolder="unet", **model_load_kwargs)
    noise_scheduler = DDPMScheduler.from_pretrained(config.base_model_id, subfolder="scheduler")
    if int(getattr(unet.config, "in_channels", 0)) != 9:
        raise RuntimeError(
            f"当前底模不是 SDXL inpainting UNet，期望 in_channels=9，实际为 {getattr(unet.config, 'in_channels', 'unknown')}"
        )
    image_encoder = None
    feature_extractor = None
    if config.enable_ip_adapter_condition:
        unet, image_encoder, feature_extractor = _attach_ip_adapter_for_training(
            config=config,
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            noise_scheduler=noise_scheduler,
        )

    lora_cfg = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_cfg)
    if config.gradient_checkpointing and hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    if image_encoder is not None:
        image_encoder.requires_grad_(False)
    unet.train()

    if not config.train_text_encoder:
        text_encoder.eval()
        text_encoder_2.eval()
    if image_encoder is not None:
        image_encoder.eval()

    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    text_encoder_2.to(device, dtype=weight_dtype)
    if image_encoder is not None:
        image_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    for name, param in unet.named_parameters():
        param.requires_grad_("lora" in name and "encoder_hid_proj" not in name)
    for param in unet.parameters():
        if param.requires_grad:
            # AdamW directly updating fp16 LoRA weights is prone to NaNs.
            # Keep trainable LoRA parameters in fp32 while frozen base weights stay in reduced precision.
            param.data = param.data.float()

    params_to_optimize = [param for param in unet.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params_to_optimize, lr=config.learning_rate)

    dataset = _SdxlLoRADataset(train_samples, resolution=config.resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=not config.use_case_grouped_sampling,
        num_workers=config.num_workers,
        collate_fn=_collate_fn,
    )
    if len(dataloader) == 0:
        raise RuntimeError("训练 DataLoader 为空。")

    effective_steps_per_epoch = max(1, math.ceil(len(dataloader) / max(1, config.gradient_accumulation_steps)))
    if config.max_train_steps > 0:
        max_train_steps = config.max_train_steps
        num_epochs = max(1, math.ceil(max_train_steps / effective_steps_per_epoch))
    else:
        num_epochs = max(1, config.num_train_epochs)
        max_train_steps = num_epochs * effective_steps_per_epoch

    global_step = 0
    running_loss = 0.0
    recent_loss = 0.0
    recent_logged_steps = 0
    started_at = time.time()
    optimizer.zero_grad(set_to_none=True)
    accumulation_counter = 0

    print(
        "[train_lora] "
        f"samples={len(train_samples)} max_steps={max_train_steps} "
        f"epochs={num_epochs} batch_size={config.train_batch_size} "
        f"grad_accum={config.gradient_accumulation_steps} resolution={config.resolution} "
        f"dtype={weight_dtype} device={device} "
        f"gradient_checkpointing={bool(config.gradient_checkpointing)} "
        f"ip_adapter_condition={bool(config.enable_ip_adapter_condition)}",
        flush=True,
    )

    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)
            mask_values = batch["mask_values"].to(device=device, dtype=weight_dtype)
            masked_pixel_values = batch["masked_pixel_values"].to(device=device, dtype=weight_dtype)
            weights = batch["weights"].to(device=device, dtype=torch.float32)

            with torch.no_grad():
                with _autocast_context(device, weight_dtype):
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    masked_image_latents = vae.encode(masked_pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                masked_image_latents = masked_image_latents * vae.config.scaling_factor
                mask = F.interpolate(mask_values, size=latents.shape[-2:], mode="nearest")

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
                dtype=torch.long,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds = _encode_prompts(
                    prompts=batch["prompts"],
                    tokenizers=(tokenizer, tokenizer_2),
                    text_encoders=(text_encoder, text_encoder_2),
                    device=device,
                )
                add_time_ids = _time_ids(latents.shape[0], config.resolution, device)
                ip_adapter_image_embeds = _encode_ip_adapter_image_embeds(
                    image_batches=batch["ip_adapter_images"],
                    image_encoder=image_encoder,
                    feature_extractor=feature_extractor,
                    unet=unet,
                    device=device,
                )

            with _autocast_context(device, weight_dtype):
                latent_model_input = torch.cat([noisy_latents, mask, masked_image_latents], dim=1)
                added_cond_kwargs = {
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids,
                }
                if ip_adapter_image_embeds is not None:
                    added_cond_kwargs["image_embeds"] = ip_adapter_image_embeds
                model_pred = unet(
                    latent_model_input,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

            target = noise
            if noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)

            loss = ((model_pred.float() - target.float()) ** 2).mean(dim=(1, 2, 3))
            loss = (loss * weights).mean() / max(1, config.gradient_accumulation_steps)
            loss_value = float(loss.detach().item())
            if not math.isfinite(loss_value):
                raise FloatingPointError(
                    f"训练 loss 非有限，停止以避免保存坏权重: "
                    f"epoch={epoch} dataloader_step={step} global_step={global_step} loss={loss_value}"
                )
            loss.backward()
            running_loss += loss_value
            recent_loss += loss_value
            accumulation_counter += 1

            # 数据量可能很小，显式维护累计步数，
            # 保证最后一轮不足 `gradient_accumulation_steps` 也能落一次优化。
            if accumulation_counter >= max(1, config.gradient_accumulation_steps):
                if config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params_to_optimize, config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                recent_logged_steps += 1
                accumulation_counter = 0

                if config.log_every_steps > 0 and (
                    global_step == 1 or global_step % config.log_every_steps == 0 or global_step >= max_train_steps
                ):
                    elapsed = max(1e-6, time.time() - started_at)
                    mean_recent_loss = recent_loss / max(1, recent_logged_steps)
                    steps_per_min = global_step / elapsed * 60.0
                    print(
                        "[train_lora] "
                        f"step={global_step}/{max_train_steps} "
                        f"loss={mean_recent_loss:.6f} "
                        f"elapsed_min={elapsed / 60.0:.1f} "
                        f"steps_per_min={steps_per_min:.2f}",
                        flush=True,
                    )
                    recent_loss = 0.0
                    recent_logged_steps = 0

                if config.save_every_steps > 0 and global_step % config.save_every_steps == 0:
                    _save_lora_weights(paths.lora_root / f"checkpoint-{global_step}", unet)
                    print(
                        f"[train_lora] saved checkpoint step={global_step} "
                        f"path={paths.lora_root / f'checkpoint-{global_step}'}",
                        flush=True,
                    )

                if global_step >= max_train_steps:
                    break

        if accumulation_counter > 0 and global_step < max_train_steps:
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params_to_optimize, config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            recent_logged_steps += 1
            accumulation_counter = 0

            if config.log_every_steps > 0 and (
                global_step == 1 or global_step % config.log_every_steps == 0 or global_step >= max_train_steps
            ):
                elapsed = max(1e-6, time.time() - started_at)
                mean_recent_loss = recent_loss / max(1, recent_logged_steps)
                steps_per_min = global_step / elapsed * 60.0
                print(
                    "[train_lora] "
                    f"step={global_step}/{max_train_steps} "
                    f"loss={mean_recent_loss:.6f} "
                    f"elapsed_min={elapsed / 60.0:.1f} "
                    f"steps_per_min={steps_per_min:.2f}",
                    flush=True,
                )
                recent_loss = 0.0
                recent_logged_steps = 0

            if config.save_every_steps > 0 and global_step % config.save_every_steps == 0:
                _save_lora_weights(paths.lora_root / f"checkpoint-{global_step}", unet)
                print(
                    f"[train_lora] saved checkpoint step={global_step} "
                    f"path={paths.lora_root / f'checkpoint-{global_step}'}",
                    flush=True,
                )

        if global_step >= max_train_steps:
            break

    final_dir = paths.lora_root / "final"
    _save_lora_weights(final_dir, unet)
    print(f"[train_lora] saved final LoRA path={final_dir}", flush=True)

    train_index_path = paths.lora_root / "train_samples.jsonl"
    train_index_path.parent.mkdir(parents=True, exist_ok=True)
    with train_index_path.open("w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(
                json.dumps(
                    {
                        "case_id": sample.case_id,
                        "view_id": sample.view_id,
                        "condition_image_path": sample.condition_image_path,
                        "target_image_path": sample.target_image_path,
                        "inpaint_mask_path": sample.inpaint_mask_path,
                        "prompt": sample.prompt,
                        "sample_type": sample.sample_type,
                        "weight": sample.weight,
                        "crop_box": sample.crop_box,
                        "ip_adapter_reference_image_paths": list(sample.ip_adapter_reference_image_paths),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    sample_type_counts: dict[str, int] = {}
    for sample in train_samples:
        sample_type_counts[sample.sample_type] = sample_type_counts.get(sample.sample_type, 0) + 1

    summary = {
        "training_mode": "sdxl_inpainting_lora",
        "train_samples": len(train_samples),
        "sample_type_counts": sample_type_counts,
        "epochs": num_epochs,
        "max_train_steps": max_train_steps,
        "completed_steps": global_step,
        "device": str(device),
        "dtype": str(weight_dtype),
        "final_lora_dir": str(final_dir),
        "mean_loss": running_loss / max(1, global_step),
        "config": dataclass_to_dict(config),
    }
    write_json(paths.summaries_root / "train_lora_summary.json", summary)
    dump_config_snapshot(
        paths.summaries_root / "train_lora_config.json",
        paths=paths,
        train=config,
    )
    return summary


def main() -> None:
    summary = train_lora()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
