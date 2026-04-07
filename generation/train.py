"""
SDXL Inpainting LoRA 训练。

默认策略：
- 只训练 UNet LoRA；
- 样本来自术后图；
- 同时使用头部参考图和鼻嘴联合 crop 图；
- 不训练 ControlNet。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from project_utils.io import load_jsonl, write_json
from .index import build_generation_manifest, part_mask_path
from .settings import (
    GenerationPaths,
    LoRATrainConfig,
    dataclass_to_dict,
    default_generation_paths,
    default_lora_train_config,
    dump_config_snapshot,
)


def _resolve_rel_path(paths: GenerationPaths, sample_path: str | Path) -> Path:
    """把绝对路径归一化为相对样本路径（用于跨目录复用 mask 路径规则）。"""
    path = Path(sample_path)
    for root in (paths.sanitized_root, paths.input_root, paths.depth_root, paths.inpaint_mask_root):
        try:
            return path.relative_to(root)
        except ValueError:
            continue
    raise ValueError(f"无法解析相对路径: {path}")

def _load_mask_bbox(path: Path) -> tuple[int, int, int, int] | None:
    """从二值 mask 反推最小外接框，返回 (x1,y1,x2,y2)。"""
    if not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _expand_box(
    box: tuple[int, int, int, int] | None,
    width: int,
    height: int,
    context_ratio: float,
) -> tuple[int, int, int, int] | None:
    """按比例给框加上下文边界，避免裁剪过紧导致语义缺失。"""
    if box is None:
        return None
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad = int(round(max(bw, bh) * max(0.0, context_ratio)))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width, x2 + pad)
    y2 = min(height, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _image_to_tensor(image: Image.Image, resolution: int):
    """RGB 图转 [-1,1] 张量，统一 pad 到正方形训练分辨率。"""
    import torch

    square = ImageOps.pad(image.convert("RGB"), (resolution, resolution), method=Image.Resampling.BICUBIC)
    arr = np.asarray(square).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _mask_to_tensor(mask: Image.Image, resolution: int):
    """Mask 转 [0,1] 单通道张量，并硬阈值为二值。"""
    import torch

    square = ImageOps.pad(mask.convert("L"), (resolution, resolution), method=Image.Resampling.NEAREST)
    arr = np.asarray(square).astype(np.float32) / 255.0
    arr = (arr > 0.5).astype(np.float32)
    return torch.from_numpy(arr[None, ...]).contiguous()


@dataclass
class _TrainSample:
    image_path: str
    inpaint_mask_path: str
    prompt: str
    sample_type: str
    weight: float
    crop_box: tuple[int, int, int, int] | None = None


class _SdxlLoRADataset:
    """训练数据集：同一 manifest 会扩展出头部参考样本和局部 inpaint 样本。"""

    def __init__(self, samples: list[_TrainSample], resolution: int):
        self.samples = samples
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """返回一条训练样本（图像、mask、masked 图、prompt、样本权重）。"""
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        mask = Image.open(sample.inpaint_mask_path).convert("L")
        if sample.crop_box is not None:
            # 局部样本模式：只在鼻嘴联合区域训练细节先验。
            image = image.crop(sample.crop_box)
            mask = mask.crop(sample.crop_box)
        tensor = _image_to_tensor(image, self.resolution)
        mask_tensor = _mask_to_tensor(mask, self.resolution)
        # SDXL inpaint 训练输入之一：被遮掉编辑区域后的上下文图。
        masked_tensor = tensor * (mask_tensor < 0.5)
        return {
            "pixel_values": tensor,
            "mask_values": mask_tensor,
            "masked_pixel_values": masked_tensor,
            "prompt": sample.prompt,
            "weight": float(sample.weight),
            "sample_type": sample.sample_type,
        }


def _collate_fn(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """DataLoader 拼批函数，保留 prompt 与样本类型等元信息。"""
    import torch

    return {
        "pixel_values": torch.stack([ex["pixel_values"] for ex in examples]),
        "mask_values": torch.stack([ex["mask_values"] for ex in examples]),
        "masked_pixel_values": torch.stack([ex["masked_pixel_values"] for ex in examples]),
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

    当前固定两类样本：
    - head_reference：用 face 框内区域学习整体头部/姿态参考
    - edit_crop：用鼻嘴局部区域学习目标重绘区域的形态偏好
    """
    samples: list[_TrainSample] = []
    for row in manifest_rows:
        # 当前训练只用术后训练集；术前样本用于推理，不进入 LoRA 训练。
        if row.get("stage") != "术后":
            continue
        if row.get("split") != "train":
            continue
        # 训练输入图优先使用 eye_mask 后的图片；没有则回退原图。
        image_path = str(row.get("sanitized_image_path") or row.get("image_path") or "").strip()
        inpaint_mask_path = str(row.get("inpaint_mask_path") or "").strip()
        if not image_path or not inpaint_mask_path:
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
        face_box = _load_mask_bbox(Path(str(row.get("face_mask_path") or "")))
        samples.append(
            _TrainSample(
                image_path=image_path,
                inpaint_mask_path=inpaint_mask_path,
                prompt=prompt,
                sample_type="head_reference",
                weight=config.full_face_weight,
                crop_box=face_box,
            )
        )

        rel_path = _resolve_rel_path(paths, image_path)
        nose_box = _load_mask_bbox(Path(str(row.get("nose_mask_path") or "")))
        mouth_box = _load_mask_bbox(part_mask_path(paths, "mouth", rel_path))
        if nose_box is None and mouth_box is None:
            continue

        # edit_crop：把鼻子 + 嘴巴合成一个局部训练区域，强化目标编辑区表达。
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        boxes = [box for box in (nose_box, mouth_box) if box is not None]
        x1 = min(box[0] for box in boxes)
        y1 = min(box[1] for box in boxes)
        x2 = max(box[2] for box in boxes)
        y2 = max(box[3] for box in boxes)
        # 鼻子和嘴巴一起裁成局部样本，并用同一张 inpaint_mask 指定重绘目标区域。
        crop_box = _expand_box(
            (x1, y1, x2, y2),
            width=width,
            height=height,
            context_ratio=config.crop_context_ratio,
        )
        if crop_box is None:
            continue
        samples.append(
            _TrainSample(
                image_path=image_path,
                inpaint_mask_path=inpaint_mask_path,
                prompt=prompt,
                sample_type="edit_crop",
                weight=config.nose_mouth_crop_weight,
                crop_box=crop_box,
            )
        )
    return samples


def _torch_dtype(name: str, device: torch.device) -> torch.dtype:
    """按配置字符串解析训练 dtype（CPU 上统一回退 fp32）。"""
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
    """根据设备与 dtype 生成 autocast 上下文。"""
    import torch

    enabled = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _encode_prompts(
    prompts: list[str],
    tokenizers,
    text_encoders,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """编码 SDXL 双文本编码器输入，返回拼接后的 prompt embeds。"""
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
    """构造 SDXL 额外条件 time_ids（原始尺寸与目标尺寸信息）。"""
    values = torch.tensor(
        [[resolution, resolution, 0, 0, resolution, resolution]],
        dtype=torch.float32,
        device=device,
    )
    return values.repeat(batch_size, 1)


def _save_lora_weights(save_dir: Path, unet) -> None:
    """仅保存 UNet LoRA 权重，避免落盘整模型。"""
    from diffusers import StableDiffusionXLInpaintPipeline
    from peft.utils import get_peft_model_state_dict

    state_dict = get_peft_model_state_dict(unet)
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
    manifest_rows = build_generation_manifest(paths)
    train_samples = _build_training_samples(manifest_rows, paths, config)
    if not train_samples:
        raise RuntimeError("没有可用的训练样本，请先完成 detection 侧产物生成并检查 manifest。")

    import torch
    import torch.nn.functional as F
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from peft import LoraConfig
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # 训练硬件与精度配置：GPU 时可走 fp16/bf16；CPU 固定 fp32。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = _torch_dtype(config.mixed_precision, device)

    # 加载 SDXL inpainting 训练所需组件。
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
    text_encoder = CLIPTextModel.from_pretrained(config.base_model_id, subfolder="text_encoder")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        config.base_model_id,
        subfolder="text_encoder_2",
    )
    vae = AutoencoderKL.from_pretrained(config.base_model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(config.base_model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(config.base_model_id, subfolder="scheduler")
    if int(getattr(unet.config, "in_channels", 0)) != 9:
        raise RuntimeError(
            f"当前底模不是 SDXL inpainting UNet，期望 in_channels=9，实际为 {getattr(unet.config, 'in_channels', 'unknown')}"
        )

    # LoRA 仅挂载在 cross-attention 常见线性层上，减少训练参数量。
    lora_cfg = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_cfg)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.train()

    if not config.train_text_encoder:
        text_encoder.eval()
        text_encoder_2.eval()

    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    text_encoder_2.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    params_to_optimize = [param for param in unet.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params_to_optimize, lr=config.learning_rate)

    dataset = _SdxlLoRADataset(train_samples, resolution=config.resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=_collate_fn,
    )
    if len(dataloader) == 0:
        raise RuntimeError("训练 DataLoader 为空。")

    effective_steps_per_epoch = max(1, math.ceil(len(dataloader) / max(1, config.gradient_accumulation_steps)))
    if config.max_train_steps > 0:
        # 手动设定 step 上限时，按每轮有效 step 推导最小 epoch 数。
        max_train_steps = config.max_train_steps
        num_epochs = max(1, math.ceil(max_train_steps / effective_steps_per_epoch))
    else:
        num_epochs = max(1, config.num_train_epochs)
        max_train_steps = num_epochs * effective_steps_per_epoch

    global_step = 0
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    accumulation_counter = 0

    # 主训练循环：
    # 1) 编码图像到 latent
    # 2) 按扩散时间步加噪
    # 3) UNet 预测噪声
    # 4) 对噪声目标做加权 MSE
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
                # inpaint UNet 输入的 mask 分辨率需与 latent 空间对齐。
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

            with _autocast_context(device, weight_dtype):
                # SDXL inpaint UNet 典型 9 通道输入：
                # noisy_latents(4) + mask(1) + masked_image_latents(4)
                latent_model_input = torch.cat([noisy_latents, mask, masked_image_latents], dim=1)
                model_pred = unet(
                    latent_model_input,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs={
                        "text_embeds": pooled_prompt_embeds,
                        "time_ids": add_time_ids,
                    },
                ).sample

            target = noise
            if noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)

            # 按样本权重聚合损失（head_reference 与 edit_crop 的权重可独立控制）。
            loss = ((model_pred.float() - target.float()) ** 2).mean(dim=(1, 2, 3))
            loss = (loss * weights).mean() / max(1, config.gradient_accumulation_steps)
            loss.backward()
            running_loss += float(loss.item())
            accumulation_counter += 1

            # 数据量可能很小，显式维护累计步数，
            # 保证最后一轮不足 `gradient_accumulation_steps` 也能落一次优化。
            if accumulation_counter >= max(1, config.gradient_accumulation_steps):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                accumulation_counter = 0

                if config.save_every_steps > 0 and global_step % config.save_every_steps == 0:
                    _save_lora_weights(paths.lora_root / f"checkpoint-{global_step}", unet)

                if global_step >= max_train_steps:
                    break

        if accumulation_counter > 0 and global_step < max_train_steps:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            accumulation_counter = 0

            if config.save_every_steps > 0 and global_step % config.save_every_steps == 0:
                _save_lora_weights(paths.lora_root / f"checkpoint-{global_step}", unet)

        if global_step >= max_train_steps:
            break

    final_dir = paths.lora_root / "final"
    _save_lora_weights(final_dir, unet)

    # 落盘训练样本清单，便于后续复现实验。
    train_index_path = paths.lora_root / "train_samples.jsonl"
    train_index_path.parent.mkdir(parents=True, exist_ok=True)
    with train_index_path.open("w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(
                json.dumps(
                    {
                        "image_path": sample.image_path,
                        "inpaint_mask_path": sample.inpaint_mask_path,
                        "prompt": sample.prompt,
                        "sample_type": sample.sample_type,
                        "weight": sample.weight,
                        "crop_box": sample.crop_box,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "training_mode": "sdxl_inpainting_lora",
        "train_samples": len(train_samples),
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
