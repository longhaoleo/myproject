"""
SDXL 推理。

固定链路：
术前图 -> 标准化重绘 mask -> 可选 depth -> SDXL inpaint -> 羽化回贴
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from project_utils.io import write_json
from .index import build_generation_manifest
from .settings import (
    GenerationPaths,
    InferenceConfig,
    dataclass_to_dict,
    default_generation_paths,
    default_inference_config,
    dump_config_snapshot,
)


def _resolve_rel_path(paths: GenerationPaths, sample_path: str | Path) -> Path:
    path = Path(sample_path)
    for root in (paths.sanitized_root, paths.input_root, paths.depth_root, paths.inference_root):
        try:
            return path.relative_to(root)
        except ValueError:
            continue
    raise ValueError(f"无法解析相对路径: {path}")


def _mask_path(root: Path, rel_path: Path) -> Path:
    return (root / rel_path).with_suffix(".png")


def _row_path(row: dict[str, Any], *keys: str) -> Path | None:
    for key in keys:
        value = str(row.get(key) or "").strip()
        if value:
            return Path(value)
    return None


def _load_rgb_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _load_gray_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("L")


def _load_mask_bbox(path: Path) -> tuple[int, int, int, int] | None:
    if not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _apply_privacy_fill(image_bgr: np.ndarray, mask: np.ndarray, fill_mode: str) -> np.ndarray:
    if not mask.any():
        return image_bgr.copy()
    output = image_bgr.copy()
    mode = str(fill_mode).strip().lower()
    if mode == "blur":
        blurred = cv2.GaussianBlur(image_bgr, (31, 31), 0)
        output[mask] = blurred[mask]
        return output
    output[mask] = 0
    return output


def _load_binary_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return mask > 0


def _blur_mask(mask: np.ndarray, blur_px: int) -> np.ndarray:
    kernel = max(1, int(blur_px))
    if kernel % 2 == 0:
        kernel += 1
    return cv2.GaussianBlur(mask.astype(np.uint8) * 255, (kernel, kernel), 0)


def _stable_seed(parts: tuple[Any, ...], modulo: int = 2**31 - 1) -> int:
    joined = "::".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(joined).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def _generator_seed(row: dict[str, Any], config: InferenceConfig, index: int) -> int:
    if not config.shared_case_seed:
        return int(config.seed + index)
    case_hash = _stable_seed(("case", row.get("case_id", "")))
    view_hash = _stable_seed(("view", row.get("view_id", "")))
    return int((config.seed + case_hash * max(1, config.case_seed_stride) + view_hash) % (2**31 - 1))


def _crop_box_or_full(
    image_shape: tuple[int, int],
    face_box: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    if face_box is None:
        return 0, 0, width, height
    x1, y1, x2, y2 = face_box
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def _crop_array(array: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return array[y1:y2, x1:x2].copy()


def _paste_array(
    base: np.ndarray,
    crop: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = box
    output = base.copy()
    output[y1:y2, x1:x2] = crop
    return output


def _composite_with_mask(
    base_bgr: np.ndarray,
    generated_bgr: np.ndarray,
    feather_mask_u8: np.ndarray,
) -> np.ndarray:
    alpha = feather_mask_u8.astype(np.float32) / 255.0
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    base_f = base_bgr.astype(np.float32)
    gen_f = generated_bgr.astype(np.float32)
    mixed = gen_f * alpha + base_f * (1.0 - alpha)
    return mixed.clip(0, 255).astype(np.uint8)


def _make_preview(
    input_bgr: np.ndarray,
    mask_u8: np.ndarray,
    depth_bgr: np.ndarray,
    raw_bgr: np.ndarray,
    composite_bgr: np.ndarray,
) -> np.ndarray:
    mask_bgr = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
    height, width = input_bgr.shape[:2]
    tiles = [input_bgr, mask_bgr, depth_bgr, raw_bgr, composite_bgr]
    resized = [cv2.resize(tile, (width, height), interpolation=cv2.INTER_NEAREST) for tile in tiles]
    return np.concatenate(resized, axis=1)


def _torch_dtype(name: str, device: torch.device) -> torch.dtype:
    import torch

    key = str(name).strip().lower()
    if device.type != "cuda":
        return torch.float32
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def _normalized_quant_components(config: InferenceConfig) -> tuple[str, ...]:
    """标准化需要量化的 pipeline 组件名。"""
    allowed = {"transformer", "unet", "controlnet", "text_encoder", "text_encoder_2", "vae"}
    components = []
    for name in config.quantize_components:
        key = str(name).strip().lower()
        if not key:
            continue
        if key not in allowed:
            raise ValueError(f"不支持的量化组件: {name}")
        if key not in components:
            components.append(key)
    return tuple(components)


def _build_quantization_config(config: InferenceConfig):
    """
    构造 diffusers 量化配置。

    当前只在推理侧使用，优先支持：
    - QuantoConfig(weights_dtype="float8")
    - TorchAoConfig("float8wo")
    """
    backend = str(config.quantization_backend).strip().lower()
    if backend in {"", "none", "off", "disable", "disabled"}:
        return None

    components = _normalized_quant_components(config)
    if not components:
        return None

    if backend == "quanto":
        from diffusers import PipelineQuantizationConfig, QuantoConfig

        quant_cfg = QuantoConfig(weights_dtype=str(config.quantization_dtype).strip().lower())
        return PipelineQuantizationConfig(quant_mapping={name: quant_cfg for name in components})

    if backend == "torchao":
        from diffusers import PipelineQuantizationConfig, TorchAoConfig

        quant_cfg = TorchAoConfig(str(config.torchao_quant_type).strip())
        return PipelineQuantizationConfig(quant_mapping={name: quant_cfg for name in components})

    raise ValueError(f"不支持的 quantization_backend: {config.quantization_backend}")


def _build_component_quant_config(config: InferenceConfig, component_name: str):
    """为独立加载的单个组件构造量化配置。"""
    if component_name not in _normalized_quant_components(config):
        return None

    backend = str(config.quantization_backend).strip().lower()
    if backend in {"", "none", "off", "disable", "disabled"}:
        return None

    if backend == "quanto":
        from diffusers import QuantoConfig

        return QuantoConfig(weights_dtype=str(config.quantization_dtype).strip().lower())

    if backend == "torchao":
        from diffusers import TorchAoConfig

        return TorchAoConfig(str(config.torchao_quant_type).strip())

    raise ValueError(f"不支持的 quantization_backend: {config.quantization_backend}")


def _load_pipeline(config: InferenceConfig, device: torch.device):
    from diffusers import StableDiffusionXLControlNetInpaintPipeline, StableDiffusionXLInpaintPipeline

    dtype = _torch_dtype(config.torch_dtype, device)
    quant_components = _normalized_quant_components(config)
    quantization_config = _build_quantization_config(config)
    controlnet_quant_cfg = _build_component_quant_config(config, "controlnet")
    pipe_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    pipe_cls = StableDiffusionXLInpaintPipeline

    if config.enable_depth_condition:
        from diffusers import ControlNetModel

        controlnet = ControlNetModel.from_pretrained(
            config.controlnet_model_id,
            torch_dtype=dtype,
            quantization_config=controlnet_quant_cfg,
        )
        pipe_cls = StableDiffusionXLControlNetInpaintPipeline
        pipe_kwargs["controlnet"] = controlnet
        if quantization_config is not None:
            if "controlnet" in quant_components:
                quant_mapping = dict(quantization_config.quant_mapping)
                quant_mapping.pop("controlnet", None)
                quantization_config = type(quantization_config)(quant_mapping=quant_mapping)
            if getattr(quantization_config, "quant_mapping", None):
                pipe_kwargs["quantization_config"] = quantization_config
    elif quantization_config is not None and getattr(quantization_config, "quant_mapping", None):
        quant_mapping = dict(quantization_config.quant_mapping)
        quant_mapping.pop("controlnet", None)
        if quant_mapping:
            pipe_kwargs["quantization_config"] = type(quantization_config)(quant_mapping=quant_mapping)
    if str(config.pipeline_device_map).strip():
        pipe_kwargs["device_map"] = str(config.pipeline_device_map).strip()
    pipe = pipe_cls.from_pretrained(
        config.base_model_id,
        **pipe_kwargs,
    )
    if config.lora_weights_path:
        # LoRA 既支持 HF repo，也支持本地目录；默认不做强绑定路径假设。
        pipe.load_lora_weights(config.lora_weights_path)
        if hasattr(pipe, "fuse_lora"):
            pipe.fuse_lora(lora_scale=config.lora_scale)
    if not str(config.pipeline_device_map).strip():
        pipe.to(device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe


def run_inference(
    paths: GenerationPaths | None = None,
    config: InferenceConfig | None = None,
) -> dict[str, Any]:
    """运行术前图批量推理。"""
    paths = paths or default_generation_paths()
    config = config or default_inference_config()
    manifest_rows = build_generation_manifest(paths)
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = _load_pipeline(config, device)

    # 推理固定读取术前多视角样本。
    # 输入图默认已经是 eye_mask 后的主数据；每个视角仍独立 inpaint，
    # 但种子会按 case_id/view_id 稳定，减小同病例多视角之间的随机漂移。
    input_rows = [row for row in manifest_rows if row.get("stage") == "术前"]
    if config.max_samples > 0:
        input_rows = input_rows[: config.max_samples]

    success_count = 0
    failed_count = 0
    issues: list[dict[str, Any]] = []
    predicted_views_by_case: dict[str, set[str]] = {}

    for idx, row in enumerate(input_rows):
        image_path = _row_path(row, "sanitized_image_path", "image_path")
        if image_path is None:
            image_path = Path("")
        if not image_path.exists():
            failed_count += 1
            issues.append(
                {
                    "kind": "missing_input_image",
                    "image_path": str(image_path),
                    "case_id": row.get("case_id"),
                    "view_id": row.get("view_id"),
                }
            )
            continue

        rel_path = _resolve_rel_path(paths, image_path)
        inpaint_mask_path = _row_path(row, "inpaint_mask_path")
        if inpaint_mask_path is None:
            inpaint_mask_path = _mask_path(paths.inpaint_mask_root, rel_path)
        feather_mask_path = _row_path(row, "feather_mask_path")
        if feather_mask_path is None:
            feather_mask_path = _mask_path(paths.feather_mask_root, rel_path)
        privacy_mask_path = _row_path(row, "eye_privacy_mask_path")
        if privacy_mask_path is None:
            privacy_mask_path = _mask_path(paths.privacy_mask_root, rel_path)
        face_mask_path = _row_path(row, "face_mask_path")
        depth_path = _row_path(row, "depth_path") or Path("")

        missing_depth = config.enable_depth_condition and not depth_path.exists()
        if not inpaint_mask_path.exists() or missing_depth:
            failed_count += 1
            issues.append(
                {
                    "kind": "missing_condition",
                    "rel_path": str(rel_path),
                    "inpaint_mask_exists": inpaint_mask_path.exists(),
                    "depth_required": bool(config.enable_depth_condition),
                    "depth_exists": depth_path.exists(),
                }
            )
            continue

        base_bgr = cv2.imread(str(image_path))
        inpaint_mask_u8 = cv2.imread(str(inpaint_mask_path), cv2.IMREAD_GRAYSCALE)
        feather_mask_u8 = cv2.imread(str(feather_mask_path), cv2.IMREAD_GRAYSCALE)
        if base_bgr is None:
            failed_count += 1
            issues.append(
                {
                    "kind": "read_failed",
                    "rel_path": str(rel_path),
                    "image_path": str(image_path),
                }
            )
            continue
        if inpaint_mask_u8 is None:
            failed_count += 1
            issues.append(
                {
                    "kind": "mask_read_failed",
                    "rel_path": str(rel_path),
                    "mask_path": str(inpaint_mask_path),
                }
            )
            continue

        if config.apply_eye_privacy_mask and not str(image_path).startswith(str(paths.sanitized_root)):
            privacy_mask = _load_binary_mask(privacy_mask_path)
            if privacy_mask is not None:
                base_bgr = _apply_privacy_fill(base_bgr, privacy_mask, config.privacy_fill)

        prompt = str(config.prompt_template).format(
            doctor_token=row.get("doctor_token", "dr_style"),
            view_token=row.get("view_token", ""),
        )
        generator = torch.Generator(device=device.type).manual_seed(_generator_seed(row, config, idx))

        face_box = _crop_box_or_full(
            base_bgr.shape,
            _load_mask_bbox(face_mask_path) if face_mask_path is not None else None,
        )
        cropped_base_bgr = _crop_array(base_bgr, face_box)
        cropped_mask_u8 = _crop_array(inpaint_mask_u8, face_box)
        if not np.any(cropped_mask_u8 > 0):
            failed_count += 1
            issues.append(
                {
                    "kind": "empty_inpaint_mask",
                    "rel_path": str(rel_path),
                    "face_box": list(face_box),
                }
            )
            continue

        if feather_mask_u8 is None:
            cropped_feather_u8 = _blur_mask(cropped_mask_u8 > 0, config.composite_blur_px)
        else:
            cropped_feather_u8 = _crop_array(feather_mask_u8, face_box)

        depth_bgr = np.zeros_like(base_bgr)
        cropped_depth_bgr = np.zeros_like(cropped_base_bgr)
        if config.enable_depth_condition and depth_path.exists():
            loaded_depth_bgr = cv2.imread(str(depth_path))
            if loaded_depth_bgr is not None:
                depth_bgr = loaded_depth_bgr
                cropped_depth_bgr = _crop_array(loaded_depth_bgr, face_box)

        input_image = Image.fromarray(cv2.cvtColor(cropped_base_bgr, cv2.COLOR_BGR2RGB))
        mask_image = Image.fromarray(cropped_mask_u8)
        pipe_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": config.negative_prompt,
            "image": input_image,
            "mask_image": mask_image,
            "num_inference_steps": config.num_inference_steps,
            "guidance_scale": config.guidance_scale,
            "strength": config.strength,
            "num_images_per_prompt": config.num_images_per_prompt,
            "generator": generator,
        }
        if config.enable_depth_condition:
            pipe_kwargs.update(
                {
                    "control_image": Image.fromarray(cv2.cvtColor(cropped_depth_bgr, cv2.COLOR_BGR2RGB)),
                    "controlnet_conditioning_scale": config.controlnet_conditioning_scale,
                    "control_guidance_start": config.control_guidance_start,
                    "control_guidance_end": config.control_guidance_end,
                }
            )
        result = pipe(**pipe_kwargs)
        generated_rgb = result.images[0]
        raw_crop_bgr = cv2.cvtColor(np.asarray(generated_rgb), cv2.COLOR_RGB2BGR)

        # 推理固定在 face crop 内运行，再把局部结果映射回整图。
        raw_bgr = _paste_array(base_bgr, raw_crop_bgr, face_box)
        composite_crop_bgr = _composite_with_mask(cropped_base_bgr, raw_crop_bgr, cropped_feather_u8)
        composite_bgr = _paste_array(base_bgr, composite_crop_bgr, face_box)

        raw_path = _mask_path(paths.inference_root / "raw", rel_path)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(raw_path), raw_bgr)

        composite_path = _mask_path(paths.inference_root / "composited", rel_path)
        composite_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(composite_path), composite_bgr)

        preview_path = _mask_path(paths.inference_root / "preview", rel_path)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview = _make_preview(base_bgr, inpaint_mask_u8, depth_bgr, raw_bgr, composite_bgr)
        cv2.imwrite(str(preview_path), preview)
        success_count += 1
        predicted_views_by_case.setdefault(str(row.get("case_id") or ""), set()).add(str(row.get("view_id") or ""))

    summary = {
        "requested_samples": len(input_rows),
        "success_count": success_count,
        "failed_count": failed_count,
        "predicted_view_count": sum(len(views) for views in predicted_views_by_case.values()),
        "predicted_cases": sorted(case_id for case_id in predicted_views_by_case if case_id),
        "inference_root": str(paths.inference_root),
        "device": str(device),
        "enable_depth_condition": bool(config.enable_depth_condition),
        "quantization_backend": str(config.quantization_backend),
        "quantize_components": list(_normalized_quant_components(config)),
        "config": dataclass_to_dict(config),
    }
    issues_path = paths.summaries_root / "infer_issues.jsonl"
    issues_path.parent.mkdir(parents=True, exist_ok=True)
    with issues_path.open("w", encoding="utf-8") as f:
        for item in issues:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    write_json(paths.summaries_root / "infer_summary.json", summary)
    dump_config_snapshot(
        paths.summaries_root / "infer_config.json",
        paths=paths,
        inference=config,
    )
    return summary


def main() -> None:
    summary = run_inference()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
