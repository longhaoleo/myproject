#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/deformity_dataset/Train}"
export GEN_INPUT_ROOT="${GEN_INPUT_ROOT:-${DATASET_ROOT}/images}"
export GEN_SAM_ROOT="${GEN_SAM_ROOT:-${DATASET_ROOT}/masks}"
export GEN_INPAINT_MASK_ROOT="${GEN_INPAINT_MASK_ROOT:-${DATASET_ROOT}/masks/inpaint_mask}"
export GEN_FEATHER_MASK_ROOT="${GEN_FEATHER_MASK_ROOT:-${DATASET_ROOT}/masks/feather_mask}"
export GEN_OUTPUT_ROOT="${GEN_OUTPUT_ROOT:-${PROJECT_ROOT}/Image_output/generation/lora_pre2post_800step}"
export GEN_LORA_ROOT="${GEN_LORA_ROOT:-${GEN_OUTPUT_ROOT}/lora}"

export BASE_MODEL_ID="${BASE_MODEL_ID:-${PROJECT_ROOT}/model/generation/stable-diffusion-xl-1.0-inpainting-0.1}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-800}"
export SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-200}"
export LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-10}"
export RESOLUTION="${RESOLUTION:-1024}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
export LEARNING_RATE="${LEARNING_RATE:-5e-5}"
export MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
export RANK="${RANK:-16}"
export ALPHA="${ALPHA:-16}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export ENABLE_IP_ADAPTER_CONDITION="${ENABLE_IP_ADAPTER_CONDITION:-1}"
export IP_ADAPTER_MODEL_ID="${IP_ADAPTER_MODEL_ID:-${PROJECT_ROOT}/model/generation/ip-adapter}"
export IP_ADAPTER_SUBFOLDER="${IP_ADAPTER_SUBFOLDER:-sdxl_models}"
export IP_ADAPTER_WEIGHT_NAME="${IP_ADAPTER_WEIGHT_NAME:-ip-adapter-plus-face_sdxl_vit-h.safetensors}"
export IP_ADAPTER_IMAGE_ENCODER_FOLDER="${IP_ADAPTER_IMAGE_ENCODER_FOLDER:-sdxl_models/image_encoder}"
export IP_ADAPTER_SCALE="${IP_ADAPTER_SCALE:-0.55}"
export IP_ADAPTER_MAX_REFERENCE_IMAGES="${IP_ADAPTER_MAX_REFERENCE_IMAGES:-6}"
CLEAR_OUTPUT="${CLEAR_OUTPUT:-1}"

if [[ "${CLEAR_OUTPUT}" == "1" ]]; then
  case "${GEN_OUTPUT_ROOT}" in
    "${PROJECT_ROOT}/Image_output/"*)
      rm -rf "${GEN_OUTPUT_ROOT}"
      ;;
    *)
      echo "Refuse to clear GEN_OUTPUT_ROOT outside project Image_output: ${GEN_OUTPUT_ROOT}" >&2
      exit 1
      ;;
  esac
fi

mkdir -p "${GEN_OUTPUT_ROOT}"

python - <<'PY'
import json
import os

from generation.settings import default_generation_paths, default_lora_train_config, update_dataclass
from generation.train import train_lora

paths = default_generation_paths()
config = update_dataclass(
    default_lora_train_config(),
    base_model_id=os.environ["BASE_MODEL_ID"],
    resolution=int(os.environ["RESOLUTION"]),
    max_train_steps=int(os.environ["MAX_TRAIN_STEPS"]),
    train_batch_size=int(os.environ["TRAIN_BATCH_SIZE"]),
    gradient_accumulation_steps=int(os.environ["GRADIENT_ACCUMULATION_STEPS"]),
    learning_rate=float(os.environ["LEARNING_RATE"]),
    rank=int(os.environ["RANK"]),
    alpha=int(os.environ["ALPHA"]),
    mixed_precision=os.environ["MIXED_PRECISION"],
    max_grad_norm=float(os.environ["MAX_GRAD_NORM"]),
    gradient_checkpointing=os.environ["GRADIENT_CHECKPOINTING"].strip() not in {"0", "false", "False", ""},
    save_every_steps=int(os.environ["SAVE_EVERY_STEPS"]),
    log_every_steps=int(os.environ["LOG_EVERY_STEPS"]),
    num_workers=int(os.environ["NUM_WORKERS"]),
    enable_ip_adapter_condition=os.environ["ENABLE_IP_ADAPTER_CONDITION"].strip() not in {"0", "false", "False", ""},
    ip_adapter_model_id=os.environ["IP_ADAPTER_MODEL_ID"],
    ip_adapter_subfolder=os.environ["IP_ADAPTER_SUBFOLDER"],
    ip_adapter_weight_name=os.environ["IP_ADAPTER_WEIGHT_NAME"],
    ip_adapter_image_encoder_folder=os.environ["IP_ADAPTER_IMAGE_ENCODER_FOLDER"],
    ip_adapter_scale=float(os.environ["IP_ADAPTER_SCALE"]),
    ip_adapter_max_reference_images=int(os.environ["IP_ADAPTER_MAX_REFERENCE_IMAGES"]),
)
summary = train_lora(paths=paths, config=config)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
