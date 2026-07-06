#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# This script is for overfit verification. It runs inference on the same Train
# split used by run_lora_pre2post_800step.sh, so preview images include:
# 术前输入 / 真实术后 / 预测术后 / 回贴结果
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/deformity_dataset/Train}"
export GEN_INPUT_ROOT="${GEN_INPUT_ROOT:-${DATASET_ROOT}/images}"
export GEN_SAM_ROOT="${GEN_SAM_ROOT:-${DATASET_ROOT}/masks}"
export GEN_INPAINT_MASK_ROOT="${GEN_INPAINT_MASK_ROOT:-${DATASET_ROOT}/masks/inpaint_mask}"
export GEN_FEATHER_MASK_ROOT="${GEN_FEATHER_MASK_ROOT:-${DATASET_ROOT}/masks/feather_mask}"
export GEN_OUTPUT_ROOT="${GEN_OUTPUT_ROOT:-${PROJECT_ROOT}/Image_output/generation/train_overfit_noip_s045}"
export GEN_INFERENCE_ROOT="${GEN_INFERENCE_ROOT:-${GEN_OUTPUT_ROOT}/infer}"
export GEN_EVAL_ROOT="${GEN_EVAL_ROOT:-${GEN_OUTPUT_ROOT}/eval}"

export BASE_MODEL_ID="${BASE_MODEL_ID:-${PROJECT_ROOT}/model/generation/stable-diffusion-xl-1.0-inpainting-0.1}"
export LORA_WEIGHTS_PATH="${LORA_WEIGHTS_PATH:-${PROJECT_ROOT}/Image_output/generation/lora_pre2post_800step/lora/final}"
export IP_ADAPTER_MODEL_ID="${IP_ADAPTER_MODEL_ID:-${PROJECT_ROOT}/model/generation/ip-adapter}"
export IP_ADAPTER_SUBFOLDER="${IP_ADAPTER_SUBFOLDER:-sdxl_models}"
export IP_ADAPTER_WEIGHT_NAME="${IP_ADAPTER_WEIGHT_NAME:-ip-adapter-plus-face_sdxl_vit-h.safetensors}"
export IP_ADAPTER_IMAGE_ENCODER_FOLDER="${IP_ADAPTER_IMAGE_ENCODER_FOLDER:-sdxl_models/image_encoder}"
export IP_ADAPTER_SCALE="${IP_ADAPTER_SCALE:-0.55}"
export IP_ADAPTER_MAX_REFERENCE_IMAGES="${IP_ADAPTER_MAX_REFERENCE_IMAGES:-6}"
export ENABLE_IP_ADAPTER="${ENABLE_IP_ADAPTER:-0}"
export STRENGTH="${STRENGTH:-0.45}"
export GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-25}"
export TORCH_DTYPE="${TORCH_DTYPE:-fp16}"
export MAX_SAMPLES="${MAX_SAMPLES:-6}"
export QUANTIZATION_BACKEND="${QUANTIZATION_BACKEND:-none}"
CLEAR_OUTPUT="${CLEAR_OUTPUT:-1}"

if [[ ! -d "${LORA_WEIGHTS_PATH}" && ! -f "${LORA_WEIGHTS_PATH}" ]]; then
  echo "LoRA weights not found: ${LORA_WEIGHTS_PATH}" >&2
  echo "Set LORA_WEIGHTS_PATH to your trained lora/final or checkpoint directory." >&2
  exit 1
fi

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

from generation.infer import run_inference
from generation.settings import default_generation_paths, default_inference_config, update_dataclass

paths = default_generation_paths()
config = update_dataclass(
    default_inference_config(),
    base_model_id=os.environ["BASE_MODEL_ID"],
    lora_weights_path=os.environ["LORA_WEIGHTS_PATH"],
    enable_ip_adapter=os.environ["ENABLE_IP_ADAPTER"].strip() not in {"0", "false", "False", ""},
    ip_adapter_model_id=os.environ["IP_ADAPTER_MODEL_ID"],
    ip_adapter_subfolder=os.environ["IP_ADAPTER_SUBFOLDER"],
    ip_adapter_weight_name=os.environ["IP_ADAPTER_WEIGHT_NAME"],
    ip_adapter_image_encoder_folder=os.environ["IP_ADAPTER_IMAGE_ENCODER_FOLDER"],
    ip_adapter_scale=float(os.environ["IP_ADAPTER_SCALE"]),
    ip_adapter_max_reference_images=int(os.environ["IP_ADAPTER_MAX_REFERENCE_IMAGES"]),
    strength=float(os.environ["STRENGTH"]),
    guidance_scale=float(os.environ["GUIDANCE_SCALE"]),
    num_inference_steps=int(os.environ["NUM_INFERENCE_STEPS"]),
    torch_dtype=os.environ["TORCH_DTYPE"],
    max_samples=int(os.environ["MAX_SAMPLES"]),
    quantization_backend=os.environ["QUANTIZATION_BACKEND"],
)
summary = run_inference(paths=paths, config=config)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
