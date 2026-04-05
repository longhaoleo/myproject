"""
SDXL 生成侧模块。

对外暴露：
- 路径配置
- LoRA 训练配置
- 推理配置
"""

from .settings import (
    GenerationPaths,
    InferenceConfig,
    LoRATrainConfig,
    default_generation_paths,
    default_inference_config,
    default_lora_train_config,
)

__all__ = [
    "GenerationPaths",
    "InferenceConfig",
    "LoRATrainConfig",
    "default_generation_paths",
    "default_inference_config",
    "default_lora_train_config",
]
