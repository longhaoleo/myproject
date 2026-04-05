# generation 模块说明

该目录负责把现有前处理产物接成一条可复现的 SDXL 生成链路。

统一入口：`python generation_pipeline.py`

支持 4 个模式：

1. `prepare_dataset`
2. `train_lora`
3. `infer`
4. `evaluate`

## 1. 当前逻辑

### 1.1 输入约定

原始数据目录沿用现有项目约定：

```text
<input_root>/
└── <case_id>/
    ├── 术前/
    │   ├── 1.JPG
    │   ├── 2.JPG
    │   └── ...
    └── 术后/
        ├── 1.JPG
        ├── 2.JPG
        └── ...
```

SAM 部位 mask 沿用 `detection/sam_mask.py` 的输出：

```text
<sam_root>/
└── parts/
    ├── face/
    ├── left_eye/
    ├── right_eye/
    ├── nose/
    └── mouth/
```

### 1.2 detection 提供的 mask 规则

这些规则由 detection 侧生成并落盘，generation 只读取：

1. 重绘区域是连续的 `nose + mouth` 联合 `inpaint_mask`。
2. 默认优先使用方框 mask，而不是直接使用 SAM 轮廓。
3. `face` 会被整理成更大的方形头框，作为头部参考区域。
4. 眼部隐私 mask 可选预先打码，并且会单独落盘成 `eye_privacy_mask` 图。
5. 若眼部隐私 mask 与鼻嘴重绘区重叠，会自动把重叠区域从眼部 mask 中抠掉，并给编辑区额外留一圈安全边距。
6. 训练与推理默认读取 detection 落盘的 `sanitized` 图，也就是已经应用过眼部 mask 的版本。
7. depth 条件图目前是可选扩展，默认不启用；若后续接回，会基于“已完成眼部隐私处理”的图生成。

### 1.3 四个模式的职责

#### `prepare_dataset`

说明：

1. 该模式实际调用的是 detection 侧的 `prepare_dataset.py`。
2. generation 不负责定义 mask 区域，只消费 detection 侧落盘的 `inpaint_mask` 与相关条件。

#### `train_lora`

负责：

1. 从 manifest 里筛选 `术后 + train split` 样本
2. 同时构建两类训练样本
   - 方形头部参考图
   - 鼻嘴联合 crop 图
3. 使用 `sanitized` 图和 `inpaint_mask` 训练 SDXL inpainting UNet LoRA
4. 落盘 checkpoint、最终 LoRA 权重和训练摘要

#### `infer`

默认链路：

```text
术前图
-> detection 生成的连续鼻嘴 inpaint mask
-> detection 生成的 sanitized 参考图
-> SDXL Inpainting + LoRA
-> 整图输出
-> 仅按羽化 mask 回贴鼻嘴联合区域
-> 输出 raw / composited / preview
```

可选扩展：

```text
+ depth 条件图
+ ControlNet-Depth
```

#### `evaluate`

负责：

1. 统计 manifest 完整率
2. 统计 nose / face mask 可用率
3. 若启用了 depth，则统计 depth 可用率
4. 统计术前图推理输出成功率
5. 导出 `术前 / 术后 / 生成结果` 三联预览

## 2. 依赖安装

生成侧依赖已经写入项目根目录 [requirements.txt](../requirements.txt)。

建议环境：

```bash
cd ~/myproject
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果是 GPU 环境，建议先装对应 CUDA 版本的 `torch / torchvision`，再装项目依赖。

如果你要跑 `FP8` 推理，当前实现优先支持两条量化后端：

1. `quanto`
2. `torchao`

对应依赖也已经写入 `requirements.txt`：

- `optimum-quanto`
- `torchao`

建议：

- 默认先用 `quanto`
- `torchao` 只在你的 `torch` / CUDA 版本已经匹配时再切换

## 3. 模型与权重

### 3.1 默认模型 ID

当前代码默认使用：

- LoRA 训练底模：`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`
- Inpainting 底模：`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`
- ControlNet-Depth（可选）：`diffusers/controlnet-depth-sdxl-1.0`
- Depth 模型（可选）：`Intel/dpt-hybrid-midas`

说明：

- 这些值定义在 [generation/settings.py](settings.py)
- `from_pretrained(...)` 同时支持 Hugging Face repo id 和本地目录
- 如果你已经把模型下载到本地，可以直接把配置改成本地路径
- 当前推理侧支持在 `from_pretrained(...)` 阶段直接挂 `FP8` 量化配置

### 3.2 推荐本地目录结构

```text
model/
└── generation/
    ├── sdxl-inpaint/
    ├── controlnet-depth-sdxl/
    └── dpt-hybrid-midas/
```

### 3.3 下载方式

先确保 Hugging Face CLI 可用：

```bash
python -m pip install -U huggingface-hub
huggingface-cli login
```

然后可按下面方式下载到本地目录：

```bash
mkdir -p model/generation

huggingface-cli download diffusers/stable-diffusion-xl-1.0-inpainting-0.1 \
  --local-dir model/generation/sdxl-inpaint

huggingface-cli download diffusers/controlnet-depth-sdxl-1.0 \
  --local-dir model/generation/controlnet-depth-sdxl

huggingface-cli download Intel/dpt-hybrid-midas \
  --local-dir model/generation/dpt-hybrid-midas
```

若你使用本地目录，请把 [generation/settings.py](settings.py) 中的相关字段改成对应本地路径，或在调用配置时覆盖：

- `LoRATrainConfig.base_model_id`
- `InferenceConfig.base_model_id`
- `InferenceConfig.controlnet_model_id`
- `DatasetBuildConfig.depth_model_id`

### 3.4 FP8 推理配置

当前代码已经在 [generation/infer.py](infer.py) 支持 `from_pretrained(...)` 直接挂量化配置。

默认配置在 [generation/settings.py](settings.py) 的 `InferenceConfig`：

- `enable_depth_condition=False`
- `quantization_backend="quanto"`
- `quantization_dtype="float8"`
- `quantize_components=("unet", "controlnet")`
- `torchao_quant_type="float8wo"`

说明：

1. 默认优先用 `QuantoConfig(weights_dtype="float8")`
2. 若你改成 `quantization_backend="torchao"`，则会走 `TorchAoConfig("float8wo")`
3. 默认 `enable_depth_condition=False`，所以实际默认只会用到 `unet`
4. 若你后续打开 depth 条件，再把 `controlnet` 保留在量化组件里即可
5. `text_encoder / vae` 默认不量化，先保证兼容性和稳定性

如果你想显式关闭量化，可把：

- `quantization_backend="none"`

如果你想手动覆盖配置，最小示例：

```python
from generation.infer import run_inference
from generation.settings import default_inference_config, update_dataclass

config = update_dataclass(
    default_inference_config(),
    enable_depth_condition=False,
    quantization_backend="quanto",
    quantization_dtype="float8",
    quantize_components=("unet",),
)

run_inference(config=config)
```

## 4. 运行顺序

### 4.1 准备数据

```bash
cd ~/myproject
GEN_RUN_MODE=prepare_dataset python generation_pipeline.py
```

### 4.2 训练 LoRA

```bash
cd ~/myproject
GEN_RUN_MODE=train_lora python generation_pipeline.py
```

### 4.3 运行推理

```bash
cd ~/myproject
GEN_RUN_MODE=infer python generation_pipeline.py
```

如果你保持默认配置，这一步会优先按 `quanto + FP8` 加载推理模型。
若当前环境的 `diffusers / torch / CUDA` 组合不支持，再把 `InferenceConfig.quantization_backend` 改成 `none` 做回退。

### 4.4 运行评估

```bash
cd ~/myproject
GEN_RUN_MODE=evaluate python generation_pipeline.py
```

## 5. 默认输出目录

默认根目录：`~/datasets/deformity_generation`

主要产物：

```text
prepared/
├── manifest.jsonl
├── issues.jsonl
├── inpaint_mask/
├── feather_mask/
├── eye_privacy_mask/
├── sanitized/
├── preview/
└── parts/

lora/
├── checkpoint-*/
└── final/

infer/
├── raw/
├── composited/
└── preview/

eval/
└── triptychs/

summaries/
├── prepare_dataset_summary.json
├── train_lora_summary.json
├── infer_summary.json
└── evaluate_summary.json
```

## 6. 当前限制

1. v1 只训练 LoRA，不训练自定义 ControlNet。
2. 默认不接 IP-Adapter。
3. 当前训练侧是“最小闭环实现”，优先保证链路打通，而不是先做复杂训练技巧。
4. 当前默认不启用 depth；如果后续重新打开，depth 生成支持 `transformers` 和 `opencv-luma fallback` 两条路。
5. 当前 `FP8` 支持只接在推理侧；训练侧仍是常规精度加载。
