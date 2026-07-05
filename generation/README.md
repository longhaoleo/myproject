# generation 模块说明

该目录负责把现有前处理产物接成一条可复现的 SDXL 生成链路。

统一入口：`python generation_pipeline.py`

支持 3 个模式：

1. `train_lora`
2. `infer`
3. `evaluate`

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

1. 训练和推理默认直接使用 `eye_mask` 后的图片；若未生成，则回退到原图。
2. `face` 会被整理成标准化头部框，用于头部参考区域裁剪。
3. 重绘目标区域是连续的 `nose + mouth` 联合 `inpaint_mask`。
4. `feather_mask` 只用于最终回贴时的羽化混合。
5. 默认优先使用方框 mask，而不是直接使用 SAM 轮廓。
6. depth 条件图目前是可选扩展，默认不启用。

### 1.3 三个模式的职责

#### `train_lora`

负责：

1. 运行时直接从 detection 输出构建 manifest，并按 `case_id` 进行 train / val / test 切分
2. 当前 baseline 只使用同病例同视角 paired 样本
3. 每个训练样本统一构造成 `pre_to_post_inpaint`
   - `condition_image`：术前图的正方形 `face` crop
   - `target_image`：同视角术后图的正方形 `face` crop
   - `mask`：术前图的连续鼻嘴 `inpaint_mask`
4. 默认训练时加载冻结的 IP-Adapter，把同病例术前最多 6 个视角 face crop 作为 reference 条件
5. 训练时在术前 face crop 里把鼻嘴区域挖空，mask 外保留术前上下文，让模型学习在当前人脸和多视角参考条件下重绘术后鼻嘴
6. face crop 和 mask crop 直接 resize 到训练分辨率，不再用额外 padding
7. 默认使用 `fp16` 和 UNet gradient checkpointing，降低显存和权重占用
8. face mask bbox 会缓存到 `summaries/mask_bbox_cache.json`，首次构造较慢，后续基本秒开
9. 缺视角利用先作为后续工作
10. 落盘 checkpoint、最终 LoRA 权重和训练摘要

#### `infer`

默认链路：

```text
术前图
-> detection 生成的眼部打码图（主输入）
-> detection 生成的连续鼻嘴 inpaint mask
-> face crop
-> 可选：同病例术前 6 视角 face crop 作为 IP-Adapter reference
-> SDXL Inpainting + LoRA
-> 映射回整图
-> 仅按羽化 mask 回贴鼻嘴联合区域
-> 输出 raw / composited / preview
```

可选扩展：

```text
+ depth 条件图
+ ControlNet-Depth
+ IP-Adapter reference：同病例术前多视角图，默认最多 6 张
```

#### `evaluate`

负责：

1. 运行时直接从 detection 输出构建 manifest
2. 统计 nose / face mask 可用率
3. 若启用了 depth，则统计 depth 可用率
4. 统计 paired / unpaired 视角数量、完整六视角病例和部分视角病例
5. 统计 `hard_identity_similarity` 与 `soft_face_similarity`
6. 导出 `术前 / 术后 / 生成结果` 三联预览，以及病例级 contact sheet

### 1.4 一致性策略（v1）

当前一致性是轻量方案，不额外引入 identity loss。

固定拆成两层：

1. `hard_identity_region = face_mask - dilated(inpaint_mask)`
   - 代表非编辑区，要求强一致
2. `soft_identity_region = face_mask`
   - 鼻子和嘴巴虽然会变化，但仍保留为弱一致性参考

说明：

1. 当前训练 baseline 先验证术前锚点 paired inpainting 能否稳定学会
2. 评估阶段通过 `hard_identity_similarity` 和 `soft_face_similarity` 做轻量验收
3. 多视角一致性和显式 identity / perceptual consistency loss 都放到后续版本

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
- IP-Adapter（可选）：默认本地目录 `model/generation/ip-adapter`，来源 repo 为 `h94/IP-Adapter`
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

hf download h94/IP-Adapter \
  sdxl_models/image_encoder/config.json \
  sdxl_models/image_encoder/model.safetensors \
  sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors \
  --local-dir model/generation/ip-adapter \
  --max-workers 1

huggingface-cli download Intel/dpt-hybrid-midas \
  --local-dir model/generation/dpt-hybrid-midas
```

若你使用本地目录，请把 [generation/settings.py](settings.py) 中的相关字段改成对应本地路径，或在调用配置时覆盖：

- `LoRATrainConfig.base_model_id`
- `InferenceConfig.base_model_id`
- `InferenceConfig.controlnet_model_id`
- `InferenceConfig.ip_adapter_model_id`

### 3.4 IP-Adapter 多视角参考

推理侧可以开启 IP-Adapter，把同病例术前最多 6 个视角作为 reference：

```text
InferenceConfig.enable_ip_adapter = True
InferenceConfig.ip_adapter_model_id = "model/generation/ip-adapter"
InferenceConfig.ip_adapter_subfolder = "sdxl_models"
InferenceConfig.ip_adapter_weight_name = "ip-adapter-plus-face_sdxl_vit-h.safetensors"
InferenceConfig.ip_adapter_image_encoder_folder = "sdxl_models/image_encoder"
InferenceConfig.ip_adapter_scale = 0.55
InferenceConfig.ip_adapter_max_reference_images = 6
```

运行时会按 `case_id` 收集 `术前` 行，按视角排序，默认用每张图自己的 `face_mask` 裁成 face crop 后传给 IP-Adapter。这样当前视角仍由 SDXL inpainting 的 `image + mask` 控制，其他术前视角作为身份和脸部风格参考。

训练脚本同样默认启用这个条件，但只训练 LoRA，不训练 IP-Adapter：

```text
ENABLE_IP_ADAPTER_CONDITION=1
IP_ADAPTER_SCALE=0.55
IP_ADAPTER_MAX_REFERENCE_IMAGES=6
MIXED_PRECISION=fp16
GRADIENT_CHECKPOINTING=1
```

也可以直接用脚本跑 `Feature_test`：

```bash
tools/run_infer_feature_test_ip_adapter.sh
```

常用覆盖项：

```bash
MAX_SAMPLES=6 STRENGTH=0.60 IP_ADAPTER_SCALE=0.55 \
tools/run_infer_feature_test_ip_adapter.sh
```

### 3.5 FP8 推理配置

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

### 4.1 训练 LoRA

```bash
cd ~/myproject
GEN_RUN_MODE=train_lora python generation_pipeline.py
```

### 4.2 运行推理

```bash
cd ~/myproject
GEN_RUN_MODE=infer python generation_pipeline.py
```

如果你保持默认配置，这一步会优先按 `quanto + FP8` 加载推理模型。
若当前环境的 `diffusers / torch / CUDA` 组合不支持，再把 `InferenceConfig.quantization_backend` 改成 `none` 做回退。

### 4.3 运行评估

```bash
cd ~/myproject
GEN_RUN_MODE=evaluate python generation_pipeline.py
```

## 5. 默认输出目录

默认根目录：`~/datasets/deformity_generation`

主要产物：

```text
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
├── manifest.jsonl
├── mask_bbox_cache.json
├── train_lora_summary.json
├── infer_summary.json
└── evaluate_summary.json
```

## 6. 当前限制

1. v1 只训练 LoRA，不训练自定义 ControlNet。
2. IP-Adapter 是冻结参考条件；当前 LoRA 训练侧会使用它的多视角条件，但不训练 IP-Adapter 权重。
3. 当前训练侧是“术前锚点 paired inpainting baseline”，优先验证 crop/mask/LoRA 链路，再逐步加入更强的身份约束。
4. 当前默认不启用 depth；如果后续重新打开，depth 生成支持 `transformers` 和 `opencv-luma fallback` 两条路。
5. 当前 `FP8` 支持只接在推理侧；训练侧默认使用 `fp16`，并把保存的 LoRA 权重转成 `fp16`。
