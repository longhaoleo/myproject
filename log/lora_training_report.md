# LoRA 训练计划：术前多视角条件下的术后鼻嘴重绘

日期：2026-07-05

## 1. 当前状态

本阶段仍是 active training plan，不是最终效果报告。

上一轮 800-step LoRA 已证明工程链路能跑通，但视觉效果不可用：mask 区域出现贴片、黑边、局部换脸感。复盘结论是不能先归因于 rank 小，而应先修正 inpainting 协议和条件输入。

当前代码已经完成下一轮训练前的关键修正：

1. 训练样本改为术前锚点的 `pre_to_post_inpaint`。
2. inpainting 条件图在 mask 内挖空，和推理语义对齐。
3. 训练和推理都接入冻结 IP-Adapter，把同病例术前最多 6 个视角作为参考条件。
4. 默认训练精度为 `fp16`，但可训练 LoRA 参数用 fp32 优化，保存时转 fp16。
5. loss 和 LoRA 保存前都会检查 NaN/Inf，避免再保存坏权重。
6. 默认开启 UNet gradient checkpointing，降低 1024 分辨率训练显存压力。

## 2. 数据拆分与用途

当前使用：

```text
/root/autodl-tmp/deformity_dataset/
├── Train/
├── Test/
└── Feature_test/
```

已确认数据情况：

1. `Train`
   - manifest 行：2254
   - 当前可构造训练样本：898 个 `pre_to_post_inpaint`
   - 用途：正式 LoRA 训练
2. `Test`
   - 主要是术前图
   - 用途：看无术后参考时的泛化重绘效果
3. `Feature_test`
   - 有一部分术前/术后同视角配对
   - 用途：固定验证集，看 `术前 / 术后 / 重绘` 对比

## 3. 当前训练样本定义

当前只有一种训练样本：

```text
sample_type = pre_to_post_inpaint
```

构造方式：

```text
condition_image_path = 同病例同视角术前图
target_image_path    = 同病例同视角术后图
inpaint_mask_path    = 术前图的鼻子+嘴巴 inpaint_mask
crop_box             = 术前图的 face_mask bbox
prompt               = doctor_token + view_token + caption_suffix
```

训练前处理：

1. condition 术前图、target 术后图、mask 都按术前 `face_mask` crop。
2. crop 后 resize 到训练分辨率，正式计划为 `1024 x 1024`。
3. `masked_pixel_values = condition_image * (mask < 0.5)`。
4. 也就是 mask 内旧鼻嘴信息被挖掉，mask 外保留术前人脸上下文。

这和上一轮失败训练的关键区别是：当前训练输入已经对齐标准 SDXL inpainting 语义，不再把完整鼻嘴旧信息直接送进 masked image latent。

## 4. IP-Adapter 条件输入

训练时默认开启冻结 IP-Adapter：

```text
ENABLE_IP_ADAPTER_CONDITION=1
IP_ADAPTER_MODEL_ID=model/generation/ip-adapter
IP_ADAPTER_WEIGHT_NAME=ip-adapter-plus-face_sdxl_vit-h.safetensors
IP_ADAPTER_MAX_REFERENCE_IMAGES=6
IP_ADAPTER_SCALE=0.55
```

每个训练样本会额外收集同病例术前视角：

```text
ip_adapter_reference_image_paths = 同病例术前最多 6 张图
```

每张参考图默认用自己的 `face_mask` 裁成 face crop。IP-Adapter 本身不训练，只作为身份、脸部结构和照片风格的参考条件。

已修正的维度问题：

1. 本地 image encoder 有 `hidden_size=1664` 和 `projection_dim=1280` 两类输出。
2. 当前 IP-Adapter 权重的投影层期望 1280 维输入。
3. 训练和推理现在都会按投影层实际维度选择 `image_embeds`，避免把 1664 维 hidden states 错喂进去。
4. 推理侧验证过 embedding 形状，例如 `(2, 5, 1, 1280)`；中间的 `5` 是该病例实际参考图数量，最多为 6。

## 5. 损失与优化

模型输入是 SDXL inpainting UNet 标准 9 通道：

```text
noisy_target_latents + mask + masked_condition_latents
```

训练目标仍是标准扩散噪声回归：

```text
loss = MSE(model_pred, target_noise)
```

当前没有额外加入：

1. identity loss
2. perceptual loss
3. mask-only reconstruction loss
4. 多视角一致性 loss
5. ControlNet 训练
6. IP-Adapter 训练

当前只训练 UNet LoRA，冻结 VAE、Text Encoder、Text Encoder 2、IP-Adapter image encoder。

## 6. 精度与稳定性策略

默认配置：

```text
base_model_variant       = fp16
mixed_precision          = fp16
gradient_checkpointing   = true
max_grad_norm            = 1.0
learning_rate            = 5e-5
rank / alpha             = 16 / 16
```

注意：这里不是让 AdamW 直接更新 fp16 LoRA。当前代码会把 trainable LoRA 参数转为 fp32 做 optimizer update，最后保存权重时再转 fp16，兼顾稳定性和磁盘占用。

保护逻辑：

1. loss 非有限时立即停止。
2. 保存 checkpoint/final 前检查所有 LoRA tensor。
3. 如果出现 NaN/Inf，直接抛错，拒绝保存坏权重。

如果正式 800 step 仍出现数值问题，第一回退选项是：

```bash
MIXED_PRECISION=bf16 CLEAR_OUTPUT=1 tools/run_lora_pre2post_800step.sh
```

最终 LoRA 仍会以 fp16 保存。

## 7. 已完成验证

### 7.1 协议 smoke

已完成普通 LoRA 5-step smoke：

```text
Image_output/generation/lora_pre2post_protocol_smoke_5step/
```

结论：术前到术后 paired inpainting 协议能跑通，样本数为 898。

### 7.2 IP-Adapter 条件 smoke

已完成 512 和 1024 分辨率的 IP-Adapter 条件版 1-step smoke：

```text
Image_output/generation/lora_pre2post_ipcond_smoke_512_1step/
Image_output/generation/lora_pre2post_ipcond_smoke_1024_1step/
```

1024 smoke 关键信息：

```text
train_samples       = 898
completed_steps     = 1
dtype               = torch.float16
resolution          = 1024
ip_adapter_condition= true
mean_loss           = 0.004477500915527344
```

权重检查：

```text
tensors = 1152
NaN     = 0
Inf     = 0
```

这说明当前代码可以开始正式训练。

## 8. 下一轮正式训练计划

### 8.1 先跑 20-step 观察日志

目的：确认日志、loss、显存和保存保护都正常。

```bash
cd /root/autodl-tmp/myproject

CLEAR_OUTPUT=1 \
MAX_TRAIN_STEPS=20 \
SAVE_EVERY_STEPS=0 \
LOG_EVERY_STEPS=1 \
tools/run_lora_pre2post_800step.sh
```

验收：

1. 能看到 `[train_lora] step=... loss=...`。
2. loss 是有限数。
3. final LoRA 能保存。
4. 保存前没有 NaN/Inf 报错。

### 8.2 正式 800-step 训练

如果 20-step 正常，跑正式训练：

```bash
cd /root/autodl-tmp/myproject
CLEAR_OUTPUT=1 tools/run_lora_pre2post_800step.sh
```

默认关键参数：

```text
resolution                  = 1024
max_train_steps             = 800
train_batch_size            = 1
gradient_accumulation_steps = 4
learning_rate               = 5e-5
rank / alpha                = 16 / 16
mixed_precision             = fp16
gradient_checkpointing      = 1
save_every_steps            = 200
log_every_steps             = 10
ip_adapter_condition        = 1
ip_adapter_max_refs         = 6
```

输出重点看：

```text
Image_output/generation/lora_pre2post_800step/lora/final/
Image_output/generation/lora_pre2post_800step/summaries/train_lora_summary.json
Image_output/generation/lora_pre2post_800step/lora/train_samples.jsonl
```

## 9. 训练后验证计划

### 9.1 Feature_test 固定验证

先用 Feature_test 做可对比验证：

```bash
cd /root/autodl-tmp/myproject

CLEAR_OUTPUT=1 \
LORA_WEIGHTS_PATH=/root/autodl-tmp/myproject/Image_output/generation/lora_pre2post_800step/lora/final \
GEN_OUTPUT_ROOT=/root/autodl-tmp/myproject/Image_output/generation/lora_pre2post_800step_feature_test_ip \
MAX_SAMPLES=0 \
STRENGTH=0.60 \
GUIDANCE_SCALE=6.0 \
NUM_INFERENCE_STEPS=25 \
tools/run_infer_feature_test_ip_adapter.sh
```

然后评估：

```bash
GEN_INPUT_ROOT=/root/autodl-tmp/deformity_dataset/Feature_test/images \
GEN_SAM_ROOT=/root/autodl-tmp/deformity_dataset/Feature_test/masks \
GEN_INPAINT_MASK_ROOT=/root/autodl-tmp/deformity_dataset/Feature_test/masks/inpaint_mask \
GEN_FEATHER_MASK_ROOT=/root/autodl-tmp/deformity_dataset/Feature_test/masks/feather_mask \
GEN_OUTPUT_ROOT=/root/autodl-tmp/myproject/Image_output/generation/lora_pre2post_800step_feature_test_ip \
GEN_RUN_MODE=evaluate \
python generation_pipeline.py
```

优先看：

```text
infer/preview/
eval/triptychs/
eval/case_sheets/
summaries/infer_summary.json
summaries/evaluate_summary.json
```

### 9.2 strength 小网格

上一轮 `strength=0.88` 太激进。新权重先比较：

```text
strength = 0.45
strength = 0.60
strength = 0.75
```

判断标准：

1. mask 内是否仍出现贴片、黑边、局部换脸。
2. 鼻嘴变化是否方向正确。
3. 非编辑区是否保持稳定。
4. raw inpaint 是否已经正常，而不是靠 composite 掩盖问题。

### 9.3 Test 泛化验证

Feature_test 通过后，再用 Test 看无术后配对时的真实使用场景。Test 主要看术前重绘的自然度，不适合做严格术前/术后形态一致性判断。

## 10. 失败时的排查顺序

如果正式训练后效果仍差，按下面顺序排查，不要直接加 rank：

1. 先看 raw inpaint 是否坏；如果 raw 已坏，问题在生成条件或训练 recipe。
2. 比较 `strength=0.45/0.60/0.75`。
3. 缩小 mask，只鼻子或鼻子+上唇过渡先跑 smoke。
4. 检查 `train_samples.jsonl` 中 condition/target/mask/crop 是否符合预期。
5. 确认 IP-Adapter reference 数量和视角是否正常。
6. 若协议和推理都稳定但变化太弱，再比较 `rank=16/32/64`。
7. 若身份漂移仍明显，再考虑显式 identity/perceptual loss，而不是继续堆步数。

## 11. 800-step IP-Adapter 条件训练结果

本节记录 2026-07-05 的实际训练和中断前 Feature_test 推理观察。结论：训练数值正常，但视觉效果失败。

训练产物：

```text
Image_output/generation/lora_pre2post_800step/lora/final/
```

训练 summary：

```text
train_samples       = 898
sample_type         = pre_to_post_inpaint
completed_steps     = 800 / 800
resolution          = 1024
mixed_precision     = fp16
gradient_checkpointing = true
ip_adapter_condition   = true
mean_loss           = 0.18639623696180935
```

权重检查：

```text
pytorch_lora_weights.safetensors = 47,929,768 bytes
tensors = 1152
NaN     = 0
Inf     = 0
```

因此本轮失败不是“训练没有跑完”或“权重全 NaN”。数值链路是通的。

## 12. Feature_test 视觉观察

推理使用：

```text
LORA_WEIGHTS_PATH = Image_output/generation/lora_pre2post_800step/lora/final
GEN_OUTPUT_ROOT   = Image_output/generation/lora_pre2post_800step_feature_test_ip
STRENGTH          = 0.60
GUIDANCE_SCALE    = 6.0
NUM_INFERENCE_STEPS = 25
TORCH_DTYPE       = fp16
QUANTIZATION_BACKEND = none
IP-Adapter        = enabled
```

推理过程中先修正了两个工程问题：

1. 启用 IP-Adapter 时必须先加载 IP-Adapter，再加载 LoRA，否则训练出的 `encoder_hid_proj` 下 LoRA key 会被忽略。
2. 当前 diffusers 版本下，`enable_attention_slicing()` 会把 IP-Adapter attention processor 换成不支持 tuple encoder states 的 sliced processor；IP-Adapter 推理时需要关闭 attention slicing。

用户观察到效果太差后中断推理，因此没有完整 `infer_summary.json` 和 evaluate 输出。中断前已落盘：

```text
raw        = 25
composited = 25
preview    = 25
```

关键 preview：

```text
Image_output/generation/lora_pre2post_800step_feature_test_ip/infer/preview/150/术前/1.png
Image_output/generation/lora_pre2post_800step_feature_test_ip/infer/preview/50/术前/1.png
Image_output/generation/lora_pre2post_800step_feature_test_ip/infer/preview/185/术前/1.png
Image_output/generation/lora_pre2post_800step_feature_test_ip/infer/preview/50/术前/4.png
```

肉眼观察：

1. raw inpaint 不是鼻嘴局部重绘，而是一整块低质量人脸贴片。
2. 贴片内常出现完整脸部结构，包括眼睛区域、额头、脸颊、头发或背景噪声。
3. composite 只是把 raw 的坏块按 mask 贴回术前图，所以最终结果呈现强烈矩形边界和局部换脸感。
4. 正脸和侧脸都失败，说明不是单个样本异常，而是系统性失败。
5. IP-Adapter 多视角参考没有稳定身份，反而像把参考脸整体语义压进 inpaint 区域。
6. `strength=0.60` 已经比上一轮 `0.88` 保守，但仍然失控；这说明单靠调低 strength 不足以修复。

本轮视觉失败形态与上一轮 800-step 不同但同样不可用：

1. 上一轮主要是硬贴片、黑边和局部换脸。
2. 本轮加入 IP-Adapter 后，raw 更明显地生成了整张脸式的局部 patch。
3. 多视角条件没有解决“个体化术后鼻嘴映射”，反而强化了错误的全脸生成倾向。

## 13. 阶段判断

当前路线不建议继续直接扩步数、加 rank 或跑 strength 网格。

理由：

1. `rank=16` 可能不够，但当前失败不是变化弱，而是 raw 生成对象错误。
2. `strength=0.60` 下已经出现整脸贴片，再试 `0.45/0.75` 只能验证程度，不能改变机制。
3. IP-Adapter reference 作为全脸语义条件太强，当前没有约束它只提供身份上下文。
4. 训练目标只有扩散噪声 MSE，没有显式惩罚 mask 内生成整脸贴片或边界不连续。
5. 现有 mask 是鼻子+嘴巴联合区域，区域对 SDXL 来说仍可能太大，容易触发“重画脸部局部块”。

本阶段结论：

```text
训练流程：通过
数值稳定：通过
IP-Adapter 技术接入：通过
视觉效果：失败
是否可展示：不可展示
是否继续当前权重：不建议
是否直接加 rank/步数：不建议
```

## 14. 下一步建议

下一阶段应先做方法重构，而不是继续这条 recipe。

优先级：

1. 先做术后自重建 sanity check：`术后 masked -> 术后`，确认 SDXL inpainting + LoRA 能在本数据 mask 上做干净局部补全。
2. 缩小编辑区域：先只重绘鼻子，或鼻子+上唇过渡，不要直接鼻嘴大块联合重绘。
3. 暂停 IP-Adapter 进入 UNet cross-attention 的强条件方式，改成更弱的 identity/context 约束，或只在推理侧做低 scale 参考。
4. 增加 mask 内/边界质量检查，而不是只看全脸 identity similarity。
5. 如果继续利用术前多视角，应考虑单独提取身份 embedding 或结构 embedding，而不是直接把多张 face crop 作为生成语义参考。
6. 等 raw inpaint 能稳定只改鼻嘴后，再比较 rank、步数和 IP-Adapter scale。
