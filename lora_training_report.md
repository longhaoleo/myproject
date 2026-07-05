# 当前 LoRA 训练方法说明

日期：2026-07-04

## 1. 方法概述

当前训练方案已经切到配对监督：

```text
术前 face crop -> 同病例同视角术后 face crop
```

模型仍然是 `SDXL Inpainting UNet LoRA`。generation 侧不重新推断编辑区域，而是消费 detection / split 后已经落盘的 `face_mask`、`inpaint_mask` 和 `feather_mask`。

当前主线是：

1. 使用 SDXL inpainting 作为基础生成框架。
2. 默认加载 fp16 SDXL inpainting 本地权重。
3. 只训练 UNet LoRA，冻结 VAE 与文本编码器。
4. 条件图使用术前图，目标图使用同视角术后图。
5. 重绘区域使用连续鼻子 + 嘴巴 `inpaint_mask`。
6. 验证优先使用有真实配对的 `Feature_test`。

## 2. 数据与训练目标

当前数据根目录：

```text
/root/autodl-tmp/deformity_dataset
```

训练集可构造：

```text
898 个 pre_to_post_inpaint 样本
```

每个训练样本统一为：

```text
condition_image = 同病例同视角术前 face crop
target_image    = 术后 face crop
mask            = 术后行对应的 nose + mouth inpaint_mask
sample_type     = pre_to_post_inpaint
```

训练目标是：给定术前 face crop 和鼻嘴编辑区域，让 LoRA 学习生成更接近术后外观的鼻嘴区域。当前没有额外 identity loss、多视角一致性 loss 或 ControlNet 训练。

## 3. 样本组织逻辑

当前只有一种训练样本：

### `pre_to_post_inpaint`

触发条件：

1. manifest 行属于 `train` split。
2. `stage == "术后"`。
3. 当前术后行存在 `paired_pre_path`。
4. 当前术后行存在 `face_mask_path` 与 `inpaint_mask_path`。
5. `face_mask` 能提取出有效 bbox。

样本内容：

1. `condition_image_path = paired_pre_path`
2. `target_image_path = 当前术后图路径`
3. `inpaint_mask_path = 当前术后图自己的 inpaint_mask`
4. `crop_box = 当前术后图 face_mask` 的 bbox
5. `weight = 1.0`

旧方案里的 `post_face_inpaint`、`paired_head_reference`、`paired_edit_crop` 和 `self_identity_head` 当前都不再生成。

## 4. 图像预处理

训练样本进入 VAE 前按同一套规则处理：

1. condition 图、target 图和 mask 先使用同一个 `face_mask` bbox crop。
2. crop 后直接 resize 到 `LoRATrainConfig.resolution`，正式训练建议 `1024 x 1024`。
3. 图片 resize 使用 `BICUBIC`。
4. mask resize 使用 `NEAREST` 并二值化。
5. 当前实现没有在像素侧把 condition 的鼻嘴区域置零，而是把完整 condition crop 作为 inpainting 条件 latent。

## 5. 模型结构

当前训练框架基于本地 SDXL inpainting 权重：

```text
/root/autodl-tmp/myproject/model/generation/stable-diffusion-xl-1.0-inpainting-0.1
```

该路径指向 fp16 权重目录：

```text
/root/autodl-tmp/models/sdxl_inpaint_fp16
```

组成仍是标准 SDXL inpainting：

1. VAE：把图像编码到 latent 空间。
2. Text Encoder / Text Encoder 2：编码 prompt。
3. UNet：预测扩散噪声，也是当前插入 LoRA 的模块。
4. Scheduler：定义加噪和去噪时间步。

当前只训练 UNet 注意力相关投影层上的 LoRA，其他模块冻结。

## 6. 前向过程

对于每个 `pre_to_post_inpaint` 样本：

1. 读取术前图作为 condition。
2. 读取同视角术后图作为 target。
3. 使用术后侧 face crop 裁出训练区域。
4. VAE 编码 target crop，得到 `latents`。
5. VAE 编码 condition crop，得到 `masked_image_latents`。
6. mask 缩放到 latent 尺寸。
7. UNet 输入为：

```text
noisy_latents + mask + masked_image_latents
```

对应 SDXL inpainting UNet 的 9 通道输入。

## 7. 损失函数

当前仍使用标准 diffusion noise regression loss：

```text
L = MSE(predicted_noise, target_noise)
```

mask 是 inpainting 条件通道，不是额外 masked-only loss。当前所有训练样本权重为 `1.0`。

## 8. 验证策略

当前建议固定两套验证：

1. `Test`
   - 主要看术前泛化重绘
   - 当前没有同病例同视角术后配对，不能做真实三列对比
2. `Feature_test`
   - 有 20 个真实配对视角
   - 用于看 `术前 / 术后 / 重绘` triptych 和 case sheet

20 step smoke LoRA 已验证链路能跑通且输出不是全黑，但不代表最终训练效果。

## 9. 下一步方向

建议下一步先训一版 800 step：

```text
resolution                  = 1024
max_train_steps             = 800
train_batch_size            = 1
gradient_accumulation_steps = 4
learning_rate               = 1e-4
rank                        = 16
alpha                       = 16
mixed_precision             = fp16
save_every_steps            = 200
```

如果 `Feature_test` 上鼻嘴变化不足，再加到 1500-2000 steps；如果非编辑区漂移明显，优先调推理 `strength`、mask 和回贴逻辑。
