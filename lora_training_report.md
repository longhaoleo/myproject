# 当前 LoRA 训练方法说明

## 1. 方法概述

当前训练方案先收敛为一个简单 baseline：只用术后图训练 `SDXL Inpainting UNet LoRA`，验证 detection 产物、face crop、鼻嘴 `inpaint_mask` 和 LoRA 训练链路是否稳定。

当前主线是：

1. 使用 `SDXL inpainting` 作为基础生成框架。
2. 仅训练 `UNet LoRA`，冻结 VAE 与文本编码器。
3. 只取 `术后` 样本进入训练。
4. 使用 detection 侧产出的正方形 `face_mask` 作为训练 crop。
5. 使用同一张术后图的连续鼻嘴 `inpaint_mask` 作为编辑区域。
6. 术前图、术前到术后配对监督、多视角参考和缺视角利用先作为后续工作。

## 2. 数据与训练目标

当前训练直接使用 detection 侧产物，不在 generation 侧重新推断编辑区域。

每个训练样本统一为：

```text
condition_image = 术后图的 face square crop
target_image    = 同一张术后图的 face square crop
mask            = 同一张术后图的 nose + mouth inpaint_mask
sample_type     = post_face_inpaint
```

训练目标是：在完整术后 face crop 的上下文里，把鼻嘴区域挖空，让模型学习补回术后鼻嘴区域。

这一步暂时不是最终的“术前预测术后”任务，而是一个更小、更稳的自监督 baseline，用来确认：

1. `face_mask` crop 是否稳定。
2. `inpaint_mask` 和 face crop 是否对齐。
3. LoRA 是否能学到术后鼻嘴区域的局部补全能力。
4. 训练流程和显存路径是否可用。

## 3. 样本组织逻辑

当前只有一种训练样本：

### `post_face_inpaint`

触发条件：

1. manifest 行属于 `train` split。
2. `stage == "术后"`。
3. `image_path / face_mask_path / inpaint_mask_path` 都存在。
4. `face_mask` 能提取出有效 bbox。

样本内容：

1. `condition_image_path = image_path`
2. `target_image_path = image_path`
3. `inpaint_mask_path = 当前术后图自己的 inpaint_mask`
4. `crop_box = face_mask` 的 bbox
5. `weight = 1.0`

旧方案里的 `paired_head_reference`、`paired_edit_crop` 和 `self_identity_head` 暂时停用。

## 4. 图像预处理

训练样本进入 VAE 前按同一套规则处理：

1. condition 图、target 图和 mask 先使用同一个 `face_mask` bbox crop。
2. crop 后直接 resize 到 `LoRATrainConfig.resolution`，默认 `1024 x 1024`。
3. 图片 resize 使用 `BICUBIC`。
4. mask resize 使用 `NEAREST`。
5. 不再使用 `ImageOps.pad` 做保持比例补边。

这样训练坐标系完全由 detection 输出的正方形 face crop 决定，`inpaint_mask` 只负责指定鼻嘴重绘区域。

## 5. 模型结构

当前训练框架基于 `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`。

组成仍是标准 SDXL inpainting：

1. VAE：把图像编码到 latent 空间。
2. Text Encoder / Text Encoder 2：编码 prompt。
3. UNet：预测扩散噪声，也是当前插入 LoRA 的模块。
4. Scheduler：定义加噪和去噪时间步。

当前只训练 UNet 注意力相关投影层上的 LoRA，其他模块冻结。

## 6. 前向过程

对于每个 `post_face_inpaint` 样本：

1. 读取同一张术后图作为 condition 和 target。
2. 使用 face crop 裁出训练区域。
3. 使用 `inpaint_mask` 在 condition 中挖掉鼻嘴区域。
4. VAE 编码 target crop，得到 `latents`。
5. VAE 编码 masked condition crop，得到 `masked_image_latents`。
6. mask 缩放到 latent 尺寸。
7. UNet 输入为：

```text
noisy_latents + mask + masked_image_latents
```

对应 SDXL inpainting UNet 的 9 通道输入。

## 7. 损失函数

当前仍使用标准 diffusion noise regression loss：

```text
L = || epsilon - epsilon_theta(z_t, t, c, m, z_cond) ||^2
```

当前所有训练样本权重为 `1.0`，不再区分 paired、局部强化或自重建补偿权重。

## 8. 当前限制

这版训练是最小闭环 baseline，限制很明确：

1. 它学习的是“术后图鼻嘴区域重建”，还不是“术前到术后预测”。
2. 术前图暂时不作为 condition。
3. 多视角一致性暂时不进入训练目标。
4. 缺失视角暂时不做低权重自重建利用。
5. 还没有显式 identity consistency loss。

## 9. 后续方向

当前 baseline 跑稳后，再逐步加回：

1. 术前 face crop 作为 condition，术后 face crop 作为 target。
2. 术前图作为额外 reference 条件。
3. 同病例多视角一致性约束。
4. paired / unpaired 样本重新分层加权。
5. 更强的 identity 或 perceptual consistency loss。
