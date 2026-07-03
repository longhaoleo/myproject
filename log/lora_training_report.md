# LoRA 训练小报告：当前思路、输入输出与阶段性结论

日期：2026-07-03

## 1. 当前目标

当前 `train_lora` 先改成一个简单 baseline：只用术后图训练 SDXL inpainting LoRA，让模型在术后 face crop 里补回被 mask 掉的鼻嘴区域。

这一步的目的不是直接完成“术前预测术后”，而是先验证：

1. detection 产出的 `face_mask` crop 是否稳定。
2. 连续鼻嘴 `inpaint_mask` 是否和 face crop 对齐。
3. SDXL inpainting LoRA 是否能学到术后鼻嘴区域的局部补全能力。
4. 训练链路、显存和权重保存是否能跑通。

## 2. 当前训练主线

训练逻辑可以概括为：

```text
术后图
-> face_mask 正方形 crop
-> 用术后 inpaint_mask 挖掉鼻嘴
-> 让 SDXL inpainting LoRA 补回同一张术后图的鼻嘴区域
```

对应样本类型统一为：

```text
post_face_inpaint
```

术前图、术前到术后配对监督、多视角参考和缺视角利用先不进入当前训练。

## 3. 训练输入

当前依赖的输入包括：

1. `术后` 图。
2. detection 侧导出的 `face_mask`。
3. detection 侧导出的连续鼻嘴 `inpaint_mask`。
4. 可选的 `eye_mask` 后图；如果存在，manifest 会优先使用它作为 image path。

训练不重新推断编辑区域，只消费 detection 已落盘的产物。

## 4. 样本构造

每个有效术后样本构造成：

```text
condition_image_path = image_path
target_image_path    = image_path
inpaint_mask_path    = 当前术后图自己的 inpaint_mask
crop_box             = face_mask 的 bbox
weight               = 1.0
```

condition、target 和 mask 使用同一个 `crop_box`。crop 后直接 resize 到训练分辨率，默认 `1024 x 1024`。

不再使用旧方案中的：

1. `paired_head_reference`
2. `paired_edit_crop`
3. `self_identity_head`

## 5. 训练过程

训练本体仍然是标准扩散噪声回归：

1. target face crop 进入 VAE，得到目标 latent。
2. condition face crop 的鼻嘴区域按 mask 挖空，再进入 VAE。
3. mask 缩放到 latent 尺寸。
4. inpainting UNet 输入 `noisy_latents + mask + masked_image_latents`。
5. UNet LoRA 学习预测噪声。

当前只训练 UNet LoRA，不训练 VAE、文本编码器或 ControlNet。

## 6. 当前局限

这版是最小训练闭环：

1. 它学习的是术后鼻嘴自重建，不是术前到术后的形态映射。
2. 还没有使用术前图作为 condition 或 reference。
3. 还没有多视角一致性损失。
4. 还没有缺视角样本利用。
5. 还没有显式 identity consistency loss。

## 7. 后续方向

baseline 稳定后，再逐步恢复更完整的训练目标：

1. `术前 face crop -> 术后 face crop` 的 paired 监督。
2. 术前图作为额外 reference 条件。
3. 同病例多视角一致性。
4. paired / unpaired 样本分层权重。
5. identity 或 perceptual consistency loss。
