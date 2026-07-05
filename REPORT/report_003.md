# 报告003：800-step SDXL Inpainting LoRA 失败复盘

日期：2026-07-05

## 1. 当前状态

这一阶段是 generation 主链的第一次 800-step 正式小跑。它完成了训练和 `Feature_test` 推理，但视觉效果失败，暂不应进入主线展示或继续基于该权重做下游评估。

用户已删除本轮正式 LoRA 权重目录，因此本报告主要基于：

1. 用户终端中的训练 summary。
2. 已落盘的 `Feature_test` 推理与评估输出。
3. 当前 `generation/train.py`、`generation/infer.py` 的实际逻辑。
4. 人工查看 `case_sheets/50.png` 和 `preview/50/术前/1.png` 的视觉结果。

本轮不应被解释为“模型已经学会一点但还不够好”。更准确地说，它暴露了当前训练/推理 recipe 的结构性问题。

## 2. 本轮配置与产物

训练配置来自用户终端 summary：

```text
train_samples              = 898
sample_type                = pre_to_post_inpaint
max_train_steps            = 800
resolution                 = 1024
train_batch_size           = 1
gradient_accumulation_steps = 4
learning_rate              = 5e-5
rank / alpha               = 16 / 16
mixed_precision            = bf16
base_model_variant         = fp16
mean_loss                  = 0.16010754398885182
```

`Feature_test` 推理输出目录：

```text
Image_output/generation/lora_pre2post_800step_feature_test/
```

推理 summary 显示：

```text
requested_samples = 27
success_count     = 27
failed_count      = 0
```

但评估和实际落盘文件显示：

```text
实际 composited 输出 = 20
triptych_count      = 18
case_sheets         = 4
```

因此本轮还有一个工程问题：旧版 `infer.py` 没有检查 `cv2.imwrite()` 返回值，summary 可能把未成功落盘的样本计入 success。这个问题已经在后续代码中加了显式写图检查。

## 3. 观察到的效果

人工查看：

```text
Image_output/generation/lora_pre2post_800step_feature_test/eval/case_sheets/50.png
Image_output/generation/lora_pre2post_800step_feature_test/eval/case_sheets/97.png
Image_output/generation/lora_pre2post_800step_feature_test/infer/preview/50/术前/1.png
```

主要现象：

1. 右侧重绘区域出现明显“贴片感”。
2. 鼻嘴区域不是自然融合，而像从另一张脸或另一个局部图块硬贴上去。
3. 多个视角出现黑边、矩形边界和局部背景纹理。
4. 预览中的 `模型直接生成` 已经包含另一张脸的大块局部特征，说明问题不是单纯回贴羽化造成的。
5. `术前重绘结果` 只是把这个坏局部再按 mask 回贴，最终视觉非常不自然。

这不是“轻微瑕疵”，而是当前方法在这个设置下没有正确学到受控鼻嘴重绘。

2026-07-05 复查补充：

```text
raw/composited/preview 实际各落盘 20 张
infer_summary requested/success/failed = 27 / 27 / 0
evaluate_summary inference_ok/predicted_view_count = 20 / 20
```

`case_sheets/50.png` 和 `case_sheets/97.png` 的失败形态一致：正脸、侧脸、仰视角都出现局部换脸感、黑色背景块、硬边界和不属于当前照片的纹理。`preview/50/术前/1.png` 里第四列 `模型直接生成` 已经生成了带头发、眼睛和整块脸的局部图，而不是只补鼻嘴；所以最终 composite 不是根因，只是把失控的 raw inpaint 区域贴回原图。

## 4. 为什么不是单纯 rank 太小

`rank=16` 的确可能偏小。它可能带来：

1. 医生风格容量不足。
2. 术后形态变化学习不充分。
3. 鼻尖、鼻梁、鼻翼、上唇等细节表达弱。

但本轮失败不像典型低 rank 失败。低 rank 常见表现是“变化不明显”“风格学不进去”“趋近底模原始输出”。本轮表现是：

1. 大块贴片。
2. 另一张脸的局部纹理。
3. 黑边和矩形边界。
4. raw inpaint 结果本身已经不可信。

这更像训练/推理协议错配、条件输入语义错误和 mask 设计不稳定，而不只是 LoRA 容量不足。直接把 rank 提到 32 或 64 可能只会让模型更强地记住错误映射，甚至放大贴片和身份漂移。

## 5. 更核心的失败原因

### 5.1 Inpainting 条件图语义错配

当前训练数据集里：

```python
masked_tensor = condition_tensor
```

也就是说，训练时送进 UNet 的 `masked_image_latents` 实际是完整术前 face crop，没有在像素侧把鼻嘴区域置空。

但 SDXL inpainting pipeline 推理时通常会根据 mask 构造 masked image latent，也就是编辑区域会被遮掉或置空。这样就产生了严重错配：

```text
训练：UNet 在 mask 区域还能看到术前鼻嘴
推理：UNet 在 mask 区域主要看到空洞/被遮区域
```

模型训练时学到的条件分布和推理时实际输入分布不一致，很容易导致 inpaint 区域失控。

### 5.2 训练 crop/mask 与推理 crop/mask 不是同一坐标系

当前训练样本以术后行作为锚点：

```text
condition_image_path = paired_pre_path
target_image_path    = 当前术后图
crop_box             = 当前术后 face_mask bbox
inpaint_mask_path    = 当前术后 inpaint_mask
```

推理时只能使用术前图的 face crop 和术前 mask。

这意味着训练和推理坐标系不同：

```text
训练：术后侧 face crop / 术后侧 inpaint_mask
推理：术前侧 face crop / 术前侧 inpaint_mask
```

如果术前术后拍摄姿态、头部位置、面部比例或 mask box 有差异，模型看到的空间关系就会漂。对于鼻嘴这种强几何区域，这种错配会比普通风格迁移更致命。

### 5.3 当前 loss 不是 mask-only，容易让 LoRA 学到整脸/局部贴图倾向

当前训练目标是标准 diffusion noise regression：

```text
L = MSE(predicted_noise, target_noise)
```

虽然 UNet 输入里有 mask 通道，但 loss 没有显式只约束编辑区域，也没有对非编辑区域加身份保持损失。对于 `pre -> post` 这种配对监督，小数据下 LoRA 可能学习到“术后目标局部纹理”而不是“在当前人脸上做局部形态变化”。

这解释了为什么 raw 输出像另一张脸局部，而不是当前脸自然变化。

### 5.4 mask 区域过大，任务从鼻子重绘变成局部人脸重生成

当前编辑区域是 nose + mouth 联合 mask。以 `case50 view1` 为例：

```text
inpaint bbox roughly covers lower mid-face
mask area about 244k pixels
mask / face bbox area about 13.2%
```

另一个失败样本 `case97 view1` 也类似：

```text
mask area about 222k pixels
mask / face bbox area about 12.2%
```

这个区域同时包含鼻、上唇、嘴唇、部分面中皮肤。对 SDXL inpainting 来说，这已经不是“小修鼻子”，而是重生成一整块面中区域。没有 identity reference、没有 depth、没有 landmark 约束时，模型容易补出一块语义上像脸但身份和光照不对的局部。

### 5.5 推理 strength 偏高

本轮推理使用：

```text
strength = 0.88
num_inference_steps = 20
guidance_scale = 6.0
```

`strength=0.88` 对鼻嘴局部医学照片来说非常激进。它让模型强烈重采样 mask 区域，原始图像约束弱，出现“新贴片”的概率会明显上升。

如果当前训练 recipe 还不稳定，高 strength 会把失败放大到肉眼非常明显。

### 5.6 评估指标没有捕捉失败点

评估 summary 里：

```text
hard_identity_similarity = 0.9999626
soft_face_similarity     = 0.9987650
```

这两个指标主要看非编辑区或整脸轻量特征，不能说明鼻嘴重绘质量好。当前失败恰好集中在 mask 区域，所以这些指标会给出虚假的稳定感。

后续必须增加 mask 区域质量检查，例如：

1. mask 区域颜色/纹理突变检测。
2. mask 边界梯度异常。
3. 生成区域与周边肤色一致性。
4. 人工固定样本预览优先级高于这类轻量 embedding 分数。

## 6. 本轮仍然有价值的部分

虽然视觉失败，本轮仍然验证了几件事：

1. bf16 训练稳定，不再出现上一轮 fp16 NaN。
2. 800-step 训练可以完整跑完。
3. LoRA 权重曾经检查过 `NaN=0 / Inf=0`。
4. `Feature_test` 的配对验证流程能暴露真实问题。
5. case sheet 和 preview 对诊断非常有用。
6. 推理 summary 计数问题已经暴露，需要保留写图检查。

所以这不是无效尝试，而是一个明确的失败分支。

## 7. 当前判断

本阶段应该暂停继续扩大训练。不要直接做：

1. 继续加步数到 1500 或 2000。
2. 只把 rank 从 16 提到 32/64。
3. 继续用 `strength=0.88` 跑更多测试集。
4. 直接拿当前输出写效果结论。

原因是当前失败形态是协议和数据语义问题。容量和步数不是第一优先级。

## 8. 建议的下一轮最小修复实验

### 8.1 先修 inpainting 条件输入

训练时应与推理保持一致，把 condition 图在 mask 区域置空：

```python
masked_tensor = condition_tensor * (mask_tensor < 0.5)
```

然后先跑很小步数验证 raw output 是否仍然贴片。

### 8.2 统一训练和推理坐标系

优先尝试以术前为锚点：

```text
condition = 术前图
target    = 同视角术后图
crop_box  = 术前 face bbox
mask      = 术前 inpaint_mask
```

这样至少训练和推理用同一侧 crop/mask 规则。术后 target 也按术前 crop box 裁剪，牺牲一点配准精度，但换来协议一致。

### 8.3 降低推理 strength

先固定同一权重或下一版权重，比较：

```text
strength = 0.45
strength = 0.60
strength = 0.75
```

在 `Feature_test` 的固定 4 到 6 张样本上看 raw/composite。不要一开始全量跑 27 张。

### 8.4 缩小或分层 mask

当前 nose + mouth 一起重绘太激进。下一轮可以比较：

1. 只鼻子。
2. 鼻子 + 上唇过渡小区域。
3. 当前 nose + mouth 大 mask。

如果只鼻子明显稳定，说明嘴部联合 mask 是主要放大器。

### 8.5 rank 放到第二阶段再调

在协议修好前，不建议先调 rank。协议修好后可以比较：

```text
rank=16
rank=32
rank=64
```

预期：

1. rank=16 可能变化弱。
2. rank=32 可能是第一档合理配置。
3. rank=64 可能更容易记忆训练集，需要更严格验证。

## 9. 建议的重启顺序

下一轮建议按这个顺序执行：

1. 改回训练 masked condition，与 SDXL inpaint 推理一致。
2. 训练 50 到 100 step smoke，不追求效果，只看是否还贴片。
3. 用 `Feature_test` 只跑 4 到 6 张固定样本。
4. 比较 `strength=0.45/0.60/0.75`。
5. 若贴片消失但变化弱，再考虑 rank=32 和更长训练。
6. 若仍贴片，优先检查 crop/mask 坐标系，而不是加容量。
7. 等小样本视觉稳定后，再跑全量 `Feature_test`。

## 10. 结论

本阶段结论很明确：

当前 800-step LoRA 不是“效果差一点”，而是训练/推理 recipe 尚未成立。rank 偏小可能存在，但不是第一解释。更关键的问题是：

1. 训练时 condition 没有按 mask 挖空，和推理输入分布不一致。
2. 训练使用术后侧 crop/mask，推理使用术前侧 crop/mask，坐标系不一致。
3. mask 区域过大，任务难度接近局部人脸重生成。
4. strength 偏高，把不稳定生成放大成明显贴片。
5. 当前轻量评估指标不能反映 mask 区域灾难。

这一路线需要先修协议，再谈 rank、步数和模型容量。
