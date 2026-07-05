# Generation 报告：对接、训练逻辑与阶段性思路说明

日期：2026-04-05
最新修订：2026-07-04

## 1. 核心结论

当前仓库已经从“仅前处理”推进到“Detection 条件构建 + 生成侧消费”的最小闭环。

这次新增的不是最终科研版本，而是一套工程骨架。当前代码已经把 generation 主链切到“小样本、少量缺视角”的术前锚点重绘方案，目标是把以下链路完整接上：

1. 读取病例目录树
2. 读取 SAM part mask
3. 在 detection 侧规范化并落盘为生成器稳定可消费的 mask 条件
4. 训练医生风格 LoRA
5. 运行 `SDXL Inpainting + LoRA`
6. 输出推理结果、病例级一致性摘要和工程验收报告

## 2. 当前推荐配置

### 2.1 编辑区域归属

当前默认重绘区域固定为：

- `nose`
- `mouth`

并采用联合编辑，而不是鼻子单独编辑。该区域由 detection 侧定义和落盘，不由 generation 侧二次推断。

原因：

1. 鼻整形后视觉变化通常会牵连鼻小柱、上唇过渡和嘴周比例。
2. 如果只改鼻子，不改嘴部邻域，容易出现术后图风格不连贯。

### 2.2 mask 策略

当前默认优先使用“方框 mask”，而不是直接使用 SAM 原始轮廓。

原因：

1. 现阶段 SAM part mask 在鼻翼、鼻孔和嘴角附近仍可能抖动。
2. 对于 inpainting 来说，过于抖动的轮廓容易放大边缘伪影。
3. 先把轮廓规范化成 box mask，再在 detection 侧合成为连续鼻嘴 `inpaint_mask`，工程稳定性更高，更适合做 v1 基线。

### 2.3 当前 generation 输入约定

当前默认规则已经变成：

1. generation 默认直接消费 `eye_mask` 后的图片；
2. 若某张图没有打码版本，才回退到原图；
3. 训练和推理都优先使用术前侧的 `face_mask / inpaint_mask / feather_mask`；
4. 当前训练只使用同病例同视角的 `术前 -> 术后` 配对样本；缺少配对的视角暂不进入训练。

## 3. 当前主逻辑与代码落点

本轮主入口：

- [generation_pipeline.py](/Users/leo/myproject/generation_pipeline.py)

本轮生成侧模块：

- [generation/settings.py](/Users/leo/myproject/generation/settings.py)
- [generation/train.py](/Users/leo/myproject/generation/train.py)
- [generation/infer.py](/Users/leo/myproject/generation/infer.py)
- [generation/eval.py](/Users/leo/myproject/generation/eval.py)
- [generation/README.md](/Users/leo/myproject/generation/README.md)

本轮新增的根级公共工具包：

- [project_utils/dataset.py](/Users/leo/myproject/project_utils/dataset.py)
- [project_utils/io.py](/Users/leo/myproject/project_utils/io.py)

## 4. 输出逻辑

### 4.1 `train_lora`

输出：

1. `checkpoint-*`
2. `final/` LoRA 权重
3. `train_samples.jsonl`
4. `train_lora_summary.json`

训练数据现在只生成一种样本：

1. `pre_to_post_inpaint`

其中：

1. 条件图是同病例同视角术前图；
2. 目标图是同病例同视角术后图；
3. crop 和 `inpaint_mask` 当前取自术后行；
4. 训练顺序默认按病例分组，尽量让同一个人的多个视角在同一梯度累积周期里出现。

### 4.1.1 当前训练逻辑

当前训练逻辑主要落在：

- [generation/index.py](/Users/leo/myproject/generation/index.py)
- [generation/train.py](/Users/leo/myproject/generation/train.py)

一句话概括：

同病例同视角术前图作为条件图，术后图作为目标图；在 `face` crop 内用鼻嘴联合 `inpaint_mask` 指示重绘区域，训练 SDXL inpainting UNet LoRA 学习术前到术后的鼻嘴局部重绘。

按实际执行顺序，训练链路可以拆成 5 层。

#### A. 文件输入层

训练前必须先有 detection 侧产物。generation 实际会扫描这几类输入：

1. 原始病例目录
   - `input_root/<case_id>/术前/<view>.jpg`
   - `input_root/<case_id>/术后/<view>.jpg`
2. `eye_mask` 产出的打码图
   - `sanitized_root/<case_id>/<stage>/<view>.jpg`
3. `sam_mask` 产出的部位和编辑条件
   - `parts/face`
   - `parts/nose`
   - `parts/mouth`
   - `inpaint_mask`
   - `feather_mask`

这里真正参与训练的主输入优先级是：

1. 条件图优先用 `sanitized_root` 下的打码图
2. 若没有打码图，才回退到原图
3. `face_mask / inpaint_mask / feather_mask` 一律从 detection 落盘结果读取

也就是说，训练不是自己再去算编辑区域，而是完全消费 detection 的现成定义。

#### B. manifest 行层

`build_generation_manifest()` 会把所有图片先整理成逐行记录。  
每一行的核心字段是：

1. 样本主键
   - `case_id`
   - `stage`
   - `view_id`
2. 图像路径
   - `image_path`
   - `original_image_path`
   - `sanitized_image_path`
3. 条件路径
   - `face_mask_path`
   - `nose_mask_path`
   - `inpaint_mask_path`
   - `feather_mask_path`
4. 配对关系
   - `paired_pre_path`
   - `paired_post_path`
   - `is_paired_view`
5. 病例级视角信息
   - `available_pre_views`
   - `available_post_views`
   - `paired_views`
   - `paired_view_count`

这一层还没变成真正训练样本，它的作用只是先回答两个问题：

1. 这一行是不是有“术前-术后同视角配对”
2. 这个病例还有哪些别的视角可以一起利用

#### C. 训练样本构造层

真正的训练样本不是 manifest 行本身，而是 `_build_training_samples()` 从 manifest 里派生出来的 `_TrainSample`。

每个 `_TrainSample` 固定包含：

1. `condition_image_path`
   - 条件图路径
2. `target_image_path`
   - 目标图路径
3. `inpaint_mask_path`
   - 编辑 mask 路径
4. `crop_box`
   - 是否只裁局部区域
5. `sample_type`
   - 样本类型
6. `weight`
   - 该样本的训练权重

当前只会生成 1 类样本：

1. `pre_to_post_inpaint`
   - 触发条件：manifest 行属于 `train` split，且 `stage == 术后`
   - 额外要求：该术后行存在 `paired_pre_path / face_mask_path / inpaint_mask_path`
   - 输入：
     - `condition_image_path = paired_pre_path`
     - `target_image_path = 当前术后图路径`
     - `crop_box = 当前术后图 face_mask` 的包围框
     - `inpaint_mask_path = 当前术后图 inpaint_mask`
   - 作用：
     - 学习同病例同视角的术前到术后鼻嘴区域重绘

换句话说，当前训练不再把一个视角拆成整头部样本和鼻嘴局部样本，也不再用 unpaired 视角做低权重自重建。

#### D. Dataset 张量层

`_SdxlLoRADataset.__getitem__()` 会把 `_TrainSample` 变成真正送进扩散模型的张量。

它做的事情按顺序是：

1. 读取条件图 `condition_image_path`
2. 读取目标图 `target_image_path`
3. 读取二值 mask `inpaint_mask_path`
4. 如果有 `crop_box`，先把条件图、目标图、mask 一起裁成同一块区域
5. 再统一 resize 到 `resolution x resolution`

最后输出 3 个核心张量：

1. `pixel_values`
   - 来自 `target_image`
   - 这是模型要拟合的目标图像
2. `mask_values`
   - 来自 `inpaint_mask`
   - 表示哪些区域允许被重绘
3. `masked_pixel_values`
   - 当前实现来自完整 `condition_image`
   - 也就是术前 face crop 本身；mask 通过额外通道告诉 UNet 哪个区域需要编辑

所以从 I/O 上看，现在训练样本的最核心关系是：

1. 条件图：术前
2. 目标图：术后
3. 编辑 mask：当前术后行对应的鼻嘴联合区域

#### E. 扩散训练层

`train_lora()` 里真正的优化逻辑是标准 SDXL inpainting LoRA 训练，只训练 UNet LoRA，不训练 text encoder，不训练 ControlNet。

它的实际输入到 UNet 的内容是：

1. `latents`
   - 由 `pixel_values` 经过 VAE 编码得到
   - 也就是目标图 latent
2. `masked_image_latents`
   - 由 `masked_pixel_values` 经过 VAE 编码得到
   - 也就是术前条件图 latent
3. `mask`
   - 由 `mask_values` 缩放到 latent 尺寸得到
4. `prompt_embeds`
   - 来自文本 prompt
   - prompt 由 `doctor_token + view_token + caption_suffix` 拼接

最终 UNet 接收的是一个 9 通道的 inpainting 输入：

1. `noisy_latents`
2. `mask`
3. `masked_image_latents`

训练目标仍然是扩散标准噪声回归：

1. 给目标 latent 加噪声
2. UNet 预测噪声
3. 用 MSE 损失回归真实噪声
4. 再乘上每个样本自己的 `weight`

所以这版训练虽然在“样本构造”上已经是术前到术后的配对监督，但在“损失形式”上仍然是标准 diffusion noise loss，没有额外 identity loss，也没有额外 masked-only loss。

#### F. 当前权重与顺序设计

当前默认权重思路很简单：

1. 所有 `pre_to_post_inpaint` 样本权重为 `1.0`
2. 暂不区分整头部样本、鼻嘴局部强化样本和 unpaired 自重建样本
3. 后续如果需要强化鼻嘴区域，可以再考虑 masked loss 或局部样本重加权

同时，`_order_train_samples_for_cases()` 还会做病例级重排：

1. 先按 `case_id` 聚合样本
2. 再优先让同一个病例的不同视角连续出现
3. 每个病例内部按 view id 排序取样

这样做的目的不是改变 loss，而是改变一个梯度累积周期里“模型看到数据的顺序”，尽量让它连续看到同一个人的多个视角。

#### G. 当前训练逻辑的本质

把上面全部压缩成一句更准确的话：

当前训练逻辑是一个“基于 detection 条件、以术前为条件图、以同视角术后为目标图、用鼻嘴联合 mask 指示编辑区域”的 SDXL inpainting UNet LoRA 训练方案。

当前一致性方案也一并说明如下：

1. 训练阶段不额外引入显式 identity loss
2. 一致性主要靠：
   - 术前/术后同视角配对监督
   - 病例级 grouped sampling
   - face crop 内的上下文保持
3. 评估阶段再输出：
   - `hard_identity_similarity`
   - `soft_face_similarity`
4. 这是一版轻量实现，后续仍可以升级成显式 perceptual / identity consistency loss

### 4.2 `infer`

输出：

1. `raw/` 映射回整图后的生成结果
2. `composited/` 仅回贴编辑区域后的最终图
3. `preview/` 五联调试图
4. `infer_issues.jsonl`
5. `infer_summary.json`

当前推理不再默认直接整图跑，而是：

1. 先按 `face_mask` 做 face crop
2. 在 crop 内运行 inpainting
3. 再映射回整图
4. 最后按 `feather_mask` 回贴

另外，同一病例的不同视角会使用稳定的病例级随机种子，减少结果风格漂移。

### 4.3 `evaluate`

输出：

1. `triptychs/`
2. `case_sheets/`
3. `evaluate_summary.json`

三联图格式仍固定为：

- 术前
- 术后
- 生成结果

此外现在还会输出：

1. `paired_view_count`
2. `unpaired_pre_view_count`
3. `unpaired_post_view_count`
4. `predicted_view_count`
5. `complete_six_view_cases`
6. `partial_view_cases`
7. `hard_identity_similarity`
8. `soft_face_similarity`

## 5. 当前尝试与阶段性效果

已在本地完成、可以确认有效的尝试包括：

1. detection 与 generation 代码可编译
2. generation 侧运行时 manifest 可正确附带 paired/unpaired 视角信息
3. `evaluate` 可统计结果并导出三联图与病例级 contact sheet
4. 本地 fp16 SDXL inpainting 权重可加载，UNet 确认为 9 通道 inpainting 结构
5. 已完成 20 step LoRA 烟测
6. 已完成 `Test` 与 `Feature_test` 推理验证

当前阶段已经得到的效果主要是工程层面的：

1. detection 与 generation 的条件接口已经打通
2. split 后的 `Train / Test / Feature_test` 数据集可以直接被 generation 消费
3. 当前训练可从 `Train` 构造 898 个 `pre_to_post_inpaint` 样本
4. 训练、推理、评估三条链路已经跑通
5. `Test` 术前 29 张推理成功，但没有真实术前/术后配对
6. `Feature_test` 术前 27 张推理成功，并生成 20 个真实配对 triptych

当前仍需要继续验证的部分：

1. 800 step 或更长训练后的有效性
2. 鼻嘴区域变化是否足够接近术后
3. 非编辑区身份和皮肤纹理是否稳定
4. 当前使用术后侧 mask/crop 训练、术前侧 mask/crop 推理之间是否需要进一步统一

## 6. 当前风险

1. 方框 mask 稳定，但会牺牲边缘精细度。
2. 目前 LoRA 训练实现优先追求小样本闭环，不是最终训练 recipe。
3. 当前只使用 paired 视角训练，unpaired 视角暂未作为低权重身份样本利用。
4. 当前 `hard_identity_similarity / soft_face_similarity` 是工程指标，不是最终科研版身份度量。

## 7. 后续可能思路

1. 先按 [generation/README.md](/Users/leo/myproject/generation/README.md) 安装依赖并下载权重。
2. 先跑 `eye_mask` 和 `sam_mask`，确认 detection 产物齐全。
3. 用 800 step 训练第一版正式小跑，并固定在 `Feature_test` 上验收。
4. 再根据真实结果决定：
   - 是否加入 masked loss 或鼻嘴局部重加权
   - 是否把一致性从轻量指标升级成显式 loss
   - 是否在推理侧引入病例级 reference 方案
