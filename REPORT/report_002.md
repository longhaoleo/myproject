# 报告002：Generation 阶段实验总结

日期：2026-04-11

## 1. 当前状态与主线关系

这一阶段对应当前主线里的 generation 部分。  
它已经不是一个“还没定义清楚的想法”，而是一条已经接上 detection 产物、可以继续迭代的 v1 主链。

当前状态是：

1. 训练、推理、评估三条链路都已成型
2. 仍然属于小样本验证阶段
3. 目标是先把术前到术后的重绘逻辑跑稳，而不是立即追求最终论文级方案

## 2. 阶段目标

这一阶段要解决的问题是：

1. 如何把 detection 侧条件真正接进训练与推理
2. 如何在小样本下最大化利用 paired 和 unpaired 视角
3. 如何在多视角下尽量稳住人物一致性

也就是说，generation 阶段不是“纯生成”，而是一个有明确条件输入和编辑区域约束的局部重绘过程。

## 3. 当前主逻辑

当前 generation 的主逻辑可以概括成：

1. detection 先提供：
   - `eye_mask` 后图片
   - `face_mask`
   - `inpaint_mask`
   - `feather_mask`
2. generation 默认直接用 `eye_mask` 后图片做主输入
3. 训练时以术前图为条件、同视角术后图为目标
4. 推理时只在 face crop 内做 inpainting，再映射回整图
5. 评估时同时看视角级结果和病例级结果

这条逻辑背后的核心判断是：

1. 编辑区域必须固定
2. 条件图和推理输入分布必须一致
3. 缺少真值的少量视角也要尽可能继续利用

## 4. 当前训练逻辑

### 4.1 文件输入

训练前要求 detection 已经产出：

1. 原始病例目录
   - `input_root/<case_id>/术前/<view>.jpg`
   - `input_root/<case_id>/术后/<view>.jpg`
2. 打码图
   - `sanitized_root/<case_id>/<stage>/<view>.jpg`
3. 条件 mask
   - `face_mask`
   - `inpaint_mask`
   - `feather_mask`
   - `nose/mouth` 部位 mask

训练时真正的主输入优先级是：

1. 先用 `eye_mask` 后图片
2. 没有才回退到原图

### 4.2 manifest 组织

generation 不再依赖独立准备步骤，而是运行时直接扫描 detection 产物并生成 manifest。

manifest 每一行至少要回答这些问题：

1. 这是哪个病例、哪个视角、术前还是术后
2. 这张图的主输入路径是什么
3. 对应的 `face_mask / inpaint_mask / feather_mask` 在哪里
4. 这一视角有没有同视角术后真值
5. 这个病例整体有哪些 paired / unpaired 视角

所以 manifest 不是训练样本本身，而是训练样本的索引层。

### 4.3 训练样本构造

当前固定三类训练样本：

1. `paired_head_reference`
   - 条件图：术前 masked 图
   - 目标图：同视角术后 masked 图
   - 区域：`face_mask` 框内
   - 作用：学习整体头面部、姿态和人物连续性

2. `paired_edit_crop`
   - 条件图：术前 masked 图
   - 目标图：同视角术后 masked 图
   - 区域：鼻子和嘴巴联合 crop
   - mask：术前侧 `inpaint_mask`
   - 作用：学习鼻嘴目标区域的风格重绘

3. `self_identity_head`
   - 条件图和目标图：同一张 masked 图
   - 区域：`face_mask` 框内
   - 作用：让少量缺配对视角继续贡献人物稳定信息

换句话说：

1. paired 视角负责真正的术前到术后监督
2. unpaired 视角负责低权重身份保持

### 4.4 Dataset 输入输出

训练时每个样本都会被转成 3 个核心张量：

1. `pixel_values`
   - 来自目标图
   - paired 时就是同视角术后图

2. `mask_values`
   - 来自术前侧 `inpaint_mask`
   - 定义允许重绘的区域

3. `masked_pixel_values`
   - 来自条件图
   - 把编辑区域按 mask 挖空后的结果

所以当前训练的 I/O 关系非常明确：

1. 条件图：术前
2. 目标图：术后
3. mask：术前侧鼻嘴联合编辑区域

### 4.5 扩散训练本体

当前训练本体仍然是标准 SDXL inpainting LoRA：

1. 目标图编码成 latent
2. 挖空后的条件图也编码成 latent
3. mask 缩放到 latent 尺寸
4. UNet 以 `noisy_latents + mask + masked_image_latents` 为输入
5. 用标准噪声回归损失训练

所以当前方案的“新意”主要在样本组织和条件设计，不在损失形式。

### 4.6 权重与顺序设计

当前三类样本权重有明显分工：

1. `paired_head_reference`
   - 标准权重
   - 负责头面部整体稳定

2. `paired_edit_crop`
   - 更高权重
   - 负责强化鼻嘴目标区域学习

3. `self_identity_head`
   - 更低权重
   - 负责利用数据，不压过 paired 监督

此外当前还做了 grouped sampling：

1. 先按病例聚合样本
2. 再尽量让同一个病例的不同视角连续出现

它改变的不是 loss，而是模型在一次梯度累积里“看见样本的顺序”。

## 5. 当前推理逻辑

当前推理不是直接整图跑，而是：

1. 读取术前 masked 图
2. 读取术前 `inpaint_mask`
3. 用 `face_mask` 先裁出 face crop
4. 在 crop 内做 SDXL inpainting
5. 再把结果映射回整图
6. 最后按 `feather_mask` 回贴

这样做的原因是：

1. 更接近训练时的区域分布
2. 更聚焦到局部重绘任务
3. 更容易控制非编辑区稳定性

同一病例不同视角的推理还会共享病例级稳定随机性，减少风格漂移。

## 6. 当前评估逻辑

评估不是只看单张图，而是分成两层：

1. 视角级
   - 三联图：术前 / 真实术后 / 预测术后
2. 病例级
   - contact sheet
   - 统计 paired/unpaired 视角
   - 统计完整六视角病例和部分视角病例

当前一致性评估也是两层：

1. `hard_identity_similarity`
   - 更强调非编辑区稳定
2. `soft_face_similarity`
   - 鼻嘴虽会变化，但仍作为弱参考

## 7. 已做尝试

generation 侧已经经历过这些关键变化：

1. 从术后单图自监督，切到术前锚点配对监督
2. 从“只用 paired 视角”，切到“paired + 低权重 unpaired”
3. 从整图思路，切到 face crop 内 inpainting 再回贴
4. 从只看单张输出，切到病例级多视角组织评估

## 8. 当前效果与阶段性结论

当前阶段已经得到的主要效果是：

1. detection 和 generation 的接口已经清晰
2. 训练、推理、评估的输入输出逻辑已经打通
3. 小样本下 paired/unpaired 数据都能被利用
4. 多视角结果已经能按病例维度组织和检查

当前阶段性结论是：

1. v1 的关键价值在于样本组织方式，而不是复杂新损失
2. detection 条件统一之后，generation 的问题更容易单独定位
3. 当前方案已经足够支撑下一轮真正的生成质量验证

## 9. 当前局限

1. 当前仍然是轻量一致性方案，没有显式 identity loss
2. 当前评估指标更偏工程验收，不是最终身份度量
3. 当前 LoRA recipe 仍然是小样本可落地方案，不等于最终最优 recipe
4. 真正的训练质量和推理效果还需要在完整环境中继续验证

## 10. 后续可能思路

1. 继续细化 paired/unpaired 样本权重
2. 继续验证 grouped sampling 是否足够，或是否要升级到联合视角损失
3. 继续观察 face crop 与整图回贴的稳定性
4. 后续再评估是否加入：
   - 显式 identity / perceptual consistency loss
   - `ControlNet-Depth`
   - 病例级 reference 方案，如 `IP-Adapter`
