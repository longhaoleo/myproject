# Report 002

日期：2026-04-07

## 1. 本轮目标

这轮改动把 generation 侧从“术后单图自监督 inpainting”调整成“术前锚点、同视角术后监督”的小样本重绘方案，并且显式处理两类现实约束：

1. 一个人通常有 6 个视角，但少量样本会缺 1 到 2 个视角；
2. 数据量不大，不能把单边缺失视角直接废掉；
3. 眼部已经先打码，因此 generation 默认直接消费 `eye_mask` 后的图片。

## 2. 当前实现思路

### 2.1 manifest 侧

`generation/index.py` 继续按图片逐行输出 manifest，但每一行都补了病例级视角信息：

1. `available_pre_views`
2. `available_post_views`
3. `paired_views`
4. `paired_view_count`
5. `is_paired_view`

这样 generation 不需要额外做一次病例聚合，就能知道当前视角是否有同视角术后真值，也能知道这个病例整体缺了哪些视角。

### 2.2 训练侧

`generation/train.py` 现在固定构造三类样本：

1. `paired_head_reference`
   - 条件图：术前 masked 图
   - 目标图：同视角术后 masked 图
   - 区域：`face_mask` 框内
   - 作用：稳住头部整体、姿态和人物连续性

2. `paired_edit_crop`
   - 条件图：术前 masked 图
   - 目标图：同视角术后 masked 图
   - 区域：鼻子和嘴巴联合 crop
   - 作用：集中学习鼻整形目标区域的风格重绘

3. `self_identity_head`
   - 条件图和目标图都用同一张 masked 图
   - 适用于单边缺失视角
   - 作用：把缺配对视角转成低权重的身份保持样本，避免浪费数据

训练输入不再默认读原图，而是直接把 `eye_mask` 产出的打码图当主数据。原图只在缺少打码图时回退。

### 2.3 病例级采样

小样本场景下，单纯按图片随机会让“完整六视角病例”天然占更多梯度预算。因此训练顺序改成按病例分组：

1. 先按 `case_id` 组织样本
2. 每轮尽量从同一个病例里连续取两个不同视角
3. 如果该病例 paired view 不够，就用 `self_identity_head` 补位

这样一次梯度累积里更容易连续看到同一个人的多个视角，提升人物稳定性，同时又不需要额外引入复杂一致性损失。

## 3. 一致性逻辑

这版没有把一致性直接做成训练 loss，而是先落成两层区域和两类验收指标。

### 3.1 两层区域

1. `hard_identity_region = face_mask - dilated(inpaint_mask)`
   - 代表非编辑区
   - 这些区域原则上应当更稳定

2. `soft_identity_region = face_mask`
   - 鼻子和嘴巴虽然会改，但仍保留局部解剖和人物线索
   - 不能从“人物连续性参考”里彻底删掉

### 3.2 两类指标

`generation/eval.py` 会输出：

1. `hard_identity_similarity`
2. `soft_face_similarity`

这两个指标目前是轻量启发式特征，不是正式的人脸识别模型分数。它们的作用是先做工程验收和模型选择，而不是替代后续科研版的一致性损失。

## 4. 推理与评估

### 4.1 推理

`generation/infer.py` 现在按下面的顺序运行：

1. 读取术前 masked 图
2. 读取术前 `inpaint_mask`
3. 用 `face_mask` 框裁出 face crop
4. 在 crop 内做 SDXL inpainting
5. 再把结果映射回整图
6. 最终按 `feather_mask` 回贴

同一个病例的推理随机性也做了稳定化：

1. `case_seed` 固定病例级风格落点
2. `view_seed` 再做视角级细分

这样不同视角不会完全共用噪声，但也不会随机漂移得太开。

### 4.2 评估

评估现在分两层：

1. 视角级
   - 有 paired post 的视角导出三联图：`术前 / 真实术后 / 预测术后`

2. 病例级
   - 导出 contact sheet
   - 标出哪些视角有真值、哪些视角只有预测、哪些视角缺失

同时汇总：

1. `paired_view_count`
2. `unpaired_pre_view_count`
3. `unpaired_post_view_count`
4. `predicted_view_count`
5. `complete_six_view_cases`
6. `partial_view_cases`

## 5. 注释与后续升级口

代码里已经把几个关键点写进注释：

1. 训练时 `target_tensor` 来自术后图，`masked_tensor` 始终来自条件图
2. 鼻嘴联合 crop 和病例级分组采样是为小样本场景服务
3. 评估里的 `hard/soft` 一致性指标是 v1 轻量方案

后续升级优先级：

1. 在 `hard_identity_region` 和 `soft_identity_region` 上增加低权重 perceptual / identity consistency loss
2. 把病例级分组采样升级成“同病例双视角联合损失”
3. 在推理侧加入病例级 reference 适配，例如 IP-Adapter 或 reference attention
4. 数据量扩大后，再考虑更强的多视角联合建模
