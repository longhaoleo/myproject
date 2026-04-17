# 基线报告：项目基线、方法论与日志书写约定

日期：2026-04-11

> 重要说明  
> `baseline_report.md` 是当前日志体系的基线文件。  
> 它既定义项目的总目标和主路线，也定义后续各份报告应该怎么写。  
> 若后续目标或方法发生重大变化，应新增 `report_00x.md` 记录，不建议频繁改写本文件。

## 1. 项目核心目标

本项目的目标，是构建一个针对特定医生风格的鼻整形术前预测系统。  
系统输入患者术前多视角照片，输出该医生风格下的术后外观预测结果。

这个目标同时要求满足 3 个约束：

1. 风格约束
   - 预测结果要体现特定医生的术式审美和鼻部重塑倾向。
2. 结构约束
   - 非手术区域不能明显漂移，头面部比例和姿态要尽量连续。
3. 人物约束
   - 预测结果仍然必须看起来是同一个人，而不是“换了一张脸”。

## 2. 当前主路线

当前代码对应的是一个分阶段落地的 v1 路线，而不是一次性把所有理想组件都塞进来。

### 2.1 detection 侧主逻辑

detection 负责定义 generation 能消费的条件，而不是 generation 自己再去推断编辑区域。

当前 detection 的职责是：

1. 人脸检测
2. 关键点和部位框构建
3. `eye_mask` 打码图生成
4. `sam_mask` 产出标准化的：
   - `face_mask`
   - `inpaint_mask`
   - `feather_mask`

也就是说，generation 只负责“怎么改”，不负责“改哪里”。

### 2.2 generation 侧主逻辑

当前 generation 的 v1 主线是：

1. 默认使用 `eye_mask` 后的术前图作为主输入
2. 训练时用“同病例、同视角”的 `术前 -> 术后` 配对监督
3. 同时利用少量 unpaired 视角做低权重身份保持样本
4. 底模固定为 `SDXL Inpainting`
5. 当前只训练 UNet LoRA
6. 推理时默认先做 face crop，再映射回整图并按 `feather_mask` 回贴

这是一个“小样本、少量缺视角”的工程可落地方案，不是最终科研完整版。

## 3. 当前方法论

这部分不是写“用了哪些模型”，而是写为什么当前主链路这样设计。

### 3.1 编辑区域由 detection 定义

原因：

1. 术后重绘区域如果在 generation 侧再临时推断，容易前后不一致。
2. 鼻整形任务本身就是局部编辑任务，更适合把编辑区域先固定下来。
3. `face_mask / inpaint_mask / feather_mask` 由 detection 统一生成后，训练、推理、评估才能共用同一套条件。

### 3.2 术前图是条件图，术后图是目标图

原因：

1. 任务本质不是“根据术后图学重建”，而是“根据术前图预测术后”。
2. 如果训练锚点放在术后图，模型会更像图像自重建，而不是术前到术后的条件生成。
3. 把术前作为条件图，推理时的输入分布才和训练时一致。

### 3.3 眼部打码图是主输入

原因：

1. 数据链路里已经先做了 `eye_mask`，训练和推理应与这条链保持一致。
2. 如果训练时用原图、推理时用打码图，会产生输入分布偏移。
3. 眼部被遮挡后，模型会更集中依赖鼻部、口周、脸型和姿态等信息。

### 3.4 配对样本与单边样本都要利用

原因：

1. 数据量本来就不大，少量缺视角样本不能简单浪费。
2. paired 视角负责真正的术前到术后监督。
3. unpaired 视角虽然没有术后真值，但仍能提供这个人的头面部稳定信息。

### 3.5 一致性先做轻量方案

当前 v1 不直接上显式 identity loss，而是先靠：

1. `paired_head_reference`
2. `paired_edit_crop`
3. `self_identity_head`
4. 病例级 grouped sampling
5. `hard_identity_similarity / soft_face_similarity` 评估

原因：

1. 当前更紧迫的问题是先把小样本链路跑稳。
2. 过早引入多种额外损失，会让调参空间爆炸。
3. 轻量方案更适合先验证“数据组织方式”和“主链路方向”是否正确。

## 4. 当前主要尝试与阶段性效果

### 4.1 detection 侧尝试

已经尝试过的主要检测路线包括：

1. `MTCNN`
2. `RetinaFace`
3. `SCRFD`
4. `BlazeFace`
5. `MediaPipe Face Mesh / FaceLandmarker`
6. `YOLOv8-Face`
7. `CenterFace`

当前阶段性结论是：

1. `SCRFD` 在稳定性上更优
2. `BlazeFace` 速度更快
3. `YOLOv8-Face` 是可接受的折中方案
4. `MTCNN / Face Mesh / CenterFace` 暂不适合作为主链路默认方案

### 4.2 generation 侧尝试

generation 侧已经从“术后单图自监督”逐步切到“术前锚点、同视角术后监督”的思路。

当前阶段性效果是：

1. detection 和 generation 的条件接口已经打通
2. manifest 已能区分 paired / unpaired 视角
3. 训练样本已明确分成：
   - `paired_head_reference`
   - `paired_edit_crop`
   - `self_identity_head`
4. 推理和评估已能围绕病例级多视角组织输出

这里的“效果”目前主要是工程链路效果，而不是最终生成质量结论。

## 5. 当前局限

当前方案仍然有明确边界：

1. detection 侧鼻部和部位框在部分大侧脸/底视图仍有偏移风险
2. generation 侧目前仍是 v1 轻量一致性方案
3. 当前评估里的相似度是工程指标，不是最终身份度量
4. 当前 LoRA recipe 侧重可落地和小样本利用，不等于最终最优训练 recipe

## 6. 后续可能思路

后续可以沿 4 条线继续迭代：

1. detection 线
   - 继续优化多视角检测稳定性
   - 优化视角微调参数
   - 继续比较 `SCRFD / YOLOv8-Face / BlazeFace`

2. generation 线
   - 继续细化 paired/unpaired 样本权重
   - 验证 face crop 与整图回贴策略的稳定性
   - 进一步提高小样本利用率

3. 一致性线
   - 从轻量评估指标升级到显式 identity / perceptual consistency loss
   - 从 grouped sampling 升级到同病例双视角联合损失

4. 条件增强线
   - 重新评估是否加入 `ControlNet-Depth`
   - 评估病例级 reference 方案，例如 `IP-Adapter` 或 reference attention

## 7. 日志书写约定

从现在开始，`log/` 下的所有报告都应优先写“逻辑、方法、思路”，而不是只写“做了什么改动”。

每份报告至少应该回答这些问题：

1. 当前要解决的核心问题是什么
2. 当前采用的逻辑和方法是什么
3. 为什么这样设计，而不是别的方案
4. 当前做过哪些尝试
5. 当前观察到的效果或阶段性结论是什么
6. 当前方案还有哪些局限
7. 后续可能怎么继续推进

不推荐把报告写成：

1. 单纯的文件改动列表
2. 没有上下文的模块清单
3. 只有最终结论，没有尝试过程和方法判断

## 8. 建议阅读顺序

1. `baseline_report.md`
   - 看项目基线、方法论和日志写法
2. `detection_report.md`
   - 看 detection 侧的方法演化、尝试和效果
3. `generation_report.md`
   - 看 generation 侧的训练、推理、评估逻辑
4. 后续 `*_report.md`
   - 看新的阶段性变化、尝试结果和后续可能思路
