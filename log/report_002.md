# 报告2：Detection 条件构建与 SDXL Pipeline 对接说明

日期：2026-04-05

## 1. 核心结论

当前仓库已经从“仅前处理”推进到“Detection 条件构建 + 生成侧消费”的最小闭环。

这次新增的不是最终科研版本，而是一套工程骨架，目标是把以下链路完整接上：

1. 读取病例目录树
2. 读取 SAM part mask
3. 在 detection 侧规范化并落盘为生成器稳定可消费的 mask 条件
4. 生成 depth 条件
5. 训练医生风格 LoRA
6. 运行 `SDXL Inpainting + ControlNet-Depth + LoRA`
7. 输出推理结果与工程验收报告

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

### 2.3 眼部隐私与编辑区域冲突策略

这是本轮最重要的逻辑补充。

当前默认规则：

1. 眼部隐私 mask 可以先打码。
2. 若眼部隐私 mask 与鼻嘴编辑区重叠：
   - 以编辑区优先
   - 自动把重叠区域从眼部 mask 中减掉
   - 再给编辑区外扩一圈安全边距

这样做的目的，是避免黑块压入鼻梁、鼻根或嘴周编辑区，导致生成器把“隐私黑块”也学进目标区域。

## 3. 当前代码成果

本轮主入口：

- [generation_pipeline.py](/Users/leo/myproject/generation_pipeline.py)

本轮 detection 侧新增/扩展模块：

- [prepare_dataset.py](/Users/leo/myproject/detection/prepare_dataset.py)

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

### 4.1 `prepare_dataset`

归属：

- `prepare_dataset` 的 mask 定义与条件构建逻辑归 detection 侧负责
- generation 侧只消费这些落盘结果

输出：

1. `manifest.jsonl`
2. `issues.jsonl`
3. 连续鼻嘴 `inpaint_mask`
4. `feather_mask`
5. `eye_privacy_mask`
6. `sanitized` 图
7. `depth` 图
8. `preview` 调试图
9. `prepare_dataset_summary.json`

其中：

- `manifest.jsonl` 是后续训练和推理的唯一真值入口
- `issues.jsonl` 会显式记录缺图、缺 mask、depth fallback、眼部冲突裁切等问题

### 4.2 `train_lora`

输出：

1. `checkpoint-*`
2. `final/` LoRA 权重
3. `train_samples.jsonl`
4. `train_lora_summary.json`

训练数据由两类样本组成：

1. 全脸样本
2. 鼻嘴局部 crop 样本

### 4.3 `infer`

输出：

1. `raw/` 生成器整图输出
2. `composited/` 仅回贴编辑区域后的最终图
3. `preview/` 五联调试图
4. `infer_issues.jsonl`
5. `infer_summary.json`

### 4.4 `evaluate`

输出：

1. `triptychs/`
2. `evaluate_summary.json`

三联图格式固定为：

- 术前
- 术后
- 生成结果

## 5. 已完成验证

已在本地完成的验证包括：

1. detection 与 generation 代码可编译
2. detection 侧 `prepare_dataset` 可在临时样本上落出 manifest 和中间产物
3. `evaluate` 可统计结果并导出三联图
4. 轻量单测已通过

当前尚未在本机完整跑通的部分：

1. 真正的 SDXL LoRA 训练
2. 真正的 SDXL + ControlNet 推理

原因不是代码结构问题，而是当前运行环境还没有安装生成侧依赖与权重。

## 6. 当前风险

1. 方框 mask 稳定，但会牺牲边缘精细度。
2. 目前 LoRA 训练实现优先追求链路闭环，不是最终训练 recipe。
3. 如果术前/术后配对存在漏视角，当前会记录 issue，但不会自动补全。
4. depth fallback 的 `opencv-luma` 只适合 debug，不适合作为正式实验结果。

## 7. 下一步建议

1. 先按 [generation/README.md](/Users/leo/myproject/generation/README.md) 安装依赖并下载权重。
2. 先运行 `prepare_dataset`，确认实际数据上的 `issues.jsonl`。
3. 用极小样本运行一次 `train_lora` 和 `infer`，只验证显存、路径和权重可用性。
4. 再根据真实结果决定：
   - 是否继续坚持 box mask
   - 是否对 mouth 区域缩小
   - 是否把眼部隐私从黑块改成 blur
