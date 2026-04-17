# Log Index

`REPORT/` 现在只承担两个作用：

1. 给人一个清晰的阅读顺序
2. 保存阶段性实验总结，而不是保存零散改动记录

## 推荐阅读顺序

1. [report_000.md](/Users/leo/myproject/log/report_000.md)
   - 项目基线
   - 当前主路线
   - 方法论
   - 报告书写约定

2. [report_001.md](/Users/leo/myproject/log/report_001.md)
   - detection 阶段总结
   - 多检测器、打码、SAM 条件构建
   - 主要尝试、效果和阶段性结论

3. [report_002.md](/Users/leo/myproject/log/report_002.md)
   - generation 阶段总结
   - 训练、推理、评估主链
   - 小样本、缺视角、多视角一致性的当前方案

## 当前写法要求

后续所有 `report_00x.md` 都优先写这些内容：

1. 当前阶段要解决的核心问题
2. 当前方法与执行逻辑
3. 已做过的尝试
4. 观察到的效果和阶段性结论
5. 当前局限与风险
6. 后续可能思路

不推荐只写：

1. 文件改动列表
2. 模块清单
3. 没有上下文的最终结论
