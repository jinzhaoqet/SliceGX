# SliceGXQ Demo 与 TKDE 扩展版范围划分

## 1. 文档目的

本文档用于基于当前已有材料，明确区分两条工作线：

- `Demo`：SliceGXQ 的系统演示与重投修改
- `TKDE`：围绕声明式查询语言与执行优化的扩展版论文

目标不是让两条线彼此竞争，而是让它们形成清晰分工：

- `Demo` 负责回答：用户如何使用系统，系统如何交互展示
- `TKDE` 负责回答：为什么 explanation analytics 需要声明式查询、组合查询和优化器


## 2. 当前已有材料

### 2.1 已有论文与评审材料

- [WWW.pdf](/home/ycb/SliceGX/WWW.pdf): 已录用的 SliceGX 算法论文
- [SliceGXQ.pdf](/home/ycb/SliceGX/SliceGXQ.pdf): SliceGXQ demo 论文版本
- [review.pdf](/home/ycb/SliceGX/review.pdf): SliceGXQ demo 拒稿评审意见

### 2.2 仓库中已有的声明式原型基础

- 查询入口：[slicegx_lang.py](/home/ycb/SliceGX/slicegx_lang.py:1)
- 查询解析：[query_parser.py](/home/ycb/SliceGX/query_parser.py:1)
- 查询校验：[query_validator.py](/home/ycb/SliceGX/query_validator.py:1)
- 查询执行：[query_executor.py](/home/ycb/SliceGX/query_executor.py:1)
- 结果对象：[result_schema.py](/home/ycb/SliceGX/result_schema.py:1)
- 会话查询：[query_session.py](/home/ycb/SliceGX/query_session.py:1)
- 计划格式化：[query_formatter.py](/home/ycb/SliceGX/query_formatter.py:1)
- 轻量优化器：[planner/optimizer.py](/home/ycb/SliceGX/planner/optimizer.py:1)
- 功能脚本：[test_features.sh](/home/ycb/SliceGX/test_features.sh:1)

### 2.3 仓库中已有的文档基础

- [声明式查询语言总览.md](/home/ycb/SliceGX/docs/声明式查询语言总览.md:1)
- [query_language_spec.md](/home/ycb/SliceGX/docs/query_language_spec.md:1)
- [query_semantics.md](/home/ycb/SliceGX/docs/query_semantics.md:1)
- [TKDE_VLDBJ_declarative_gap_analysis.md](/home/ycb/SliceGX/docs/TKDE_VLDBJ_declarative_gap_analysis.md:1)
- [下一步实现文档_中文版.md](/home/ycb/SliceGX/docs/下一步实现文档_中文版.md:1)


## 3. Demo 的定位与归属

### 3.1 Demo 的核心目标

Demo 不再强调 SliceGX 算法本身，而是强调：

- SliceGXQ 如何被用户使用
- GUI 与查询接口如何配合
- 系统如何支持 layer-wise explanation 的浏览、诊断与交互分析

一句话概括：

- `Demo` 的主问题是 “how to use the system”

### 3.2 Demo 应承担的内容

- 真实 UI 展示与交互路径
- 数据集选择、模型选择、查询输入、输出可视化
- end-to-end 使用场景
- query 和 GUI 的映射关系
- 输出结果如何解释
- 轻量系统证据：
  - 响应时间
  - 默认 `k`
  - explanation size
  - hardware
  - query latency

### 3.3 Demo 不应承担的核心内容

这些内容可以提到，但不应成为 demo 主贡献：

- formal query semantics
- result algebra
- optimizer 理论
- benchmark suite
- routed vs naive 等系统型消融
- 声明式查询语言的完整理论化

### 3.4 Demo 当前已经有的内容

- SliceGXQ 系统框架与前后端模块描述
- GUI + query-based explanation 的系统方向
- 多个 scenario 草稿
- query-like 示例
- 可视化与 layer-wise explanation 的展示基础

### 3.5 Demo 下一步准备补什么

基于 [review.pdf](/home/ycb/SliceGX/review.pdf:1)，Demo 修改集中在“讲清楚”和“展示一致性”上。

#### 必改项

- 重写 `Section 4`，统一 narrative：
  - dataset/context
  - analysis question
  - user operations
  - system outputs
  - interpretation
- 重构 `Figure 1`，确保与 `Section 2` 的模块命名和流程完全一致
- 重做 `Figure 4`，使用真实 UI 截图，或与视频严格对应
- 在正文中展开 `Figure 2` 的 query examples，并与具体 scenario 对应
- 明确说明 `SliceGXQ` 与 `SliceGX` 的区别
- 明确强调 demo 的 declarative interface / system integration / interaction contribution

#### 建议补充

- 给每个 scenario 加 dataset 背景
- 给关键 scenario 加 query latency、默认 `k`、explanation size、hardware
- 修正 Scenario 2 中 “错误来自 hop 3 / layer 3” 的推理链说明
- 将 Scenario 3 改写成真正的 usage scenario，或明确标注为 scalability showcase
- 补 query examples 的输出解释与用户操作路径

#### 视频修改

- 控制在 5 分钟内
- 使用自然配音
- 按明确 story 录制：
  - 任务背景
  - 用户操作
  - 系统输出
  - 输出解读


## 4. TKDE 扩展版的定位与归属

### 4.1 TKDE 的核心目标

TKDE 扩展版不再围绕“单个算法”和“单个 demo”写，而是围绕：

- explanation analytics 的声明式查询模型
- 结果查询与结果集合查询
- 组合查询
- optimizer-backed execution engine

一句话概括：

- `TKDE` 的主问题是 “why explanation analytics needs a declarative query model and optimized execution”

### 4.2 TKDE 的重点范围

目前建议将 TKDE 聚焦在三条主线：

- 结果查询
- 组合集查询
- optimizer

这三条线都比“选哪个节点”“选哪一层”更适合构成系统论文的主卖点。

### 4.3 TKDE 应承担的内容

#### A. 结果查询

重点不是基础选择器，而是结果对象和结果集上的查询。

建议重点写：

- `EXPLAIN ALL`
- `EXPLAIN CLASS`
- `WHERE FACTUAL = TRUE`
- `WHERE FIDELITY_PLUS > x`
- `COMPARE BY FIDELITY_PLUS`
- `COMPARE BY COMMON_NODES`
- `RANK BY FIDELITY_PLUS`

需要回答：

- explanation result 的 schema 是什么
- result set 的语义是什么
- compare / rank / filter 的输出对象是什么

#### B. 组合集查询

这是声明式语言区别于 GUI 和脚本接口的关键。

建议重点写：

- `LET`
- `FILTER`
- `RANK`
- `COMPARE <name> BY ...`

这部分可以形成一个最小 result algebra，用于支持：

- 保存结果
- 基于结果继续筛选
- 基于结果继续比较
- 复用中间分析对象

#### C. Optimizer

这是 TKDE 中最具系统论文特征的一部分。

建议重点写：

- logical plan
- physical plan
- query routing
- cache reuse
- approximate execution
- multi-node shared execution
- multi-layer execution

建议重点组织为以下 plan family：

- `SingleNodeExactPlan`
- `SharedCandidateMultiNodePlan`
- `MultiLayerExplorationPlan`
- `ApproximateSamplePlan`
- `CacheResumePlan`

### 4.4 TKDE 不应成为主线的内容

这些内容可作为背景或基础功能，但不宜作为 TKDE 核心 novelty：

- 仅仅支持节点选择
- 仅仅支持层选择
- 仅仅支持参数调节
- 单个 explanation 的 GUI 可视化
- 单次交互式使用流程

这些内容更适合归属给 demo。

### 4.5 TKDE 当前已经有的内容

- 查询入口和 DSL 雏形
- 基础 parser / validator / executor 分层
- `EXPLAIN PLAN FOR ...`
- 轻量 logical / physical plan
- 最小会话组合查询：
  - `LET`
  - `FILTER`
  - `RANK`
  - `COMPARE`
- JSON 输出与统一结果对象雏形
- cache / approximate / routing 的初步实现

### 4.6 TKDE 下一步准备补什么

#### 第一优先级：结果查询与结果模型

- 明确 `ExplanationResult`、`QueryExecutionResult`、`ComparisonResult`
- 把结果字段和语义写成正式 schema
- 补足 query 与 result 的映射说明
- 增强结果序列化与后处理能力

#### 第二优先级：组合集查询

- 强化 `LET / FILTER / RANK / COMPARE`
- 让组合查询不只停留在 REPL feature
- 形成可写入论文的 minimal result algebra
- 明确组合查询的语义与合法性规则

#### 第三优先级：optimizer

- 将当前 rule-based routing 进一步模块化
- 增加更清晰的 logical plan / physical plan 抽象
- 增加更明确的 plan reason 输出
- 加入轻量 cost model
- 做最小系统级 ablation：
  - routed vs naive
  - cache vs no-cache
  - exact vs approximate

#### 第四优先级：benchmark 与评估

- 设计 benchmark query suite
- 形成固定 workload 集
- 增加 latency / plan choice / cache hit / approximation deviation 等统计
- 让实验从 feature demo 变成 systems evaluation


## 5. Demo 与 TKDE 的边界总结

### 5.1 适合归给 Demo 的

- GUI
- 用户操作路径
- 视频
- 真实 UI 截图
- 单次交互与可视化
- 场景讲解
- 输出解释

### 5.2 适合归给 TKDE 的

- 结果查询
- 结果集查询
- 组合查询
- query semantics
- result schema
- optimizer
- plan 输出
- benchmark 与系统评估

### 5.3 一句话区分

- `Demo` 关注 “怎么用”
- `TKDE` 关注 “为什么需要这样查询、这样执行、这样优化”


## 6. 建议的近期推进顺序

### 6.1 Demo 线

先完成改稿所需的最小闭环：

1. 重写 Scenario narrative
2. 重做 Figure 1 / 2 / 4
3. 统一论文、UI、视频三者的一致性
4. 补 dataset context 与轻量 system evidence
5. 重录 5 分钟视频

### 6.2 TKDE 线

优先集中在三件事，不要继续发散：

1. 结果查询与结果模型
2. 组合集查询
3. optimizer 与最小 benchmark


## 7. 参考文献建议

以下文献按用途分组，优先服务于 `TKDE` 扩展版，也可部分用于 Demo related work。

### 7.1 Declarative ML / 查询 ML 工件

1. Piero Molino, Christopher Ré. Declarative Machine Learning Systems: The future of machine learning will depend on it being in the hands of the rest of us. ACM Queue, 2021.
   链接: https://dblp.org/rec/journals/queue/MolinoR21

2. Maximilian E. Schüle, Matthias Bungeroth, Dimitri Vorona, Alfons Kemper, Stephan Günnemann, Thomas Neumann. ML2SQL: Compiling a Declarative Machine Learning Language to SQL and Python. EDBT 2019.
   链接: https://dblp.org/rec/conf/edbt/SchuleBVKG019

3. Sebastian Schelter. Reconstructing and Querying ML Pipeline Intermediates. CIDR 2023.
   链接: https://dblp.org/rec/conf/cidr/Schelter23

### 7.2 查询语言 / 图学习视角

4. Floris Geerts. A Query Language Perspective on Graph Learning. PODS 2023.
   链接: https://dblp.org/rec/conf/pods/Geerts23

5. Laura State, Salvatore Ruggieri, Franco Turini. Declarative Reasoning on Explanations Using Constraint Logic Programming. JELIA 2023.
   链接: https://dblp.org/rec/conf/jelia/StateRT23

### 7.3 Explanation 优化与系统执行

6. Supun Nakandala, Arun Kumar, Yannis Papakonstantinou. Incremental and Approximate Inference for Faster Occlusion-based Deep CNN Explanations. SIGMOD 2019.
   链接: https://dblp.org/rec/conf/sigmod/NakandalaKP19

7. Supun Nakandala, Arun Kumar, Yannis Papakonstantinou. Query Optimization for Faster Deep CNN Explanations. SIGMOD Record, 2020.
   链接: https://dblp.org/rec/journals/sigmod/NakandalaKP20

### 7.4 GNN explanation 代表方法

8. Zhitao Ying, Dylan Bourgeois, Jiaxuan You, Marinka Zitnik, Jure Leskovec. GNNExplainer: Generating Explanations for Graph Neural Networks. NeurIPS 2019.
   链接: https://dblp.org/rec/conf/nips/YingBYZL19

9. Hao Yuan, Haiyang Yu, Jie Wang, Kang Li, Shuiwang Ji. On Explainability of Graph Neural Networks via Subgraph Explorations. ICML 2021.
   链接: https://dblp.org/rec/conf/icml/YuanYWLJ21

10. Dongsheng Luo, Wei Cheng, Dongkuan Xu, Wenchao Yu, Bo Zong, Haifeng Chen, Xiang Zhang. Parameterized Explainer for Graph Neural Network. NeurIPS 2020.
    链接: https://arxiv.org/abs/2011.04573

11. Michael Schlichtkrull, Nicola De Cao, Ivan Titov. Interpreting Graph Neural Networks for NLP With Differentiable Edge Masking. arXiv, 2020.
    链接: https://arxiv.org/abs/2010.00577

### 7.5 本项目自身论文

12. Cibo Yu, Tingting Zhu, Tingyang Chen, Yinghui Wu, Arijit Khan, Xiangyu Ke. SliceGX: Layer-wise GNN Explanation with Model-slicing. WWW 2026.

13. Haitong Tang, Yinghui Wu, Arijit Khan, Tingting Zhu, Tingyang Chen, Xiangyu Ke. Declarative Explanations for Graph Neural Networks: A Demonstration. SliceGXQ demo paper draft.


## 8. 当前结论

基于现有材料，建议采用如下策略：

- Demo 线：以“修改好并重投”为目标，重点解决 narrative、界面一致性、query example 落地、用户操作路径和视频问题
- TKDE 线：以“做成声明式 explanation analytics 系统论文”为目标，重点推进结果查询、组合集查询和 optimizer

目前最不建议做的事情是：

- 继续同时扩很多新关键词
- 让 Demo 和 TKDE 两条线继续混用同一套叙事
- 把 GUI 能轻松完成的基础选择器当作 TKDE 的核心贡献

