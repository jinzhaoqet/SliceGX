# SliceGX 查询语义说明

## 1. 文档目标

本文档定义 SliceGX 声明式查询语言的语义层含义，也就是：

- 一条 query 想表达什么
- 每个 clause 约束的对象是什么
- 哪些子句影响逻辑结果
- 哪些子句只影响执行策略

语法定义见 [query_language_spec.md](/home/ycb/SliceGX/docs/query_language_spec.md:1)。


## 2. 基本语义对象

当前系统的核心语义对象包括：

- 图对象 `G`
  - 节点集合
  - 边集合
  - 节点特征
  - 标签与 mask
- 目标集合 `T`
  - query 想解释的节点集合
- explanation result `E`
  - 针对某个目标节点产生的解释结果
- explanation result set `R`
  - 一条 query 执行后的结果集合
- comparison result `C`
  - 对 `R` 做比较或聚合后得到的附加结果
- named result `N`
  - 通过 `LET` 在当前会话中保存的结果集合


## 3. 一条查询的总体语义流程

从逻辑语义上，一条查询可分为 6 步：

1. 目标解析
   - 从图和 query 中确定目标集合 `T`
2. 执行参数确定
   - 将 query 覆盖参数与配置默认值合并
3. explanation 生成
   - 对 `T` 中的目标产生 explanation results
4. 结果过滤
   - 应用 `WHERE` 约束得到过滤后的结果集合 `R`
5. 结果比较/聚合
   - 若存在 `COMPARE BY`，则对 `R` 计算 `C`
6. 结果排序
   - 若存在 `RANK BY`，则对结果集合按指定指标重排

注意：

- `EXPLAIN`、`WHERE`、`COMPARE BY` 主要影响逻辑结果
- `WITH APPROXIMATE` 主要影响物理执行策略
- 优化器选择 `SS / MS / MM` 不改变 query 的逻辑意图，只改变执行方式
- `LET / FILTER / RANK / COMPARE <name>` 作用在已保存结果集上


## 4. `EXPLAIN` 子句语义

### 4.1 `EXPLAIN NODE <id>`

语义：

- 目标集合 `T = {id}`

结果：

- 返回该节点对应的 explanation result

### 4.2 `EXPLAIN NODES <id1>,<id2>,...`

语义：

- 目标集合 `T = {id1, id2, ...}`

结果：

- 返回多个 explanation result，通常每个 target 对应一个结果

### 4.3 `EXPLAIN ALL`

语义：

- `T` 为当前数据集 test set 中可解释的全部目标节点

说明：

- 对 `tree_grid` / `tree_cycle` 这类数据集，系统采用当前实现约定的 test-mask 规则
- 对其他数据集，采用 `test_mask` 为真对应的全部节点

### 4.4 `EXPLAIN CLASS <label>`

语义：

- `T` 为 test set 中标签等于 `<label>` 的全部目标节点

## 4.5 `LET <name> = <expr>`

语义：

- 将表达式 `<expr>` 的结果绑定到当前 REPL 会话中的名字 `<name>`

说明：

- `LET` 不改变 `<expr>` 的结果内容
- 它只是引入一个可复用的结果引用
- 当前实现是内存态 session binding，不做磁盘持久化


## 4.6 `FILTER <name> WHERE ...`

语义：

- 取出已保存结果集 `N(name)`
- 对其中的 explanation results 应用 `WHERE` 风格过滤
- 返回新的结果集 `R'`

这一步不会重新跑底层 explanation 算法。


## 4.7 `RANK <name> BY FIDELITY_PLUS`

语义：

- 取出已保存结果集 `N(name)`
- 按 `fidelity_plus` 从高到低排序
- 返回排序后的结果集 `R'`

当前版本中：

- `RANK` 只改变结果顺序
- 不丢弃记录
- 不重新执行 explanation


## 4.8 `COMPARE <name> BY ...`

语义：

- 取出已保存结果集 `N(name)`
- 在该结果集上执行比较或聚合算子
- 返回带 comparison summary 的结果对象


## 5. `WHERE` 子句语义

当前 `WHERE` 是结果过滤语义，而不是前置搜索约束语义。

这意味着：

- 系统先生成候选 explanation results
- 然后根据 `WHERE` 条件对结果做筛选

### 5.1 `WHERE FACTUAL = TRUE/FALSE`

语义：

- 仅保留 `factual` 字段匹配给定布尔值的 explanation result

### 5.2 `WHERE COUNTERFACTUAL = TRUE/FALSE`

语义：

- 仅保留 `counterfactual` 字段匹配给定布尔值的 explanation result

### 5.3 `WHERE FIDELITY_PLUS > x`

语义：

- 仅保留 `fidelity_plus > x` 的 explanation result

### 5.4 `WHERE FIDELITY_MINUS < x`

语义：

- 仅保留 `fidelity_minus < x` 的 explanation result

### 5.5 `WHERE SUBGRAPH_SIZE <= k`

当前实现语义是弱约束：

- 它通过限制 `K` 的上界来间接约束 explanation 的规模

这意味着它当前更接近：

- 执行参数约束

而不是：

- 对最终结果大小的严格 post-filter

后续如果要形式化增强，建议把它拆分为：

- `WITH K`
- `WHERE RESULT_SIZE <= k`

对于 `FILTER <name> WHERE ...`：

- `WHERE` 被复用为 result algebra 中的过滤谓词
- 语义仍然是 post-filter，而不是重新搜索


## 6. `AT` 子句语义

### 6.1 `AT LAYER n`

语义：

- explanation 在单层视角下执行
- 逻辑上表示仅关注某一层对应的 explanation scope

当前实现中：

- 它会映射为相应的切层或 hop 范围

### 6.2 `AT ALL LAYERS`

语义：

- explanation 在多层范围上执行
- 结果集保留层维度

当前实现中：

- 系统走 `MM` 路径
- 返回结果中会记录 `layer`

因此从语义上：

- 结果不是单一 explanation 集合
- 而是一个带层标签的 explanation result 集合


## 7. `INCLUDE` / `EXCLUDE` 子句语义

### 7.1 `INCLUDE`

语义：

- 在 explanation 搜索空间中强制包含指定节点

说明：

- 这是搜索约束，不是结果后处理
- 若节点不在可达范围内，最终效果受底层搜索空间限制

### 7.2 `EXCLUDE`

语义：

- 在 explanation 搜索空间中排除指定节点

说明：

- 这是执行前约束，不是结果后过滤


## 8. `WITH` 子句语义

### 8.1 `WITH K`

语义：

- 覆盖 explanation size / budget 参数

### 8.2 `WITH H`

语义：

- 覆盖 influence threshold

### 8.3 `WITH THETA`

语义：

- 覆盖 diversity threshold

### 8.4 `WITH GAMMA`

语义：

- 覆盖评分目标中 influence / diversity 的权重

### 8.5 `WITH APPROXIMATE <ratio>`

语义：

- 不改变用户的目标对象和逻辑意图
- 只改变系统生成 explanation 的物理执行方式

更明确地说：

- 逻辑目标仍然是“解释这些节点”
- approximate 只是允许系统用采样近似部分搜索或统计过程

因此它属于：

- execution preference / physical hint

而不是：

- logical result operator


## 9. `COMPARE BY` 子句语义

`COMPARE BY` 的输入对象是过滤后的结果集合 `R`。

对于 `COMPARE <name> BY ...`：

- 输入对象是已保存结果集 `N(name)` 对应的当前结果列表
- 不会重新触发 explanation 生成

### 9.1 `COMPARE BY FIDELITY_PLUS`

语义：

- 在 `R` 中寻找 `fidelity_plus` 最大的结果

输出：

- 最优结果节点
- 最优 `fidelity_plus`
- 对应 explanation subgraph

### 9.2 `COMPARE BY COMMON_NODES`

语义：

- 在 `R` 中统计 explanation node 的出现频率
- 识别支持度较高的共同模式

当前实现中：

- 支持度阈值为 `>= 50%`

输出：

- common nodes
- total explanations
- top support 分布


## 10. 优化器与查询语义的关系

当前系统已经引入优化器原型。

优化器负责：

- 从 query 生成 logical plan
- 选择 physical plan
- 决定走 `SS / MS / MM`

对于普通查询：

- 优化器会参与执行路径选择
- 系统仍会继续调用后端 explanation executor

但是从语义上必须强调：

- 优化器不改变 query 的逻辑意图
- 优化器只改变执行策略

这也是 declarative system 的关键属性之一。


## 11. 结果对象语义

每条 explanation result 至少包含：

- `node_id`
- `nodes`
- `factual`
- `counterfactual`
- `both`
- `score`
- `fidelity_plus`
- `fidelity_minus`
- 可选 `layer`

一个 query 的完整输出由以下部分组成：

- `query`：归一化后的查询摘要
- `plan`：优化器选择的逻辑/物理计划
- `results`：过滤后的 explanation results
- `comparison`：比较或聚合结果
- `cache_stats`：缓存统计
- `time_seconds`：执行时长

## 12. 文本输出与 JSON 输出语义

### 文本输出

文本输出用于：

- REPL 交互
- 人类快速浏览
- feature demo

### JSON 输出

JSON 输出用于：

- benchmark
- 脚本化实验
- 结果复现
- 后续论文图表生成

对于 `LET / FILTER / RANK / COMPARE <name>`：

- JSON / 文本输出仍然基于同一个 `QueryExecutionResult`
- 区别在于 `query` 字段会标记 `source_name`、`filter_applied`、`rank_by` 或 `compare_by`


## 13. 当前语义边界

当前系统仍有几个语义上尚未完全形式化的地方：

1. `WHERE SUBGRAPH_SIZE <= k` 当前更像参数裁剪
2. approximate 模式尚未提供误差界或等价性说明
3. 多条件 `WHERE` 尚未支持
4. 命名查询和组合语义尚未引入
5. 当前组合查询仅支持 REPL 内存态结果，不支持跨进程持久化
6. 不同 backend 之间尚未形成统一 backend-independent semantics
7. 当前 planner 仍是 explain-query 级规划，不是多查询 workload 级规划

这些都是后续版本应继续补足的部分。


## 14. 当前版本的语义结论

就当前实现而言，可以把 SliceGX query language 理解为：

- 一个面向 explanation workload 的声明式查询接口
- 一个将目标声明、结果约束、结构约束、比较分析和执行偏好统一到单条 query 中的 DSL
- 一个开始具备最小 result algebra 的 explanation query language

并且它已经具备 declarative system 的关键雏形：

- 用户描述 what
- 系统决定 how

下一步需要继续加强的，是：

- 更正式的语义边界
- 更清晰的 logical / physical separation
- 更强的结果代数与组合性
