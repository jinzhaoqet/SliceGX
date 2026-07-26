# SliceGX 查询语言规范

## 1. 目标

本文档定义 SliceGX 声明式查询语言的表层语法，用于统一：

- 查询写法
- clause 顺序
- 参数格式
- 合法与非法输入的边界

本文档只描述语法，不描述执行语义。  
执行语义见 [query_semantics.md](/home/ycb/SliceGX/docs/query_semantics.md:1)。


## 2. 设计原则

当前语言面向 GNN explanation workload，强调：

- 单条查询即可表达完整意图
- 支持集合级目标而非仅单节点
- 支持结构约束、结果过滤、结果比较、近似执行
- 支持基于已保存结果的轻量组合查询
- 保持 REPL 和脚本调用写法一致


## 3. 核心查询形态

一条标准查询由以下部分组成：

```text
EXPLAIN <TARGET>
[WHERE <PREDICATE>]
[AT LAYER <n> | AT ALL LAYERS]
[INCLUDE <NODE_LIST>]
[EXCLUDE <NODE_LIST>]
[COMPARE BY <COMPARE_METRIC>]
[RANK BY <RANK_METRIC>]
[WITH <PARAM> <VALUE>]
```

说明：

- `EXPLAIN <TARGET>` 是必选项
- 其他 clause 为可选项
- `WITH` 子句可出现多次
- `INCLUDE` 和 `EXCLUDE` 最多各出现一次
- `COMPARE BY` 最多出现一次
- `RANK BY` 最多出现一次


## 4. EBNF 定义

```text
QUERY           := EXPLAIN_CLAUSE
                 | LET_CLAUSE
                 | FILTER_CLAUSE
                 | RANK_CLAUSE
                 | COMPARE_RESULT_CLAUSE

EXPLAIN_CLAUSE  := "EXPLAIN" TARGET
                   [WHERE_CLAUSE]
                   [LAYER_CLAUSE]
                   [INCLUDE_CLAUSE]
                   [EXCLUDE_CLAUSE]
                   [COMPARE_CLAUSE]
                   [INLINE_RANK_CLAUSE]
                   {WITH_CLAUSE}

LET_CLAUSE      := "LET" IDENT "=" QUERY_EXPR
QUERY_EXPR      := EXPLAIN_CLAUSE
                 | FILTER_CLAUSE
                 | RANK_CLAUSE
                 | COMPARE_RESULT_CLAUSE

FILTER_CLAUSE   := "FILTER" IDENT "WHERE" PREDICATE
RANK_CLAUSE     := "RANK" IDENT "BY" "FIDELITY_PLUS"
COMPARE_RESULT_CLAUSE
                := "COMPARE" IDENT "BY" COMPARE_METRIC

TARGET          := "NODE" INT
                 | "NODES" INT_LIST
                 | "ALL"
                 | "CLASS" INT
                 | INT

WHERE_CLAUSE    := "WHERE" PREDICATE

PREDICATE       := "FACTUAL" "=" BOOL
                 | "COUNTERFACTUAL" "=" BOOL
                 | "FIDELITY_PLUS" ">" FLOAT
                 | "FIDELITY_MINUS" "<" FLOAT
                 | "SUBGRAPH_SIZE" "<=" INT

LAYER_CLAUSE    := "AT" "LAYER" INT
                 | "AT" "ALL" "LAYERS"
                 | "AT" INT

INCLUDE_CLAUSE  := "INCLUDE" INT_LIST
EXCLUDE_CLAUSE  := "EXCLUDE" INT_LIST

COMPARE_CLAUSE  := "COMPARE" "BY" COMPARE_METRIC
COMPARE_METRIC  := "FIDELITY_PLUS"
                 | "COMMON_NODES"
INLINE_RANK_CLAUSE
                := "RANK" "BY" RANK_METRIC
RANK_METRIC     := "FIDELITY_PLUS"

WITH_CLAUSE     := "WITH" PARAM_NAME PARAM_VALUE
PARAM_NAME      := "K"
                 | "H"
                 | "THETA"
                 | "GAMMA"
                 | "APPROXIMATE"

PARAM_VALUE     := INT
                 | FLOAT

INT_LIST        := INT {"," INT}
IDENT           := letter {letter | digit | "_"}
BOOL            := "TRUE" | "FALSE"
INT             := digit {digit}
FLOAT           := INT ["." INT]
```


## 5. Clause 说明

### 5.1 `EXPLAIN`

表示 explanation workload 的目标范围。

支持形式：

- `EXPLAIN NODE 519`
- `EXPLAIN NODES 519,537,556`
- `EXPLAIN ALL`
- `EXPLAIN CLASS 1`
- `EXPLAIN 519`

其中 `EXPLAIN 519` 是 `EXPLAIN NODE 519` 的简写。

### 5.2 `WHERE`

用于表达结果过滤条件。

当前只支持单个 predicate。  
未来版本可扩展为多条件布尔组合。

合法示例：

- `WHERE FACTUAL = TRUE`
- `WHERE COUNTERFACTUAL = FALSE`
- `WHERE FIDELITY_PLUS > 0.5`
- `WHERE SUBGRAPH_SIZE <= 6`

### 5.3 `AT`

用于指定层范围。

支持形式：

- `AT LAYER 2`
- `AT ALL LAYERS`
- `AT 2`

### 5.4 `INCLUDE`

强制解释结果包含给定节点列表。

示例：

- `INCLUDE 517,516`

### 5.5 `EXCLUDE`

强制解释结果排除给定节点列表。

示例：

- `EXCLUDE 8,517`

### 5.6 `COMPARE BY`

用于对结果集合执行比较或聚合分析。

支持形式：

- `COMPARE BY FIDELITY_PLUS`
- `COMPARE BY COMMON_NODES`

### 5.6 `RANK BY`

用于对当前查询结果按某个指标排序。

支持形式：

- `RANK BY FIDELITY_PLUS`

### 5.7 `WITH`

用于覆盖默认执行参数。

支持形式：

- `WITH K 6`
- `WITH H 0.2`
- `WITH THETA 0.1`
- `WITH GAMMA 0.5`
- `WITH APPROXIMATE 0.3`

说明：

- `WITH APPROXIMATE` 后可带采样比例
- 若不提供比例，则使用系统默认值

### 5.8 `LET`

用于给一条 query 或一个组合表达式命名，并将结果保存在当前 REPL 会话中。

支持形式：

- `LET Q1 = EXPLAIN NODE 519`
- `LET Q2 = FILTER Q1 WHERE FACTUAL = TRUE`
- `LET Q3 = RANK Q2 BY FIDELITY_PLUS`

说明：

- `LET` 当前只支持 REPL 会话内存态
- 当前只存储真实查询结果，不存储内部调试输出

### 5.9 `FILTER`

用于对一个已保存结果集应用 `WHERE` 风格过滤。

支持形式：

- `FILTER Q1 WHERE FACTUAL = TRUE`
- `FILTER Q1 WHERE FIDELITY_PLUS > 0.5`

### 5.10 `RANK`

用于对一个已保存结果集按某个指标排序。

当前支持：

- `RANK Q1 BY FIDELITY_PLUS`

### 5.11 `COMPARE <name> BY ...`

用于对一个已保存结果集做比较或聚合分析。

支持形式：

- `COMPARE Q1 BY FIDELITY_PLUS`
- `COMPARE Q1 BY COMMON_NODES`


## 6. 参数类型约束

建议的参数约束如下：

- `K`：正整数
- `H`：浮点数，建议范围 `[0, +inf)`
- `THETA`：浮点数，建议范围 `[0, +inf)`
- `GAMMA`：浮点数，建议范围 `[0, 1]`
- `APPROXIMATE` ratio：浮点数，建议范围 `(0, 1]`
- `LAYER`：整数；`-1` 仅保留给内部表示 “all layers”


## 7. 典型合法示例

```text
EXPLAIN NODE 519
EXPLAIN NODES 519,537,556
EXPLAIN ALL WHERE FACTUAL = TRUE
EXPLAIN ALL WHERE FACTUAL = TRUE RANK BY FIDELITY_PLUS
EXPLAIN CLASS 1 COMPARE BY COMMON_NODES
EXPLAIN NODE 519 INCLUDE 518,517 WITH K 6
EXPLAIN NODE 556 WITH APPROXIMATE 0.3
EXPLAIN ALL AT ALL LAYERS
LET Q1 = EXPLAIN NODE 519
FILTER Q1 WHERE FACTUAL = TRUE
RANK Q1 BY FIDELITY_PLUS
COMPARE Q1 BY COMMON_NODES
```


## 8. 当前不支持或不推荐的写法

以下写法当前未正式支持：

- 多个 `WHERE` 条件联写
- `AND / OR / NOT`
- 子查询
- 多个 `COMPARE BY`
- `GROUP BY`
- `ORDER BY`
- 持久化 session 结果
- 在单次命令模式下使用 `LET/FILTER/RANK/COMPARE <name>`

这些能力可在后续语言版本中引入。


## 9. REPL 与命令行模式

支持两种使用方式。

### 9.1 REPL 模式

```bash
python slicegx_lang.py
```

当前组合查询能力主要工作在 REPL 模式下。

### 9.2 单次查询模式

```bash
python slicegx_lang.py --query="EXPLAIN NODE 519"
```

### 9.3 JSON 输出模式

```bash
python slicegx_lang.py --query="EXPLAIN NODE 519" --output-format=json
```

该模式会输出结构化结果，包括：

- query summary
- selected plan
- filtered results
- comparison output
- cache stats

### 9.4 REPL 组合查询示例

```text
LET Q1 = EXPLAIN NODE 519
FILTER Q1 WHERE FACTUAL = TRUE
LET Q2 = RANK Q1 BY FIDELITY_PLUS
COMPARE Q2 BY COMMON_NODES
list
```


## 10. 当前语言边界

当前语言是一个面向 explanation workload 的声明式 DSL 原型。  
它已经支持：

- 目标声明
- 约束表达
- 结果比较
- 参数覆盖
- 近似执行控制
- 会话内 `LET / FILTER / RANK / COMPARE` 组合查询

但仍然不是通用数据库式查询语言。后续版本可考虑扩展：

- 多条件布尔表达式
- 命名结果
- query chaining
- 结果聚合代数
