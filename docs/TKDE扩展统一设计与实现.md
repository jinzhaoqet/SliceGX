# SliceGX TKDE 扩展阶段汇报

## 1. 当前工作概括

我把 WWW 中的 SliceGX 分层解释算法，扩展成了一套面向 GNN explanation analytics 的声明式查询系统。

现在系统不只是“输入节点并生成 explanation”，还可以：

- 用查询语言描述分析需求；
- 对 explanation result set 做过滤、排序、比较和模式发现；
- 自动选择 SS、MS、MM 或缓存执行；
- 复用历史结果和增量状态；
- 运行统一 benchmark；
- 使用自然语言输入查询，并把结构化结果转换为自然语言说明。

整体流程：

```text
自然语言或 SliceGX DSL
        ↓
Parser + Validator
        ↓
Logical Plan + Optimizer
        ↓
SS / MS / MM / Cache
        ↓
Filter / Rank / Compare / Pattern / Materialize
        ↓
结构化结果或 Result2NL
```


## 2. 已完成的五部分

### 2.1 Explanation Data Model

以前 explanation 主要以 dictionary 表示，现在正式定义了：

- `Graph`
- `Model`
- `Prediction`
- `Explanation`
- `LayerExplanation`
- `ExplanationSet`
- `Pattern`

这样每个查询操作都有明确的数据对象和字段。

实现：`analytics/data_model.py`


### 2.2 Query Language and Result Algebra

新增查询能力：

- 多条件过滤：`WHERE ... AND ...`
- 字段投影：`PROJECT`
- 排序：`RANK`
- 比较：`COMPARE`
- 分组模式发现：`GROUP BY ... PATTERN`
- 结果物化：`MATERIALIZE`
- 会话复用：`LET / FILTER / RANK / COMPARE`

实现了带输入输出类型的操作符：

| 操作符 | 输入 | 输出 |
|---|---|---|
| Explain | Graph + Model | ExplanationSet |
| Filter | ExplanationSet | ExplanationSet |
| Project | ExplanationSet | Table |
| Rank | ExplanationSet | ExplanationSet |
| Compare | ExplanationSet | Comparison |
| GroupPattern | ExplanationSet | PatternSet |
| Materialize | Result | MaterializedResult |

实现：

- `analytics/algebra.py`
- `query_parser.py`
- `query_validator.py`


### 2.3 Optimizer-backed Execution

原来的简单 SS/MS/MM 路由已经扩展为：

- logical operator rewriting；
- 多个 physical plan 候选枚举；
- cost-based plan selection；
- SS/MS/MM 自动选择；
- 多节点共享；
- 多层共享；
- generation result cache；
- 显式物化；
- K 增量执行；
- 质量约束近似执行。

当前物理计划包括：

| Physical Plan | 用途 |
|---|---|
| SingleNodeExactPlan | 单节点解释 |
| IncrementalSingleNodePlan | 从较小 K 的状态继续执行 |
| MultiNodeSharedCandidatePlan | 多节点共享候选计算 |
| MultiLayerSharedSlicePlan | 多层共享 model slice |
| ConstraintAwareLayeredSingleNodePlan | 带结构约束的多层查询 |
| MaterializedGenerationScan | 直接复用已有 generation result |

实现：

- `planner/rewrites.py`
- `planner/cost_model.py`
- `planner/quality.py`
- `planner/optimizer.py`
- `query_executor.py`


### 2.4 Explanation Analytics Benchmark

建立了 9 类 workload：

| Workload | 内容 |
|---|---|
| W1 | 单节点解释 |
| W2 | 类别级解释和排序 |
| W3 | 全测试集解释 |
| W4 | 多节点、多层诊断 |
| W5 | 结果过滤和投影 |
| W6 | 公共模式发现 |
| W7 | 会话式 LET/FILTER/RANK/COMPARE |
| W8 | 质量约束近似执行 |
| W9 | 物化结果复用 |

runner 会记录：

- latency；
- result count；
- selected algorithm；
- estimated cost；
- cache statistics；
- error。

实现：

- `benchmark/workloads.py`
- `benchmark/runner.py`
- `benchmark/session_adapter.py`
- `benchmark/quality.py`


### 2.5 NL2Query 和 Result2NL

NL2Query 流程：

```text
自然语言
  → QueryIntent JSON
  → 确定性 DSL 编译
  → Parser
  → Validator
  → Optimizer
  → Executor
```

主要处理：

- 自然语言转 SliceGX 查询；
- 意图不明确时先澄清；
- 非法参数通过 validator feedback 修复；
- LLM 输出不能绕过 parser 和 validator。

Result2NL 只读取真实结构化结果：

- 不允许虚构节点、层和指标；
- 不允许把统计关联直接写成因果结论；
- evidence 不足时必须说明无法判断。

实现：

- `llm/provider.py`
- `llm/intent_schema.py`
- `llm/nl2query.py`
- `llm/result2nl.py`


## 3. 一个完整例子

用户输入自然语言：

```text
分析类别1中所有层的解释，只保留factual且fidelity大于0.6的结果，
按层找公共模式，并把结果保存下来。
```

NL2Query 生成：

```text
EXPLAIN CLASS 1 AT ALL LAYERS
WHERE FACTUAL = TRUE AND FIDELITY_PLUS > 0.6
GROUP BY LAYER
PATTERN MIN_SUPPORT 0.5
MATERIALIZE AS CLASS1_ANALYSIS
```

系统内部执行：

```text
1. Parser 生成 ExplainQuery
2. Validator 检查类别、层、阈值和操作符
3. LogicalRewriter 合并两个 Filter
4. Optimizer 判断这是类别级、多节点、多层查询
5. 选择 MultiLayerSharedSlicePlan
6. MM 共享多个节点和多个层的计算
7. Filter 保留 factual 且 fidelity-plus > 0.6 的结果
8. GroupPattern 按 layer 发现公共节点模式
9. Materialize 保存结果，供后续查询复用
10. Result2NL 根据真实结果生成自然语言说明
```

结果结构示例：

```json
{
  "algorithm": "MM",
  "filtered_results": 18,
  "analytics": {
    "patterns": {
      "group_by": "layer",
      "items": [
        {
          "nodes": [17, 23],
          "support": 0.72,
          "group_key": 2
        }
      ]
    }
  },
  "materialized_as": "CLASS1_ANALYSIS"
}
```

Result2NL 可以表述为：

```text
在当前过滤后的解释集合中，节点17和23组成的模式在第2层支持率为0.72。
该结果说明这一模式在当前 explanation set 中较常见，但不能单独证明因果关系。
```

上面的数值只是结果格式示例，最终数值必须由真实数据集执行产生，LLM 不能自行填写。


## 4. 两个优化例子

### 4.1 跨查询复用

第一次执行：

```text
EXPLAIN NODES 519,537 WITH K 6 MATERIALIZE AS BASE
```

第二次执行：

```text
EXPLAIN NODES 519,537 WITH K 6
WHERE FIDELITY_PLUS > 0.5
```

两条查询的 generation 参数相同，因此第二条查询可以选择：

```text
MaterializedGenerationScan
```

只执行新的结果过滤，不重新运行 SliceGX。


### 4.2 增量 K

```text
EXPLAIN NODE 519 WITH K 4
EXPLAIN NODE 519 WITH K 8
```

第二条查询从 K=4 的 greedy state 继续扩展，对应：

```text
IncrementalSingleNodePlan
```


## 5. 当前验证情况

目前完成：

- 38 项单元测试全部通过；
- 9 类 benchmark workload 全部通过 parser/validator 校验；
- 所有新增 Python 文件静态编译通过；
- `git diff --check` 通过。

验证命令：

```bash
python -m unittest discover -s tests -v

python -m benchmark.validate_suite \
  --nodes=519,537 \
  --class-label=1
```


## 6. 下一步

代码框架已经完成，下一步主要是产生投稿所需的真实实验数据：

1. 在正确 PyTorch/PyG 环境中跑 SS、MS、MM 和约束多层计划；
2. 比较 optimizer 与 fixed SS/MS/MM；
3. 做 cache、incremental、cross-node、cross-layer 消融；
4. 为每个数据集建立 approximation quality profile；
5. 跑 9 类 workload 的 latency、memory 和 throughput；
6. 建立 NL2Query 标注数据集；
7. 对比至少两个 LLM 的 translation accuracy 和 Result2NL hallucination rate。


## 7. 准备形成的 TKDE 新贡献

1. 正式定义 GNN explanation analytics 的数据对象；
2. 提出支持生成、过滤、排序、比较、模式发现和物化的查询语言与结果代数；
3. 提出支持 SS/MS/MM、共享计算、缓存、增量和质量约束近似执行的 optimizer；
4. 建立 explanation analytics benchmark；
5. 增加经过 parser/validator 约束的 NL2Query 和基于真实证据的 Result2NL。
