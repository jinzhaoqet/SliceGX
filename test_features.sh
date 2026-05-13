#!/bin/bash
# SliceGX 声明式语言 — 五大 Feature 测试脚本
# 用法: cd /root/SliceGX && bash test_features.sh

source /root/autodl-tmp/myenv/bin/activate
export LD_LIBRARY_PATH=/root/autodl-tmp/myenv/lib:$LD_LIBRARY_PATH

SEP="============================================================"

run_query() {
    local label="$1"
    local query="$2"
    echo ""
    echo "$SEP"
    echo "[$label]"
    echo "Query: $query"
    echo "$SEP"
    printf '%s\nexit\n' "$query" | python slicegx_lang.py 2>&1 \
        | grep -E '(=== |Algorithm|Results:|Time:|Cache:|Comparison|Best|Common|support|node=|\[0\]|\[1\]|\[2\]|Router|Resuming)'
}

echo "SliceGX Declarative Language — Feature Test Suite"
echo "Dataset: tree_cycle (871 nodes, 35 test nodes)"

# ====== Feature 1: 包含/排除指定节点 ======

run_query "Feature 1a: 基础查询 (对照组)" \
    "EXPLAIN NODE 519"

run_query "Feature 1b: INCLUDE 517,516 — 强制包含" \
    "EXPLAIN NODE 519 INCLUDE 517,516"

run_query "Feature 1c: EXCLUDE 8,517 — 排除节点" \
    "EXPLAIN NODE 519 EXCLUDE 8,517"

# ====== Feature 2: 结果对比 ======

run_query "Feature 2a: COMPARE BY FIDELITY_PLUS" \
    "EXPLAIN ALL WHERE FACTUAL = TRUE COMPARE BY FIDELITY_PLUS"

run_query "Feature 2b: COMPARE BY COMMON_NODES" \
    "EXPLAIN ALL COMPARE BY COMMON_NODES"

# ====== Feature 3: 自动路由 (观察 Router 输出) ======

run_query "Feature 3a: 单节点 → SS" \
    "EXPLAIN NODE 519"

run_query "Feature 3b: 多节点 → MS" \
    "EXPLAIN NODES 519,537,556"

# ====== Feature 4: K=4 → K=6 增量复用 ======

echo ""
echo "$SEP"
echo "[Feature 4: 增量复用 — K=4 → K=6 缓存续跑]"
echo "Query 1: EXPLAIN NODE 519 WITH K 4"
echo "Query 2: EXPLAIN NODE 519 WITH K 6"
echo "$SEP"
printf 'EXPLAIN NODE 519 WITH K 4\nEXPLAIN NODE 519 WITH K 6\nexit\n' \
    | python slicegx_lang.py 2>&1 \
    | grep -E '(Cache|Resuming|node=|Time)'

# ====== Feature 5: 近似采样 ======

run_query "Feature 5: 近似采样 30%" \
    "EXPLAIN NODE 556 WITH APPROXIMATE 0.3"

echo ""
echo "$SEP"
echo "All tests completed."
echo "$SEP"
