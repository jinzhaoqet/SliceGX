# SliceGX Declarative System Gap Analysis for TKDE/VLDBJ

## Purpose

This document evaluates the current SliceGX declarative prototype against the standards typically expected by strong systems/database venues such as `TKDE` and `VLDBJ`.

It answers four concrete questions:

1. What the repository already has.
2. What is still missing.
3. Why the missing parts matter for a publishable declarative system paper.
4. Which requirements are difficult to satisfy with a GUI or page-click workflow, and therefore justify a declarative language.

The goal is not to criticize the current system as a prototype. The goal is to identify the gap between:

- a useful research prototype with a DSL-like interface, and
- a venue-level declarative data system contribution.


## Repository Evidence

The current assessment is based on concrete artifacts already present in the repository:

- Config-driven declarative layer via Hydra/OmegaConf:
  - [config/config.yaml](/home/ycb/SliceGX/config/config.yaml:1)
  - [config/datasets/tree_cycle.yaml](/home/ycb/SliceGX/config/datasets/tree_cycle.yaml:1)
  - [config/models/gcn.yaml](/home/ycb/SliceGX/config/models/gcn.yaml:1)
- Query-language prototype:
  - [slicegx_lang.py](/home/ycb/SliceGX/slicegx_lang.py:1)
- Execution backends and execution semantics:
  - [SliceGX.py](/home/ycb/SliceGX/SliceGX.py:1)
  - [Slice_MS.py](/home/ycb/SliceGX/Slice_MS.py:1)
  - [Slice_MM.py](/home/ycb/SliceGX/Slice_MM.py:1)
- Functional feature test script:
  - [test_features.sh](/home/ycb/SliceGX/test_features.sh:1)
- Project-level motivation and usage notes:
  - [README.md](/home/ycb/SliceGX/README.md:1)


## Executive Assessment

### Short verdict

The repository already contains a meaningful declarative prototype, not just a wrapper script:

- It has a domain-facing query interface.
- It supports query parsing and execution dispatch.
- It includes constrained querying, comparison, approximate execution, and limited state reuse.

However, by `TKDE/VLDBJ` standards, the current system is still closer to:

- a DSL-enabled experimental prototype, or
- a feature-rich command interface for one family of explanation algorithms,

than to:

- a fully articulated declarative data system with formal semantics, optimization theory, generalizable abstractions, and venue-grade evaluation.

### Publication readiness snapshot

| Dimension | Current status | Assessment |
|---|---|---|
| Problem motivation | Present | Good start |
| Declarative surface syntax | Present | Prototype-level |
| Execution engine | Present | Moderate |
| Optimizer | Very limited | Weak |
| Formal semantics | Largely absent | Major gap |
| Generality beyond one backend family | Weak | Major gap |
| Theory or formal guarantees | Absent | Major gap |
| Systems evaluation | Feature demo only | Major gap |
| Reproducibility infrastructure | Partial | Moderate gap |
| GUI-vs-language justification | Implicit | Needs explicit framing |


## What the Current System Already Has

This section is intentionally specific. A strong paper starts by recognizing what is already genuinely valuable.

### 1. A real query surface, not just config files

The repository does already expose a language-like interface in [slicegx_lang.py](/home/ycb/SliceGX/slicegx_lang.py:1).

Supported concepts include:

- target specification:
  - `EXPLAIN NODE <id>`
  - `EXPLAIN NODES <id1>,<id2>,...`
  - `EXPLAIN ALL`
  - `EXPLAIN CLASS <label>`
- constraints:
  - `WHERE FACTUAL = TRUE/FALSE`
  - `WHERE COUNTERFACTUAL = TRUE/FALSE`
  - `WHERE FIDELITY_PLUS > x`
  - `WHERE FIDELITY_MINUS < x`
  - `WHERE SUBGRAPH_SIZE <= x`
- execution scope:
  - `AT LAYER n`
  - `AT ALL LAYERS`
- structural constraints:
  - `INCLUDE`
  - `EXCLUDE`
- result analytics:
  - `COMPARE BY FIDELITY_PLUS`
  - `COMPARE BY COMMON_NODES`
- execution modifiers:
  - `WITH K`
  - `WITH H`
  - `WITH THETA`
  - `WITH GAMMA`
  - `WITH APPROXIMATE`

This is already beyond “just expose flags in argparse”.

### 2. Query execution is separated from parsing

The code distinguishes:

- query representation: `ExplainQuery`
- parsing: `QueryParser`
- execution: `SliceGXExecutor`

This is a good systems design signal because it creates a path toward:

- logical planning,
- validation,
- cost-based routing,
- future backends.

### 3. There is already an execution-routing idea

The current `auto_route` mechanism in [slicegx_lang.py](/home/ycb/SliceGX/slicegx_lang.py:270) selects among `SS`, `MS`, and `MM`.

This matters because declarative systems are fundamentally about:

- users specifying `what`,
- the system determining `how`.

Even though the current routing logic is simple, it is still the seed of an optimizer story.

### 4. There is already limited physical optimization

The current implementation includes:

- cache reuse for subfunction/model slice/explanatory states
- approximate execution via sampling
- multi-node and multi-layer execution paths

Those are not yet full optimizer contributions, but they are valid “physical execution” ingredients.

### 5. The system already supports set-level analysis

This is important and should not be undersold.

The language is not limited to one-shot explanation generation. It already allows:

- class-level targeting
- all-test-node targeting
- comparison across results
- common-node extraction

That moves the work from “one explanation at a time” toward “explanation analytics”, which is closer to a database-style contribution.


## What TKDE/VLDBJ Would Expect But Is Still Missing

Below is the core checklist. Each subsection has:

- what the venue expects,
- what the repository currently has,
- what is still missing,
- why it matters.


## 1. Clear Problem Formulation

### Venue expectation

The paper must define a crisp systems problem, not just showcase features.

Typical acceptable formulations would be:

- declarative querying for GNN explanations
- declarative explanation analytics over graph ML outputs
- query optimization for explanation generation and comparison workloads

### Current state

The motivation is present but diffuse.

The repository suggests several parallel stories:

- a language for explanation requests
- a convenience layer over SliceGX
- a multi-mode execution wrapper
- a feature testbed for five DSL features

### Missing

The paper must choose one primary problem statement and commit to it.

Examples:

- “We study declarative querying over explanation objects.”
- “We propose a declarative workload model for GNN explanation analytics.”
- “We design a query engine that compiles explanation intent into optimized execution plans.”

### Why this matters

Without a sharp problem statement, reviewers will classify the work as:

- a UI idea,
- a scripting convenience,
- or an engineering wrapper around an existing algorithm.


## 2. Formal Data Model

### Venue expectation

A declarative language must define the objects it queries.

At minimum, the paper should define:

- graph data object
- model object
- prediction object
- explanation object
- explanation set object
- layer-specific explanation object
- comparison result object

### Current state

These entities exist operationally in code, but they are not defined as a formal data model.

For example:

- nodes and classes are query targets
- explanations are dictionaries with fields like `nodes`, `Fid+`, `Fid-`, `factual`
- layer-indexed results exist in `MM`

### Missing

A formal schema is needed.

Example sketch:

- `Graph(V, E, X, Y)`
- `Prediction(v, label, score)`
- `Explanation(v, layer, subgraph, fidelity_plus, fidelity_minus, factual, counterfactual)`
- `ExplanationSet(Q)` as the result of query `Q`
- `Pattern(nodes, support, source_explanations)`

### Why this matters

If objects are not formally defined, then:

- semantics remain implementation-dependent,
- comparisons are ambiguous,
- optimizer correctness becomes difficult to argue.


## 3. Formal Query Language Definition

### Venue expectation

A paper at this level should define the language explicitly:

- grammar
- syntax categories
- typing rules
- well-formedness conditions

### Current state

The language exists informally through imperative parsing logic in [slicegx_lang.py](/home/ycb/SliceGX/slicegx_lang.py:84).

### Missing

At least the following are needed:

- EBNF or BNF grammar
- statement classes:
  - target clause
  - predicate clause
  - structural clause
  - comparison clause
  - approximation clause
  - routing or backend hints if any
- invalid-query rules
- type checks:
  - `K` must be integer
  - `H`, `THETA`, `GAMMA` must be floats in valid ranges
  - `CLASS` labels must be valid

### Why this matters

Without a formal language definition, the system looks like:

- a hand-coded parser with keywords,

rather than:

- a declarative query language.


## 4. Declarative Semantics

### Venue expectation

The language must define what a query means independently of how it is executed.

That means the semantics of:

- target selection
- filtering
- comparison
- layer scoping
- approximation

must be specified as logical meaning, not just “whatever this function returns”.

### Current state

Semantics are mostly operational:

- parse query
- route query
- call backend
- filter results
- print output

### Missing

A proper semantics section should define:

- the denotation of `EXPLAIN ALL`
- the denotation of `WHERE FACTUAL = TRUE`
- the meaning of `COMPARE BY COMMON_NODES`
- whether `AT ALL LAYERS` returns a set union, Cartesian family, or ordered multi-layer result
- whether `WITH APPROXIMATE 0.3` changes semantics or only execution strategy

### Why this matters

This is one of the biggest differences between:

- a real declarative system paper, and
- a practical command interface.


## 5. Separation Between Logical Intent and Physical Execution

### Venue expectation

Users should state intent.
The system should choose execution.

This separation is central in database-style work.

### Current state

There is a partial version of this separation:

- query intent is expressed via DSL
- the system selects `SS/MS/MM`

### Missing

The language currently exposes many backend-facing knobs directly:

- `K`
- `H`
- `THETA`
- `GAMMA`
- explicit layer behavior

These are useful for power users, but they are still low-level physical or algorithmic parameters.

The paper needs a cleaner distinction:

- logical user goals:
  - minimal faithful explanation
  - representative shared explanation
  - counterfactual under budget
  - cross-layer stable explanation
- physical execution:
  - exact search
  - cached search
  - approximate sampling
  - multi-start sharing
  - multi-layer hop-jumping

### Why this matters

If users must reason directly about algorithm parameters, reviewers may say:

- this is a parameterized API, not a declarative system.


## 6. Query Optimizer

### Venue expectation

A strong systems paper normally needs more than a parser and executor.
It needs some notion of planning or optimization.

### Current state

The current optimizer story is very limited:

- rule-of-thumb routing via `auto_route`
- cache reuse
- approximate mode

### Missing

A stronger optimizer should include several of the following:

- logical-to-physical plan mapping
- rule-based rewriting
- cost model
- workload-aware plan selection
- adaptive approximate planning
- cache-aware planning
- multi-query optimization
- plan explainability

A realistic optimizer section could define:

- `single-node exact plan`
- `shared-candidate multi-node plan`
- `multi-layer exploration plan`
- `approximate sampled neighborhood plan`
- `cache-resume incremental plan`

with decision rules such as:

- choose `MS` when many targets share overlapping neighborhoods
- choose approximate execution when candidate space exceeds threshold
- reuse partial greedy states for monotonic `K` expansions

### Why this matters

For `TKDE/VLDBJ`, the optimizer is often where the system novelty becomes strongest.
Right now this is your biggest growth opportunity.


## 7. Cost Model

### Venue expectation

If the system claims optimized execution, it should justify its choices using either:

- a formal cost model,
- a heuristic model with empirical validation,
- or an adaptive planning scheme.

### Current state

No explicit cost model is defined.

### Missing

A cost model should estimate query execution cost as a function of:

- number of target nodes
- neighborhood size
- hop count / layer count
- candidate-set overlap
- value of `K`
- cache state
- approximate sampling ratio

Even a simple analytical or learned cost model would help.

### Why this matters

Without a cost model, routing looks ad hoc.
Reviewers may view `auto_route` as a manual heuristic rather than a systems contribution.


## 8. Query Compositionality

### Venue expectation

A mature declarative system supports composition:

- one query feeding another
- reusable result objects
- nested or chained analytics

### Current state

The current language supports individual queries but not query composition.

### Missing

Examples of missing compositional features:

- named query results:
  - `LET Q1 = EXPLAIN ALL WHERE FACTUAL = TRUE`
- downstream use:
  - `COMPARE Q1 BY COMMON_NODES`
- query chaining:
  - explain, then filter, then aggregate, then compare
- reusable subqueries
- result materialization

### Why this matters

Without composition, the language behaves more like a command shell than a query language.


## 9. Result Model and Output Algebra

### Venue expectation

Declarative systems usually define not only input/query syntax, but also result structure and supported downstream operations.

### Current state

Results are emitted as dictionaries and formatted strings.

### Missing

The paper should define:

- result schema
- sortable fields
- filterable fields
- aggregatable fields
- comparison operators over result sets
- provenance metadata

Possible result algebra:

- `FILTER(result_set, predicate)`
- `RANK(result_set, metric)`
- `GROUP(result_set, key)`
- `SUMMARIZE(result_set, pattern_metric)`
- `COMPARE(result_set_1, result_set_2, comparator)`

### Why this matters

The current system can produce answers, but it does not yet present them as a principled queryable result space.


## 10. Backend Generality

### Venue expectation

A strong declarative paper should either:

- support multiple backends, or
- argue convincingly that the abstraction is backend-independent.

### Current state

The repository includes multiple explanation-related components, but the DSL is mostly tied to SliceGX execution patterns and associated execution classes.

### Missing

You need one of two stronger positions:

1. General framework position:
   - define an explainer backend interface
   - adapt at least 2 to 4 different explainers
   - show the same query language works across them

2. Domain-specialized system position:
   - argue that the paper is about declarative querying over layered slicing-based GNN explanations specifically
   - then formalize why this narrower scope still merits a systems paper

### Why this matters

Otherwise reviewers may ask:

- is this really a declarative system, or just a front-end for one algorithm family?


## 11. Approximation Semantics and Guarantees

### Venue expectation

If approximation is part of the language, the paper should state what guarantees exist.

### Current state

Approximate mode is implemented via sampling and exposed in the query language.

### Missing

The paper must answer:

- what exactly is approximated
- which parts of the plan are approximate
- how approximation affects output semantics
- whether any quality bound, monotonicity, or error characterization exists

At minimum, provide:

- empirical tradeoff curves
- stability analysis
- sensitivity analysis over sampling ratios

### Why this matters

Approximation is an asset only if the user can understand its consequences.
Otherwise it looks like an unsafe speed hack.


## 12. Formal Properties

### Venue expectation

Theoretical depth varies by paper, but some formal properties are usually helpful.

### Current state

The system currently does not expose formal claims.

### Missing

Potential formal properties worth proving:

- query well-formedness
- semantics preservation under specific rewrites
- monotonicity for size-bounded expansion under cache-resume
- safe pruning conditions
- complexity analysis for key query classes
- correctness of common-node aggregation semantics

### Why this matters

Even lightweight propositions can significantly improve reviewer confidence that the system is principled.


## 13. Evaluation Methodology

### Venue expectation

A `TKDE/VLDBJ` systems paper needs more than feature demos.

### Current state

The repository has:

- data
- checkpoints
- a feature-oriented test script

This is a strong prototype baseline, but not yet a full evaluation package.

### Missing

Evaluation should cover at least four axes:

1. Expressiveness
   - which analysis tasks are supported by the language but awkward in GUI/API workflows

2. Efficiency
   - planning overhead
   - execution speed
   - cache benefits
   - approximate speedup

3. Scalability
   - more targets
   - larger neighborhoods
   - more layers
   - larger graphs

4. Quality/utility
   - whether analysts can discover meaningful patterns more effectively

Required experiments likely include:

- exact vs approximate
- routed vs no routing
- cache vs no cache
- set-level query vs repeated single-node calls
- language workload vs handwritten scripts

### Why this matters

Feature demonstrations show possibility.
Venue-grade evaluation shows contribution.


## 14. Reproducibility and Artifact Design

### Venue expectation

The artifact should be runnable, testable, and benchmarkable.

### Current state

The repository already has:

- environment descriptions
- datasets
- checkpoints
- example commands
- a feature test script

### Missing

For paper-grade reproducibility, add:

- stable environment setup instructions
- benchmark query suite
- expected outputs or summary checks
- workload generator
- standardized logging
- result serialization
- automated tests beyond shell-based demonstrations

### Why this matters

A declarative system paper is stronger when readers can reproduce:

- both correctness-oriented results and systems-oriented metrics.


## 15. Usability and Human-Centered Justification

### Venue expectation

If the paper claims declarative usability advantages, it should support them.

### Current state

The usability argument is intuitive but mostly implicit.

### Missing

You should explicitly evaluate:

- lines of code saved versus Python scripting
- number of GUI actions avoided
- number of repeated experiments expressed by one query
- reproducibility benefits of text-based queries
- case studies showing analyst workflow simplification

### Why this matters

“The language is nicer” is too weak.
“The language enables workloads that are difficult to author, reproduce, and optimize otherwise” is much stronger.


## Where the Current System Is Still Too Naive

This section isolates the most likely reviewer objections.

### 1. The parser is keyword-driven rather than language-defined

This is acceptable for a prototype, but too weak for a paper centered on language design.

### 2. The language exposes many algorithm knobs directly

That makes it feel closer to a parameterized CLI than a high-level declarative layer.

### 3. The semantics are embedded in code paths

A declarative paper needs semantics that can be described outside the implementation.

### 4. The optimizer is shallow

Current routing is useful, but not yet substantial enough to be a central systems contribution.

### 5. The result model is underdeveloped

Results are printable, but not yet modeled as a rich algebra over explanation objects.

### 6. The evaluation story is feature-centric rather than systems-centric

The current setup is good for demonstration but not yet for venue-grade validation.


## Why a Declarative Language Is Needed Instead of Pure GUI or Page Clicking

This section is important because a strong paper must answer:

“Why not just build a web UI?”

### 1. Complex constraint combinations are hard to express cleanly in a GUI

Example:

`EXPLAIN ALL WHERE FACTUAL = TRUE AND FIDELITY_PLUS > 0.4 AND SUBGRAPH_SIZE <= 6 EXCLUDE 8,17 COMPARE BY COMMON_NODES`

A GUI can technically represent this, but usually only through:

- many disconnected controls,
- hidden state across panels,
- difficult-to-share interaction history.

The language keeps the entire intent in one reproducible artifact.

### 2. Set-level and workload-level reasoning do not map naturally to page clicks

Language is especially valuable when users want:

- all nodes of a class
- shared motifs across explanations
- ranking over explanation sets
- batch constrained explanation generation
- layered comparison over many nodes

GUIs are good for inspection.
Languages are better for workloads.

### 3. Reproducibility is much better with text than with interaction traces

In papers, logs, benchmarks, and multi-run studies, text queries are easier to:

- version,
- compare,
- rerun,
- cite,
- and automate.

### 4. Query optimization is easier to insert under a declarative interface

Once the user specifies only intent, the system can:

- route to different backends,
- choose exact or approximate plans,
- reuse cache,
- batch targets,
- prune or materialize intermediates.

This is far harder to explain and validate in a click-driven workflow.

### 5. LLMs, agents, and scripts can generate language queries directly

A declarative query interface is machine-friendly.
This matters for future directions such as:

- automated explanation exploration,
- benchmark generation,
- agent-assisted analysis,
- reproducible pipelines.

### 6. High-dimensional option spaces become unmanageable in GUIs

When the task combines:

- target scope,
- layer scope,
- constraints,
- approximation,
- result comparison,
- and execution reuse,

the GUI tends to become cluttered and hard to reason about.
A language scales better with complexity.


## What a TKDE/VLDBJ-Level Contribution Would Need to Claim

To become publishable at that level, the paper should likely make a contribution statement closer to one of the following:

### Option A: General declarative explanation analytics engine

Claim:

- A unified declarative language for querying, comparing, and summarizing GNN explanations across multiple explainers and workloads.

Needs:

- backend abstraction
- formal semantics
- optimizer
- broad experiments

### Option B: Query optimizer for explanation workloads

Claim:

- A workload-aware optimizer that compiles declarative explanation queries into efficient execution plans using routing, approximation, cache reuse, and set-sharing.

Needs:

- strong planning model
- cost model
- ablation-heavy systems evaluation

### Option C: Declarative system for layered explanation analytics

Claim:

- A language and execution engine specialized for multi-layer explanation exploration, comparison, and pattern mining in GNNs.

Needs:

- strong layer-oriented abstractions
- formal semantics for layer-scoped queries
- rich layer-centric workloads


## Priority Checklist: What to Build Next

If the goal is to maximize the chance of a strong paper, the highest-priority work items are below.

### Priority 1: Formalize the query language

Build:

- grammar
- type rules
- well-formedness checks
- result schema

Expected outcome:

- the language becomes a research object, not just a parser implementation.

### Priority 2: Separate logical query intent from physical execution

Build:

- logical operators
- physical operators
- logical-to-physical compilation pipeline

Expected outcome:

- the paper can clearly articulate declarative semantics and execution independence.

### Priority 3: Turn `auto_route` into a real optimizer

Build:

- cost features
- routing rules
- plan selection
- cache-aware decisions
- approximation planner

Expected outcome:

- a much stronger systems contribution.

### Priority 4: Strengthen set-level and compositional querying

Build:

- named queries
- query chaining
- result reuse
- aggregation primitives

Expected outcome:

- the language feels like a query language, not a command launcher.

### Priority 5: Add cross-backend or cross-task generality

Build either:

- multi-explainer support,

or:

- a very strong, clearly justified specialization argument.

Expected outcome:

- better defense against “this is only a wrapper around your own algorithm”.

### Priority 6: Design a systems-grade evaluation suite

Build:

- benchmark workload set
- exact vs approximate evaluation
- routed vs naive evaluation
- cache vs no-cache evaluation
- language vs GUI/API workflow comparison

Expected outcome:

- measurable evidence of systems value.


## Publishability Matrix

The following matrix is designed to help decide when the work is ready to submit.

| Requirement | Current | Needed before submission |
|---|---|---|
| Problem statement | Partial | Sharpen and narrow |
| Formal query grammar | No | Yes |
| Formal semantics | No | Yes |
| Logical/physical separation | Weak | Yes |
| Optimizer | Weak | Stronger |
| Cost model | No | Yes or strong heuristic validation |
| Composition | No | Preferably yes |
| Result algebra | Weak | Stronger |
| Approximation semantics | Weak | Yes |
| Theory/propositions | No | Preferably some |
| Multi-backend support | Weak | Preferably yes |
| Systems evaluation | Weak | Yes |
| Reproducibility artifact | Partial | Stronger |
| GUI-vs-language justification | Implicit | Explicit |


## Suggested Paper Structure

If this project is developed further, a strong paper outline could look like this:

1. Introduction
2. Motivation and workload analysis
3. Data model and query language
4. Formal semantics
5. Query planning and optimization
6. Execution engine
7. Approximate and cache-aware execution
8. Experimental evaluation
9. Related work
10. Conclusion

The most important shift is this:

- do not frame the paper as “we added five DSL features”,
- frame it as “we define a declarative query model and optimized execution engine for explanation workloads”.


## Final Bottom-Line Judgment

### Current position

The current SliceGX declarative system is a promising prototype with several nontrivial ideas:

- domain-facing query syntax
- execution routing
- set-level operations
- approximation support
- cache-aware reuse

### Current limitation

It is not yet strong enough, in its present form, to satisfy the expectations of a top-tier declarative systems paper because it still lacks:

- formal language definition
- declarative semantics
- optimizer depth
- cost modeling
- compositionality
- strong evaluation

### Most realistic path forward

The best route to `TKDE/VLDBJ` quality is not to keep adding isolated query keywords.
The best route is to elevate the work into a full systems story:

- formal query model,
- optimizer-backed execution,
- compositional explanation analytics,
- and rigorous benchmark-driven evidence.

