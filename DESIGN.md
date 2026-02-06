# Synthex Design Document

**Memory architectures that evolve with your agent**

---

## Part I: Vision

### 1.1 The Three Hypotheses

Synthex is built on three core beliefs about agent memory:

#### Hypothesis 1: No One-Size-Fits-All Memory Architecture

A customer service bot, a therapy companion, and a coding assistant all need fundamentally different memory structures. The existence of Mem0, Letta, Zep, and LangMem as co-existing solutions is itself evidence — if there were a "correct" architecture, the market would have converged.

#### Hypothesis 2: Memory Needs Evolve Over an Application's Lifecycle

This has a weak and strong version:

**Weak (obviously true):** As data accumulates, you need summarization and hierarchy or the context window drowns. This is a scaling problem.

**Strong (the real bet):** The *type* of cognition changes qualitatively as the agent-user relationship matures:

- **Early:** Episodic memory dominates — "what did we discuss Tuesday?"
- **Later:** Semantic memory emerges — "this person values directness"
- **Eventually:** Procedural memory crystallizes — "when this user does X, respond with Y"

These aren't different amounts of the same thing — they're structurally different architectures. The memory system that works at 10 conversations may fail at 1,000.

#### Hypothesis 3: Memory Architecture Is a Runtime Concern

You don't pick an architecture on day one and leave it. You need to evolve the pipeline as you learn what the agent actually needs. The tool that lets you discover the right architecture fastest wins.

### 1.2 The Vision

**Synthex is a memory workbench, not a memory system.**

The pitch isn't "our pipeline is better than RAG." It's: "Nobody knows what the right memory architecture is for your domain yet — including you. Synthex lets you find out."

What if you could:

- **Define memory architectures in Python** — composable, testable, versionable
- **Branch and experiment** — try different prompts or processing strategies in isolation
- **Measure and compare** — run benchmarks against different architectures
- **Evolve without data loss** — change the architecture, keep the source data
- **Track full provenance** — trace any memory back to its origin

### 1.3 The Killer Feature: Architecture Migration

The under-articulated superpower: **architecture migration without data loss.**

If you're on Letta and it's not working, your options today are:
1. Rip it out and start over
2. Write custom glue code
3. Cope

Synthex gives you: edit the pipeline definition, run it again, and your same raw data produces a completely different memory structure. The data doesn't move — the *lens* changes.

**The pitch to engineering teams:** "You're not locked into a memory architecture decision on day one. Start simple, evolve as you learn, never lose data in the transition."

### 1.4 Tagline

**"Memory architectures that evolve with your agent"**

The name "Synthex" comes from **synthesis** — the core action of transforming raw information into processed understanding. In chip design, synthesis turns high-level descriptions into gate-level implementations. Synthex does the same for agent memory.

---

## Part II: Conceptual Grounding

### 2.1 Naming Journey

We explored terminology from multiple domains to find the right conceptual frame:

#### Cognitive Science

| Term | Origin | Relevance |
|------|--------|-----------|
| **Basal cognition** | Michael Levin | Cognition at cellular level, scale-free — memory primitives at any granularity |
| **Merkwelt/Umwelt** | Jakob von Uexküll | Organism's perception-world — each agent has its own memory world |
| **Enactive cognition** | Varela, Thompson, Rosch | Cognition through action/interaction — memory shapes future actions |

#### Chip Design & Lithography

| Term | Meaning | Our Equivalent |
|------|---------|----------------|
| **Schematic** | Circuit diagram | Pipeline definition |
| **Synthesis** | HDL → gates | Pipeline → execution plan |
| **Cell/Standard Cell** | Reusable primitives | Step types (transform, fold, etc.) |
| **Fab/Foundry** | Where chips are made | The execution engine |

**Key insight:** Agent memory architectures are like neural circuits — you wire together primitives into a schematic, synthesize it, and fab it.

### 2.2 Design Inspirations

#### dbt (data build tool)

**What we take:**
- DAG-based processing with explicit dependencies
- Incremental materialization
- Layered architecture (staging → marts)

#### CDK vs. CloudFormation

CloudFormation (YAML/JSON) came first, but CDK (imperative code that generates infrastructure) won with developers. Real infrastructure is conditional, looped, parameterized, and composed — all things painful in config and natural in code.

**Key decision:** Python-first, not config-first. Experimentation is a code activity, not a config activity.

#### lakeFS / DVC

**What we take:**
- Branching for experiments
- Copy-on-write semantics
- Reproducible pipelines

### 2.3 Competitive Landscape

| Product | Approach | Strength | Weakness |
|---------|----------|----------|----------|
| **Mem0** | Simple API: add/search | Dead simple, AWS-selected | No processing, no provenance |
| **Letta/MemGPT** | Agent self-manages memory | Dynamic, autonomous | Opinionated hierarchy |
| **Zep (Graphiti)** | Temporal knowledge graph | Excellent temporal reasoning | Complex, graph-specific |
| **LangMem** | Semantic/Procedural/Episodic | Clear taxonomy | Fixed categories |

**Synthex positioning:**

We don't compete on simplicity (Mem0 wins) or temporal reasoning (Zep wins). We compete on **flexibility**, **provenance**, and **evolvability**:

- Define any architecture you can imagine
- Trace any memory back to its source
- Change the architecture without losing data
- Measure which architecture works best

### 2.4 Red Team Critiques (Honest Assessment)

#### "Over-engineered" (Mem0's perspective)

**Validity:** Completely fair. More concepts to learn.

**Counter:** Different use cases. Mem0 is for simple session memory. Synthex is for bulk import, custom pipelines, provenance requirements, and architecture experimentation.

#### "Branching is academic"

**Validity:** Most individual developers won't branch.

**Counter:** Enterprise and research users will. And branching enables the migration story — the sharpest differentiator.

#### "Just RAG with extra steps"

**Validity:** The core is embed-and-retrieve.

**Counter:** The "extra steps" are the value. Summarization reduces noise, aggregation creates hierarchy, folding builds world models, provenance enables debugging. Raw RAG on 1,000 conversations is worse than hierarchical processed memory.

### 2.5 Market Timing

The "no one has converged" window is real right now — the agent memory space is in active churn. But if the ecosystem converges on 2-3 standard patterns (like web dev converged on REST + Postgres), the workbench value shrinks.

Capture winning patterns as templates before that happens.

---

## Part III: Domain Model

### 3.1 Core Entities

```
┌─────────────────────────────────────────────────────────────────┐
│  PIPELINE                                                        │
│  A named container for an agent's memory processing.            │
│                                                                  │
│  Fields:                                                         │
│  - name: string (unique identifier)                              │
│  - agent: string (who this memory belongs to)                    │
│  - description: string (optional)                                │
│  - created_at: timestamp                                         │
└─────────────────────────────────────────────────────────────────┘
         │
         ├─────────────────┬─────────────────┬─────────────────┐
         ▼                 ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  SOURCE         │ │  STEP           │ │  OUTPUT         │ │  BRANCH         │
│                 │ │                 │ │                 │ │                 │
│  Input that     │ │  Processing     │ │  Queryable      │ │  Variant for    │
│  brings data    │ │  operation      │ │  projection     │ │  experiments    │
│  into pipeline  │ │                 │ │                 │ │                 │
│                 │ │  Types:         │ │  Types:         │ │                 │
│  Types:         │ │  - transform    │ │  - projection   │ │                 │
│  - file         │ │  - aggregate    │ │  - search       │ │                 │
│  - api          │ │  - fold         │ │                 │ │                 │
│  - stream       │ │  - merge        │ │                 │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘
```

### 3.2 Runtime Entities

```
┌─────────────────────────────────────────────────────────────────┐
│  RECORD                                                          │
│  A single piece of content produced by the system.              │
│                                                                  │
│  Fields:                                                         │
│  - id: uuid (globally unique)                                    │
│  - content: string (the actual text/data)                        │
│  - step: string (which step produced this)                       │
│  - branch: string (which branch this belongs to)                 │
│  - sources: list[uuid] (input record IDs — provenance)           │
│  - materialization_key: string (see below)                       │
│  - run_id: uuid (which run produced this)                        │
│  - created_at: timestamp                                         │
│  - metadata: { ... } (see reserved keys)                         │
│  - audit: {                                                      │
│      prompt_template_hash: string,                               │
│      rendered_prompt_hash: string,                               │
│      model: string,                                              │
│      temperature: float,                                         │
│      raw_response: string,                                       │
│    }                                                             │
│  - embedding: vector (optional)                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  RUN                                                             │
│  An execution of one or more steps.                             │
│                                                                  │
│  Fields:                                                         │
│  - id: uuid                                                      │
│  - pipeline: string                                              │
│  - branch: string                                                │
│  - steps: list[string]                                           │
│  - status: pending | running | completed | failed                │
│  - started_at / completed_at: timestamp                          │
│  - stats: { input, output, skipped, errors, tokens, duration }   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Materialization Keys (Incremental Processing)

The original Cursor design ("last_processed_id") is broken — UUIDs have no natural ordering, and time-ordered IDs break on late-arriving data and backfills.

**Solution:** Materialization keys + step version hashing.

Every output record has a **materialization key**:
```
(branch, step_name, step_version_hash, input_fingerprints, group_key)
```

Where:
- `step_version_hash` = hash(step_type + config + prompt_template + model)
- `input_fingerprints` = hash of source record content(s)
- `group_key` = period for aggregates, sequence for folds

**Incremental semantics:** If the materialization key exists, skip. If it doesn't, process.

**Why this works:**
- Changing a prompt → step_version_hash changes → automatic reprocess
- Late-arriving data → new input_fingerprint → gets processed
- No `--full` flag needed for prompt changes

### 3.4 Reserved Metadata Keys

```python
# Time
meta.time.created_at      # source timestamp
meta.time.period          # for aggregates: "2024-03"

# Chat context
meta.chat.conversation_id
meta.chat.message_id
meta.chat.author          # "user" | "assistant"

# Source tracking
meta.source.type          # "chatgpt-export" | "claude-export"
meta.source.path          # original file path

# Step tracking
meta.step.version_hash    # step version that produced this
meta.step.sequence        # for folds: position in sequence
```

User-defined metadata goes in `meta.custom.*` to avoid collisions.

### 3.5 Processing Types

| Type | Input | Output | Description |
|------|-------|--------|-------------|
| **transform** | 1 record | 1 record | Map operation (summarize, extract, enrich) |
| **aggregate** | N records | 1 record | Group by period/key, then reduce |
| **fold** | N records (ordered) | 1 record | Sequential accumulation with state |
| **merge** | N records | N records | Combine sources, deduplicate |

#### Transform (1:1)

The workhorse. Strictly one record in, one record out.

**V1 decision:** No `batch_size`. Batching transforms multiple inputs into a parsing problem — if the LLM doesn't return aligned output, you get silent corruption. Keep it simple.

#### Aggregate (N:1 by key)

Group records by a key (typically time period), reduce to one record.

#### Fold (N:1 sequential)

Sequential processing with accumulated state. The key difference from aggregate: **order matters**, and each step sees previous state.

**Open design questions (see Part X):**
- Backfill behavior when inserting records in the middle
- State growth and truncation strategy
- Error recovery mid-fold

#### Merge (N:N)

Combine records from multiple sources with deduplication.

### 3.6 Determinism and Audit

LLM calls are non-deterministic. We pick a lane: **audit determinism**.

Every record stores:
- `prompt_template_hash` — which template version
- `rendered_prompt_hash` — the actual prompt sent
- `model`, `temperature` — model parameters
- `raw_response` — what the LLM returned

You can always explain *why* a record exists (inputs + prompt + response), even if you can't reproduce it exactly.

**Future:** `pipeline.run(replay=True)` reuses cached responses for free rebuilds.

---

## Part IV: Python API

### 4.1 Design Decision: Python-First

TOML configuration contradicts the core thesis. TOML says "I know what I want." The whole point of Synthex is "you don't know the right architecture yet."

The CDK vs. CloudFormation parallel is exact: declarative config is painful for things that are conditional, looped, parameterized, and composed — all natural in code.

The target user is already writing Python. A Python API is:
- Just as readable for simple cases
- Far more powerful for experimentation
- One less layer to implement
- Composable with the rest of their agent code

### 4.2 Pipeline Definition

```python
from synthex import Pipeline

# Create pipeline
pipeline = Pipeline("personal-memory", agent="mark")

# Sources — where data comes from
pipeline.source("chatgpt",
    file="~/exports/chatgpt.json",
    format="chatgpt-export"
)
pipeline.source("claude",
    file="~/exports/claude.json",
    format="claude-export"
)

# Steps — processing operations
pipeline.merge("unified", from_=["chatgpt", "claude"])

pipeline.transform("transcripts",
    from_="unified",
    prompt=extract_transcript
)

pipeline.transform("summaries",
    from_="transcripts",
    prompt=summarize_conversation
)

pipeline.aggregate("monthly",
    from_="summaries",
    period="month",
    prompt=monthly_reflection
)

pipeline.fold("world-model",
    from_="monthly",
    prompt=world_model_prompt
)

# Outputs — queryable projections
pipeline.output("context",
    from_="world-model",
    surface="projection"
)

pipeline.output("search",
    from_=["summaries", "monthly"],
    surface="search"
)
```

### 4.3 Prompt Functions

Prompts are Python functions, not template files:

```python
def summarize_conversation(record: Record) -> str:
    return f"""
Summarize this conversation in 2-3 sentences.

Include:
- Main topic or question
- Key conclusions or decisions
- Any action items mentioned

## Conversation
{record.content}

## Summary
"""

def monthly_reflection(records: list[Record], period: str) -> str:
    summaries = "\n".join(f"- {r.content}" for r in records)
    return f"""
Reflect on this month's conversations.

Month: {period}
Conversations: {len(records)}

## Summaries
{summaries}

## Reflection
"""

def world_model_prompt(record: Record, state: str) -> str:
    return f"""
Update your understanding based on new information.

Current understanding:
{state}

New reflection ({record.metadata.get('period', 'unknown')}):
{record.content}

## Updated Understanding
"""
```

### 4.4 Execution

```python
# Run full pipeline
pipeline.run()

# Run specific step (dependencies run automatically)
pipeline.run("summaries")

# Run from a step onwards
pipeline.run(from_="transcripts")

# Force full reprocess (ignores materialization keys)
pipeline.run(full=True)

# Dry run — show what would execute
plan = pipeline.plan()
print(plan.steps)      # ['chatgpt', 'claude', 'unified', ...]
print(plan.estimates)  # {'tokens': 1_200_000, 'cost': 1.50}
```

### 4.5 Branching

```python
# Create branch for experimentation
branch = pipeline.branch("better-summaries")

# Modify configuration on branch
branch.transform("summaries",
    from_="transcripts",
    prompt=improved_summarize,  # different prompt
    model="gpt-4o"              # different model
)

# Run on branch
branch.run("summaries", full=True)

# Compare with main
diff = branch.diff()
print(diff.changed_steps)  # ['summaries', 'monthly', 'world-model']

# Evaluate both
main_score = pipeline.eval("locomo")
branch_score = branch.eval("locomo")

# If better, promote
if branch_score > main_score:
    branch.promote()  # merge to main
```

### 4.6 Querying

```python
# Surface search — pick your altitude
results = pipeline.search("Rust", step="monthly")

# Drill down — follow provenance
for hit in results:
    print(hit.content)
    print(hit.sources())        # conversation summaries
    print(hit.leaves())         # raw transcripts

# Leaf search — straight to bottom
raw = pipeline.search("FastAPI", step="transcripts", exact=True)

# Lineage — full provenance chain
record = pipeline.get("record-uuid")
lineage = record.lineage()      # tree of sources
```

### 4.7 The Hello World Experience

Five minutes from export to searchable memory:

```bash
# One command to initialize and import
synthex init personal-memory --from ~/exports/chatgpt.json

# Process everything
synthex run

# Search
synthex search "that rust conversation"
```

Behind the scenes, `synthex init --from` creates a sensible default pipeline.

### 4.8 TOML as Export Format

TOML isn't the primary interface, but pipelines can serialize for sharing:

```python
# Export for documentation/version control
pipeline.to_toml("pipeline.toml")

# Import (creates Python equivalent)
pipeline = Pipeline.from_toml("pipeline.toml")
```

---

## Part V: CLI Design

### 5.1 Quick Start

```bash
# Fastest path: import + run + search
synthex init my-memory --from ~/exports/chatgpt.json
synthex run
synthex search "python projects"
```

### 5.2 Full Command Reference

```bash
# ============================================
# PIPELINE LIFECYCLE
# ============================================

synthex init <name>                    # Create new pipeline
synthex init <name> --from <file>      # Create with source
synthex status                         # Show pipeline status
synthex plan                           # Dry-run, show estimates

# ============================================
# EXECUTION
# ============================================

synthex run                            # Run full pipeline
synthex run <step>                     # Run specific step
synthex run --from <step>              # Run from step onwards
synthex run --full                     # Ignore materialization, reprocess all

# ============================================
# BRANCHING
# ============================================

synthex branch list
synthex branch create <name>
synthex branch switch <name>
synthex branch diff <name>
synthex branch promote <name>
synthex branch delete <name>

# ============================================
# QUERY
# ============================================

synthex search <query>                 # Search across outputs
synthex search <query> --step <step>   # Search specific step
synthex lineage <record-id>            # Show provenance tree
synthex get <record-id>                # Get single record

# ============================================
# EVALUATION
# ============================================

synthex eval <benchmark>               # Run benchmark
synthex eval locomo --compare mem0     # Compare against baseline
synthex eval report                    # Generate report

# ============================================
# STATUS
# ============================================

synthex stats                          # Record counts by step
synthex runs                           # List recent runs
synthex runs <run-id>                  # Run details
```

---

## Part VI: Output Semantics

### 6.1 Two Output Types

#### Projection Output

A single record (e.g., the latest world-model) formatted and pushed somewhere.

```python
pipeline.output("context",
    from_="world-model",
    surface="projection",
    on_update=webhook_url  # optional: trigger on change
)
```

Use cases:
- Agent system prompt injection
- Export to file
- Webhook notification

#### Search Output

A queryable index over one or more steps' records.

```python
pipeline.output("search",
    from_=["summaries", "monthly"],
    surface="search"
)
```

### 6.2 DAG-Aware Query Model

The search system is DAG-aware, supporting three query modes:

#### Surface Search

Search within a specific step's outputs at a chosen level of abstraction.

```python
# "Show me monthly summaries mentioning Rust"
results = pipeline.search("Rust", step="monthly")
```

The user picks their altitude in the processing hierarchy.

#### Provenance Drill-Down

From any search hit, follow the DAG downward through sources.

```python
for hit in results:
    # This monthly summary mentions Rust
    sources = hit.sources()     # → conversation summaries
    leaves = hit.leaves()       # → raw transcripts
```

The drill-down path is determined by the pipeline structure, not hardcoded.

#### Leaf Search

Skip the hierarchy, search raw source records directly.

```python
# Find every transcript containing literal "FastAPI"
raw = pipeline.search("FastAPI", step="transcripts", exact=True)
```

Full-text search at the bottom of the DAG.

**This is novel.** Mem0 gives flat search. Zep gives temporal search. Nobody gives "search at an abstraction level, then drill through the processing hierarchy." And it falls out naturally from the provenance model.

---

## Part VII: Implementation Phases

### Revised Phasing

Given the "experimentable memory" thesis, evaluation and branching move up:

### Phase 1: Foundation + Processing Engine (Week 1-4)

**Goal:** Prove the primitives work.

- [ ] Project structure
  ```
  src/synthex/
  ├── __init__.py
  ├── cli.py           # Click CLI
  ├── pipeline.py      # Pipeline class
  ├── steps.py         # Step implementations
  ├── models.py        # SQLAlchemy models
  ├── db.py            # Database engine
  └── prompts.py       # Prompt utilities
  ```
- [ ] Pipeline class with Python API
- [ ] Step types: transform, aggregate, fold, merge
- [ ] Source importers: ChatGPT, Claude exports
- [ ] SQLite storage with materialization keys
- [ ] Record creation with provenance
- [ ] Run tracking and stats
- [ ] Basic CLI: init, run, status

**Deliverable:** Can define pipeline in Python, run it, query results

### Phase 2: Eval Harness (Week 5-6)

**Goal:** Prove you can measure architecture quality.

- [ ] Benchmark runner framework
- [ ] LoCoMo integration
- [ ] LongMemEval integration
- [ ] Memory freshness metric
- [ ] Comparison mode (vs Mem0, vs raw RAG)
- [ ] `synthex eval` command
- [ ] Report generation

**Rationale:** If the pitch is "find the right architecture through trial and error," then eval is load-bearing. The "just RAG with extra steps" critique gets answered by data.

**Deliverable:** `synthex eval locomo --compare mem0` produces comparison table

### Phase 3: Branching (Week 7-8)

**Goal:** Prove you can cheaply compare architectures.

- [ ] Branch model and storage
- [ ] Copy-on-write semantics
- [ ] Branch create/switch/delete
- [ ] Branch diff (compare records)
- [ ] Branch promote (merge to main via materialization keys)
- [ ] Eval on branches

**Rationale:** Branching enables the migration story — the sharpest differentiator. Cheap A/B comparison is what nobody else offers.

**Deliverable:** Can branch, modify prompts, run, compare eval scores, promote

### Phase 4: Query + Polish (Week 9-10)

**Goal:** Complete the MVP experience.

- [ ] DAG-aware search (surface, drill-down, leaf)
- [ ] Lineage visualization
- [ ] Output surfaces (projection, search)
- [ ] `synthex plan` with estimates
- [ ] Hello World experience (`init --from`)
- [ ] Error handling and progress bars
- [ ] Documentation

**Deliverable:** Complete MVP ready for users

---

## Part VIII: MVP Reference Implementation

### 8.1 Use Case: Personal Memory from ChatGPT History

**Scenario:** 2 years of ChatGPT conversations → searchable memory with provenance.

### 8.2 Pipeline Definition

```python
from synthex import Pipeline

pipeline = Pipeline("personal-memory", agent="mark")

# Source
pipeline.source("chatgpt",
    file="~/exports/chatgpt.json",
    format="chatgpt-export"
)

# Processing
pipeline.transform("transcripts",
    from_="chatgpt",
    prompt=extract_transcript
)

pipeline.transform("summaries",
    from_="transcripts",
    prompt=summarize_conversation
)

pipeline.aggregate("monthly",
    from_="summaries",
    period="month",
    prompt=monthly_reflection
)

pipeline.fold("world-model",
    from_="monthly",
    prompt=world_model_prompt
)

# Outputs
pipeline.output("context", from_="world-model", surface="projection")
pipeline.output("search", from_=["summaries", "monthly"], surface="search")
```

### 8.3 Execution

```bash
$ synthex run
Running pipeline 'personal-memory' on branch 'main'...

[1/5] chatgpt ████████████████████ 487/487 (parsed)
[2/5] transcripts ████████████████████ 487/487 (2m 15s)
[3/5] summaries ████████████████████ 487/487 (3m 42s)
[4/5] monthly ████████████████████ 28/28 (45s)
[5/5] world-model ████████████████████ 1/1 (12s)

✓ Completed in 7m 14s
  Records: 1,490 created
  Tokens: 1,247,832 input / 94,521 output
```

### 8.4 Experimentation

```python
# Try a different summarization approach
branch = pipeline.branch("gpt4-summaries")

branch.transform("summaries",
    from_="transcripts",
    prompt=detailed_summary,  # different prompt
    model="gpt-4o"            # different model
)

branch.run(from_="summaries", full=True)

# Compare
main_eval = pipeline.eval("locomo")
branch_eval = branch.eval("locomo")

print(f"Main: {main_eval.precision_at_5:.2f}")
print(f"Branch: {branch_eval.precision_at_5:.2f}")

# If better, promote
if branch_eval.precision_at_5 > main_eval.precision_at_5:
    branch.promote()
```

---

## Part IX: Evaluation Strategy

### 9.1 Standard Benchmarks

#### LoCoMo (Long Context Memory)

- Multi-session conversation memory
- Retrieval accuracy, temporal reasoning, entity tracking
- Metrics: Precision@k, temporal accuracy

#### LongMemEval

- Long-horizon memory evaluation
- Information extraction, summarization quality
- Metrics: ROUGE, factual accuracy

### 9.2 Memory Freshness

Critical for the "evolving architecture" thesis: can the system distinguish yesterday vs. two years ago?

- Surface recent information appropriately
- Don't let old memories drown current context
- Metric: recency-weighted precision

### 9.3 Custom Benchmark: SynthexBench

Existing benchmarks test flat retrieval. They have no concept of hierarchical search or provenance. **Our most differentiating capability has no benchmark.**

#### What SynthexBench Tests

**Multi-altitude retrieval:** Can the system find the right answer at different abstraction levels? Are answers appropriately different at summary vs. raw transcript level?

**Provenance navigation:** Can the system trace hits to specific source records? Is lineage complete and correct?

**Leaf bypass:** Can the system find verbatim phrases in raw transcripts even when summaries abstracted them away?

**Emergent vs. source knowledge:** Some information only exists at higher levels — a monthly reflection might identify a trend ("shifting from web dev to systems programming") that doesn't appear in any single conversation. The benchmark tests whether the system distinguishes:
- Information at a specific altitude (emergent synthesis)
- Information that traces to a source record
- Information that appears differently at different altitudes

This is the killer differentiator. No existing benchmark touches this.

**Status:** Discovery phase. Define once pipeline and query model are working.

### 9.4 Comparison Targets

| System | Configuration |
|--------|---------------|
| **Mem0** | Same data, add() then search() |
| **Raw RAG** | Embed sources directly, no processing |
| **Synthex** | Full pipeline with hierarchy |

### 9.5 Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Precision@5 | Top 5 retrieval accuracy | >0.80 |
| Temporal Accuracy | Correct time-based queries | >0.75 |
| Memory Freshness | Recency-appropriate surfacing | >0.70 |
| Provenance Completeness | Full lineage available | 100% |
| Incremental Speedup | Re-run / full run time | <0.1x |

---

## Part X: Open Design Questions

### 10.1 Fold Primitive Challenges

Fold is the most novel and most dangerous primitive. Unresolved questions:

#### Backfill

What happens when you insert a new month in the middle of the fold sequence?

Options:
1. Re-run entire fold from insertion point (expensive)
2. Mark downstream as stale, rebuild on next access (lazy)
3. Require explicit `--full` for mid-sequence insertions

#### State Growth

The world model prompt passes `{{ state }}` which grows unboundedly.

Options:
1. Truncation strategy (keep last N tokens)
2. Compression step (summarize state periodically)
3. Sliding window (only last N inputs visible)

#### Error Recovery

If step 14 of 28 fails, what happens?

Options:
1. Checkpoint every N steps, resume from checkpoint
2. Mark as failed, require manual intervention
3. Retry with exponential backoff

### 10.2 Semantic Search Strategy

MVP says "SQLite for everything" but search results show similarity scores, implying semantic search. SQLite FTS is not semantic.

**Decision needed:**
- FTS only for v1 (simpler, but less useful)
- sqlite-vss (finicky but possible)
- Require external vector store (Postgres+pgvector, Pinecone)

This cascades into whether eval numbers are meaningful.

---

## Part XI: Security and Privacy

### 11.1 The PII Problem

Bulk importing years of ChatGPT/Claude history means ingesting names, emails, medical conversations, financial details — everything a user ever discussed with an LLM.

Provenance makes this *worse*: sensitive data isn't just stored, it's linked and traceable through lineage.

### 11.2 Required Capabilities

#### PII Detection/Redaction

First-class primitive, not a user-defined transform.

```python
pipeline.transform("transcripts",
    from_="chatgpt",
    prompt=extract_transcript,
    pii="redact"  # or: "detect", "flag"
)
```

#### Encryption at Rest

SQLite with no encryption is a non-starter for enterprise.

Options:
- SQLCipher for SQLite encryption
- Require Postgres for sensitive deployments
- Field-level encryption for content

#### Selective Purge (GDPR)

Delete a source record and cascade through all derived records.

```python
# Delete and cascade
pipeline.purge("record-uuid", cascade=True)

# Preview what would be deleted
cascade = pipeline.purge("record-uuid", dry_run=True)
print(cascade.affected_records)  # all downstream records
```

### 11.3 Design Implications

Not v1-blocking, but data model must support:
- Cascade deletion via provenance links
- Encryption hooks at storage layer
- PII processor as built-in step type

---

## Part XII: Relationship to KP3

Synthex is a more compelling formulation of what KP3 was building toward.

KP3 built: programmable knowledge processing, passages with provenance, semantic search.

Synthex reframes it as: **declarative memory architecture with evolution as a first-class concern**.

The reframing elevates the concept from "a pipeline tool" to "infrastructure for a problem that doesn't have a settled answer yet."

Key insights carried forward:
- Passages → Records
- Refs/Branches → Branch model
- Processors → Step types
- Provenance tracking → Materialization keys + source links

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Pipeline** | Named container for memory processing |
| **Source** | Input that brings data into the system |
| **Step** | Processing operation (transform, aggregate, fold, merge) |
| **Output** | Queryable projection (search or projection) |
| **Record** | Single piece of content with provenance |
| **Run** | One execution of steps |
| **Branch** | Isolated variant for experimentation |
| **Materialization Key** | Unique identifier for incremental processing |
| **Provenance** | Chain from record back to sources |
| **DAG** | Directed Acyclic Graph of step dependencies |

## Appendix B: Why Not Just Use...

### "Why not just use Mem0?"

Use Mem0 if you need simple add/search. Use Synthex if you need bulk import, custom processing, provenance, or architecture experimentation.

### "Why not just use a vector database?"

Vector DBs are storage, not processing. They don't summarize, aggregate, track provenance, or run pipelines. Synthex uses vector DBs as a backend.

### "Why not just use dbt?"

dbt is for SQL transformations on structured data. Synthex is for LLM transformations on unstructured text with provenance tracking.

## Appendix C: Design Decisions Log

### Python over TOML

**Decision:** Primary interface is Python code, not TOML config.

**Rationale:** Experimentation is a code activity. Config contradicts "you don't know the right architecture yet."

### Materialization Keys over Cursors

**Decision:** Replace cursor-as-last-id with materialization keys.

**Rationale:** Cursors break on late-arriving data and prompt changes. Materialization keys handle both automatically.

### Audit Determinism over Reproducibility

**Decision:** Store audit trail rather than guaranteeing reproducible builds.

**Rationale:** LLM outputs are non-deterministic. We can explain why a record exists, even if we can't reproduce it.

### No Batch Transforms in V1

**Decision:** Transform is strictly 1:1, no batch_size.

**Rationale:** Batching creates parsing problems. Keep it simple until structured output is robust.
