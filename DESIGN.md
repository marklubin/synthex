# Synthex Design Document

**Neural schematic primitives for agent memory**

---

## Part I: Vision

### 1.1 The Problem

Agents need memory that persists across sessions. The current landscape of agent memory solutions presents a frustrating dichotomy:

**Too Simple:**
- Mem0 offers `add()` and `search()` - that's it
- No processing pipeline, no provenance tracking
- You can't transform or summarize memories
- AWS selected Mem0, validating the "simple API" approach

**Too Opinionated:**
- Letta/MemGPT prescribes a specific memory hierarchy
- Fixed categories: core memory, archival memory, recall memory
- Agent self-manages, but within rigid constraints
- Can't experiment with alternative architectures

**What's Missing:**
- **Declarative architecture definition** - describe memory structure in config, not code
- **Full provenance** - trace any memory back to its source
- **Experimentation** - branch and compare different architectures
- **Bulk import** - process years of conversation history
- **Custom processing** - your prompts, your hierarchy, your rules

### 1.2 The Vision

**"dbt/Terraform for Cognitive Architecture"**

What if you could:

- **Define memory architectures in TOML** (like Terraform resources)
  ```toml
  [steps.world-model]
  type = "fold"
  from = ["monthly-reflections"]
  prompt = "world-model"
  ```

- **Compose processing pipelines declaratively** (like dbt models)
  - Sources → transforms → aggregations → outputs
  - Dependencies resolved automatically
  - Incremental processing built-in

- **Branch and experiment** (like git)
  - Try a different summarization prompt on a branch
  - Compare results before promoting to main
  - Rollback if experiments fail

- **Track full provenance** (like lakeFS)
  - Any memory traces back to source documents
  - `synthex lineage <record-id>` shows the full chain
  - Debugging and auditing built-in

- **Benchmark architectures** (like ML experiment tracking)
  - Run LoCoMo or LongMemEval against your pipeline
  - Compare Synthex vs Mem0 vs raw RAG
  - Data-driven architecture decisions

**Synthex is assembly language for agent memory** - low-level primitives that can express any cognitive architecture.

### 1.3 Tagline

**"Neural schematic primitives for agent memory"**

The name "Synthex" comes from **synthesis** - the core action of transforming raw information into processed understanding. In chip design, synthesis is the process of turning high-level descriptions (HDL) into gate-level implementations. Synthex does the same for agent memory: turn declarative configs into executable processing pipelines.

---

## Part II: Conceptual Grounding

### 2.1 Naming Journey

We explored terminology from multiple domains to find the right conceptual frame:

#### Cognitive Science / Comparative Cognition

| Term | Origin | Relevance |
|------|--------|-----------|
| **Basal cognition** | Michael Levin | Cognition at cellular level, scale-free - memory primitives at any granularity |
| **Merkwelt/Umwelt** | Jakob von Uexküll | Organism's perception-world - each agent has its own memory world |
| **Enactive cognition** | Varela, Thompson, Rosch | Cognition through action/interaction - memory shapes future actions |
| **4E cognition** | Various | Embodied, embedded, enacted, extended - memory extends the agent |
| **Morphic fields** | Sheldrake | Form and pattern memory - memory as structure, not just content |

#### Neuroscience

| Term | Meaning | Relevance |
|------|---------|-----------|
| **Engram** | Physical trace of memory in the brain | Our Records are engrams |
| **Neural circuit** | Connected processing pathways | Our Step DAG is a circuit |
| **Consolidation** | Short-term → long-term memory | Our fold/aggregate steps consolidate |
| **Retrieval** | Accessing stored memories | Our Output surfaces enable retrieval |

#### Chip Design & Lithography

| Term | Meaning | Our Equivalent |
|------|---------|----------------|
| **Schematic** | Circuit diagram | synthex.toml config |
| **Synthesis** | HDL → gates | Config → execution plan |
| **Cell/Standard Cell** | Reusable primitives | Step types (transform, fold, etc.) |
| **Fab/Foundry** | Where chips are made | The execution engine |
| **Tapeout** | Shipping to production | `synthex apply` |
| **Mask** | Pattern template | Prompt templates |

**Key insight:** Agent memory architectures are like neural circuits - you wire together primitives (gates, transforms) into a schematic (config), synthesize it (compile to execution plan), and fab it (execute the pipeline).

### 2.2 Design Inspirations

#### dbt (data build tool)

dbt revolutionized analytics engineering with declarative SQL models:

```sql
-- models/marts/orders.sql
SELECT * FROM {{ ref('stg_orders') }}
WHERE status = 'completed'
```

**What we take from dbt:**
- **DAG-based processing** - dependencies via `ref()` / `from`
- **Declarative definitions** - describe what, not how
- **Incremental materialization** - process only new/changed data
- **Layered architecture** - staging → intermediate → marts

#### Terraform / CloudFormation

Infrastructure-as-code for cloud resources:

```hcl
resource "aws_instance" "web" {
  ami           = "ami-12345"
  instance_type = "t2.micro"
}
```

**What we take from Terraform:**
- **Desired state configuration** - declare target, system figures out how
- **Plan before apply** - see what will change
- **Drift detection** - detect manual changes

#### lakeFS / DVC

Git-like versioning for data:

```bash
lakefs branch create experiment-1
lakefs commit -m "Try new embeddings"
lakefs merge experiment-1 main
```

**What we take from lakeFS:**
- **Branching for experiments** - try different approaches in isolation
- **Copy-on-write semantics** - efficient forking
- **Reproducible pipelines** - same config = same results

#### DSPy

Declarative prompts as optimizable programs:

```python
class RAG(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=3)
        self.generate = dspy.ChainOfThought("context, question -> answer")
```

**What we take from DSPy:**
- **Prompts as configurable components** - not hardcoded strings
- **Composable modules** - build complex from simple

### 2.3 Competitive Landscape

| Product | Approach | Strength | Weakness |
|---------|----------|----------|----------|
| **Mem0** | Simple API: add/search/get_all | Dead simple, AWS-selected | No processing, no provenance, no hierarchy |
| **Letta/MemGPT** | Agent self-manages memory | Dynamic, autonomous, research-backed | Opinionated hierarchy, hard to customize |
| **Zep (Graphiti)** | Temporal knowledge graph | Excellent temporal reasoning | Complex, graph-specific, steep learning curve |
| **LangMem** | Semantic/Procedural/Episodic types | Clear type taxonomy | Fixed categories, limited flexibility |
| **Cognee** | Knowledge graph + embeddings | Rich knowledge representation | Heavyweight, complex setup |

**Synthex positioning:**

We don't compete on simplicity - Mem0 wins that handily. We don't compete on temporal reasoning - Zep/Graphiti excels there. We don't compete on autonomous self-management - Letta owns that space.

We compete on **flexibility** and **provenance**:
- Define any architecture you can imagine
- Trace any memory back to its source
- Experiment with branches
- Process bulk historical data

**Target users:**
- Researchers exploring memory architectures
- Enterprise teams with specific compliance/provenance needs
- Power users with years of conversation history to process
- Anyone who's hit the limits of Mem0's simplicity

### 2.4 Red Team Critiques (Honest Assessment)

We stress-tested our concept against realistic objections:

#### "Over-engineered" (Mem0's perspective)

**The critique:** "Most developers just need add/search. Why learn Sources, Steps, Outputs, Records, Branches? That's five concepts vs two."

**Validity:** Completely fair. Synthex has a steeper learning curve.

**Counter-argument:** Different use cases. Mem0 is for simple session memory. Synthex is for:
- Bulk import of historical data
- Custom processing pipelines
- Provenance requirements
- Architecture experimentation

If you just need add/search, use Mem0. We're not competing for that use case.

#### "Branching is academic" (Letta's perspective)

**The critique:** "In practice, 99% of users will never create a branch. It's complexity for researchers, not practitioners."

**Validity:** Probably true for individual developers.

**Counter-argument:** Enterprise and research users will use it. Even if branching is 5% of users, it's 95% of some users' workflows. And it's optional - you can ignore branching entirely.

#### "No real-time updates" (LangMem's perspective)

**The critique:** "Batch processing is 2020. Modern agents need real-time memory updates as conversations happen."

**Validity:** True limitation of V1.

**Counter-argument:** Scope management. V1 focuses on bulk import and batch processing. Real-time is a future extension. Many use cases (historical import, nightly processing) don't need real-time.

#### "Just RAG with extra steps" (General skepticism)

**The critique:** "At the end of the day, you're chunking, embedding, and retrieving. The 'processing pipeline' is lipstick on RAG."

**Validity:** The core is indeed embed-and-retrieve.

**Counter-argument:** The "extra steps" are the value:
- Summarization reduces noise
- Aggregation creates hierarchy
- Folding builds world models
- Provenance enables debugging

Raw RAG on 1000 conversations is worse than hierarchical processed memory.

---

## Part III: Domain Model

### 3.1 Core Entities

```
┌─────────────────────────────────────────────────────────────────┐
│  PROJECT                                                         │
│  A named container for an agent's memory processing.            │
│                                                                  │
│  Fields:                                                         │
│  - name: string (unique identifier)                              │
│  - agent: string (who this memory belongs to)                    │
│  - description: string (optional human description)              │
│  - created_at: timestamp                                         │
│  - config_path: string (path to synthex.toml)                    │
└─────────────────────────────────────────────────────────────────┘
         │
         ├─────────────────┬─────────────────┬─────────────────┐
         ▼                 ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  SOURCE         │ │  STEP           │ │  OUTPUT         │ │  BRANCH         │
│                 │ │                 │ │                 │ │                 │
│  Input that     │ │  Processing     │ │  Queryable      │ │  Variant for    │
│  brings data    │ │  operation      │ │  projection     │ │  experiments    │
│  into project   │ │  that transforms│ │  of processed   │ │                 │
│                 │ │  data           │ │  data           │ │                 │
│  Fields:        │ │                 │ │                 │ │  Fields:        │
│  - name: string │ │  Fields:        │ │  Fields:        │ │  - name: string │
│  - type: enum   │ │  - name: string │ │  - name: string │ │  - parent: str  │
│  - format: str  │ │  - type: enum   │ │  - from: list   │ │  - created_at   │
│  - config: {}   │ │  - from: list   │ │  - surface: str │ │  - status: enum │
│                 │ │  - prompt: str  │ │  - config: {}   │ │                 │
│  Types:         │ │  - config: {}   │ │                 │ │  Status:        │
│  - file         │ │                 │ │  Surfaces:      │ │  - active       │
│  - api          │ │  Types:         │ │  - search       │ │  - merged       │
│  - stream       │ │  - transform    │ │  - context      │ │  - abandoned    │
│                 │ │  - aggregate    │ │  - export       │ │                 │
│                 │ │  - fold         │ │                 │ │                 │
│                 │ │  - merge        │ │                 │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘
```

### 3.2 Runtime Entities

```
┌─────────────────────────────────────────────────────────────────┐
│  RECORD                                                          │
│  A single piece of content produced by the system.              │
│                                                                  │
│  This is the fundamental unit of memory. Every piece of data    │
│  in Synthex is a Record - from raw source imports to final      │
│  world model outputs.                                           │
│                                                                  │
│  Fields:                                                         │
│  - id: uuid (globally unique)                                    │
│  - content: string (the actual text/data)                        │
│  - step: string (which step produced this)                       │
│  - branch: string (which branch this belongs to)                 │
│  - sources: list[uuid] (input record IDs - provenance chain)     │
│  - run_id: uuid (which run produced this)                        │
│  - created_at: timestamp                                         │
│  - metadata: {                                                   │
│      fingerprint: string,  # content hash for dedup              │
│      period: string,       # for aggregates: "2024-03"           │
│      sequence: int,        # for folds: position in sequence     │
│      ...                   # step-specific metadata              │
│    }                                                             │
│  - embedding: vector (optional, for semantic search)             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  RUN                                                             │
│  An execution of one or more steps.                             │
│                                                                  │
│  Runs are the unit of execution. Each `synthex run` creates     │
│  a new Run record that tracks what was processed.               │
│                                                                  │
│  Fields:                                                         │
│  - id: uuid                                                      │
│  - project: string                                               │
│  - branch: string                                                │
│  - steps: list[string] (which steps were run)                    │
│  - status: enum (pending | running | completed | failed)         │
│  - started_at: timestamp                                         │
│  - completed_at: timestamp (null if still running)               │
│  - error: string (null if successful)                            │
│  - stats: {                                                      │
│      input_records: int,   # records consumed                    │
│      output_records: int,  # records created                     │
│      skipped: int,         # already processed (incremental)     │
│      errors: int,          # processing failures                 │
│      tokens_used: int,     # LLM tokens consumed                 │
│      duration_ms: int,     # wall clock time                     │
│    }                                                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  CURSOR                                                          │
│  Tracks incremental processing state for a step.                │
│                                                                  │
│  Enables efficient re-runs - only process new records.          │
│                                                                  │
│  Fields:                                                         │
│  - step: string                                                  │
│  - branch: string                                                │
│  - last_processed_id: uuid (highest record ID processed)         │
│  - last_run_id: uuid (which run updated this)                    │
│  - updated_at: timestamp                                         │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Processing Types

| Type | Input | Output | Use Case | Example |
|------|-------|--------|----------|---------|
| **transform** | 1 record | 1 record | Map operation | Summarize a conversation |
| **aggregate** | N records | 1 record | Group + reduce | Monthly reflection from daily summaries |
| **fold** | N records (ordered) | 1 record | Sequential accumulation | Build world model from monthly reflections |
| **merge** | N records | N records | Combine sources | Union ChatGPT + Claude exports, dedupe |

#### Transform (1:1)

```
┌────────┐     ┌───────────┐     ┌────────┐
│ Record │ ──▶ │ transform │ ──▶ │ Record │
└────────┘     └───────────┘     └────────┘
```

The workhorse operation. One record in, one record out. Examples:
- Extract clean transcript from raw conversation JSON
- Summarize a conversation
- Extract entities from text
- Translate to another language

#### Aggregate (N:1 by key)

```
┌────────┐
│ Record │ ──┐
└────────┘   │
┌────────┐   │   ┌───────────┐     ┌────────┐
│ Record │ ──┼──▶│ aggregate │ ──▶ │ Record │
└────────┘   │   └───────────┘     └────────┘
┌────────┐   │
│ Record │ ──┘
└────────┘
     (same period/key)
```

Group records by a key (typically time period), then reduce to one record. Examples:
- Weekly summary from daily summaries
- Monthly reflection from all conversations that month
- Quarterly review from monthly reflections

#### Fold (N:1 sequential)

```
┌────────┐     ┌────────┐     ┌────────┐
│ Rec 1  │ ──▶ │ Rec 2  │ ──▶ │ Rec 3  │ ──▶ ...
└────────┘     └────────┘     └────────┘
     │              │              │
     ▼              ▼              ▼
┌────────┐     ┌────────┐     ┌────────┐
│State 1 │ ──▶ │State 2 │ ──▶ │State 3 │ ──▶ Final
└────────┘     └────────┘     └────────┘
```

Sequential processing with accumulated state. The key difference from aggregate: order matters, and each step sees the previous state. Examples:
- Build world model by processing reflections chronologically
- Maintain running entity list
- Evolving personality profile

#### Merge (N:N)

```
┌────────┐              ┌────────┐
│ Src A  │ ──┐      ┌──▶│ Output │
└────────┘   │      │   └────────┘
┌────────┐   │ merge│   ┌────────┐
│ Src A  │ ──┼──────┼──▶│ Output │
└────────┘   │      │   └────────┘
┌────────┐   │      │   ┌────────┐
│ Src B  │ ──┘      └──▶│ Output │
└────────┘              └────────┘
```

Combine records from multiple sources. Handles deduplication and conflict resolution. Examples:
- Union ChatGPT + Claude exports
- Merge memories from multiple sessions
- Combine different data sources

### 3.4 Key Relationships

#### Record → Step

- Every Record is produced by exactly one Step
- A Step can produce many Records
- The `step` field on Record links to the producing Step

#### Record → Sources (Provenance)

- A Record has zero or more source Records
- Source records are the inputs that produced this record
- Provenance is the full chain: Record → sources → sources' sources → ...
- `synthex lineage <record-id>` traverses this graph

```
world-model-001
    └── monthly-2024-03
        ├── summary-0042
        │   └── transcript-0042
        │       └── chatgpt-raw-0042
        ├── summary-0043
        │   └── transcript-0043
        │       └── chatgpt-raw-0043
        └── summary-0044
            └── transcript-0044
                └── chatgpt-raw-0044
```

#### Branch Isolation

- Records belong to exactly one Branch
- Branches are isolated namespaces
- Querying on `main` doesn't see `experiment-1` records
- `branch promote` copies records from experiment → main

#### Copy-on-Write Semantics

- Creating a branch doesn't copy records
- New branch references parent's records
- Modifications create new records on the branch
- Efficient forking even with large datasets

---

## Part IV: Configuration Format

### 4.1 synthex.toml Structure

```toml
# ============================================
# PROJECT METADATA
# ============================================

[project]
name = "mark-memory"
agent = "mark"
description = "Personal memory architecture for my AI assistant"

# ============================================
# SOURCES - where data comes from
# ============================================

[sources.chatgpt]
type = "file"
format = "chatgpt-export"
path = "~/exports/chatgpt.json"
description = "ChatGPT data export (2022-2024)"

[sources.claude]
type = "file"
format = "claude-export"
path = "~/exports/claude.json"
description = "Claude.ai conversation export"

[sources.notes]
type = "file"
format = "markdown"
path = "~/notes/**/*.md"
description = "Personal markdown notes"

# ============================================
# STEPS - processing pipeline
# ============================================

# Step 1: Merge all sources into unified format
[steps.unified]
type = "merge"
from = ["chatgpt", "claude"]
description = "Merge conversation sources, dedupe by content hash"

# Step 2: Extract clean transcripts
[steps.transcripts]
type = "transform"
from = ["unified"]
prompt = "extract-transcript"
description = "Extract clean conversation text from raw exports"

# Step 3: Summarize each conversation
[steps.summaries]
type = "transform"
from = ["transcripts"]
prompt = "summarize-conversation"
description = "One-paragraph summary per conversation"

# Step 4: Monthly aggregation
[steps.monthly]
type = "aggregate"
from = ["summaries"]
period = "month"
prompt = "monthly-reflection"
description = "Reflect on each month's conversations"

# Step 5: Build world model
[steps.world-model]
type = "fold"
from = ["monthly"]
prompt = "world-model"
description = "Accumulated understanding across all time"

# ============================================
# OUTPUTS - queryable projections
# ============================================

[outputs.context]
from = ["world-model"]
surface = "agent-context"
description = "World model formatted for agent system prompt"

[outputs.search]
from = ["summaries", "monthly"]
surface = "semantic-search"
description = "Searchable index of summaries and reflections"

[outputs.timeline]
from = ["summaries"]
surface = "export"
format = "jsonl"
path = "~/exports/memory-timeline.jsonl"
description = "Chronological export for external tools"
```

### 4.2 Source Types

#### File Source

```toml
[sources.chatgpt]
type = "file"
format = "chatgpt-export"  # or: claude-export, markdown, jsonl, csv
path = "~/exports/chatgpt.json"  # supports glob patterns
encoding = "utf-8"  # optional, default utf-8
```

Supported formats:
- `chatgpt-export` - Official ChatGPT JSON export
- `claude-export` - Claude.ai conversation export
- `markdown` - Markdown files (one record per file or per heading)
- `jsonl` - JSON Lines (one record per line)
- `csv` - CSV with configurable content column

#### API Source (Future)

```toml
[sources.slack]
type = "api"
format = "slack"
token = "${SLACK_TOKEN}"  # environment variable
channels = ["general", "random"]
since = "2024-01-01"
```

#### Stream Source (Future)

```toml
[sources.webhook]
type = "stream"
format = "webhook"
endpoint = "/ingest"
auth = "bearer"
```

### 4.3 Step Configuration

#### Transform Step

```toml
[steps.summaries]
type = "transform"
from = ["transcripts"]  # input step(s)
prompt = "summarize-conversation"  # prompt template name

# Optional configuration
[steps.summaries.config]
model = "gpt-4o-mini"  # override default model
max_tokens = 500
temperature = 0.3
batch_size = 10  # process N records per LLM call
retry_count = 3
```

#### Aggregate Step

```toml
[steps.monthly]
type = "aggregate"
from = ["summaries"]
period = "month"  # day, week, month, quarter, year
prompt = "monthly-reflection"

[steps.monthly.config]
group_by = "metadata.date"  # field to extract period from
min_records = 3  # skip periods with fewer records
max_input_tokens = 50000  # truncate if too many records
```

#### Fold Step

```toml
[steps.world-model]
type = "fold"
from = ["monthly"]
prompt = "world-model"

[steps.world-model.config]
order_by = "metadata.period"  # chronological order
initial_state = ""  # starting state (empty by default)
checkpoint_every = 10  # save intermediate state
```

#### Merge Step

```toml
[steps.unified]
type = "merge"
from = ["chatgpt", "claude"]

[steps.unified.config]
dedupe_by = "fingerprint"  # field to check for duplicates
conflict = "prefer_latest"  # or: prefer_first, keep_all
```

### 4.4 Prompt Templates

Prompts are stored in `prompts/` directory:

```
~/synthex-project/
├── synthex.toml
└── prompts/
    ├── extract-transcript.md
    ├── summarize-conversation.md
    ├── monthly-reflection.md
    └── world-model.md
```

Example prompt template:

```markdown
<!-- prompts/summarize-conversation.md -->
# Summarize Conversation

Summarize the following conversation in one concise paragraph.
Focus on: main topics discussed, decisions made, action items.

## Conversation

{{ content }}

## Summary
```

Template variables:
- `{{ content }}` - The record content
- `{{ metadata.* }}` - Record metadata fields
- `{{ sources }}` - List of source record contents (for merge/aggregate)
- `{{ state }}` - Previous state (for fold)

### 4.5 DAG Visualization

The config above produces this DAG:

```
┌──────────┐   ┌──────────┐   ┌──────────┐
│ chatgpt  │   │  claude  │   │  notes   │     SOURCES
└────┬─────┘   └────┬─────┘   └────┬─────┘
     │              │              │
     └──────┬───────┘              │
            ▼                      │
     ┌─────────────┐               │
     │   unified   │  (merge)      │            STEPS
     └──────┬──────┘               │
            ▼                      │
     ┌─────────────┐               │
     │ transcripts │  (transform)  │
     └──────┬──────┘               │
            ▼                      │
     ┌─────────────┐               │
     │  summaries  │  (transform)  │
     └──────┬──────┘               │
            │                      │
     ┌──────┴──────┐               │
     ▼             ▼               │
┌─────────┐  ┌─────────────┐       │
│ monthly │  │   search    │ ◀─────┘
│  (agg)  │  │  (output)   │
└────┬────┘  └─────────────┘
     ▼
┌─────────────┐
│ world-model │  (fold)
└──────┬──────┘
       ▼
┌─────────────┐
│   context   │  (output)                       OUTPUTS
└─────────────┘
```

---

## Part V: CLI Design

### 5.1 Command Structure

```bash
# ============================================
# PROJECT LIFECYCLE
# ============================================

synthex init <name>
# Create new project with template synthex.toml
# Creates: ./synthex.toml, ./prompts/, ./.synthex/

synthex validate
# Check config syntax and DAG validity
# Errors: missing prompts, circular deps, invalid types

synthex plan
# Show what would run (dry-run)
# Output: steps to execute, estimated records, token estimate

synthex apply
# Apply config changes to database
# Creates/updates: project, sources, steps, outputs

# ============================================
# EXECUTION
# ============================================

synthex run <step>
# Run a specific step (and dependencies if needed)
# Example: synthex run summaries

synthex run --all
# Run full pipeline in DAG order
# Respects incremental: only processes new records

synthex run --from <step>
# Run from step onwards (step + all downstream)
# Example: synthex run --from transcripts

synthex run --full
# Ignore cursors, reprocess everything
# Use after prompt changes

# ============================================
# BRANCHING
# ============================================

synthex branch list
# List all branches with status
# Output: name, parent, created, record count

synthex branch create <name>
# Create branch forked from current
# Copy-on-write: no data copied initially

synthex branch switch <name>
# Switch to branch
# All subsequent commands operate on this branch

synthex branch promote <name>
# Merge branch to main
# Copies new/modified records

synthex branch delete <name>
# Delete branch and its unique records
# Confirms if records would be lost

synthex branch diff <name>
# Show differences from parent
# Output: new records, modified prompts

# ============================================
# QUERY
# ============================================

synthex search <query>
# Semantic search across outputs
# Options: --output <name>, --limit N, --threshold 0.7

synthex lineage <record-id>
# Show provenance chain
# Output: tree of record → sources → sources

synthex inspect <step>
# Show step outputs
# Options: --limit N, --format json|table

synthex get <record-id>
# Get single record with full details
# Output: content, metadata, sources, created_at

# ============================================
# STATUS & MONITORING
# ============================================

synthex status
# Show project status
# Output: current branch, pending changes, last run

synthex stats
# Show record counts by step
# Output: step, total, new since last run

synthex runs
# List recent runs
# Output: run_id, status, steps, duration, records

synthex runs <run-id>
# Show run details
# Output: full stats, errors, timing breakdown
```

### 5.2 Example Session

```bash
# Initialize project
$ synthex init personal-memory
Created project 'personal-memory'
  synthex.toml - edit this to configure your pipeline
  prompts/ - add your prompt templates here
  .synthex/ - internal state (add to .gitignore)

# Edit config and prompts...

# Validate configuration
$ synthex validate
✓ Config syntax valid
✓ All prompts found
✓ DAG is acyclic
✓ 2 sources, 5 steps, 2 outputs

# See what would happen
$ synthex plan
Planning run for branch 'main'...

Steps to execute (in order):
  1. chatgpt (source) - 847 conversations
  2. claude (source) - 234 conversations
  3. unified (merge) - ~1081 records
  4. transcripts (transform) - ~1081 records
  5. summaries (transform) - ~1081 records
  6. monthly (aggregate) - ~24 records
  7. world-model (fold) - 1 record

Estimated tokens: ~2.4M input, ~108K output
Estimated cost: ~$3.20 (gpt-4o-mini)

# Apply config to database
$ synthex apply
Applied configuration to database
  Created project: personal-memory
  Created 2 sources, 5 steps, 2 outputs

# Run the full pipeline
$ synthex run --all
Running pipeline on branch 'main'...

[1/7] chatgpt ████████████████████ 847/847
[2/7] claude ████████████████████ 234/234
[3/7] unified ████████████████████ 1081/1081
[4/7] transcripts ████████████████████ 1081/1081
[5/7] summaries ████████████████████ 1081/1081
[6/7] monthly ████████████████████ 24/24
[7/7] world-model ████████████████████ 1/1

Run completed in 12m 34s
  Records created: 3,349
  Tokens used: 2,847,293
  Cost: $3.42

# Search memories
$ synthex search "python web frameworks"
Found 7 results:

1. [0.92] summary-0847 (2024-03-15)
   Discussion about FastAPI vs Flask for new project...

2. [0.89] monthly-2024-03
   March focused on backend architecture decisions...

3. [0.85] summary-0234 (2023-11-02)
   Compared Django, Flask, and FastAPI...

# Check provenance
$ synthex lineage summary-0847
summary-0847 "Discussion about FastAPI vs Flask..."
└── transcript-0847 "User: I'm starting a new Python web..."
    └── unified-0847 (from chatgpt)
        └── chatgpt-raw-0847 {"id": "abc123", "create_time": ...}

# Experiment with a new prompt
$ synthex branch create better-summaries
Created branch 'better-summaries' from 'main'

$ synthex branch switch better-summaries
Switched to branch 'better-summaries'

# Edit prompts/summarize-conversation.md...

$ synthex run summaries --full
Running summaries on branch 'better-summaries'...
[1/1] summaries ████████████████████ 1081/1081

# Compare results
$ synthex branch diff better-summaries
Branch 'better-summaries' vs 'main':
  summaries: 1081 records differ
  monthly: 24 records differ (downstream)
  world-model: 1 record differs (downstream)

# If happy, promote
$ synthex branch promote better-summaries
Promoted 'better-summaries' to 'main'
  Copied 1106 records
  Updated cursors
```

---

## Part VI: Implementation Phases

### Phase 1: Foundation (Week 1-2)

**Goal:** Basic project structure and data model

- [ ] Project structure
  ```
  src/synthex/
  ├── __init__.py
  ├── cli.py           # Click CLI
  ├── config.py        # Pydantic settings
  ├── models.py        # SQLAlchemy models
  ├── db.py            # Database engine
  └── schemas.py       # Config schemas
  ```
- [ ] CLI skeleton with Click
  - `synthex init`, `synthex validate`, `synthex status`
- [ ] Config parsing
  - TOML → Pydantic models
  - Validation: required fields, types, DAG acyclicity
- [ ] SQLite database (MVP storage)
  - Models: Project, Source, Step, Output, Record, Run, Cursor
  - Basic CRUD operations
- [ ] `synthex apply` - persist config to DB

**Deliverable:** Can create project, validate config, see status

### Phase 2: Processing Engine (Week 3-4)

**Goal:** Execute transform steps with LLM

- [ ] Step execution engine
  - Fetch pending records
  - Apply processor
  - Store results with provenance
- [ ] Transform processor
  - Load prompt template
  - Call LLM (OpenAI-compatible)
  - Parse response
- [ ] Source importers
  - ChatGPT export parser
  - Claude export parser
  - Generic JSONL/Markdown
- [ ] Record creation
  - Content + metadata
  - Source linking (provenance)
  - Fingerprint generation
- [ ] Run tracking
  - Create Run on start
  - Update stats during execution
  - Mark complete/failed

**Deliverable:** `synthex run transcripts` processes source files

### Phase 3: DAG & Incremental (Week 5-6)

**Goal:** Full pipeline with incremental processing

- [ ] DAG resolution
  - Topological sort of steps
  - Dependency tracking
- [ ] `synthex run --all`
  - Execute steps in correct order
  - Skip if no pending records
- [ ] Cursor tracking
  - Track last processed record per step
  - Resume from cursor on re-run
- [ ] Fingerprint-based caching
  - Hash prompt + config
  - Invalidate if prompt changes
- [ ] Aggregate processor
  - Group records by period
  - Combine into single LLM call
- [ ] Fold processor
  - Order records chronologically
  - Process with accumulating state

**Deliverable:** Full pipeline runs incrementally

### Phase 4: Branching (Week 7-8)

**Goal:** Experiment with different configurations

- [ ] Branch model
  - Name, parent, created_at, status
  - Current branch tracking
- [ ] `branch create`
  - Copy-on-write: reference parent records
  - Fork cursors
- [ ] `branch switch`
  - Update current branch
  - Isolate subsequent operations
- [ ] `branch promote`
  - Copy new records to main
  - Update main cursors
- [ ] `branch diff`
  - Compare records between branches
  - Show changed outputs

**Deliverable:** Can experiment with prompts on branches

### Phase 5: Query & Polish (Week 9-10)

**Goal:** Complete MVP with search and UX

- [ ] Search implementation
  - Full-text search (SQLite FTS)
  - Optional: semantic search with embeddings
- [ ] Lineage traversal
  - Recursive source lookup
  - Tree visualization
- [ ] Output surfaces
  - `agent-context`: Format for system prompts
  - `semantic-search`: Queryable index
  - `export`: JSONL/JSON output
- [ ] `synthex plan`
  - Dry-run mode
  - Estimate records/tokens/cost
- [ ] Error handling
  - Graceful LLM failures
  - Retry with backoff
  - Partial run recovery
- [ ] UX polish
  - Progress bars
  - Rich console output
  - Helpful error messages
- [ ] Documentation
  - README with quickstart
  - Config reference
  - Prompt template guide

**Deliverable:** Complete MVP ready for evaluation

---

## Part VII: MVP Reference Implementation

### 7.1 Use Case: Personal Memory from ChatGPT History

**Scenario:** You have 2 years of ChatGPT conversations. You want to:
1. Import all conversations
2. Extract clean transcripts
3. Summarize each conversation
4. Create monthly reflections
5. Build an evolving world model
6. Search with full provenance

**Input:** `~/exports/chatgpt.json` (ChatGPT data export, ~500 conversations)

### 7.2 Project Configuration

```toml
# ~/personal-memory/synthex.toml

[project]
name = "personal-memory"
agent = "mark"
description = "Memory from 2 years of ChatGPT conversations"

[sources.chatgpt]
type = "file"
format = "chatgpt-export"
path = "~/exports/chatgpt.json"

[steps.transcripts]
type = "transform"
from = ["chatgpt"]
prompt = "extract-transcript"

[steps.summaries]
type = "transform"
from = ["transcripts"]
prompt = "summarize-conversation"

[steps.monthly]
type = "aggregate"
from = ["summaries"]
period = "month"
prompt = "monthly-reflection"

[steps.world-model]
type = "fold"
from = ["monthly"]
prompt = "world-model"

[outputs.context]
from = ["world-model"]
surface = "agent-context"

[outputs.search]
from = ["summaries", "monthly"]
surface = "semantic-search"
```

### 7.3 Prompt Templates

**prompts/extract-transcript.md:**
```markdown
Extract a clean conversation transcript from this ChatGPT export.

Format as:
User: [message]
Assistant: [message]

Remove system messages and metadata. Keep the conversation flow natural.

## Raw Export
{{ content }}

## Transcript
```

**prompts/summarize-conversation.md:**
```markdown
Summarize this conversation in 2-3 sentences.

Include:
- Main topic or question
- Key conclusions or decisions
- Any action items mentioned

## Conversation
{{ content }}

## Summary
```

**prompts/monthly-reflection.md:**
```markdown
Reflect on this month's conversations.

Synthesize into a cohesive narrative covering:
- Main themes and interests
- Projects or goals worked on
- Key learnings or insights
- How thinking evolved

Month: {{ metadata.period }}
Number of conversations: {{ sources | length }}

## Conversation Summaries
{% for s in sources %}
- {{ s.content }}
{% endfor %}

## Monthly Reflection
```

**prompts/world-model.md:**
```markdown
Update your understanding of this person based on new information.

Current understanding:
{{ state }}

New monthly reflection ({{ metadata.period }}):
{{ content }}

Write an updated, comprehensive understanding. Include:
- Core interests and values
- Current projects and goals
- Technical skills and preferences
- Communication style
- How they've evolved over time

## Updated Understanding
```

### 7.4 Execution Walkthrough

```bash
# Step 1: Initialize project
$ cd ~/personal-memory
$ synthex init personal-memory
Created project 'personal-memory'

# Step 2: Configure (edit synthex.toml and prompts/)

# Step 3: Validate
$ synthex validate
✓ Config syntax valid
✓ All prompts found (4)
✓ DAG is acyclic
✓ 1 source, 4 steps, 2 outputs

# Step 4: Plan
$ synthex plan
Planning run for branch 'main'...

Source analysis:
  chatgpt: 487 conversations (2022-03 to 2024-06)

Steps to execute:
  1. chatgpt (source) → 487 records
  2. transcripts (transform) → 487 records
  3. summaries (transform) → 487 records
  4. monthly (aggregate) → 28 records
  5. world-model (fold) → 1 record

Estimated: ~1.2M input tokens, ~98K output tokens
Estimated cost: ~$1.50 (gpt-4o-mini)

# Step 5: Run
$ synthex run --all
Running pipeline on branch 'main'...

[1/5] chatgpt ████████████████████ 487/487 (parsed)
[2/5] transcripts ████████████████████ 487/487 (2m 15s)
[3/5] summaries ████████████████████ 487/487 (3m 42s)
[4/5] monthly ████████████████████ 28/28 (45s)
[5/5] world-model ████████████████████ 1/1 (12s)

✓ Run completed in 7m 14s
  Records: 1,490 created
  Tokens: 1,247,832 input / 94,521 output
  Cost: $1.68

# Step 6: Query
$ synthex search "learning rust"
Found 5 results:

1. [0.94] summary-0312 (2024-01-15)
   "Asked about learning Rust coming from Python. Discussed
   ownership model, recommended 'The Book' and Exercism..."

2. [0.91] monthly-2024-01
   "January focused on systems programming exploration..."

$ synthex lineage summary-0312
summary-0312 "Asked about learning Rust..."
└── transcript-0312 "User: I'm a Python developer wanting..."
    └── chatgpt-raw-0312 {"id": "conv_abc123", ...}

# Step 7: Get context for agent
$ synthex inspect context
Output 'context' (from world-model):

Mark is a software engineer with deep Python expertise who's
been expanding into systems programming (Rust) and AI/ML
development. Core interests include:

- Developer tools and productivity
- AI assistants and agents
- Database systems (PostgreSQL, vector search)
- Clean code architecture

Currently working on:
- KP3: Knowledge processing pipeline
- Personal AI memory systems
- Learning Rust for performance-critical code

Values pragmatic solutions over theoretical purity. Prefers
explicit over implicit. Strong opinions on code readability...
```

### 7.5 Incremental Updates

```bash
# One month later, new export with 50 more conversations

$ synthex run --all
Running pipeline on branch 'main'...

[1/5] chatgpt ████████████████████ 50/50 (new records)
[2/5] transcripts ████████████████████ 50/50
[3/5] summaries ████████████████████ 50/50
[4/5] monthly ████████████████████ 1/1 (new month)
[5/5] world-model ████████████████████ 1/1 (updated)

✓ Run completed in 52s
  Records: 103 created (skipped 1,387 existing)
```

---

## Part VIII: Evaluation Strategy

### 8.1 Parity Benchmarks

We'll evaluate Synthex against established memory benchmarks:

#### LoCoMo (Long Context Memory)

- **Source:** [LoCoMo Paper](https://arxiv.org/abs/2402.01677)
- **Task:** Multi-session conversation memory
- **Tests:**
  - Retrieval accuracy across sessions
  - Temporal reasoning ("what did we discuss last week?")
  - Entity tracking across conversations
- **Metrics:** Precision@k, temporal accuracy

#### LongMemEval

- **Source:** [LongMemEval Paper](https://arxiv.org/abs/2401.14166)
- **Task:** Long-horizon memory evaluation
- **Tests:**
  - Information extraction from past sessions
  - Summarization quality
  - Consistency over time
- **Metrics:** ROUGE, factual accuracy, coherence

#### Custom: Provenance Accuracy

Since provenance is our differentiator, we need a custom benchmark:

- **Task:** Given a query result, can we trace to source?
- **Test:** Query for information, follow lineage, verify source contains info
- **Metric:** % of results with complete, accurate lineage

### 8.2 Comparison Targets

| System | How We Test |
|--------|-------------|
| **Mem0** | Same data, simple add() then search() |
| **Zep (Graphiti)** | If temporal queries are tested |
| **Raw RAG** | Embed source docs directly, no processing |
| **Synthex** | Full pipeline with hierarchy |

### 8.3 Metrics Dashboard

| Metric | Description | Target |
|--------|-------------|--------|
| **Precision@5** | Top 5 retrieval accuracy | >0.80 |
| **Temporal Accuracy** | Correct time-based queries | >0.75 |
| **Provenance Completeness** | Full lineage available | 100% |
| **Processing Efficiency** | Tokens per source record | <3000 |
| **Incremental Speedup** | Re-run time / full run time | <0.1x |
| **World Model Quality** | Human eval of coherence | >4/5 |

### 8.4 Eval Harness

```bash
# Run benchmark suite
$ synthex eval run --benchmark locomo
Running LoCoMo benchmark...
  Loading test data: 500 queries
  Testing retrieval: ████████████████████ 500/500

Results:
  Precision@1: 0.73
  Precision@5: 0.89
  Precision@10: 0.94
  Temporal Accuracy: 0.81

# Compare against baseline
$ synthex eval run --benchmark locomo --compare mem0,rag
Running LoCoMo benchmark (comparative)...

Results:
                 Synthex    Mem0      RAG
  Precision@5     0.89      0.72     0.68
  Temporal Acc    0.81      0.45     0.41
  Provenance      100%      0%       0%

# Generate report
$ synthex eval report --output results.md
Generated report: results.md
```

### 8.5 Qualitative Evaluation

Beyond metrics, we'll assess:

1. **World Model Quality**
   - Does the world model capture the person accurately?
   - Human eval: rate coherence, accuracy, usefulness

2. **Search Experience**
   - Are results relevant and useful?
   - Is provenance actually helpful for debugging?

3. **Developer Experience**
   - How long to set up a project?
   - How intuitive is the config format?
   - Error messages helpful?

---

## Part IX: Future Extensions

### 9.1 Pluggable Storage Backends

**MVP:** SQLite for everything (records, search, state)

**Future:**
```toml
[storage]
records = "postgres"
vectors = "pgvector"  # or: pinecone, qdrant
cache = "redis"

[storage.postgres]
url = "${DATABASE_URL}"

[storage.pgvector]
url = "${DATABASE_URL}"
dimensions = 1536
```

### 9.2 Real-Time Ingestion

**MVP:** Batch import from files

**Future:**
```toml
[sources.live]
type = "stream"
format = "webhook"
endpoint = "/api/ingest"
auth = "bearer:${WEBHOOK_SECRET}"

# Or Kafka/Redis streams
[sources.events]
type = "stream"
format = "kafka"
topic = "conversation-events"
bootstrap = "${KAFKA_BROKERS}"
```

Online updates flow through pipeline in near-real-time.

### 9.3 Agent Self-Management

**MVP:** External pipeline, agent queries outputs

**Future:**
```python
# Agent can trigger processing
from synthex import Client

synthex = Client("personal-memory")

# After a conversation
synthex.ingest(conversation_text)
synthex.run_incremental()

# Agent decides what's important
synthex.mark_important(record_id)
synthex.forget(record_id)  # soft delete
```

### 9.4 Additional Processors

| Type | Description | Use Case |
|------|-------------|----------|
| **cluster** | HDBSCAN clustering | Group similar memories |
| **tree_cluster** | Hierarchical online clustering (MemTree) | Latent memory organization |
| **graph** | Entity/relationship extraction | Knowledge graph |
| **embed** | Generate embeddings without LLM | Semantic search prep |
| **custom** | User-defined Python function | Anything else |

```toml
[steps.topics]
type = "cluster"
from = ["summaries"]
algorithm = "hdbscan"
min_cluster_size = 5

[steps.entities]
type = "graph"
from = ["transcripts"]
extract = ["people", "projects", "technologies"]
```

### 9.5 Multi-Agent Memory

**Scenario:** Multiple agents share a memory system

```toml
[project]
name = "team-memory"
agents = ["assistant", "researcher", "coder"]

[branches]
shared = "main"  # all agents read
assistant = "assistant-private"  # assistant-only
researcher = "researcher-private"
coder = "coder-private"

[outputs.assistant-context]
from = ["world-model"]
branches = ["main", "assistant-private"]
surface = "agent-context"
agent = "assistant"
```

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Project** | A named container for memory processing configuration and data |
| **Source** | Input that brings data into the system (files, APIs, streams) |
| **Step** | Processing operation that transforms data (transform, aggregate, fold, merge) |
| **Output** | Queryable projection of processed data (search, context, export) |
| **Record** | Single piece of content with provenance and metadata |
| **Run** | One execution of one or more steps |
| **Branch** | Isolated variant for experimentation |
| **Cursor** | Tracks incremental processing state |
| **Provenance** | Chain from any record back to its sources |
| **DAG** | Directed Acyclic Graph of step dependencies |
| **Fingerprint** | Content hash for deduplication and caching |

## Appendix B: Why Not Just Use...

### "Why not just use Mem0?"

Mem0 is great for simple use cases. Use it if:
- You just need add/search
- You don't care about provenance
- You don't need custom processing
- You're building a simple chatbot

Use Synthex if:
- You have bulk historical data to import
- You need custom processing pipelines
- Provenance/audit is important
- You want to experiment with architectures

### "Why not just use LangChain/LlamaIndex?"

Those are orchestration frameworks, not memory systems. They can help you build memory, but don't provide:
- Declarative configuration
- Incremental processing
- Branching for experiments
- Built-in provenance

Synthex could integrate with them as a memory backend.

### "Why not just use a vector database?"

Vector DBs are storage, not processing. They answer "find similar" but don't:
- Summarize or aggregate
- Build hierarchical memory
- Track provenance
- Run processing pipelines

Synthex uses vector DBs as a backend, adding the processing layer.

## Appendix C: Design Decisions

### Why TOML?

- More readable than YAML for this use case
- Better support for inline tables (step config)
- Native in Python ecosystem (tomllib in 3.11+)
- Familiar from pyproject.toml

### Why SQLite for MVP?

- Zero configuration
- Single file database
- Good enough for thousands of records
- Easy migration to Postgres later

### Why not graph database?

- Adds complexity
- Most use cases don't need graph queries
- Provenance is simple tree traversal
- Can add as optional backend later

### Why prompts as files?

- Version control with git
- Easy to edit with any editor
- Support template syntax (Jinja2)
- Separate concerns: config vs prompts
