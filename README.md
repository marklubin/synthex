# Synthex

**Neural schematic primitives for agent memory**

Synthex is a declarative framework for building custom cognitive architectures. Define your memory processing pipeline in TOML, compose transformations with LLM prompts, and query with full provenance.

## Why Synthex?

Current agent memory solutions are either too simple (Mem0: add/search) or too opinionated (Letta: fixed hierarchy). Synthex gives you primitives to build any architecture:

- **Declarative configuration** - Define memory pipelines in TOML, like Terraform for cognition
- **Full provenance** - Trace any memory back to its source
- **Incremental processing** - Only process new data on re-runs
- **Branching** - Experiment with different prompts/architectures
- **Bulk import** - Process years of conversation history

## Quick Example

```toml
# synthex.toml
[project]
name = "personal-memory"
agent = "mark"

[sources.chatgpt]
type = "file"
format = "chatgpt-export"
path = "~/exports/chatgpt.json"

[steps.summaries]
type = "transform"
from = ["chatgpt"]
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
```

```bash
synthex init personal-memory
synthex run --all
synthex search "python projects"
synthex lineage <record-id>
```

## Installation

```bash
pip install synthex  # coming soon
```

## Documentation

See [DESIGN.md](./DESIGN.md) for the full design document including:

- Vision and motivation
- Domain model and concepts
- Configuration reference
- CLI commands
- Implementation roadmap
- Evaluation strategy

## Status

🚧 **Early Development** - Architecture and design phase. Not yet functional.

## License

MIT
