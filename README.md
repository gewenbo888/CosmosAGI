# CosmosAGI

An open-source AGI prototype that combines autonomous agent loops, world modeling, multi-agent collaboration, and self-learning into a single modular system.

## What It Does

CosmosAGI takes a task, decomposes it into subtasks, predicts consequences before acting, executes with chain-of-thought reasoning, self-evaluates, and learns from every interaction.

```
Task → Plan → Simulate → Execute → Observe → Reflect → Improve → (loop)
                                                            ↓
                                                     Learn & Evolve
```

## Key Features

- **Autonomous Agent Loop** — LangGraph state machine with planning, execution, reflection, and self-improvement cycles
- **World Model** — Transformer-based next-state predictor + LLM-powered consequence simulation + causal graph with counterfactual reasoning
- **Multi-Agent Collaboration** — Dynamic team assembly with sequential pipelines and debate/consensus patterns. Specialist agents: Researcher, Coder, Reviewer, FactChecker
- **Self-Learning** — Experience replay buffer, multi-dimensional self-reward scoring, strategy evolution that deactivates failing approaches and reinforces successful ones
- **Safety & Alignment** — Output content filtering, action blocklists, sandboxed tool execution, human-in-the-loop approval gates
- **Provider Agnostic** — Works with Claude, OpenAI, Ollama (local models), and 100+ providers via LiteLLM

## Quick Start

```bash
# Clone
git clone https://github.com/gewenbo888/CosmosAGI.git
cd CosmosAGI

# Install
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY=sk-ant-...
# or: export OPENAI_API_KEY=sk-...
# or: ollama pull llama3 (no key needed)

# Run
python main.py "Explain the three laws of thermodynamics"
```

## Usage

### Single Task

```bash
python main.py "What causes ocean acidification and what are the effects?"
```

### Interactive Mode

```bash
python main.py
>>> Research the top 5 programming languages for AI development
>>> stats
>>> quit
```

### Multi-Agent Team Mode

Automatically assembles a team of specialist agents based on task requirements:

```bash
python main.py --team "Build a Python REST API for a todo app with error handling and tests"
```

### Choose Your LLM

```bash
python main.py --model claude-sonnet-4-20250514 "your task"    # Claude (default)
python main.py --model gpt-4o "your task"                      # OpenAI
python main.py --model ollama/llama3 "your task"               # Local (free)
```

### Benchmarks

```bash
python main.py --benchmark                    # Run all 7 tasks
python main.py --benchmark reasoning coding   # Filter by category
```

### Performance Stats

```bash
python main.py --stats
```

### Docker

```bash
cp .env.example .env   # Add your API key
docker compose up --build
```

## CLI Reference

| Flag | Description |
|------|-------------|
| `--model MODEL` | LLM model (default: `claude-sonnet-4-20250514`) |
| `--team` | Enable multi-agent team mode |
| `--max-iterations N` | Max agent loop iterations (default: 20) |
| `--no-hitl` | Disable human-in-the-loop approval |
| `--no-learn` | Disable post-task self-assessment |
| `--benchmark [CAT...]` | Run benchmark suite (categories: reasoning, coding, research, planning, safety) |
| `--stats` | Show learning performance statistics |
| `--log-level LEVEL` | Logging level: DEBUG, INFO, WARNING, ERROR |

## Architecture

```
cosmos_agi/
├── config/settings.py              # Pydantic settings (LLM, memory, safety)
├── core/
│   ├── state.py                    # AgentState — central state machine state
│   ├── llm.py                      # LiteLLM wrapper (provider-agnostic)
│   ├── graph.py                    # LangGraph orchestrator
│   └── safety.py                   # Output filtering, action blocking, HITL
├── agents/
│   ├── base.py                     # BaseAgent ABC
│   ├── planner.py                  # Task decomposition
│   ├── executor.py                 # Subtask execution with chain-of-thought
│   ├── critic.py                   # Result evaluation
│   ├── specialists.py             # Researcher, Coder, Reviewer, FactChecker
│   ├── spawner.py                  # Dynamic team assembly
│   ├── team.py                     # Sequential + debate orchestration
│   └── communication.py           # MessageBus + Blackboard
├── world_model/
│   ├── state_representation.py    # WorldState, Entity, Transition
│   ├── predictor.py                # StateTransformer (PyTorch) + LLMPredictor
│   ├── causal.py                   # CausalGraph with counterfactuals
│   └── integration.py             # WorldModelAgent bridge
├── evaluation/
│   ├── experience.py               # ExperienceReplayBuffer (persistent)
│   ├── self_reward.py              # Multi-dimensional self-assessment
│   ├── strategy_evolution.py      # Meta-learning from experience
│   ├── benchmark.py                # Standardized evaluation suite
│   └── learning_loop.py           # Self-improvement orchestrator
├── memory/vector_store.py          # ChromaDB vector memory
└── tools/
    ├── registry.py                 # Tool registry + builtins
    ├── code_executor.py            # Sandboxed Python execution
    ├── file_ops.py                 # Path-safe file operations
    └── web_search.py               # DuckDuckGo search
```

## Agent Loop Flow

```
PLAN ──→ SIMULATE ──→ EXECUTE ──→ OBSERVE ──→ REFLECT ──→ COMPLETE
  ↑                                              │
  └──────────── IMPROVE ←────────────────────────┘
```

1. **Plan** — Decomposes the task into ordered subtasks
2. **Simulate** — World model predicts consequences, flags risks
3. **Execute** — Solves each subtask with chain-of-thought reasoning
4. **Observe** — Collects and consolidates results
5. **Reflect** — Critic evaluates quality (confidence ≥ 0.7 → done)
6. **Improve** — Feeds reflection back into re-planning

After completion, the **Learning Loop** records the episode, self-scores on 5 dimensions (correctness, completeness, efficiency, clarity, safety), and evolves strategies for future tasks.

## Extending

### Add a Tool

```python
# cosmos_agi/tools/my_tool.py
from cosmos_agi.tools.registry import register_tool

def my_tool(input: str) -> str:
    return f"Processed: {input}"

register_tool("my_tool", my_tool, "Description of what it does")
```

### Add a Specialist Agent

```python
# cosmos_agi/agents/specialists.py
class MyAgent(BaseAgent):
    name = "my_agent"
    system_prompt = "You are a specialist in X..."

    def run(self, state: AgentState) -> AgentState:
        # Your logic here
        return state

# cosmos_agi/agents/spawner.py
AGENT_REGISTRY["my_agent"] = MyAgent
```

### Add Benchmark Tasks

```python
# cosmos_agi/evaluation/benchmark.py
BUILTIN_BENCHMARKS.append(BenchmarkTask(
    id="custom_001",
    category="reasoning",
    prompt="Your evaluation question",
    difficulty=0.5,
    expected_keywords=["expected", "answer", "keywords"],
))
```

## Tech Stack

- **Python 3.11+**
- **LangGraph** — Agent orchestration
- **LiteLLM** — Unified LLM interface (Claude, OpenAI, Ollama, 100+ providers)
- **ChromaDB** — Vector memory
- **PyTorch** — Neural world model
- **Pydantic v2** — Data validation

## Tests

```bash
pip install pytest
python -m pytest cosmos_agi/tests/ -v
```

122 tests covering all modules.

## License

MIT
