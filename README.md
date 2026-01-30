# Stratus SDK (Python)

**Give your LLM agent a planning brain** · Add M-JEPA-G world model in 3 lines of code

[![PyPI version](https://img.shields.io/pypi/v/stratus-sdk)](https://pypi.org/project/stratus-sdk/)
[![Python Support](https://img.shields.io/pypi/pyversions/stratus-sdk)](https://pypi.org/project/stratus-sdk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## The Problem

Your LLM agent (GPT-4, Claude, whatever) is smart. But it's **slow**, **expensive**, and **guesses** its way through multi-step tasks.

You're burning tokens on reasoning loops. Plans fail halfway through. You retry and hope.

## The Fix

Add a **world model** that predicts action sequences **before** your agent executes.

**M-JEPA-G plans. Your LLM executes.** Best of both worlds.

- **120ms planning** (vs 5-20s LLM reasoning)
- **$0.10 per 1M tokens** (30x cheaper than Claude for planning)
- **Validates before executing** (catch errors before expensive calls)
- **Works with your LLM** (GPT-4, Claude, Gemini - you choose)

---

## Quick Start

### Install

```bash
pip install stratus-sdk
```

### Add to Your Agent (3 lines)

**Before:**
```python
# Your agent guesses the plan, crosses fingers
response = await openai_client.chat.completions.create(
    messages=[{"role": "user", "content": "Book a flight to NYC"}],
    model="gpt-4"
)
await agent.execute(response)  # Hope it works
```

**After:**
```python
from stratus_sdk import MJepaGClient

# Add world model
planner = MJepaGClient(api_key="sk-stratus-...")

# Get validated plan
plan = await planner.rollout(
    goal="Book a flight to NYC",
    initial_state="On airline homepage",
    max_steps=5
)

# Execute only if safe
if plan.summary.outcome == "success":
    for step in plan.predictions:
        await your_agent.execute(step.action.action_text)
```

**That's it.** Your agent now thinks before it acts.

---

## What You Get

### 1. **Faster Iteration**
Stop burning 20 seconds per planning loop. Get plans in 120ms.

### 2. **Way Cheaper**
Planning: $0.10 per 1M tokens (M-JEPA-G)
Execution: $3 per 1M tokens (Claude) **only when needed**

### 3. **Fewer Retries**
World model catches errors **before** you execute. No more "oops, start over."

### 4. **Keep Your LLM**
Works with GPT-4, Claude, Gemini, Llama - whatever you're already using. Just add the planning layer.

---

## Drop Into Your Stack

### LangChain

```python
from stratus_sdk import MJepaGClient
from langchain.chat_models import ChatOpenAI

# Your existing setup
llm = ChatOpenAI(...)

# Add planning layer
planner = MJepaGClient(api_key="sk-stratus-...")

# Plan first, then execute
plan = await planner.rollout(
    goal="Complete user task",
    initial_state="Current state: ...",
    max_steps=5
)

# Let your LLM execute the validated plan
for step in plan.predictions:
    await llm.call([{"role": "user", "content": step.action.action_text}])
```

### AutoGPT / Custom Agents

```python
from stratus_sdk import MJepaGClient, TrajectoryPredictor

class MyAgent:
    def __init__(self):
        client = MJepaGClient(api_key="sk-stratus-...")
        self.planner = TrajectoryPredictor(client)
        self.executor = YourLLM()  # GPT-4, Claude, etc.

    async def execute_task(self, goal: str):
        # 1. World model plans the trajectory
        plan = await self.planner.predict(
            initial_state=self.get_current_state(),
            goal=goal,
            max_steps=10
        )

        # 2. Validate quality
        if plan.summary["qualityScore"] < 80:
            return {"error": "Plan quality too low"}

        # 3. Execute with your LLM
        for action in plan.summary["actions"]:
            await self.executor.run(action)

        return {"success": plan.summary["goalAchieved"]}
```

### CrewAI / Multi-Agent Systems

```python
from stratus_sdk import MJepaGClient

planner = MJepaGClient(api_key="sk-stratus-...")

# Before assigning tasks to agents, validate the workflow
workflow_plan = await planner.rollout(
    goal="Research, write, and publish article",
    initial_state="Topic: AI agents",
    max_steps=8
)

# Assign validated steps to your crew
if workflow_plan.summary.outcome == "success":
    for step, agent in zip(workflow_plan.predictions, crew):
        await agent.execute(step.action.action_text)
```

---

## The Pattern

**Plan → Validate → Execute**

```python
# 1. PLAN (120ms, $0.0001)
plan = await planner.rollout(
    goal="Book flight and hotel for NYC",
    initial_state="On travel site",
    max_steps=10
)

# 2. VALIDATE (instant)
if plan.summary.outcome != "success":
    print("Plan failed:", plan.summary.outcome)
    return

if plan.summary.qualityScore < 85:
    print("Quality too low, trying different approach")
    return

# 3. EXECUTE (your LLM does the work)
for step in plan.predictions:
    result = await your_llm.execute(step.action.action_text)
    # Update state, continue
```

---

## Advanced Features

### Parallel Planning (Try Multiple Approaches)

```python
from stratus_sdk import TrajectoryPredictor

predictor = TrajectoryPredictor(client)

# Generate 3 different plans in parallel
plans = await predictor.predict_many([
    {"initial_state": "...", "goal": "Fast approach", "max_steps": 3},
    {"initial_state": "...", "goal": "Safe approach", "max_steps": 5},
    {"initial_state": "...", "goal": "Optimal approach", "max_steps": 4},
])

# Pick the best one
best = predictor.find_optimal(plans)
print(f"Using: {best.summary['outcome']}")
print(f"Quality: {best.summary['qualityScore']}/100")
```

### Streaming (Real-Time Plans)

```python
# Stream plans as they're generated
async for chunk in client.chat.completions.stream(
    messages=[{"role": "user", "content": "Plan the deployment"}],
    model="stratus-x1-ac"
):
    content = chunk.choices[0].delta.get("content", "")
    if content:
        print(content, end="")
```

### Production Ready Out of the Box

**Automatic Retries:**
```python
client = MJepaGClient(
    api_key="sk-stratus-...",
    retries=3,  # Exponential backoff
    timeout=30.0
)
```

**Caching (Reduce Costs):**
```python
from stratus_sdk import SimpleCache

cache = SimpleCache(ttl_seconds=300)

async def get_plan(goal: str):
    cached = cache.get(goal)
    if cached:
        return cached

    result = await client.rollout(goal=goal, initial_state="...")
    cache.set(goal, result)
    return result
```

**Rate Limiting:**
```python
from stratus_sdk import RateLimiter

limiter = RateLimiter(max_requests_per_second=10)
await limiter.wait()
response = await client.chat.completions.create(...)
```

**Health Checks:**
```python
from stratus_sdk import HealthChecker

health = HealthChecker(client)

# Check on startup
status = await health.check()
if not status["healthy"]:
    raise Exception("M-JEPA-G API unavailable")

# Monitor continuously
await health.start_monitoring()  # Checks every 60s
```

---

## Why M-JEPA-G + Your LLM?

**You don't replace your LLM. You unlock it.**

| Task | Best Tool | Why |
|------|-----------|-----|
| **Multi-step planning** | M-JEPA-G | 120ms, $0.0001, world model |
| **Natural language** | Your LLM | Best at generation & interaction |
| **Action validation** | M-JEPA-G | Catches errors before execution |
| **Execution** | Your LLM | Does what it's best at |

**The result:** Faster, cheaper, more reliable agents.

---

## OpenAI-Compatible API

Drop-in replacement for planning tasks:

**Before (OpenAI):**
```python
import openai

client = openai.OpenAI(api_key="sk-...")
response = await client.chat.completions.create(
    messages=[{"role": "user", "content": "Plan the next steps"}],
    model="gpt-4"
)
```

**After (M-JEPA-G):**
```python
from stratus_sdk import MJepaGClient

client = MJepaGClient(api_key="sk-stratus-...")
response = await client.chat.completions.create(
    messages=[{"role": "user", "content": "Plan the next steps"}],
    model="stratus-x1-ac"
)
```

Same API. 10x faster. 10x cheaper.

---

## Complete API

### MJepaGClient

```python
# OpenAI-compatible
await client.chat.completions.create(messages, model)
async for chunk in client.chat.completions.stream(messages, model):
    ...

# M-JEPA-G specific (trajectory prediction)
await client.rollout(goal, initial_state, max_steps)
await client.health()
```

### TrajectoryPredictor

```python
await predictor.predict(initial_state, goal, max_steps)
await predictor.predict_many([...])  # Parallel
predictor.find_optimal(plans)        # Pick best
predictor.compare(trajectories)       # Compare
```

### Production Utilities

```python
SimpleCache(ttl_seconds)              # Cache results
RateLimiter(max_requests_per_second)  # Throttle requests
HealthChecker(client, ...)            # Monitor API
await retry_with_backoff(fn, ...)     # Auto-retry
```

---

## Examples

Check out [examples/](./examples) for complete implementations:

- `basic_usage.py` - Quick start guide
- `agent_planning.py` - Agent framework integration
- `langchain_integration.py` - LangChain example
- `parallel_planning.py` - Multiple trajectory exploration

---

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=stratus_sdk
```

---

## Development

```bash
# Clone repository
git clone https://github.com/formthefog/stratus-sdk-py
cd stratus-sdk-py

# Install in editable mode
pip install -e ".[dev]"

# Format
black stratus_sdk tests

# Type checking
mypy stratus_sdk

# Lint
ruff check stratus_sdk
```

---

## Links

- **Homepage:** https://stratus.run
- **Documentation:** https://docs.stratus.run/sdk
- **PyPI:** https://pypi.org/project/stratus-sdk/
- **GitHub:** https://github.com/formthefog/stratus-sdk-py
- **Issues:** https://github.com/formthefog/stratus-sdk-py/issues

---

## Support

- 📖 [Full Documentation](https://docs.stratus.run)
- 💬 [Discord Community](https://discord.gg/stratus)
- 📧 [Email Support](mailto:support@stratus.run)

---

## For AI Agents (Claude Code)

**Using this SDK with Claude Code?** Check out [`CLAUDE.md`](./CLAUDE.md) for:
- SDK-specific patterns and best practices
- Common mistakes to avoid
- Integration examples with LangChain/LlamaIndex
- Type safety patterns with Pydantic
- Error handling and async/await patterns

The `CLAUDE.md` file provides context-aware instructions that Claude automatically loads when working with this SDK.

---

## License

MIT License - see [LICENSE](./LICENSE) file for details.

---

**Built by [Formation](https://formation.ai)** · Making AI agents better, one world model at a time.
