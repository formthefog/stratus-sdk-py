# Stratus SDK (Python)

**Official Python SDK for Stratus X1** - Add a planning layer to your LLM agent in 3 lines of code.

[![PyPI version](https://img.shields.io/pypi/v/stratus-sdk)](https://pypi.org/project/stratus-sdk/)
[![Python Support](https://img.shields.io/pypi/pyversions/stratus-sdk)](https://pypi.org/project/stratus-sdk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Features

- 🔌 **OpenAI-Compatible** - Drop-in replacement for OpenAI client
- 🚀 **World Model Integration** - Built-in trajectory prediction & state rollout
- 📦 **Automatic Compression** - 15-20x smaller embeddings with 99%+ quality
- 🛠️ **Production-Ready** - Async/await, retries, caching, rate limiting
- 🎯 **Type-Safe** - Full type hints with Pydantic models
- 🐍 **Pythonic** - Follows Python best practices and conventions

---

## Installation

```bash
pip install stratus-sdk
```

**Requirements:** Python 3.8+

---

## Quick Start

### Drop-In OpenAI Replacement

**Before (OpenAI):**
```python
import openai

client = openai.OpenAI(api_key="sk-...")

response = await client.chat.completions.create(
    messages=[{"role": "user", "content": "Plan the next steps."}],
    model="gpt-4"
)
```

**After (M-JEPA-G):**
```python
from stratus_sdk import MJepaGClient

client = MJepaGClient(api_key="sk-stratus-...")

response = await client.chat.completions.create(
    messages=[{"role": "user", "content": "Plan the next steps."}],
    model="stratus-x1-ac"
)
```

**That's it.** Same API, 10x faster, 10x cheaper.

---

## Common Use Cases

### 1. Agent Planning

Before executing actions, ask M-JEPA-G what will happen:

```python
from stratus_sdk import MJepaGClient

client = MJepaGClient(api_key="sk-stratus-...")

# Predict multi-step trajectory
result = await client.rollout(
    goal="Book a flight to NYC",
    initial_state="On airline homepage",
    max_steps=5
)

# Execute only if plan looks good
if result.summary.outcome == "success":
    print("Plan validated:")
    for pred in result.predictions:
        print(f"Step {pred.step}: {pred.action.action_text}")
```

### 2. Workflow Validation

Validate multi-step workflows before running them:

```python
from stratus_sdk import TrajectoryPredictor

predictor = TrajectoryPredictor(client)

result = await predictor.predict(
    initial_state="Database: 1000 users, API: healthy",
    goal="Migrate to new database without downtime",
    max_steps=10
)

if result.summary["goalAchieved"] and result.summary["qualityScore"] > 90:
    print("Safe to proceed")
    print(f"Steps: {' → '.join(result.summary['actions'])}")
else:
    print("Plan has issues:", result.summary["outcome"])
```

### 3. Add Planning Layer to Your LLM

```python
# Before: LLM plans and executes (slow, expensive, error-prone)
response = await openai_client.chat.completions.create(
    messages=[{"role": "user", "content": "Book a flight to NYC and find a hotel"}],
    model="gpt-4"
)
await execute_actions(response)  # Hope it works!

# After: M-JEPA-G plans, your LLM executes (fast, cheap, validated)
plan = await client.rollout(
    goal="Book flight and hotel for NYC",
    initial_state="On travel site",
    max_steps=10
)

# Validate before executing
if plan.summary.outcome == "success":
    for step in plan.predictions:
        await your_llm.execute(step.action.action_text)  # GPT-4, Claude, etc.
```

---

## Framework Integration

### LangChain

```python
from stratus_sdk import MJepaGClient
from langchain.chat_models import ChatOpenAI

# Your existing LangChain setup
llm = ChatOpenAI(...)

# Add M-JEPA-G for planning
planner = MJepaGClient(api_key="sk-stratus-...")

# Validate actions before executing
result = await planner.rollout(
    goal="Complete the user task",
    initial_state="Current context: ...",
    max_steps=5
)

# Execute the validated plan
for step in result.predictions:
    await llm.call([{"role": "user", "content": step.action.action_text}])
```

### Custom Agent Framework

```python
from stratus_sdk import MJepaGClient, TrajectoryPredictor

class MyAgent:
    def __init__(self):
        client = MJepaGClient(api_key="sk-stratus-...")
        self.planner = TrajectoryPredictor(client)

    async def execute_task(self, goal: str):
        # 1. Use M-JEPA-G to plan
        plan = await self.planner.predict(
            initial_state=self.get_current_state(),
            goal=goal,
            max_steps=10
        )

        # 2. Execute validated plan
        for action in plan.summary["actions"]:
            await self.execute(action)

        return plan.summary["goalAchieved"]
```

---

## Advanced Features

### Streaming

Stream chat completions for real-time responses:

```python
async for chunk in client.chat.completions.stream(
    messages=[{"role": "user", "content": "Plan the deployment."}],
    model="stratus-x1-ac"
):
    content = chunk.choices[0].delta.get("content", "")
    if content:
        print(content, end="")
```

### Parallel Planning

Explore multiple approaches simultaneously:

```python
from stratus_sdk import TrajectoryPredictor

predictor = TrajectoryPredictor(client)

# Try 3 different approaches
plans = await predictor.predict_many([
    {"initial_state": "state A", "goal": "Fast path", "max_steps": 3},
    {"initial_state": "state B", "goal": "Safe path", "max_steps": 5},
    {"initial_state": "state C", "goal": "Optimal", "max_steps": 4},
])

# Pick the best one automatically
best = predictor.find_optimal(plans)

print(f"Using plan: {best.summary['outcome']}")
print(f"Quality: {best.summary['qualityScore']}/100")
print(f"Steps: {' → '.join(best.summary['actions'])}")
```

### Automatic Compression

M-JEPA-G embeddings are compressed automatically:

```python
from stratus_sdk import MJepaGClient, CompressionLevel

client = MJepaGClient(
    api_key="sk-stratus-...",
    compression_profile=CompressionLevel.MEDIUM  # 16x compression, 99.7% quality
)

# Embeddings are compressed behind the scenes
response = await client.chat.completions.create(...)

# Storage savings:
print(client.get_compression_ratio())  # "16.8x"
print(client.get_quality_score())      # 99.7
```

**Compression levels:**
- `CompressionLevel.LOW` - 15x compression, 99.9% quality
- `CompressionLevel.MEDIUM` - 16x compression, 99.7% quality (default)
- `CompressionLevel.HIGH` - 18x compression, 99.5% quality
- `CompressionLevel.VERY_HIGH` - 20x compression, 99.0% quality

---

## Production Features

### Automatic Retries

Built-in exponential backoff:

```python
client = MJepaGClient(
    api_key="sk-stratus-...",
    retries=3,  # Automatic retry on failure
    timeout=30.0
)

# Retries happen automatically
response = await client.chat.completions.create(...)
```

### Caching

Reduce costs with in-memory caching:

```python
from stratus_sdk import SimpleCache

cache = SimpleCache(ttl_seconds=300)  # 5-minute TTL

async def get_plan(goal: str):
    cached = cache.get(goal)
    if cached:
        return cached

    result = await client.rollout(goal=goal, initial_state="...")
    cache.set(goal, result)
    return result
```

### Rate Limiting

Automatic request throttling:

```python
from stratus_sdk import RateLimiter

limiter = RateLimiter(max_requests_per_second=10)

await limiter.wait()
response = await client.chat.completions.create(...)
```

### Health Checks

Monitor API availability:

```python
from stratus_sdk import HealthChecker

health = HealthChecker(
    client,
    on_unhealthy=lambda: print("M-JEPA-G API is down!")
)

# Check on startup
status = await health.check()
if not status["healthy"]:
    raise Exception("API unavailable")

# Monitor continuously
await health.start_monitoring()  # Checks every 60s
```

---

## How M-JEPA-G Complements Your LLM

**The Pattern:** Plan with M-JEPA-G, Execute with Your LLM

```python
# 1. M-JEPA-G generates the plan (120ms, $0.0001)
plan = await mjepa_client.rollout(
    goal="Book flight and hotel for NYC trip",
    initial_state="On travel site homepage",
    max_steps=10
)

# 2. Your LLM executes each step (using GPT-4, Claude, etc.)
for step in plan.predictions:
    await your_llm.execute(step.action.action_text)
```

### Why This Works

- **Planning is cheap** - M-JEPA-G: $0.10 per 1M tokens (30x cheaper than Claude)
- **Execution quality** - Use GPT-4/Claude for what they're best at: language & interaction
- **Faster iteration** - Validate plans before expensive LLM calls
- **Fewer retries** - World model catches errors before execution

### Use M-JEPA-G For

✅ Multi-step workflow planning
✅ Predicting action sequences before execution
✅ Validating agent plans (catch errors early)
✅ High-frequency planning tasks

### Use Your LLM For

✅ Executing the validated plan
✅ Natural language generation
✅ Complex reasoning and creativity
✅ Domain-specific knowledge

---

## API Reference

### MJepaGClient

```python
# OpenAI-compatible
await client.chat.completions.create(messages, model)
async for chunk in client.chat.completions.stream(messages, model):
    ...

# M-JEPA-G specific
await client.rollout(goal, initial_state, max_steps)
await client.health()
```

### TrajectoryPredictor

```python
await predictor.predict(initial_state, goal, max_steps)
await predictor.predict_many([...])  # Parallel execution
predictor.find_optimal(plans)        # Pick best trajectory
predictor.compare(trajectories)       # Compare results
```

### Production Utilities

```python
SimpleCache(ttl_seconds)              # Cache results
RateLimiter(max_requests_per_second)  # Throttle requests
HealthChecker(client, ...)            # Monitor API
await retry_with_backoff(fn, ...)     # Auto-retry with backoff
```

---

## Examples

Check out complete examples in the [examples/](./examples) directory:

- `basic_usage.py` - Quick start guide
- `agent_planning.py` - Agent framework integration
- `langchain_integration.py` - LangChain example

---

## Testing

Run the test suite:

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

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Format code
black stratus_sdk tests

# Type checking
mypy stratus_sdk

# Linting
ruff check stratus_sdk
```

---

## License

MIT License - see [LICENSE](./LICENSE) file for details.

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

**Built by [Formation](https://formation.ai)** · Making AI infrastructure better, one vector at a time.
