<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/formthefog/stratus-sdk-py/main/assets/stratus-x1-dark.png">
    <img src="https://raw.githubusercontent.com/formthefog/stratus-sdk-py/main/assets/stratus-x1-light.png" alt="Stratus X1" width="400">
  </picture>
</p>

<div align="center">

  <p><strong>Python SDK for Stratus X1 — the predictive world model for AI agents.</strong></p>

  [![PyPI version](https://img.shields.io/pypi/v/stratus-sdk-py)](https://pypi.org/project/stratus-sdk-py/)
  [![Python](https://img.shields.io/pypi/pyversions/stratus-sdk-py)](https://pypi.org/project/stratus-sdk-py/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Tests](https://img.shields.io/badge/tests-76%20passing-brightgreen)]()

</div>

---

## What is Stratus?

Stratus X1 is a **predictive action model** that sits between your LLM and the environment. It understands where an agent is, simulates what happens next, and sequences actions toward a goal — before a single real action executes.

Current LLM-based agents achieve **10–20% success rates** on real-world benchmarks. Human performance on the same tasks: **78%.** The gap isn't a prompting problem — it's structural. Agents fail because they have no state representation, no consequence prediction, and no way to plan across steps.

Stratus solves this without replacing your LLM:

- **State encoder** — compresses any observation (webpage, UI, tool response) into a rich semantic representation
- **World model** — simulates what the environment looks like after each candidate action, in representation space, before anything executes
- **Planning layer** — sequences actions toward the goal, returning a ranked plan with confidence at each step

The result: **68% fewer tokens. 2–3× faster. 2× higher task success rate.**

This SDK gives you full access to the Stratus API — chat completions (OpenAI and Anthropic formats), trajectory rollout, embeddings, LLM key management, and credits — plus compression profiles and production utilities.

**Docs:** [stratus.run/docs](https://stratus.run/docs) · **API:** [api.stratus.run](https://api.stratus.run) · **Dashboard:** [stratus.run](https://stratus.run)

---

## Install

```bash
pip install stratus-sdk-py
```

---

## Quick Start

### Chat completions (OpenAI-compatible)

```python
import asyncio
from stratus_sdk import StratusClient

client = StratusClient(api_key="sk-stratus-...")

response = await client.chat.completions.create(
    model="stratus-x1ac-base-gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)

print(response.choices[0].message.content)
```

### Streaming

```python
async for chunk in client.chat.completions.stream(
    model="stratus-x1ac-base-gpt-4o",
    messages=[{"role": "user", "content": "Count to 5"}],
):
    content = chunk.choices[0].get("delta", {}).get("content", "")
    if content:
        print(content, end="", flush=True)
```

### Trajectory prediction (rollout)

```python
result = await client.rollout(
    goal="Book a flight to NYC",
    initial_state="On airline homepage",
    max_steps=5,
)

print(result.summary.outcome)
for step in result.predictions:
    print(step.action.action_text, step.state_change)
```

### Embeddings

```python
result = await client.embeddings(
    model="stratus-x1ac-base",
    input=["Hello world", "Another sentence"],
)

print(result.data[0].embedding)  # list[float]
```

---

## API Reference

### `StratusClient`

```python
from stratus_sdk import StratusClient

client = StratusClient(
    api_key="sk-stratus-...",   # Required. Sent as Bearer token and x-api-key header.
    api_url="https://api.stratus.run",  # Default
    timeout=30.0,               # Default (seconds)
    retries=3,                  # Default (with exponential backoff)
    compression_profile=CompressionLevel.MEDIUM,  # Default
)
```

Supports async context manager:

```python
async with StratusClient(api_key="sk-stratus-...") as client:
    response = await client.chat.completions.create(...)
```

---

### `GET /health`

```python
status = await client.health()
# status["status"] == "healthy"
# status["model_loaded"] == True
```

---

### `GET /v1/models`

```python
models = await client.list_models()
# models[0].id == "stratus-x1ac-base-gpt-4o"
```

---

### `POST /v1/chat/completions`

```python
response = await client.chat.completions.create(
    model="stratus-x1ac-base-gpt-4o",
    messages=[{"role": "user", "content": "..."}],
    temperature=0.7,
    max_tokens=1000,
    # Stratus hybrid orchestration (optional)
    stratus={
        "mode": "plan",                # "plan" | "validate" | "rank" | "hybrid"
        "validation_threshold": 0.8,
        "return_action_sequence": True,
    },
    # Inline LLM keys (alternative to vault)
    openai_key="sk-...",
    anthropic_key="sk-ant-...",
    openrouter_key="sk-or-...",
    # Tool use
    tools=[{"type": "function", "function": {"name": "get_weather", ...}}],
    tool_choice="auto",
)
```

Streaming:

```python
async for chunk in client.chat.completions.stream(
    model="stratus-x1ac-base-gpt-4o",
    messages=[{"role": "user", "content": "..."}],
):
    print(chunk.choices[0].get("delta", {}).get("content", ""), end="")
```

---

### `POST /v1/messages` (Anthropic format)

```python
response = await client.messages(
    model="stratus-x1ac-base-claude-3-5-sonnet",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=1024,
    system="You are a helpful assistant.",
)

# response.content[0].type == "text"
# response.content[0].text == "..."
```

---

### `POST /v1/rollout`

```python
result = await client.rollout(
    goal="Complete the checkout",
    initial_state="Cart has 3 items",
    max_steps=10,
    return_intermediate=True,
)

# result.summary.outcome: str
# result.summary.total_steps: int
# result.predictions[0].action.confidence: float (0-1)
# result.predictions[0].state_change: float
```

---

### `POST /v1/embeddings`

```python
result = await client.embeddings(
    model="stratus-x1ac-base",
    input="Hello world",           # or list[str]
)

# result.data[0].embedding: list[float]
# result.data[0].index: int
```

---

### `POST /v1/account/llm-keys`

```python
result = await client.account.llm_keys.set(
    openai_key="sk-...",
    anthropic_key="sk-ant-...",
    openrouter_key="sk-or-...",
)
# result.success: bool
# result.configured.openai: bool
```

### `GET /v1/account/llm-keys`

```python
status = await client.account.llm_keys.get()
# status.openai: bool
# status.anthropic: bool
# status.openrouter: bool
```

### `DELETE /v1/account/llm-keys`

```python
# Delete a specific provider
await client.account.llm_keys.delete(provider="openai")

# Delete all keys
await client.account.llm_keys.delete()
```

---

### `GET /v1/credits/packages`

```python
packages = await client.credits.packages()
# packages[0].name: str
# packages[0].credits: float
# packages[0].price_usd: float
```

### `POST /v1/credits/purchase`

```python
result = await client.credits.purchase(
    package_name="starter",
    payment_header=base64_payment_header,
)
# result.credits_added: float
# result.new_balance: float
```

---

## Error Handling

```python
from stratus_sdk import StratusAPIError, StratusErrorType

try:
    await client.chat.completions.create(...)
except StratusAPIError as e:
    print(e.status_code)   # HTTP status
    print(e.error_type)    # StratusErrorType enum
    print(str(e))          # Human-readable message
```

**Error types:**

| Type | Meaning |
|------|---------|
| `authentication_error` | Invalid or missing API key |
| `insufficient_credits` | Not enough credits |
| `rate_limit` | Too many requests |
| `invalid_model` | Model ID not recognized |
| `model_not_loaded` | Model exists but not currently loaded |
| `llm_provider_not_configured` | No LLM key set for the requested provider |
| `llm_provider_error` | Upstream LLM call failed |
| `planning_failed` | World model planning failed |
| `validation_error` | Request validation failed |
| `internal_error` | Server error |

---

## Trajectory Predictor

Higher-level wrapper for multi-step rollout operations:

```python
from stratus_sdk import StratusClient, TrajectoryPredictor

client = StratusClient(api_key="sk-stratus-...")
predictor = TrajectoryPredictor(client, quality_threshold=80.0)

# Single prediction
result = await predictor.predict(
    initial_state="On checkout page",
    goal="Complete purchase",
    max_steps=5,
)

print(result.summary["goalAchieved"])   # bool
print(result.summary["qualityScore"])   # float 0-100
print(result.summary["actions"])        # list[str]

# Parallel predictions — try multiple approaches at once
plans = await predictor.predict_many(
    [
        {"initial_state": "...", "goal": "Fast approach", "max_steps": 3},
        {"initial_state": "...", "goal": "Safe approach", "max_steps": 5},
        {"initial_state": "...", "goal": "Optimal approach", "max_steps": 4},
    ],
    on_progress=lambda done, total: print(f"{done}/{total}"),
)

# Find the best plan
best = predictor.find_optimal(plans, min_quality=75.0, max_steps=10)

# Compare all plans
comparison = predictor.compare(plans)
print(f"Best quality: {comparison['best'].summary['qualityScore']}")
print(f"Avg steps: {comparison['average']['steps']}")

# Human-readable summary
print(predictor.get_summary(best))
```

---

## Drop Into Your Stack

### LangChain

```python
from stratus_sdk import StratusClient
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(...)
planner = StratusClient(api_key="sk-stratus-...")

plan = await planner.rollout(
    goal="Complete user task",
    initial_state="Current state: ...",
    max_steps=5,
)

if plan.summary.outcome == "success":
    for step in plan.predictions:
        await llm.call([{"role": "user", "content": step.action.action_text}])
```

### AutoGPT / Custom Agents

```python
from stratus_sdk import StratusClient, TrajectoryPredictor

class MyAgent:
    def __init__(self):
        client = StratusClient(api_key="sk-stratus-...")
        self.planner = TrajectoryPredictor(client)
        self.executor = YourLLM()

    async def execute_task(self, goal: str):
        plan = await self.planner.predict(
            initial_state=self.get_current_state(),
            goal=goal,
            max_steps=10,
        )

        if plan.summary["qualityScore"] < 80:
            return {"error": "Plan quality too low"}

        for action in plan.summary["actions"]:
            await self.executor.run(action)

        return {"success": plan.summary["goalAchieved"]}
```

### CrewAI / Multi-Agent Systems

```python
from stratus_sdk import StratusClient

planner = StratusClient(api_key="sk-stratus-...")

workflow = await planner.rollout(
    goal="Research, write, and publish article",
    initial_state="Topic: AI agents",
    max_steps=8,
)

if workflow.summary.outcome == "success":
    for step, agent in zip(workflow.predictions, crew):
        await agent.execute(step.action.action_text)
```

---

## Production Utilities

### Caching

```python
from stratus_sdk import SimpleCache

cache = SimpleCache(ttl_seconds=300)  # 5-min TTL

async def get_plan(goal: str):
    cached = cache.get(goal)
    if cached:
        return cached
    result = await client.rollout(goal=goal, initial_state="...")
    cache.set(goal, result)
    return result
```

### Rate Limiting

```python
from stratus_sdk import RateLimiter

limiter = RateLimiter(max_requests_per_second=10.0)
await limiter.wait()
response = await client.chat.completions.create(...)
```

### Health Checks

```python
from stratus_sdk import HealthChecker

checker = HealthChecker(
    client,
    check_interval_seconds=60,
    on_unhealthy=lambda: print("Stratus API unavailable"),
)

status = await checker.check()
if not status["healthy"]:
    raise RuntimeError("API unavailable")

await checker.start_monitoring()   # polls every 60s
await checker.stop_monitoring()
```

### Credit Monitoring

```python
from stratus_sdk import CreditMonitor

monitor = CreditMonitor(
    client,
    warning_threshold=10.0,
    critical_threshold=2.0,
    on_warning=lambda balance: print(f"Low credits: {balance}"),
    on_critical=lambda balance: print(f"CRITICAL: {balance} credits left"),
)

balance = await monitor.check()
print(f"Current balance: {balance}")
```

### Retry with Backoff

```python
from stratus_sdk import retry_with_backoff

result = await retry_with_backoff(
    lambda: client.chat.completions.create(...),
    max_retries=5,
    initial_delay_ms=500,
    max_delay_ms=10000,
    backoff_multiplier=2.0,
)
```

---

## Types

All request and response objects are fully typed with Pydantic v2:

```python
from stratus_sdk import (
    # Chat
    Message, ChatCompletionResponse, ChatCompletionChunk,

    # Anthropic
    AnthropicRequest, AnthropicResponse, AnthropicContentBlock,

    # Embeddings
    EmbeddingRequest, EmbeddingResponse, EmbeddingObject,

    # Rollout
    RolloutResponse, StatePrediction, Action, TrajectoryResult,

    # Models
    ModelInfo,

    # Account
    LLMKeyStatus, LLMKeySetRequest, LLMKeySetResponse,

    # Credits
    CreditPackage, CreditPurchaseResponse,

    # Stratus metadata
    StratusMetadata,

    # Errors
    StratusErrorType,
)
```

---

## Development

```bash
# Clone
git clone https://github.com/formthefog/stratus-sdk-py
cd stratus-sdk-py

# Install in editable mode with dev deps
pip install -e ".[dev]"

# Run tests (76 passing)
pytest tests/ -v

# Format
black stratus_sdk tests

# Type check
mypy stratus_sdk

# Lint
ruff check stratus_sdk
```

---

## Links

- **Homepage:** https://stratus.run
- **Documentation:** https://docs.stratus.run/sdk
- **PyPI:** https://pypi.org/project/stratus-sdk-py/
- **GitHub:** https://github.com/formthefog/stratus-sdk-py
- **Issues:** https://github.com/formthefog/stratus-sdk-py/issues

---

**Built by [Formation](https://formation.ai)**
