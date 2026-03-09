# Stratus Python SDK - Claude Instructions

**Context:** When working with Stratus Python SDK for Stratus X1 world model integration.

## What This SDK Does

The Stratus Python SDK provides:
- **Stratus X1 World Model** - Trajectory prediction & state rollout for AI agents
- **OpenAI-Compatible API** - Drop-in replacement for OpenAI client
- **Agent Planning** - Predict multi-step action sequences before execution
- **Workflow Validation** - Validate multi-step workflows before running them
- **Production Features** - Async/await, retries, caching, rate limiting

## When to Use This SDK

Use the Stratus Python SDK when:
1. Building AI agents that need multi-step planning
2. Need to predict action outcomes before executing
3. Validating complex workflows before running them
4. Want OpenAI-compatible API with world model capabilities
5. Working in Python environments (3.8+)

## Core Patterns

### 1. Drop-In OpenAI Replacement

```python
from stratus_sdk import StratusClient

client = StratusClient(api_key="sk-stratus-...")

response = await client.chat.completions.create(
    messages=[{"role": "user", "content": "Plan the next steps."}],
    model="stratus-x1-ac"
)
```

### 2. Multi-Step Rollout

```python
# Predict multi-step trajectory
result = await client.rollout(
    goal="Book a flight to NYC",
    initial_state="On airline homepage",
    max_steps=5
)

# Check if plan is valid
if result.summary.outcome == "success":
    for pred in result.predictions:
        print(f"Step {pred.step}: {pred.action.action_text}")
```

### 3. Trajectory Prediction

```python
from stratus_sdk import TrajectoryPredictor

predictor = TrajectoryPredictor(client)

result = await predictor.predict(
    initial_state="Database: 1000 users, API: healthy",
    goal="Migrate to new database without downtime",
    constraints=["No user-facing downtime", "Data integrity maintained"]
)
```

## Key Files

| File | Purpose |
|------|---------|
| `stratus_sdk/client.py` | Main Stratus X1 client |
| `stratus_sdk/models.py` | Pydantic models for requests/responses |
| `stratus_sdk/trajectory.py` | Trajectory prediction utilities |
| `demo.py` | Usage examples |

## Common Use Cases

### Agent Planning
Before executing actions, use Stratus X1 to predict what will happen.

### Workflow Validation
Validate multi-step workflows before running them to catch issues early.

### State Rollout
Generate complete action sequences for complex goals.

## Installation

```bash
pip install stratus-sdk
```

**Requirements:** Python 3.8+

## API Reference

### StratusClient

```python
client = StratusClient(
    api_key="sk-stratus-...",
    base_url="https://api.stratus.chat",  # optional
    timeout=30,  # optional
    max_retries=3  # optional
)
```

### Methods

- `chat.completions.create()` - OpenAI-compatible chat completions
- `rollout()` - Multi-step trajectory prediction
- `predict()` - Single-step state prediction

## Error Handling

```python
from stratus_sdk.exceptions import StratusError, RateLimitError, AuthenticationError

try:
    result = await client.rollout(goal="...", initial_state="...")
except RateLimitError:
    # Handle rate limiting
    await asyncio.sleep(5)
    result = await client.rollout(goal="...", initial_state="...")
except AuthenticationError:
    # Handle invalid API key
    print("Invalid API key")
except StratusError as e:
    # Handle other Stratus errors
    print(f"Stratus error: {e}")
```

## Best Practices

1. **Use async/await** - All SDK methods are async
2. **Set constraints** - Define constraints for trajectory prediction
3. **Check outcome** - Always check `result.summary.outcome` before executing
4. **Handle retries** - SDK handles retries automatically, but add your own for critical paths
5. **Cache results** - Trajectory predictions can be cached for similar inputs

## Type Hints

The SDK is fully typed with Pydantic models:

```python
from stratus_sdk.models import RolloutRequest, RolloutResponse, Prediction

# Full type safety
request: RolloutRequest = RolloutRequest(
    goal="Book flight",
    initial_state="Homepage",
    max_steps=5
)

response: RolloutResponse = await client.rollout(**request.dict())
```

## Integration with LangChain/LlamaIndex

The SDK works with popular frameworks:

```python
from langchain.llms import BaseLLM
from stratus_sdk import StratusClient

class StratusLLM(BaseLLM):
    client: StratusClient

    def _call(self, prompt: str, **kwargs) -> str:
        response = await self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="stratus-x1-ac"
        )
        return response.choices[0].message.content
```

## Related Documentation

- **README.md** - Installation and quick start
- **demo.py** - Working examples
- **API Spec** - See `references/stratus-docs` for full API reference

## When Writing Code

When implementing features with this SDK:

1. **Always use async/await** - This is an async-first SDK
2. **Import from stratus_sdk** - Use `from stratus_sdk import ...`
3. **Check result outcomes** - Validate predictions before executing actions
4. **Handle errors gracefully** - Use try/except for network/API errors
5. **Use type hints** - Leverage Pydantic models for type safety

## Common Mistakes to Avoid

1. ❌ Forgetting `await` on async methods
2. ❌ Not checking `result.summary.outcome` before executing
3. ❌ Using sync code in async context
4. ❌ Not handling rate limits
5. ❌ Ignoring trajectory prediction constraints

## Environment Variables

```bash
STRATUS_API_KEY=sk-stratus-...
STRATUS_BASE_URL=https://api.stratus.chat  # optional
STRATUS_TIMEOUT=30  # optional
```

## Testing

```python
# Mock the client for testing
from unittest.mock import AsyncMock

client = StratusClient(api_key="test")
client.rollout = AsyncMock(return_value=RolloutResponse(...))
```
