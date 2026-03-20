"""
Tests for TrajectoryPredictor.
"""

import pytest
import respx
import httpx

from stratus_sdk import StratusClient, TrajectoryPredictor
from stratus_sdk.types import StatePrediction, Action, RolloutSummary, Usage, TrajectoryResult


BASE_URL = "https://api.stratus.run"


def make_rollout_response(steps: int = 3, outcome: str = "success") -> dict:
    predictions = [
        {
            "step": i + 1,
            "predicted_state": {"step": i + 1, "magnitude": 0.5 + i * 0.1, "confidence": "high"},
            "action": {
                "action_id": i,
                "action_name": f"action {i}",
            },
            "state_change": 0.3 * (i + 1),
            "brain_confidence": 0.8 + i * 0.05,
        }
        for i in range(steps)
    ]
    return {
        "predictions": predictions,
        "summary": {
            "total_steps": steps,
            "outcome": outcome,
            "final_magnitude": 0.9,
        },
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


# --- predict ---

@pytest.mark.asyncio
@respx.mock
async def test_predict_success():
    respx.post(f"{BASE_URL}/v1/rollout").mock(
        return_value=httpx.Response(200, json=make_rollout_response(steps=3))
    )
    async with StratusClient(api_key="sk-test") as client:
        predictor = TrajectoryPredictor(client)
        result = await predictor.predict(initial_state="start", goal="reach goal")

    assert len(result.predictions) == 3
    assert result.summary["outcome"] == "success"


@pytest.mark.asyncio
@respx.mock
async def test_predict_quality_scoring():
    respx.post(f"{BASE_URL}/v1/rollout").mock(
        return_value=httpx.Response(200, json=make_rollout_response(steps=5))
    )
    async with StratusClient(api_key="sk-test") as client:
        predictor = TrajectoryPredictor(client, quality_threshold=50.0)
        result = await predictor.predict(initial_state="start", goal="goal", max_steps=5)

    assert "qualityScore" in result.summary
    assert result.summary["qualityScore"] >= 0.0


# --- predict_many ---

@pytest.mark.asyncio
@respx.mock
async def test_predict_many_parallel():
    respx.post(f"{BASE_URL}/v1/rollout").mock(
        return_value=httpx.Response(200, json=make_rollout_response(steps=2))
    )
    async with StratusClient(api_key="sk-test") as client:
        predictor = TrajectoryPredictor(client)
        results = await predictor.predict_many(
            [
                {"initial_state": "s1", "goal": "g1"},
                {"initial_state": "s2", "goal": "g2"},
                {"initial_state": "s3", "goal": "g3"},
            ]
        )

    assert len(results) == 3
    for r in results:
        assert isinstance(r, TrajectoryResult)


@pytest.mark.asyncio
@respx.mock
async def test_predict_many_progress_callback():
    respx.post(f"{BASE_URL}/v1/rollout").mock(
        return_value=httpx.Response(200, json=make_rollout_response(steps=2))
    )
    progress_calls = []

    async with StratusClient(api_key="sk-test") as client:
        predictor = TrajectoryPredictor(client)
        await predictor.predict_many(
            [{"initial_state": "s1", "goal": "g1"}, {"initial_state": "s2", "goal": "g2"}],
            on_progress=lambda done, total: progress_calls.append((done, total)),
        )

    assert len(progress_calls) > 0
    assert progress_calls[-1][0] == progress_calls[-1][1]


# --- find_optimal ---

def make_trajectory(quality: float, steps: int) -> TrajectoryResult:
    predictions = [
        StatePrediction(
            step=i + 1,
            predicted_state=None,
            action=Action(action_id=i, action_name="act"),
            state_change=0.3,
            brain_confidence=0.9,
        )
        for i in range(steps)
    ]
    return TrajectoryResult(
        predictions=predictions,
        summary={
            "qualityScore": quality,
            "totalSteps": steps,
            "goalAchieved": quality >= 80.0,
            "actions": ["act"] * steps,
            "outcome": "success",
            "finalMagnitude": 0.9,
        },
        usage=Usage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
    )


def test_find_optimal_returns_best():
    predictor = TrajectoryPredictor(None)  # type: ignore
    t1 = make_trajectory(quality=95.0, steps=3)
    t2 = make_trajectory(quality=70.0, steps=4)
    t3 = make_trajectory(quality=85.0, steps=2)

    result = predictor.find_optimal([t1, t2, t3], min_quality=80.0)
    assert result is not None
    assert result.summary["qualityScore"] == 95.0


def test_find_optimal_no_input_returns_none():
    predictor = TrajectoryPredictor(None)  # type: ignore
    result = predictor.find_optimal([], min_quality=80.0)
    assert result is None


def test_find_optimal_relaxes_when_none_qualify():
    predictor = TrajectoryPredictor(None)  # type: ignore
    t1 = make_trajectory(quality=50.0, steps=2)
    t2 = make_trajectory(quality=60.0, steps=3)

    # Falls back to all trajectories when none meet min_quality
    result = predictor.find_optimal([t1, t2], min_quality=80.0)
    assert result is not None
    # Returns best of all (highest quality)
    assert result.summary["qualityScore"] == 60.0


def test_find_optimal_filters_by_max_steps():
    predictor = TrajectoryPredictor(None)  # type: ignore
    t1 = make_trajectory(quality=95.0, steps=15)  # exceeds max_steps=10
    t2 = make_trajectory(quality=85.0, steps=5)

    result = predictor.find_optimal([t1, t2], min_quality=80.0, max_steps=10)
    assert result is not None
    assert result.summary["qualityScore"] == 85.0


# --- compare ---

def test_compare_best_and_worst():
    predictor = TrajectoryPredictor(None)  # type: ignore
    t1 = make_trajectory(quality=95.0, steps=3)
    t2 = make_trajectory(quality=60.0, steps=5)
    t3 = make_trajectory(quality=80.0, steps=4)

    result = predictor.compare([t1, t2, t3])
    assert "best" in result
    assert "worst" in result
    assert "average" in result
    assert result["average"]["qualityScore"] > 0


def test_compare_averages():
    predictor = TrajectoryPredictor(None)  # type: ignore
    t1 = make_trajectory(quality=100.0, steps=2)
    t2 = make_trajectory(quality=80.0, steps=4)

    result = predictor.compare([t1, t2])
    assert result["average"]["qualityScore"] == pytest.approx(90.0)
    assert result["average"]["steps"] == pytest.approx(3.0)


# --- get_summary ---

def test_get_summary_returns_string():
    predictor = TrajectoryPredictor(None)  # type: ignore
    t = make_trajectory(quality=88.0, steps=3)
    summary = predictor.get_summary(t)
    assert isinstance(summary, str)
    assert len(summary) > 0
