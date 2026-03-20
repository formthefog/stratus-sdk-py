"""
Trajectory prediction and analysis.

High-level tools for M-JEPA-G world modeling.
"""

from typing import Callable, List, Optional

from .client import MJepaGClient
from .types import StatePrediction, TrajectoryResult, Usage


class TrajectoryPredictor:
    """
    Trajectory predictor for multi-step planning.

    Provides high-level tools for:
    - Single trajectory prediction
    - Batch prediction with concurrency
    - Quality scoring
    - Trajectory optimization
    """

    def __init__(
        self, client: MJepaGClient, quality_threshold: float = 80.0
    ):
        """
        Initialize trajectory predictor.

        Args:
            client: M-JEPA-G client instance
            quality_threshold: Minimum quality score (0-100)
        """
        self.client = client
        self.quality_threshold = quality_threshold

    async def predict(
        self,
        initial_state: str,
        goal: str,
        max_steps: int = 10,
        quality_threshold: Optional[float] = None,
    ) -> TrajectoryResult:
        """
        Predict a single trajectory.

        Args:
            initial_state: Initial state description
            goal: Goal to achieve
            max_steps: Maximum steps to predict
            quality_threshold: Override default quality threshold

        Returns:
            TrajectoryResult with predictions and metrics

        Example:
            >>> result = await predictor.predict(
            ...     initial_state="stability: 45%",
            ...     goal="Reach 80% stability",
            ...     max_steps=5
            ... )
        """
        response = await self.client.rollout(
            goal=goal, initial_state=initial_state, max_steps=max_steps
        )

        quality_score = self._score_trajectory(response.predictions, max_steps)

        threshold = quality_threshold or self.quality_threshold
        goal_achieved = quality_score >= threshold

        return TrajectoryResult(
            predictions=response.predictions,
            summary={
                "totalSteps": response.summary.total_steps,
                "goalAchieved": goal_achieved,
                "qualityScore": quality_score,
                "actions": [p.action.action_name for p in response.predictions],
                "outcome": response.summary.outcome,
                "finalMagnitude": response.summary.final_magnitude,
            },
            usage=response.usage or Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

    async def predict_many(
        self,
        trajectories: List[dict],
        quality_threshold: Optional[float] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[TrajectoryResult]:
        """
        Predict multiple trajectories in parallel.

        Args:
            trajectories: List of trajectory configs
            quality_threshold: Filter by minimum quality
            on_progress: Progress callback (completed, total)

        Returns:
            List of TrajectoryResult objects

        Example:
            >>> results = await predictor.predict_many([
            ...     {"initial_state": "state A", "goal": "goal X", "max_steps": 5},
            ...     {"initial_state": "state B", "goal": "goal Y", "max_steps": 5},
            ... ])
        """
        results = []
        total = len(trajectories)

        for i, traj in enumerate(trajectories):
            result = await self.predict(
                initial_state=traj["initial_state"],
                goal=traj["goal"],
                max_steps=traj.get("max_steps", 10),
                quality_threshold=quality_threshold,
            )
            results.append(result)

            if on_progress:
                on_progress(i + 1, total)

        # Filter by quality if threshold specified
        if quality_threshold:
            results = [r for r in results if r.summary["qualityScore"] >= quality_threshold]

        return results

    def find_optimal(
        self,
        trajectories: List[TrajectoryResult],
        min_quality: float = 80.0,
        max_steps: int = 10,
        cost_function: Optional[Callable[[StatePrediction], float]] = None,
    ) -> Optional[TrajectoryResult]:
        """
        Find optimal trajectory given criteria.

        Args:
            trajectories: List of trajectory results
            min_quality: Minimum quality score
            max_steps: Maximum allowed steps
            cost_function: Custom cost function (higher = better)

        Returns:
            Best trajectory or None

        Example:
            >>> best = predictor.find_optimal(
            ...     trajectories,
            ...     min_quality=85,
            ...     cost_function=lambda p: (p.brain_confidence or 0.0) * p.state_change
            ... )
        """
        if not trajectories:
            return None

        # Filter by criteria
        candidates = [
            t
            for t in trajectories
            if t.summary["qualityScore"] >= min_quality
            and t.summary["totalSteps"] <= max_steps
        ]

        if not candidates:
            candidates = trajectories  # Relax constraints

        # Apply cost function or default scoring
        if cost_function:
            best = max(
                candidates,
                key=lambda t: sum(cost_function(p) for p in t.predictions)
                / len(t.predictions),
            )
        else:
            # Default: highest quality, then fewest steps
            best = max(
                candidates,
                key=lambda t: (
                    t.summary["qualityScore"],
                    -t.summary["totalSteps"],
                ),
            )

        return best

    def compare(self, trajectories: List[TrajectoryResult]) -> dict:
        """
        Compare multiple trajectories.

        Args:
            trajectories: List of trajectory results

        Returns:
            Comparison dict with best, worst, and averages

        Example:
            >>> comparison = predictor.compare(trajectories)
            >>> print(f"Best quality: {comparison['best'].summary['qualityScore']}")
        """
        if not trajectories:
            return {
                "best": None,
                "worst": None,
                "average": {"qualityScore": 0.0, "steps": 0.0, "confidence": 0.0},
            }

        sorted_by_quality = sorted(
            trajectories, key=lambda t: t.summary["qualityScore"], reverse=True
        )

        avg_quality = sum(t.summary["qualityScore"] for t in trajectories) / len(
            trajectories
        )
        avg_steps = sum(t.summary["totalSteps"] for t in trajectories) / len(trajectories)

        avg_confidence = sum(
            sum((p.brain_confidence or 0.0) for p in t.predictions) / len(t.predictions)
            for t in trajectories
        ) / len(trajectories)

        return {
            "best": sorted_by_quality[0],
            "worst": sorted_by_quality[-1],
            "average": {
                "qualityScore": avg_quality,
                "steps": avg_steps,
                "confidence": avg_confidence,
            },
        }

    def get_summary(self, trajectory: TrajectoryResult) -> str:
        """
        Get human-readable summary.

        Args:
            trajectory: Trajectory result

        Returns:
            Formatted summary string
        """
        s = trajectory.summary
        lines = [
            f"Steps: {s['totalSteps']}",
            f"Goal: {'Achieved ✓' if s['goalAchieved'] else 'Not achieved ✗'}",
            f"Quality: {s['qualityScore']:.1f}/100",
            f"Actions: {' → '.join(s['actions'])}",
            f"Outcome: {s['outcome']}",
        ]
        return "\n".join(lines)

    def _score_trajectory(
        self, predictions: List[StatePrediction], max_steps: int
    ) -> float:
        """
        Score trajectory quality.

        Components:
        - Confidence: average confidence across steps (40%)
        - Progress: total state change (40%)
        - Efficiency: fewer steps = higher score (20%)

        Args:
            predictions: List of state predictions
            max_steps: Maximum allowed steps

        Returns:
            Quality score (0-100)
        """
        if not predictions:
            return 0.0

        # Average confidence (0-1 -> 0-100)
        avg_confidence = (
            sum((p.brain_confidence or 0.0) for p in predictions) / len(predictions)
        ) * 100

        # Total progress (normalized to 0-100)
        total_progress = sum(p.state_change for p in predictions)
        normalized_progress = min(total_progress * 10, 100)

        # Efficiency (fewer steps = better, but completing matters)
        efficiency = 100 * (1 - len(predictions) / max_steps) if predictions else 0

        # Weighted combination
        score = avg_confidence * 0.4 + normalized_progress * 0.4 + efficiency * 0.2

        return max(0.0, min(score, 100.0))
