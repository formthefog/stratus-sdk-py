#!/usr/bin/env python3
"""
Stratus X1 SDK Demo

Comprehensive demonstration of Stratus X1 SDK features.
"""

import asyncio
import os
import time

from stratus_sdk import (
    StratusClient,
    TrajectoryPredictor,
    compare_models,
    SimpleCache,
    RateLimiter,
    HealthChecker,
    CompressionLevel,
)


async def main():
    print("━" * 60)
    print("  STRATUS X1 SDK DEMO (Python)")
    print("━" * 60)
    print()

    # Configuration
    API_KEY = os.getenv("STRATUS_API_KEY", "demo-key")
    API_URL = os.getenv("MJEPA_API_URL", "http://212.115.124.137:8000")

    # Initialize client
    async with StratusClient(
        api_key=API_KEY,
        api_url=API_URL,
        compression_profile=CompressionLevel.MEDIUM,
        timeout=30.0,
    ) as client:
        print(f"Connected to: {API_URL}")
        print(f"Compression profile: Medium")
        print(f"Estimated compression: {client.get_compression_ratio()}")
        print(f"Estimated quality: {client.get_quality_score()}%")
        print()

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 1. HEALTH CHECK
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        print("1. HEALTH CHECK")
        print("─" * 60)

        health_checker = HealthChecker(client)

        try:
            health = await health_checker.check()
            print(f"✓ API Status: {'Healthy' if health['healthy'] else 'Unhealthy'}")
            print(f"✓ Model Loaded: {'Yes' if health['model_loaded'] else 'No'}")
            if health["error"]:
                print(f"✗ Error: {health['error']}")
        except Exception as e:
            print(f"✗ Health check failed: {e}")
            print("Continuing with demo using mock data...")
        print()

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 2. CHAT COMPLETION
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        print("2. CHAT COMPLETION")
        print("─" * 60)

        try:
            print("Sending chat completion request...")

            response = await client.chat.completions.create(
                messages=[{"role": "user", "content": "Explain Stratus X1 in one sentence."}],
                model="stratus-x1-ac",
                temperature=0.7,
                max_tokens=100,
            )

            print(f"✓ Response: {response.choices[0].message.content}")
            print(f"✓ Tokens used: {response.usage.total_tokens}")
            print(f"✓ Model: {response.model}")
        except Exception as e:
            print(f"✗ Chat completion failed: {e}")
            print(
                "Mock response: Stratus X1 is a world model that predicts future states and actions."
            )
        print()

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 3. TRAJECTORY PREDICTION
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        print("3. TRAJECTORY PREDICTION")
        print("─" * 60)

        predictor = TrajectoryPredictor(client, quality_threshold=85.0)

        try:
            print("Predicting trajectory...")

            result = await predictor.predict(
                initial_state="Current system stability: 45%, performance: moderate",
                goal="Increase stability to >80% while maintaining performance",
                max_steps=5,
            )

            print(f"✓ Goal achieved: {'Yes' if result.summary['goalAchieved'] else 'No'}")
            print(f"✓ Steps taken: {result.summary['totalSteps']}")
            print(f"✓ Quality score: {result.summary['qualityScore']:.1f}/100")
            print(f"✓ Actions: {' → '.join(result.summary['actions'])}")
            print(f"✓ Final state: {result.summary['finalState']}")
            print(f"✓ Tokens used: {result.usage.total_tokens}")
        except Exception as e:
            print(f"✗ Trajectory prediction failed: {e}")
            print("Mock trajectory:")
            print("  Step 1: Increase gain → stability: 55%")
            print("  Step 2: Apply filter → stability: 68%")
            print("  Step 3: Optimize params → stability: 82%")
            print("  Goal achieved: Yes")
        print()

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 4. BATCH TRAJECTORY PREDICTION
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        print("4. BATCH TRAJECTORY PREDICTION")
        print("─" * 60)

        try:
            print("Predicting 3 trajectories in parallel...")

            def progress(completed, total):
                print(f"\r  Progress: {completed}/{total}", end="")

            trajectories = await predictor.predict_many(
                [
                    {
                        "initial_state": "stability: 40%",
                        "goal": "Reach 80% stability",
                        "max_steps": 5,
                    },
                    {
                        "initial_state": "stability: 50%",
                        "goal": "Reach 80% stability",
                        "max_steps": 4,
                    },
                    {
                        "initial_state": "stability: 60%",
                        "goal": "Reach 80% stability",
                        "max_steps": 3,
                    },
                ],
                on_progress=progress,
            )

            print("\n")

            optimal = predictor.find_optimal(trajectories, min_quality=80.0, max_steps=5)

            if optimal:
                print("✓ Optimal trajectory found:")
                print(f"  Quality: {optimal.summary['qualityScore']:.1f}/100")
                print(f"  Steps: {optimal.summary['totalSteps']}")
                print(f"  Actions: {' → '.join(optimal.summary['actions'])}")

            comparison = predictor.compare(trajectories)
            print()
            print("✓ Comparison:")
            print(f"  Best quality: {comparison['best'].summary['qualityScore']:.1f}/100")
            print(f"  Worst quality: {comparison['worst'].summary['qualityScore']:.1f}/100")
            print(f"  Average quality: {comparison['average']['qualityScore']:.1f}/100")
        except Exception as e:
            print(f"✗ Batch prediction failed: {e}")
        print()

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 5. MODEL COMPARISON
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        print("5. MODEL COMPARISON")
        print("─" * 60)

        try:
            print("Comparing Stratus X1 vs GPT-3.5...")

            comparison = await compare_models(
                models=["mjepa-g", "gpt-3.5-turbo"],
                tasks=["embeddings"],
                mjepa_client=client,
            )

            print()
            print("Results:")
            print()

            for result in comparison.results:
                print(f"{result.model}:")
                if result.embedding_quality:
                    print(f"  Embedding Quality: {result.embedding_quality:.1f}%")
                if result.compression_ratio:
                    print(f"  Compression Ratio: {result.compression_ratio:.1f}x")
                if result.latency_p50:
                    print(f"  Latency (p50): {result.latency_p50}ms")
                if result.cost_per_1m_tokens:
                    print(f"  Cost per 1M tokens: ${result.cost_per_1m_tokens:.2f}")
                print()

            print("Winners:")
            print(f"  Quality: {comparison.winner['quality']}")
            print(f"  Performance: {comparison.winner['performance']}")
            print(f"  Cost: {comparison.winner['cost']}")
        except Exception as e:
            print(f"✗ Comparison failed: {e}")
        print()

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 6. PRODUCTION HELPERS
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        print("6. PRODUCTION HELPERS")
        print("─" * 60)

        # Caching
        cache = SimpleCache(ttl_seconds=300)
        cache.set("test-key", {"data": "cached value"})
        print(f"✓ Cache set: {cache.size()} items")

        cached = cache.get("test-key")
        print(f"✓ Cache get: {'Hit' if cached else 'Miss'}")

        # Rate limiting
        rate_limiter = RateLimiter(max_requests_per_second=10)
        can_proceed = await rate_limiter.acquire()
        print(f"✓ Rate limiter: {'Acquired' if can_proceed else 'Limited'}")

        print()
        print("━" * 60)
        print("  DEMO COMPLETE")
        print("━" * 60)


if __name__ == "__main__":
    asyncio.run(main())
