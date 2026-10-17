"""Benchmark statistics, recommendation and API tests (fake measurements only here;
genuine-run measurement is covered by the bounded integration test)."""

import time

import pytest
from pydantic import ValidationError

from rocket_league_ai.benchmark import BenchmarkSpec, aggregate_iterations, recommend
from rocket_league_ai.config import TrainingConfig


def iterations(count, *, collection=0.4, learning=0.1, episodes_each=2):
    return [
        {"steps": (i + 1) * 128, "collection_seconds": collection, "learning_seconds": learning,
         "iteration_seconds": collection + learning, "episodes": (i + 1) * episodes_each}
        for i in range(count)
    ]


def test_spec_rejects_unbounded_or_duplicate_candidates():
    with pytest.raises(ValidationError):
        BenchmarkSpec(devices=["cpu", "cpu"])
    with pytest.raises(ValidationError):  # (1 + 20) * 4096 exceeds the per-run step cap
        BenchmarkSpec(batch_size=4096, measure_iterations=20)
    with pytest.raises(ValidationError):  # 2 devices x 4 n_proc x ... > 8 candidates is fine, 5 values is not
        BenchmarkSpec(n_proc_values=[1, 2, 3, 4, 5, 6, 7, 8, 1][:5], devices=["cpu", "cuda:0"], batch_size=128)


def test_measurement_config_is_bounded_and_preserves_strategy():
    spec = BenchmarkSpec(batch_size=256, warmup_iterations=1, measure_iterations=3)
    config = spec.measurement_config(TrainingConfig(seed=7, learning_rate=0.001), {"device": "cpu", "n_proc": 2})
    assert config.total_timesteps == 4 * 256
    assert config.seed == 7 and config.learning_rate == 0.001
    TrainingConfig.model_validate(config.model_dump())  # still a valid training config


def test_aggregate_excludes_warmup_and_splits_throughputs():
    measured = [
        {"steps": 128, "collection_seconds": 9.9, "learning_seconds": 9.9, "iteration_seconds": 19.8, "episodes": 2},
        *[{"steps": (i + 2) * 128, "collection_seconds": 0.4, "learning_seconds": 0.1,
           "iteration_seconds": 0.5, "episodes": (i + 2) * 2} for i in range(3)],
    ]
    aggregate = aggregate_iterations(measured, warmup=1, batch_size=128)
    assert aggregate["iterations_measured"] == 3
    # simulation throughput uses collection time only; PPO uses learning time only
    assert aggregate["simulation_steps_per_second"]["median"] == pytest.approx(128 / 0.4)
    assert aggregate["ppo_steps_per_second"]["median"] == pytest.approx(128 / 0.1)
    assert aggregate["agent_steps_per_second"]["median"] == pytest.approx(128 / 0.5)
    assert aggregate["episodes_measured"] == 6


def result(candidate, *, median=200.0, cv=0.05, status="completed", cpu=50.0, memory=40.0, error=None):
    return {"candidate": candidate, "status": status, "error": error,
            "aggregate": {"agent_steps_per_second": {"median": median, "cv": cv},
                          "simulation_steps_per_second": {"median": median},
                          "ppo_steps_per_second": {"median": median},
                          "episodes_per_minute": 3.0, "iterations_measured": 6} if status == "completed" else None,
            "resources": {"cpu_percent_peak": cpu, "memory_percent_peak": memory}}


def test_recommend_picks_fastest_stable_and_explains_exclusions():
    spec = BenchmarkSpec()
    results = [
        result({"device": "cpu", "n_proc": 1}, median=100),
        result({"device": "cpu", "n_proc": 4}, median=300),
        result({"device": "cuda:0", "n_proc": 4}, median=900, cv=0.9),  # unstable
        result({"device": "cuda:0", "n_proc": 8}, median=950, memory=99.0),  # over memory limit
    ]
    recommendation = recommend(results, spec)
    assert recommendation["candidate"] == {"device": "cpu", "n_proc": 4}
    assert "fastest stable" in recommendation["explanation"]
    excluded = {tuple(sorted(item["candidate"].items())) for item in recommendation["excluded"]}
    assert len(excluded) == 2


def test_recommend_refuses_when_nothing_is_stable():
    spec = BenchmarkSpec()
    recommendation = recommend([result({"device": "cpu", "n_proc": 1}, status="failed", error="boom")], spec)
    assert recommendation["candidate"] is None
    assert "No candidate" in recommendation["explanation"]


def test_benchmark_api_runs_injected_measurements(client, manager, monkeypatch):
    fake_iterations = iterations(1, collection=9.0, learning=9.0) + iterations(3)

    def fake_measure(config, spec):
        return fake_iterations, {"cpu_percent_peak": 42.0, "memory_percent_peak": 30.0,
                                 "gpu_percent_peak": None, "gpu_memory_used_mib_peak": None,
                                 "gpu_peak_bytes": None}, "f" * 32, None

    monkeypatch.setattr(manager, "_measure_once", fake_measure)
    spec = {"devices": ["cpu"], "n_proc_values": [1, 2], "batch_size": 128,
            "warmup_iterations": 1, "measure_iterations": 3, "repeats": 1}
    response = client.post("/api/benchmarks", json=spec)
    assert response.status_code == 200
    session_id = response.json()["id"]
    deadline = time.time() + 30
    while time.time() < deadline:
        session = manager.store.benchmark(session_id)
        if session["status"] == "completed":
            break
        time.sleep(0.05)
    assert session["status"] == "completed"
    results = manager.store.benchmark_results(session_id)
    assert len(results) == 2 and all(item["status"] == "completed" for item in results)
    assert session["recommendation"]["candidate"] in ({"device": "cpu", "n_proc": 1}, {"device": "cpu", "n_proc": 2})
    snapshot = client.get("/api/snapshot").json()
    assert snapshot["benchmark"]["session"]["id"] == session_id
    # A second benchmark cannot start while... it is finished, so it can; but not concurrently with itself
    assert client.post("/api/benchmarks/stop").status_code == 409


def test_benchmark_api_validation(client):
    assert client.post("/api/benchmarks", json={"devices": ["tpu"]}).status_code == 422
    assert client.post("/api/benchmarks", json={"batch_size": 4096, "measure_iterations": 20}).status_code == 422
