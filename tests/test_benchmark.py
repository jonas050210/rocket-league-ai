"""Benchmark statistics, recommendation and API tests (fake measurements only here;
genuine-run measurement is covered by the bounded integration test)."""

import time

import pytest
from conftest import requires_training
from pydantic import ValidationError

from rocket_league_ai.benchmark import (
    MAX_MEASUREMENT_STEPS,
    BenchmarkSpec,
    aggregate_iterations,
    recommend,
)
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
    with pytest.raises(ValidationError):  # 2 devices x 4 n_proc x ... > 8 candidates is fine, 5 values is not
        BenchmarkSpec(n_proc_values=[1, 2, 3, 4, 5, 6, 7, 8, 1][:5], devices=["cpu", "cuda:0"], batch_size=128)


def test_measurement_ceiling_is_clamped_to_the_step_cap():
    # The whole runbook compares batch 512/1024/2048; a high ceiling must be
    # clamped, never reject the sweep.
    assert BenchmarkSpec(batch_size=512).effective_measure_iterations == 1900
    assert BenchmarkSpec(batch_size=2048).effective_measure_iterations == 487
    assert BenchmarkSpec(batch_size=4096, measure_iterations=500).effective_measure_iterations == 243
    assert BenchmarkSpec(batch_size=4096, measure_iterations=10).effective_measure_iterations == 10
    spec = BenchmarkSpec(batch_size=2048)
    budget = spec.measurement_config(TrainingConfig(), {"device": "cpu", "n_proc": 2}).total_timesteps
    # Every warm-up and measured iteration must fit inside the run budget.
    assert (spec.warmup_iterations + spec.effective_measure_iterations) * 2048 <= budget
    assert budget <= MAX_MEASUREMENT_STEPS


def test_measurement_config_is_bounded_and_preserves_strategy():
    spec = BenchmarkSpec(batch_size=256, warmup_iterations=1, measure_iterations=3)
    config = spec.measurement_config(TrainingConfig(seed=7, learning_rate=0.001), {"device": "cpu", "n_proc": 2})
    # The budget must outlast the time window: a fast host reaches the 3
    # measured iterations long before the step budget, so the stop is time.
    assert config.total_timesteps == (MAX_MEASUREMENT_STEPS // 256) * 256
    assert config.total_timesteps > (1 + 3) * 256
    assert config.total_timesteps % config.timesteps_per_iteration == 0
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


def result(candidate, *, median=200.0, cv=0.05, status="completed", cpu=50.0, memory=40.0,
           gpu=None, vram=None, error=None):
    resources = {"cpu_percent_peak": cpu, "memory_percent_peak": memory}
    if gpu is not None:
        resources["gpu_percent_peak"] = gpu
    if vram is not None:
        resources["gpu_memory_percent_peak"] = vram
    return {"candidate": candidate, "status": status, "error": error,
            "aggregate": {"agent_steps_per_second": {"median": median, "cv": cv},
                          "simulation_steps_per_second": {"median": median},
                          "ppo_steps_per_second": {"median": median},
                          "episodes_per_minute": 3.0, "iterations_measured": 6} if status == "completed" else None,
            "resources": resources}


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


@requires_training
def test_benchmark_api_runs_injected_measurements(client, manager, monkeypatch):
    fake_iterations = iterations(1, collection=9.0, learning=9.0) + iterations(3)

    def fake_measure(config, spec):
        return fake_iterations, {"cpu_percent_peak": 42.0, "memory_percent_peak": 30.0,
                                 "gpu_percent_peak": None, "gpu_memory_used_mib_peak": None,
                                 "gpu_peak_bytes": None}, "f" * 32, None

    monkeypatch.setattr(manager, "_measure_once", fake_measure)
    spec = {"devices": ["cpu"], "n_proc_values": [1, 2], "batch_size": 128,
            "warmup_iterations": 1, "measure_iterations": 3, "repeats": 2}
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
    assert client.post("/api/benchmarks", json={"repeats": 9}).status_code == 422
    assert client.post("/api/benchmarks", json={"timeout_seconds": 30, "measurement_seconds": 40}).status_code == 422
    assert client.post("/api/benchmarks", json={"batch_size": 128, "n_proc_values": [1, 2, 4, 8, 6]}).status_code == 422
    assert client.post("/api/benchmarks", json={"measurement_seconds": 30, "timeout_seconds": 30}).status_code == 422


def test_combine_repeats_excludes_each_repeats_warmup_and_counts_episodes_per_run():
    from rocket_league_ai.benchmark import combine_repeats

    def repeat(warmup_collection, measured_collection, episodes):
        iterations = [{"steps": 128, "collection_seconds": warmup_collection, "learning_seconds": 0.1,
                       "iteration_seconds": warmup_collection + 0.1, "episodes": 0}]
        iterations += [{"steps": (i + 2) * 128, "collection_seconds": measured_collection,
                        "learning_seconds": 0.1, "iteration_seconds": measured_collection + 0.1,
                        "episodes": episodes} for i in range(2)]
        return aggregate_iterations(iterations, warmup=1, batch_size=128)

    combined = combine_repeats([repeat(99.0, 0.4, 4), repeat(88.0, 0.2, 6)])
    # Neither repeat's slow warm-up leaks into the combined simulation rate.
    assert combined["simulation_steps_per_second"]["repeat_medians"] == pytest.approx([128 / 0.4, 128 / 0.2])
    # Episode spans are counted inside each run, never across run boundaries.
    assert combined["episodes_measured"] == 10
    assert combined["repeats"] == 2 and combined["iterations_measured"] == 4


def test_episodes_per_minute_uses_last_warmup_as_baseline():
    iterations = [
        {"steps": 128, "collection_seconds": 1.0, "learning_seconds": 0.1, "iteration_seconds": 1.1, "episodes": 0},
        {"steps": 256, "collection_seconds": 1.0, "learning_seconds": 0.1, "iteration_seconds": 1.1, "episodes": 4},
        {"steps": 384, "collection_seconds": 1.0, "learning_seconds": 0.1, "iteration_seconds": 1.1, "episodes": 8},
        {"steps": 512, "collection_seconds": 1.0, "learning_seconds": 0.1, "iteration_seconds": 1.1, "episodes": 12},
    ]
    aggregate = aggregate_iterations(iterations, warmup=2, batch_size=128)
    # Episodes 0 and 4 belong to warm-up; only 8 and 12 are measured.
    assert aggregate["episodes_measured"] == 8
    assert aggregate["episodes_per_minute"] == pytest.approx(8 / 2.2 * 60)


def test_single_repeat_keeps_within_run_variation():
    from rocket_league_ai.benchmark import combine_repeats

    iterations = [{"steps": 128, "collection_seconds": 9.0, "learning_seconds": 0.1,
                   "iteration_seconds": 9.1, "episodes": 0}]
    iterations += [{"steps": (i + 2) * 128, "collection_seconds": collection,
                    "learning_seconds": 0.1, "iteration_seconds": collection + 0.1,
                    "episodes": i * 2} for i, collection in enumerate([0.4, 0.2, 0.4])]
    combined = combine_repeats([aggregate_iterations(iterations, warmup=1, batch_size=128)])
    # One repeat has zero between-repeat variance, but this run was not stable.
    assert combined["agent_steps_per_second"]["cv_within_repeats"] > 0.25
    assert combined["agent_steps_per_second"]["cv"] > 0.25


def test_auto_default_requires_tight_cv_repeats_and_gpu_telemetry():
    candidate = {"device": "cpu", "n_proc": 1}
    spec = BenchmarkSpec(devices=["cpu"], n_proc_values=[1], batch_size=128,
                         repeats=3, auto_default=True)
    stable = recommend([result(candidate, cv=0.05)], spec)
    assert stable["auto_default"] is True and stable["safety"]["gpu_verified"] is True
    loose = recommend([result(candidate, cv=0.15)], spec)
    assert loose["auto_default"] is False
    assert any("CV" in reason for reason in loose["auto_default_blockers"])
    fewer = recommend([result(candidate, cv=0.05)],
                      BenchmarkSpec(devices=["cpu"], n_proc_values=[1], batch_size=128,
                                    repeats=2, auto_default=True))
    assert fewer["auto_default"] is False
    cuda = recommend([result({"device": "cuda:0", "n_proc": 1}, cv=0.05, gpu=None)],
                     BenchmarkSpec(devices=["cuda:0"], n_proc_values=[1], batch_size=128,
                                   repeats=3, auto_default=True))
    assert cuda["candidate"] is None
    assert "GPU telemetry" in cuda["excluded"][0]["reasons"][0]


def test_benchmark_spec_enforces_wall_clock_and_wave_validation():
    assert BenchmarkSpec().measurement_seconds == 20
    with pytest.raises(ValidationError):
        BenchmarkSpec(n_proc_values=[3], batch_size=128)
    with pytest.raises(ValidationError):
        BenchmarkSpec(measurement_seconds=30, timeout_seconds=30)
    with pytest.raises(ValidationError):
        BenchmarkSpec(min_iterations=5, measure_iterations=4)


def test_auto_adoption_changes_only_execution_fields(manager, monkeypatch):
    plan = TrainingConfig(
        name="Adoption test", device="cpu", n_proc=1, hidden_sizes=(32, 32),
        total_timesteps=512, timesteps_per_iteration=128, batch_size=128,
        buffer_size=384, n_minibatches=2, episode_timeout_seconds=5,
        no_touch_timeout_seconds=5,
    )
    manager.store.set_settings({"plan": plan.model_dump(mode="json"), "plan_revision": 4})
    monkeypatch.setattr(
        "rocket_league_ai.orchestration.package_versions",
        lambda: {"torch": "test", "numpy": "test", "rlgym-learn": "test"},
    )
    measured = iterations(4)
    calls = []

    def fake_measure(config, spec):
        calls.append((config.device, config.n_proc, config.batch_size))
        return measured, {"cpu_percent_peak": 25.0, "memory_percent_peak": 12.0,
                          "gpu_percent_peak": None, "gpu_memory_used_mib_peak": None,
                          "gpu_memory_percent_peak": None, "gpu_peak_bytes": None,
                          "startup_seconds": 1.0, "measurement_seconds": 20.0,
                          "stop_reason": "time"}, "r" * 32, None

    monkeypatch.setattr(manager, "_measure_once", fake_measure)
    spec = BenchmarkSpec(devices=["cpu"], n_proc_values=[2], batch_size=128,
                         warmup_iterations=1, measure_iterations=3, repeats=3,
                         measurement_seconds=20, auto_default=True)
    session = manager.start_benchmark(spec)
    deadline = time.time() + 10
    while time.time() < deadline:
        if manager.store.benchmark(session["id"])["status"] != "running":
            break
        time.sleep(0.05)
    stored = manager.store.benchmark(session["id"])
    assert stored["status"] == "completed", stored
    assert stored["recommendation"]["auto_default"] is True
    assert stored["recommendation"]["adoption"]["status"] == "adopted"
    adopted = TrainingConfig.model_validate(manager.store.setting("plan"))
    assert adopted.n_proc == 2 and adopted.device == "cpu"
    # Batch, network, reward and seed are not throughput evidence.
    assert adopted.batch_size == 128 and adopted.hidden_sizes == (32, 32)
    assert adopted.total_timesteps == 512 and adopted.seed == plan.seed
    assert manager.store.setting("plan_revision") == 5
    record = manager.store.setting("benchmark_adoption")
    assert record["previous_plan"]["n_proc"] == 1
    assert len(calls) == 3 and all(call[2] == 128 for call in calls)
