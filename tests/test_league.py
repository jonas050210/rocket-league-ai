"""Opponent pool: bounded storage, matchmaking weights and Elo updates."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rlai.league import DEFAULT_ELO, OpponentPool, expected_score


def test_pool_is_bounded_and_drops_oldest():
    pool = OpponentPool(capacity=3)
    for index in range(5):
        pool.add(f"snap-{index}", index * 10, {"w": index})
    assert len(pool) == 3
    assert [entry.id for entry in pool.entries] == ["snap-2", "snap-3", "snap-4"]


def test_states_are_not_serialised_by_default():
    pool = OpponentPool(capacity=2)
    pool.add("a", 1, {"weight": np.zeros(3, dtype=np.float32)})
    data = pool.to_dict()
    assert "state" not in data["entries"][0]
    restored = OpponentPool.from_dict(data)
    assert restored.entries[0].state is None
    assert restored.agent_elo == pool.agent_elo
    # a restored pool keeps ratings but has nothing playable yet
    assert restored.usable == 0
    with pytest.raises(ValueError, match="playable"):
        restored.sample()


def test_elo_moves_toward_results_and_is_zero_sum():
    pool = OpponentPool(capacity=2)
    pool.add("copy", 0, {})
    before_agent, before_entry = pool.agent_elo, pool.entries[0].elo
    pool.record("copy", wins=8, losses=2, draws=0)
    assert pool.agent_elo > before_agent
    assert pool.entries[0].elo < before_entry
    assert pool.agent_elo + pool.entries[0].elo == pytest.approx(
        before_agent + before_entry
    )
    # a losing record moves the rating the other way
    after_wins = pool.agent_elo
    pool.record("copy", wins=0, losses=10, draws=0)
    assert pool.agent_elo < after_wins
    assert pool.entries[0].elo > before_entry


def test_unknown_entry_is_an_error():
    pool = OpponentPool()
    with pytest.raises(KeyError):
        pool.record("missing", 1, 0, 0)
    with pytest.raises(ValueError):
        pool.sample()


def test_pfsp_prefers_even_matchups():
    pool = OpponentPool(capacity=2)
    weak = pool.add("weak", 0, {})
    even = pool.add("even", 0, {})
    pool.agent_elo = 1000.0
    weak.elo = 600.0
    even.elo = 1000.0
    assert pool._weight(even) > pool._weight(weak)

    rng = np.random.default_rng(7)
    assert pool.usable == 2
    picks = [pool.sample(rng).id for _ in range(400)]
    assert picks.count("even") > picks.count("weak")


def test_min_weight_keeps_hopeless_opponents_sampleable():
    pool = OpponentPool(capacity=1, min_weight=0.05)
    pool.add("stronger", 0, {})
    pool.entries[0].elo = 3000.0
    assert pool._weight(pool.entries[0]) == pytest.approx(0.05)


def test_expected_score_is_symmetric():
    assert expected_score(1000, 1000) == pytest.approx(0.5)
    assert expected_score(1200, 1000) > 0.5
    assert expected_score(1200, 1000) + expected_score(1000, 1200) == pytest.approx(1.0)


def test_snapshot_reports_games_and_win_rate():
    pool = OpponentPool()
    entry = pool.add("snap", 42, {})
    assert entry.summary()["games"] == 0
    pool.record("snap", 3, 1, 0)
    summary = pool.snapshot()["entries"][0]
    assert summary["games"] == 4
    assert summary["win_rate"] == pytest.approx(0.75)
    assert summary["step"] == 42
    assert DEFAULT_ELO == 1000.0


def test_pool_states_round_trip_and_stay_frozen():
    """Self-play continuation needs the real opponents, not only their ratings."""
    pool = OpponentPool(capacity=3)
    states = [
        {"actor.0.weight": torch.full((2, 2), float(index + 1))} for index in range(3)
    ]
    for index, state in enumerate(states):
        pool.add(f"snapshot-{index}", step=100 * (index + 1), state=state)
    pool.record("snapshot-2", wins=2, losses=1, draws=0)

    # default serialisation keeps ratings only (old behaviour, small checkpoints)
    ratings_only = OpponentPool.from_dict(pool.to_dict())
    assert len(ratings_only) == 3 and ratings_only.usable == 0
    assert ratings_only.agent_elo == pytest.approx(pool.agent_elo)
    assert ratings_only.to_dict()["states_included"] is False

    # with states the restored pool is immediately playable
    restored = OpponentPool.from_dict(pool.to_dict(include_states=True))
    assert restored.usable == 3
    assert [entry.id for entry in restored.entries] == [entry.id for entry in pool.entries]
    assert [entry.elo for entry in restored.entries] == [entry.elo for entry in pool.entries]
    for original, copy in zip(pool.entries, restored.entries):
        assert torch.equal(original.state["actor.0.weight"], copy.state["actor.0.weight"])
    # the restored weights are copies, so training cannot mutate the pool
    assert (
        restored.entries[0].state["actor.0.weight"].data_ptr()
        != pool.entries[0].state["actor.0.weight"].data_ptr()
    )
    with torch.no_grad():
        pool.entries[0].state["actor.0.weight"].add_(10.0)
    assert torch.equal(
        restored.entries[0].state["actor.0.weight"], torch.full((2, 2), 1.0)
    )
    entry = restored.sample(np.random.default_rng(0))
    assert entry.state is not None


def test_pool_states_can_be_capped():
    pool = OpponentPool(capacity=4)
    for index in range(4):
        pool.add(f"snapshot-{index}", step=index, state={"w": torch.zeros(2)})
    data = pool.to_dict(include_states=True, max_states=2)
    assert data["states_included"] is True
    kept = [entry["has_state"] for entry in data["entries"]]
    assert kept == [False, False, True, True]  # newest snapshots win
    restored = OpponentPool.from_dict(data)
    assert restored.usable == 2 and len(restored) == 4
    assert pool.to_dict(include_states=True, max_states=0)["states_included"] is False
