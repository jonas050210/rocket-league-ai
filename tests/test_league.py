"""Opponent pool: bounded storage, matchmaking weights and Elo updates."""

from __future__ import annotations

import numpy as np
import pytest

from rlai.league import DEFAULT_ELO, OpponentPool, expected_score


def test_pool_is_bounded_and_drops_oldest():
    pool = OpponentPool(capacity=3)
    for index in range(5):
        pool.add(f"snap-{index}", index * 10, {"w": index})
    assert len(pool) == 3
    assert [entry.id for entry in pool.entries] == ["snap-2", "snap-3", "snap-4"]


def test_transient_states_are_not_serialised():
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
