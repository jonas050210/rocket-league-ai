"""The new feature modules: Bot Brain, difficulties, teams, memory, rivals.

These modules are the "missing features" of the plan.  They are pure data
and pure logic, so they can be tested exhaustively without a simulator and
without a single training step.

The rules that the tests protect:

* a difficulty percentage is a *design target*, never a measured skill -
  the code has to say so;
* the behaviour layer is deterministic and does not touch the simulator;
* the team builder rejects invalid input with a message a user can act on;
* the player memory stays bounded (entries and age) and is separate from
  the model weights;
* the rival system only reports what was actually measured.
"""

from __future__ import annotations

import pytest

from rlai.adaptive import (
    AdaptationLimits,
    AdaptationPolicy,
    AdaptiveController,
)
from rlai.botbrain import (
    PROFILES,
    BrainContext,
    BotBrain,
    available_profiles,
    merge_overrides,
    profiles_differ,
)
from rlai.difficulty import (
    DIFFICULTY_NAMES,
    brain_parameters,
    custom_profile,
    describe_all,
    get_profile,
    list_profiles,
    skill_estimate,
)
from rlai.player_memory import (
    MAX_ENTRIES,
    PlayerMemory,
    memory_from_rows,
    rows_from_memory,
)
from rlai.rivals import RivalRegistry
from rlai.team_builder import (
    BotSlot,
    Team,
    default_team,
    describe_team,
    opponent_names,
    team_brains,
    validate_team,
)


# ------------------------------------------------------------------ bot brain
def test_every_shipped_profile_is_offered():
    names = {entry["name"] for entry in available_profiles()}
    assert names == set(PROFILES)
    assert names == {"aggressive", "defensive", "balanced", "adaptive"}


def test_an_unknown_profile_is_rejected():
    with pytest.raises(ValueError) as excinfo:
        BotBrain(profile="kamikaze")
    assert "kamikaze" in str(excinfo.value)


def test_the_brain_is_deterministic():
    brain = BotBrain("balanced")
    context = BrainContext(distance_to_ball=300.0, can_shoot=True)
    first = brain.decide(context)
    second = brain.decide(context)
    assert first.intent == second.intent
    assert first.parameters == second.parameters


def test_the_brain_shoots_when_the_geometry_allows_it():
    brain = BotBrain("aggressive")
    decision = brain.decide(BrainContext(can_shoot=True, distance_to_ball=100.0))
    assert decision.intent == "shoot"


def test_the_brain_rotates_when_the_opponent_is_closer():
    brain = BotBrain("defensive")
    decision = brain.decide(
        BrainContext(opponents_closer=1, ball_in_attacking_third=False, can_shoot=False)
    )
    assert decision.intent == "rotate"


def test_a_bot_without_boost_cannot_be_aggressive():
    brain = BotBrain("aggressive")
    decision = brain.decide(
        BrainContext(boost=0.0, ball_in_attacking_third=True, can_shoot=False)
    )
    assert decision.parameters["aggression"] <= 0.35
    assert "low boost" in decision.reason


def test_the_adaptive_profile_reacts_to_the_score():
    brain = BotBrain("adaptive")
    behind = brain.decide(BrainContext(score_diff=-2, time_remaining=300.0))
    ahead = brain.decide(BrainContext(score_diff=2, time_remaining=300.0))
    assert behind.parameters["aggression"] > ahead.parameters["aggression"]
    assert "behind" in behind.reason


def test_the_balanced_profile_ignores_the_context():
    brain = BotBrain("balanced")
    behind = brain.decide(BrainContext(score_diff=-2, time_remaining=10.0))
    ahead = brain.decide(BrainContext(score_diff=2, time_remaining=300.0))
    assert behind.parameters["aggression"] == ahead.parameters["aggression"]


def test_parameters_are_clamped_into_their_range():
    brain = BotBrain("balanced", {"aggression": 5.0, "safety": -2.0})
    assert brain.parameters.aggression == 1.0
    assert brain.parameters.safety == 0.0


def test_overrides_are_merged_but_unknown_keys_are_ignored():
    merged = merge_overrides(PROFILES["balanced"], {"aggression": 0.9, "nonsense": 1})
    assert merged.aggression == 0.9
    assert "nonsense" not in merged.to_dict()


def test_profiles_differ_lists_the_changed_parameters():
    changed = profiles_differ(
        PROFILES["aggressive"].to_dict(), PROFILES["defensive"].to_dict()
    )
    assert "aggression" in changed
    assert "rotation" in changed
    assert profiles_differ(PROFILES["balanced"].to_dict(), PROFILES["balanced"].to_dict()) == []


# ----------------------------------------------------------------- difficulty
def test_the_four_difficulty_levels_exist():
    assert DIFFICULTY_NAMES == ("rock", "tiny", "medium", "pro")
    profiles = {entry["name"] for entry in list_profiles()}
    assert profiles == set(DIFFICULTY_NAMES)


def test_the_percentage_is_a_target_and_not_a_measurement():
    profile = get_profile("medium")
    assert profile.target_win_rate == 0.75
    assert profile.target_is_measured is False
    catalogue = describe_all()
    assert catalogue["targets_are_measured"] is False
    assert "target" in catalogue["note"].lower()


def test_skill_estimate_separates_target_from_measurement():
    without = skill_estimate("medium")
    assert without["measured_available"] is False
    assert without["measured_win_rate"] is None
    assert without["target_is_measured"] is False
    assert "not been measured" in without["note"]

    with_measurement = skill_estimate("medium", games=20, measured_win_rate=0.8)
    assert with_measurement["measured_available"] is True
    assert with_measurement["measured_win_rate"] == 0.8
    assert with_measurement["target_win_rate"] == 0.75
    assert with_measurement["delta_to_target"] == pytest.approx(0.05)


def test_the_difficulty_mechanism_gets_stronger_with_the_level():
    previous = None
    for name in DIFFICULTY_NAMES:
        parameters = brain_parameters(name)
        assert parameters.mistake_rate >= 0.0
        assert parameters.reaction_delay >= 0.0
        if previous is not None:
            # a stronger level errs less, reacts faster and uses more boost
            assert parameters.mistake_rate <= previous.mistake_rate
            assert parameters.reaction_delay <= previous.reaction_delay
        previous = parameters


def test_a_custom_level_is_bounded():
    profile = custom_profile("my level", mistake_rate=9.0, reaction_delay=-1.0)
    assert profile.mistake_rate == 1.0
    assert profile.reaction_delay == 0.0
    assert profile.target_is_measured is False


def test_an_unknown_difficulty_is_rejected():
    with pytest.raises(ValueError):
        get_profile("godlike")


# ---------------------------------------------------------------- team builder
def test_a_team_is_validated():
    team = validate_team(
        {
            "name": "duo",
            "mode": "soccer-2v2",
            "bots": [
                {"name": "striker", "difficulty": "pro", "behaviour": "aggressive"},
                {"name": "keeper", "difficulty": "rock", "behaviour": "defensive"},
            ],
        }
    )
    assert isinstance(team, Team)
    assert len(team.bots) == 2
    assert team.bots[0].difficulty == "pro"


def test_a_team_needs_at_least_one_bot():
    with pytest.raises(ValueError) as excinfo:
        validate_team({"name": "empty", "bots": []})
    assert "at least" in str(excinfo.value)


def test_a_team_cannot_exceed_the_supported_size():
    bots = [{"name": f"b{index}"} for index in range(4)]
    with pytest.raises(ValueError) as excinfo:
        validate_team({"name": "huge", "bots": bots})
    assert "at most" in str(excinfo.value)


def test_an_unknown_difficulty_in_a_team_is_reported_with_the_bot_name():
    with pytest.raises(ValueError) as excinfo:
        validate_team({"name": "t", "bots": [{"name": "ghost", "difficulty": "legend"}]})
    assert "ghost" in str(excinfo.value)
    assert "legend" in str(excinfo.value)


def test_the_described_team_carries_the_actual_behaviour():
    team = default_team("soccer-2v2")
    described = describe_team(team)
    assert described["size"] == 2
    assert described["details"][0]["parameters"]["mistake_rate"] >= 0.0
    assert "design targets" in described["target_note"]


def test_the_team_brains_include_the_per_bot_overrides():
    team = Team(
        name="t",
        bots=[
            BotSlot(
                name="b1",
                difficulty="pro",
                behaviour="aggressive",
                overrides={"aggression": 0.11},
            )
        ],
    )
    brains = team_brains(team)
    assert len(brains) == 1
    assert brains[0].parameters.aggression == pytest.approx(0.11)


def test_the_opponent_names_address_model_backed_bots():
    team = Team(
        name="t",
        bots=[
            BotSlot(name="a", difficulty="medium"),
            BotSlot(name="b", difficulty="pro", model="model-0001"),
        ],
    )
    assert opponent_names(team) == ["difficulty:medium", "model:model-0001"]


# ------------------------------------------------------------------ memory
def test_a_remembered_player_is_counted():
    memory = PlayerMemory()
    memory.remember("Nitro", won=True, goals_for=3, goals_against=1, tag="pro")
    memory.remember("Nitro", won=False, goals_for=0, goals_against=2)
    record = memory.records["Nitro"]
    assert record.games == 2
    assert record.wins == 1
    assert record.losses == 1
    assert record.goals_for == 3
    assert record.win_rate == pytest.approx(0.5)
    assert record.tags == ["pro"]


def test_a_player_needs_a_name():
    memory = PlayerMemory()
    with pytest.raises(ValueError):
        memory.remember("   ")


def test_the_memory_is_bounded_by_the_number_of_entries():
    memory = PlayerMemory(max_entries=5)
    for index in range(20):
        memory.remember(f"p{index}", timestamp=1000.0 + index)
    memory.prune(now=2000.0)
    assert len(memory.records) <= 5
    # the oldest entries are the ones that go
    assert "p0" not in memory.records
    assert "p19" in memory.records


def test_the_memory_is_bounded_by_age():
    memory = PlayerMemory(max_age_days=1.0)
    now = 1_000_000.0
    memory.remember("old", timestamp=now - 10 * 86400)
    memory.remember("new", timestamp=now - 60)
    removed = memory.prune(now=now)
    assert removed["removed_by_age"] == 1
    assert "old" not in memory.records
    assert "new" in memory.records


def test_the_default_limit_is_documented():
    assert MAX_ENTRIES == 500
    assert PlayerMemory().max_entries == MAX_ENTRIES


def test_a_note_is_truncated():
    memory = PlayerMemory()
    memory.remember("Nitro")
    record = memory.set_note("Nitro", "x" * 5000)
    assert len(record.note) == 500


def test_the_memory_survives_an_export_and_import():
    memory = PlayerMemory()
    memory.remember("Nitro", won=True, goals_for=2)
    memory.remember("Zen", won=False)
    exported = memory.export()
    restored = PlayerMemory()
    count = restored.import_data(exported)
    assert count == 2
    assert restored.records["Nitro"].goals_for == 2
    assert restored.summary()["games"] == 2


def test_the_memory_survives_a_database_round_trip():
    memory = PlayerMemory()
    memory.remember("Nitro", won=True, goals_for=2)
    memory.set_note("Nitro", "fast")
    rows = rows_from_memory(memory)
    restored = memory_from_rows(rows)
    assert restored.records["Nitro"].note == "fast"
    assert restored.records["Nitro"].wins == 1


def test_forgetting_and_resetting_the_memory():
    memory = PlayerMemory()
    memory.remember("a")
    memory.remember("b")
    assert memory.forget("a") is True
    assert memory.forget("a") is False
    assert memory.reset() == 1
    assert memory.records == {}


def test_a_note_on_an_unknown_player_is_an_error():
    with pytest.raises(KeyError):
        PlayerMemory().set_note("nobody", "hello")


# ------------------------------------------------------------------- rivals
def test_a_rival_without_games_has_no_win_rate():
    registry = RivalRegistry()
    registry.register("pro-bot")
    rival = registry.get("pro-bot")
    assert rival is not None
    assert rival.win_rate is None
    assert registry.bottleneck()["available"] is False


def test_the_registry_is_built_from_the_evaluations():
    registry = RivalRegistry().build_from_evaluations(
        [
            {
                "opponent": "difficulty:pro",
                "games": 10,
                "wins": 8,
                "losses": 2,
                "draws": 0,
                "goals_for": 20,
                "goals_against": 7,
                "created_at": "2026-01-01T00:00:00+00:00",
                "model_id": "m1",
                "details": {"protocol": "soccer-1v1"},
            }
        ]
    )
    rival = registry.get("difficulty:pro")
    assert rival is not None
    assert rival.games == 10
    assert rival.win_rate == pytest.approx(0.8)
    assert rival.goal_diff == 13
    assert "protocol" in rival.description


def test_the_ranking_puts_the_best_measured_rival_first():
    registry = RivalRegistry()
    registry.record("weak", games=10, wins=2, losses=8, draws=0)
    registry.record("strong", games=10, wins=9, losses=1, draws=0)
    ranking = registry.ranking()
    assert ranking[0]["name"] == "strong"
    assert ranking[0]["measured"] is True


def test_the_bottleneck_is_the_worst_measured_rival():
    registry = RivalRegistry()
    registry.record("easy", games=10, wins=9, losses=1, draws=0)
    registry.record("hard", games=10, wins=1, losses=9, draws=0)
    bottleneck = registry.bottleneck()
    assert bottleneck["available"] is True
    assert bottleneck["name"] == "hard"
    assert bottleneck["win_rate"] == pytest.approx(0.1)


def test_the_trend_of_a_rival():
    registry = RivalRegistry()
    registry.record("r", games=10, wins=4, losses=6, draws=0, timestamp=1.0)
    registry.record("r", games=20, wins=16, losses=4, draws=0, timestamp=2.0)
    trend = registry.trend("r")
    assert trend["available"] is True
    assert trend["first_win_rate"] == pytest.approx(0.4)
    assert trend["last_win_rate"] == pytest.approx(0.8)
    assert trend["delta"] == pytest.approx(0.4)


def test_the_trend_of_an_unmeasured_rival_says_so():
    registry = RivalRegistry()
    registry.register("r")
    trend = registry.trend("r")
    assert trend["available"] is False
    assert "no measured evaluation" in trend["reason"]


# ---------------------------------------------------------------- adaptation
def test_the_adaptation_is_off_by_default():
    controller = AdaptiveController("medium")
    result = controller.observe(wins=10, games=10)
    assert result["adapted"] is False
    assert "disabled" in result["reason"]
    assert controller.events == []


def test_the_adaptation_needs_enough_games():
    controller = AdaptiveController(
        "medium",
        policy=AdaptationPolicy(enabled=True, min_games_between_changes=10, min_seconds_between_changes=0),
    )
    result = controller.observe(wins=10, games=5)
    assert result["adapted"] is False
    assert "not enough new games" in result["reason"]


def test_a_strong_agent_makes_the_opponent_harder():
    controller = AdaptiveController(
        "medium",
        policy=AdaptationPolicy(
            enabled=True,
            target_win_rate=0.5,
            tolerance=0.1,
            min_games_between_changes=1,
            min_seconds_between_changes=0,
        ),
    )
    before = controller.current
    result = controller.observe(wins=10, games=10)
    assert result["adapted"] is True
    assert controller.current["mistake_rate"] < before["mistake_rate"]
    assert controller.current["boost_efficiency"] > before["boost_efficiency"]


def test_a_weak_agent_makes_the_opponent_easier():
    controller = AdaptiveController(
        "pro",
        policy=AdaptationPolicy(
            enabled=True,
            target_win_rate=0.8,
            tolerance=0.1,
            min_games_between_changes=1,
            min_seconds_between_changes=0,
        ),
    )
    before = controller.current
    result = controller.observe(wins=1, games=10)
    assert result["adapted"] is True
    assert controller.current["mistake_rate"] > before["mistake_rate"]


def test_the_adaptation_stays_inside_the_limits():
    limits = AdaptationLimits(
        min_mistake_rate=0.0,
        max_mistake_rate=0.2,
        min_reaction_delay=0.05,
        max_reaction_delay=0.15,
        min_boost_efficiency=0.5,
        max_boost_efficiency=0.9,
    )
    controller = AdaptiveController(
        "rock",
        policy=AdaptationPolicy(
            enabled=True,
            target_win_rate=0.5,
            tolerance=0.05,
            min_games_between_changes=1,
            min_seconds_between_changes=0,
            step=0.5,
        ),
        limits=limits,
    )
    # even the starting point respects the limits ("rock" would err more)
    assert controller.current["mistake_rate"] == pytest.approx(0.2)
    assert controller.current["reaction_delay"] == pytest.approx(0.15)

    for _ in range(10):
        controller.observe(wins=0, games=10, at=1000.0)
    current = controller.current
    assert limits.min_mistake_rate <= current["mistake_rate"] <= limits.max_mistake_rate
    assert limits.min_reaction_delay <= current["reaction_delay"] <= limits.max_reaction_delay
    assert (
        limits.min_boost_efficiency
        <= current["boost_efficiency"]
        <= limits.max_boost_efficiency
    )


def test_the_adaptation_does_not_oscillate():
    controller = AdaptiveController(
        "medium",
        policy=AdaptationPolicy(
            enabled=True,
            target_win_rate=0.5,
            tolerance=0.1,
            min_games_between_changes=10,
            min_seconds_between_changes=60,
        ),
    )
    first = controller.observe(wins=10, games=10, at=100.0)
    second = controller.observe(wins=10, games=20, at=110.0)
    assert first["adapted"] is True
    assert second["adapted"] is False
    assert "too soon" in second["reason"]


def test_the_adaptation_state_survives_a_resume():
    controller = AdaptiveController(
        "medium",
        policy=AdaptationPolicy(
            enabled=True,
            target_win_rate=0.5,
            tolerance=0.1,
            min_games_between_changes=1,
            min_seconds_between_changes=0,
        ),
    )
    controller.observe(wins=10, games=10, at=100.0)
    state = controller.state()
    restored = AdaptiveController("medium")
    restored.restore(state)
    assert restored.current == controller.current
    assert len(restored.events) == 1
    assert restored.policy.enabled is True


def test_the_adaptation_never_touches_a_policy():
    controller = AdaptiveController("medium", policy=AdaptationPolicy(enabled=True))
    described = controller.to_dict()
    assert "never modified" in described["note"]
    assert set(described["current"]) == {
        "mistake_rate",
        "reaction_delay",
        "boost_efficiency",
    }


def test_the_adapted_parameters_reach_the_opponent():
    """The adaptation has to change the opponent, not just the log."""
    from rlai.opponents import build_opponent

    opponent = build_opponent("difficulty:medium")
    assert opponent.mistake_rate == pytest.approx(0.18)
    opponent.apply_execution(
        {"mistake_rate": 0.5, "reaction_delay": 0.9, "boost_efficiency": 0.2}
    )
    assert opponent.mistake_rate == pytest.approx(0.5)
    assert opponent.reaction_delay == pytest.approx(0.9)
    assert opponent.boost_efficiency == pytest.approx(0.2)


def test_the_execution_override_keeps_the_behaviour_of_the_opponent():
    """A plain scripted opponent keeps its behaviour when only the
    execution quality is adapted."""
    from rlai.opponents import build_opponent

    plain = build_opponent("chase")
    adapted = build_opponent(
        "chase", execution={"mistake_rate": 0.4, "reaction_delay": 0.3, "boost_efficiency": 0.5}
    )
    assert type(adapted.base) is type(plain)
    assert adapted.mistake_rate == pytest.approx(0.4)
    assert adapted.boost_efficiency == pytest.approx(0.5)


def test_the_execution_override_is_bounded():
    from rlai.opponents import build_opponent

    opponent = build_opponent(
        "difficulty:pro", execution={"mistake_rate": 9.0, "reaction_delay": -5.0}
    )
    assert opponent.mistake_rate == 1.0
    assert opponent.reaction_delay == 0.0


def test_the_adaptation_controller_is_built_by_the_trainer(data_root):
    """The trainer's wiring of the controller must work (it is created lazily)."""
    from rlai.config import TrainingConfig
    from rlai.store import Store
    from rlai.trainer import Trainer

    store = Store(data_root / "rlai.sqlite3")
    config = TrainingConfig(
        run_name="adaptive",
        total_steps=2000,
        rollout_steps=64,
        n_workers=1,
        minibatch_size=64,
        eval_interval=0,
        policy={"hidden": [16]},
        device="cpu",
        opponent="difficulty:medium",
        adaptive_dynamics=True,
    )
    trainer = Trainer(config, "adaptive-run", store)
    controller = trainer._adaptive_controller()
    assert controller is not None
    assert controller.policy.enabled is True
    assert controller.difficulty == "medium"
    # a second call returns the same controller (state is kept)
    assert trainer._adaptive_controller() is controller


def test_the_adaptation_needs_a_difficulty_opponent(data_root):
    """A plain scripted opponent keeps its behaviour and is not adapted."""
    from rlai.config import TrainingConfig
    from rlai.store import Store
    from rlai.trainer import Trainer

    store = Store(data_root / "rlai.sqlite3")
    config = TrainingConfig(
        run_name="adaptive-chase",
        total_steps=2000,
        rollout_steps=64,
        n_workers=1,
        minibatch_size=64,
        eval_interval=0,
        policy={"hidden": [16]},
        device="cpu",
        opponent="chase",
        adaptive_dynamics=True,
    )
    trainer = Trainer(config, "adaptive-chase-run", store)
    assert trainer._adaptive_controller() is None
    assert trainer._opponent_execution() is None


def test_the_adaptation_is_off_unless_it_is_enabled(data_root):
    from rlai.config import TrainingConfig
    from rlai.store import Store
    from rlai.trainer import Trainer

    store = Store(data_root / "rlai.sqlite3")
    config = TrainingConfig(
        run_name="no-adaptive",
        total_steps=2000,
        rollout_steps=64,
        n_workers=1,
        minibatch_size=64,
        eval_interval=0,
        policy={"hidden": [16]},
        device="cpu",
    )
    trainer = Trainer(config, "no-adaptive-run", store)
    assert trainer._adaptive_controller() is None
