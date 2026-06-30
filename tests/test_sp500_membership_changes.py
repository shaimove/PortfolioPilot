import datetime as dt

import pytest

from portfoliopilot.data import universe
from portfoliopilot.simulation.engine import SimulationEngine
from portfoliopilot.utils import simulation_month_ends


# --------------------------------------------------------------------------- #
# Membership data model (fast unit tests)
# --------------------------------------------------------------------------- #
def test_synthetic_membership_has_additions_and_removals():
    months = simulation_month_ends(total_months=120)
    df = universe.build_synthetic_membership(months)
    uni = universe.Universe(df)
    removed_any = added_any = False
    me = uni.month_ends
    for prev, cur in zip(me, me[1:]):
        if uni.removed_between(prev, cur):
            removed_any = True
        if uni.added_between(prev, cur):
            added_any = True
    assert removed_any, "expected at least one index removal over the window"
    assert added_any, "expected at least one index addition over the window"


def test_current_constituents_fallback_is_constant():
    months = simulation_month_ends(total_months=24)
    df = universe.build_membership_from_current(months)
    uni = universe.Universe(df)
    first = uni.members_on(uni.month_ends[0])
    last = uni.members_on(uni.month_ends[-1])
    assert first == last  # survivorship-biased: membership never changes


# --------------------------------------------------------------------------- #
# Full engine integration (run once per module)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def engine_run(offline_data):
    eng = SimulationEngine()
    eng.load()
    while eng.step() is not None:
        pass
    return eng


def test_simulation_runs_120_steps(engine_run):
    assert len(engine_run.history) == 120
    assert engine_run.state.finished is True
    assert engine_run.state.month_index == 120


def test_forced_sells_are_executed(engine_run):
    # No month may end with an un-executed forced sell.
    assert all(r["forced_sell_not_executed"] == 0 for r in engine_run.history)
    # And at least one forced sell actually occurred during the run.
    total_forced = sum(r["forced_sell_count"] for r in engine_run.history)
    assert total_forced >= 1


def test_new_entrants_become_eligible(engine_run):
    considered = any(r["new_entrants_considered"] for r in engine_run.history)
    assert considered, "expected new S&P 500 entrants to be considered at least once"


def test_local_monitor_writes_events(engine_run):
    steps = engine_run.monitor.steps()
    assert len(steps) == 120
    # alerts table should be reachable and return a list
    assert isinstance(engine_run.monitor.alerts(), list)


def test_dashboard_state_is_valid(engine_run):
    state = engine_run.get_state()
    for key in ("controls", "aggregates", "series", "alerts", "memory", "latest"):
        assert key in state
    assert len(state["series"]["portfolio_value"]) == 120
    assert len(state["series"]["benchmark_value"]) == 120
    fin = state["aggregates"]["financial"]
    assert fin["portfolio_value"] > 0
    assert state["controls"]["progress"] == "120 / 120"
