"""Checkpoint saver: unit behavior + engine point-in-time rewind."""
from portfoliopilot.simulation.checkpoint import CheckpointSaver
from portfoliopilot.simulation.engine import SimulationEngine


def test_checkpoint_save_load_list_prune(tmp_path):
    cp = CheckpointSaver(directory=tmp_path)
    for i in range(5):
        cp.save(i, {"month_index": i + 1, "current_date": f"2020-{i+1:02d}-28",
                    "portfolio_value": 100000 + i, "simulation_id": "sim_x"})
    assert cp.load(2)["completed_month"] == 2
    assert cp.load_latest()["completed_month"] == 4
    assert len(cp.list_checkpoints()) == 5

    cp.prune_after(2)
    months = [c["completed_month"] for c in cp.list_checkpoints()]
    assert months == [0, 1, 2]

    cp.clear()
    assert cp.list_checkpoints() == []
    assert cp.load_latest() is None


def test_engine_checkpoints_each_month_and_rewinds(offline_data):
    eng = SimulationEngine()
    eng.reset()          # clean monitor + checkpoints
    eng.load()
    for _ in range(6):
        eng.step()

    # a checkpoint exists for every completed month
    cps = eng.checkpointer.list_checkpoints()
    assert [c["completed_month"] for c in cps] == [0, 1, 2, 3, 4, 5]
    assert eng.state.month_index == 6

    # capture state at month 3 for comparison
    snap3 = eng.checkpointer.load(3)

    # rewind to end of month 2
    assert eng.restore_to(2) is True
    assert eng.state.month_index == 3
    assert len(eng.history) == 3
    # checkpoints after month 2 were pruned
    assert [c["completed_month"] for c in eng.checkpointer.list_checkpoints()] == [0, 1, 2]
    # monitor steps truncated too
    assert len(eng.monitor.steps()) == 3

    # continuing forward re-creates month 3 deterministically (same value)
    rec = eng.step()
    assert rec is not None
    assert eng.state.month_index == 4
    assert abs(rec["portfolio_value"] - snap3["portfolio_value"]) < 1e-6

    eng.reset()
