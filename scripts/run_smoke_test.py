"""End-to-end offline smoke test.

Ingests deterministic synthetic data, builds features, then runs the full
120-month simulation headlessly and prints a summary. No network, no LLM
required (uses the deterministic agent + judge fallbacks).

    python scripts/run_smoke_test.py
"""
import _bootstrap  # noqa: F401
import time

from portfoliopilot.data import ingestion
from portfoliopilot.features.feature_engine import build_all_features
from portfoliopilot.simulation.engine import SimulationEngine


def main() -> None:
    t0 = time.time()
    print("[1/4] Ingesting synthetic constituents (with index changes)...")
    ingestion.ingest_constituents(offline=True, mode="synthetic")
    print("[2/4] Ingesting synthetic prices + fundamentals...")
    ingestion.ingest_fundamentals(offline=True)
    ingestion.ingest_prices(offline=True)
    print("[3/4] Building monthly features...")
    build_all_features()

    print("[4/4] Running 120-month simulation (headless)...")
    eng = SimulationEngine()
    eng.load()
    steps = 0
    while True:
        rec = eng.step()
        if rec is None:
            break
        steps += 1

    state = eng.get_state()
    agg = state["aggregates"]
    fin = agg["financial"]
    print("\n================ SMOKE TEST SUMMARY ================")
    print(f"Months simulated:        {steps}")
    print(f"Final portfolio value:   ${fin['portfolio_value']:,.0f}")
    print(f"Final benchmark value:   ${fin['benchmark_value']:,.0f}")
    print(f"Total return:            {fin['total_return']*100:.2f}%")
    print(f"Benchmark return:        {fin['benchmark_return']*100:.2f}%")
    print(f"Excess return:           {fin['excess_return']*100:.2f}%")
    print(f"Max drawdown:            {fin['max_drawdown']*100:.2f}%")
    print(f"Transaction cost drag:   {fin['transaction_cost_drag']*100:.2f}%")
    print(f"Total transactions:      {agg['agent']['transaction_count']}")
    print(f"Forced sells:            {agg['agent']['forced_sell_count']}")
    print(f"Risk modifications:      {agg['agent']['risk_modified_count']}")
    print(f"Constraint violations:   {agg['agent']['constraint_violation_count']}")
    print(f"Memories stored:         {state['memory']['total']}")
    print(f"Incidents recorded:      {len(state['alerts'])}")
    print(f"Elapsed:                 {time.time()-t0:.1f}s")
    print("===================================================")
    assert steps == 120, f"expected 120 months, got {steps}"
    print("SMOKE TEST PASSED ✔")


if __name__ == "__main__":
    main()
