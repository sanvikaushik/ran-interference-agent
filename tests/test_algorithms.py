from agent.dtw_signature_engine import DTWSignatureEngine
from agent.kpi_simulator import KpiConfig, generate_kpi_series
from agent.graph_optimizer import plan_load_shift


def test_dtw_signature_engine_classifies_sequences():
    cfg = KpiConfig(timesteps=200, seed=21)
    df = generate_kpi_series("congestion", cfg=cfg)
    engine = DTWSignatureEngine.from_synthetic(n_per_scenario=5, timesteps=200, seed=9)

    match = engine.match(df.tail(200))

    assert set(match.distances.keys()) == {
        "normal",
        "external_interference",
        "congestion",
    }
    assert match.confidence > 0
    assert match.distance == match.distances[match.label]


def test_graph_optimizer_builds_plan_for_overload():
    cell_loads = {"Cell-1": 95.0, "Cell-2": 55.0, "Cell-3": 50.0}
    capacities = {cid: 100.0 for cid in cell_loads}
    neighbor_costs = {
        ("Cell-1", "Cell-2"): 1.0,
        ("Cell-1", "Cell-3"): 1.2,
        ("Cell-2", "Cell-3"): 0.8,
    }

    plan, cost = plan_load_shift(
        focal_cell="Cell-1",
        cell_loads=cell_loads,
        capacities=capacities,
        neighbor_costs=neighbor_costs,
        target_utilization=70.0,
    )

    assert any(move[0] == "Cell-1" for move in plan)
    assert cost >= 0