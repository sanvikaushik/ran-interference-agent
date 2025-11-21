# tests/test_reasoner_metrics.py
from agent.metrics import evaluate_on_synthetic
from agent.reasoner import InterferenceAgent
from agent.kpi_simulator import generate_kpi_series, KpiConfig
from agent.metrics import evaluate_ml_classifier

def test_agent_runs_end_to_end():
    cfg = KpiConfig(timesteps=300, seed=123)
    df = generate_kpi_series("congestion", cfg=cfg)
    agent = InterferenceAgent()
    out = agent.run_on_sequence(df)

    assert "diagnosis" in out
    assert "action" in out
    assert "explanation" in out
    assert isinstance(out["explanation"], str)
    assert len(out["explanation"]) > 10
    assert "dtw" in out
    assert set(out["dtw"].distances.keys()) == {
        "normal",
        "external_interference",
        "congestion",
    }



def test_metrics_reasonable_accuracy():
    metrics = evaluate_on_synthetic(n_per_scenario=10, timesteps=300)
    assert metrics["accuracy"] > 0.75
    # At least one scenario should have > 0.8 accuracy
    assert any(acc > 0.8 for acc in metrics["per_scenario_accuracy"].values())

def test_evaluate_ml_classifier_metric_ranges():
    metrics = evaluate_ml_classifier(n_per_scenario=20)
    assert 0.0 <= metrics["overall_accuracy"] <= 1.0
    assert metrics["n_test_samples"] > 0
    for label, acc in metrics["per_class_accuracy"].items():
        assert label in {"normal", "external_interference", "congestion"}
        assert 0.0 <= acc <= 1.0