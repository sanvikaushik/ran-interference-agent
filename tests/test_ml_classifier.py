from agent.ml_classifier import (
    build_ml_dataset,
    train_and_eval_split,
    predict_sequence,
)
from agent.kpi_simulator import generate_kpi_series, KpiConfig


def test_build_ml_dataset_shapes():
    X, y, id_to_label = build_ml_dataset(n_per_scenario=5)
    # 3 scenarios * 5 each = 15 samples
    assert X.shape[0] == 15
    assert X.shape[1] > 0
    assert y.shape[0] == 15
    assert set(id_to_label.keys()) == {0, 1, 2}


def test_train_and_eval_split_accuracy_reasonable():
    model, metrics = train_and_eval_split(n_per_scenario=30, test_size=0.3)

    assert 0.0 <= metrics["overall_accuracy"] <= 1.0
    # aim for decent performance on synthetic data
    assert metrics["overall_accuracy"] > 0.75

    per_class = metrics["per_class_accuracy"]
    # each scenario should have some accuracy signal
    for label in ["normal", "external_interference", "congestion"]:
        assert label in per_class
        assert 0.0 <= per_class[label] <= 1.0


def test_predict_sequence_runs():
    # quick smoke test that prediction works on a fresh sequence
    model, _ = train_and_eval_split(n_per_scenario=20, test_size=0.3)

    cfg = KpiConfig(timesteps=300, seed=999)
    df = generate_kpi_series("external_interference", cfg=cfg)

    label, conf = predict_sequence(model, df)
    assert label in {"normal", "external_interference", "congestion"}
    assert 0.0 <= conf <= 1.0
