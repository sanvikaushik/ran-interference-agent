# main.py
from agent.kpi_simulator import generate_kpi_series, KpiConfig
from agent.reasoner import InterferenceAgent
from agent.metrics import evaluate_on_synthetic


def demo_single_run():
    cfg = KpiConfig(timesteps=300, seed=123)
    df = generate_kpi_series("external_interference", cfg=cfg)

    agent = InterferenceAgent()
    out = agent.run_on_sequence(df)

    print("=== Single-sequence demo ===")
    print("Diagnosis:", out["diagnosis"]["root_cause"])
    print("Confidence:", out["diagnosis"]["confidence"])
    print("Action intent:", out["action"]["intent"])
    print("Explanation:")
    print(out["explanation"])
    print()

def demo_metrics():
    print("=== Rule-based synthetic evaluation ===")
    metrics_rb = evaluate_on_synthetic(n_per_scenario=15, timesteps=300)
    print(f"Overall accuracy (rules): {metrics_rb['accuracy'] * 100:.1f}%")
    print("Per-scenario accuracy (rules):")
    for s, acc in metrics_rb["per_scenario_accuracy"].items():
        print(f"  {s}: {acc * 100:.1f}%")
    print(f"Total samples: {metrics_rb['total_samples']}")
    print()

    print("=== ML classifier evaluation ===")
    metrics_ml = evaluate_ml_classifier(n_per_scenario=50)
    print(f"Overall accuracy (ML): {metrics_ml['overall_accuracy'] * 100:.1f}%")
    print("Per-class accuracy (ML):")
    for s, acc in metrics_ml["per_class_accuracy"].items():
        print(f"  {s}: {acc * 100:.1f}%")
    print(f"Test samples: {metrics_ml['n_test_samples']}")



from agent.metrics import evaluate_on_synthetic, evaluate_ml_classifier

if __name__ == "__main__":
    demo_single_run()
    demo_metrics()
