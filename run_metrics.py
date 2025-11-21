from agent.metrics import evaluate_on_synthetic, evaluate_ml_classifier
from agent.ml_classifier import train_and_eval_split


def main():
    print("=== Rule-based engine metrics ===")
    rb = evaluate_on_synthetic(n_per_scenario=30, timesteps=300)
    print(f"Overall accuracy (rules): {rb['accuracy'] * 100:.2f}%")
    print("Per-scenario accuracy:")
    for s, acc in rb["per_scenario_accuracy"].items():
        print(f"  {s}: {acc * 100:.2f}%")
    print(f"Total samples: {rb['total_samples']}")
    print()

    print("=== ML classifier metrics ===")
    model, ml = train_and_eval_split(n_per_scenario=80, test_size=0.3, random_state=123)
    print(f"Overall accuracy (ML): {ml['overall_accuracy'] * 100:.2f}%")
    print("Per-class accuracy:")
    for s, acc in ml["per_class_accuracy"].items():
        print(f"  {s}: {acc * 100:.2f}%")
    print(f"Test samples: {ml['n_test_samples']}")


if __name__ == "__main__":
    main()
