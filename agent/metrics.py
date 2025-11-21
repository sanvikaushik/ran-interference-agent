from __future__ import annotations

from typing import Literal

from .kpi_simulator import generate_dataset
from .reasoner import InterferenceAgent
from .ml_classifier import train_and_eval_split

Scenario = Literal["normal", "external_interference", "congestion"]

def evaluate_on_synthetic(
    n_per_scenario: int = 20,
    timesteps: int = 300,
) -> dict:
    """
    Evaluate the agent on synthetic KPI sequences.
    """
    data = generate_dataset(n_per_scenario=n_per_scenario)
    agent = InterferenceAgent()

    correct = 0
    total = len(data)
    per_scenario_counts: dict[str, int] = {}
    per_scenario_correct: dict[str, int] = {}

    for entry in data:
        scenario: Scenario = entry["scenario"]  # type: ignore
        df = entry["df"]

        out = agent.run_on_sequence(df)
        pred = out["diagnosis"]["root_cause"]

        per_scenario_counts[scenario] = per_scenario_counts.get(scenario, 0) + 1
        if scenario == "normal" and pred == "normal":
            correct += 1
            per_scenario_correct[scenario] = per_scenario_correct.get(scenario, 0) + 1
        elif scenario == "external_interference" and pred == "external_interference":
            correct += 1
            per_scenario_correct[scenario] = per_scenario_correct.get(scenario, 0) + 1
        elif scenario == "congestion" and pred == "congestion":
            correct += 1
            per_scenario_correct[scenario] = per_scenario_correct.get(scenario, 0) + 1

    accuracy = correct / total if total else 0.0
    per_scenario_accuracy = {
        s: per_scenario_correct.get(s, 0) / c for s, c in per_scenario_counts.items()
    }

    return {
        "accuracy": accuracy,
        "per_scenario_accuracy": per_scenario_accuracy,
        "total_samples": total,
    }

def evaluate_ml_classifier(
    n_per_scenario: int = 50,
) -> dict:
    """
    Train/test an ML anomaly classifier on synthetic KPI features.
    Returns accuracy metrics.
    """
    _, metrics = train_and_eval_split(n_per_scenario=n_per_scenario)
    return metrics