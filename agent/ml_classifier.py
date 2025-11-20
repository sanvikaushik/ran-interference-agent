from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from .kpi_simulator import generate_dataset
from .rulebase import _window_stats  # reuse same feature logic


ScenarioLabel = Literal["normal", "external_interference", "congestion"]


@dataclass
class AnomalyModel:
    clf: RandomForestClassifier
    id_to_label: Dict[int, ScenarioLabel]


def _features_from_df(df: pd.DataFrame) -> np.ndarray:
    """
    Turn a full KPI sequence into a fixed-length feature vector.
    Uses the same window stats as the rule-based engine +
    derived features.
    """
    stats = _window_stats(df)
    sinr_drop = stats["sinr_before"] - stats["sinr_now"]
    noise_rise = stats["noise_now"] - stats["noise_before"]

    feat = np.array(
        [
            stats["sinr_before"],
            stats["sinr_now"],
            sinr_drop,
            stats["noise_before"],
            stats["noise_now"],
            noise_rise,
            stats["prb_util_now"],
            stats["bler_now"],
            stats["ho_fail_rate_now"],
        ],
        dtype=float,
    )
    return feat


def build_ml_dataset(
    n_per_scenario: int = 50,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, ScenarioLabel]]:
    """
    Generate synthetic KPI sequences and turn them into an ML dataset:
        X: feature matrix (n_samples, n_features)
        y: integer labels
        id_to_label: mapping int -> original scenario label
    """
    data = generate_dataset(n_per_scenario=n_per_scenario)

    label_to_id: Dict[ScenarioLabel, int] = {
        "normal": 0,
        "external_interference": 1,
        "congestion": 2,
    }
    id_to_label: Dict[int, ScenarioLabel] = {v: k for k, v in label_to_id.items()}

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for entry in data:
        scenario: ScenarioLabel = entry["scenario"]  # type: ignore
        df = entry["df"]
        feat = _features_from_df(df)
        X_list.append(feat)
        y_list.append(label_to_id[scenario])

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)
    return X, y, id_to_label


def train_anomaly_classifier(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    random_state: int = 42,
) -> AnomalyModel:
    """
    Train a RandomForest classifier on the KPI feature matrix.
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
    )
    clf.fit(X, y)

    # we assume labels are 0..K-1 and correspond to scenarios from build_ml_dataset
    id_to_label: Dict[int, ScenarioLabel] = {
        0: "normal",
        1: "external_interference",
        2: "congestion",
    }
    return AnomalyModel(clf=clf, id_to_label=id_to_label)


def train_and_eval_split(
    n_per_scenario: int = 50,
    test_size: float = 0.3,
    random_state: int = 123,
) -> Tuple[AnomalyModel, dict]:
    """
    Convenience function: build synthetic dataset, train/test split,
    train model, and compute accuracy metrics.
    """
    X, y, id_to_label = build_ml_dataset(n_per_scenario=n_per_scenario)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = train_anomaly_classifier(X_train, y_train)

    y_pred = model.clf.predict(X_test)
    overall_acc = (y_pred == y_test).mean()

    # per-class accuracy
    per_class_correct: dict[int, int] = {}
    per_class_total: dict[int, int] = {}
    for yt, yp in zip(y_test, y_pred):
        per_class_total[yt] = per_class_total.get(yt, 0) + 1
        if yt == yp:
            per_class_correct[yt] = per_class_correct.get(yt, 0) + 1

    per_class_acc: dict[str, float] = {}
    for class_id, total in per_class_total.items():
        label = id_to_label[class_id]
        correct = per_class_correct.get(class_id, 0)
        per_class_acc[label] = correct / total if total else 0.0

    metrics = {
        "overall_accuracy": float(overall_acc),
        "per_class_accuracy": per_class_acc,
        "n_test_samples": int(len(y_test)),
    }

    # attach mapping from model's ids to labels
    model.id_to_label = id_to_label
    return model, metrics


def predict_sequence(
    model: AnomalyModel,
    df: pd.DataFrame,
) -> Tuple[ScenarioLabel, float]:
    """
    Predict scenario label + confidence for a single KPI sequence.
    """
    feat = _features_from_df(df).reshape(1, -1)
    probs = model.clf.predict_proba(feat)[0]
    pred_id = int(np.argmax(probs))
    label = model.id_to_label[pred_id]
    confidence = float(probs[pred_id])
    return label, confidence
