"""Dynamic Time Warping-based signature matcher for KPI sequences."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .kpi_simulator import Scenario, generate_dataset

KPI_COLUMNS: Tuple[str, ...] = ("sinr", "prb_util", "pusch_noise", "bler")


@dataclass
class SignatureMatch:
    label: Scenario
    distance: float
    distances: Dict[Scenario, float]
    confidence: float


def _multivariate_dtw(a: np.ndarray, b: np.ndarray) -> float:
    """Compute multivariate DTW distance between two sequences.

    Uses squared Euclidean cost and a classic O(n^2) dynamic program.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")

    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.sum((a[i - 1] - b[j - 1]) ** 2)
            dp[i, j] = cost + min(
                dp[i - 1, j],       # insertion
                dp[i, j - 1],       # deletion
                dp[i - 1, j - 1],   # match
            )
    return float(dp[n, m])


def _aggregate_sequences(sequences: List[pd.DataFrame]) -> np.ndarray:
    """Build a prototype by averaging aligned KPI vectors across runs."""
    stacked = np.stack([df[list(KPI_COLUMNS)].to_numpy() for df in sequences], axis=0)
    return stacked.mean(axis=0)


class DTWSignatureEngine:
    """Learn class templates and classify KPI windows via DTW matching."""

    def __init__(self, templates: Dict[Scenario, np.ndarray]):
        self.templates = templates

    @classmethod
    def from_synthetic(
        cls,
        n_per_scenario: int = 8,
        timesteps: int = 200,
        seed: int = 7,
    ) -> "DTWSignatureEngine":
        dataset = generate_dataset(
            n_per_scenario=n_per_scenario,
            cfg=None,
        )
        grouped: Dict[Scenario, List[pd.DataFrame]] = {
            "normal": [],
            "external_interference": [],
            "congestion": [],
        }
        for entry in dataset:
            grouped[entry["scenario"]].append(entry["df"].tail(timesteps))

        templates = {
            label: _aggregate_sequences(runs)
            for label, runs in grouped.items()
        }
        return cls(templates=templates)

    def match(self, window: pd.DataFrame) -> SignatureMatch:
        seq = window[list(KPI_COLUMNS)].to_numpy()
        distances: Dict[Scenario, float] = {}
        for label, template in self.templates.items():
            distances[label] = _multivariate_dtw(seq, template)

        best_label = min(distances, key=distances.get)
        best_dist = distances[best_label]
        denom = sum(distances.values()) or 1.0
        confidence = 1.0 - (best_dist / denom)
        return SignatureMatch(
            label=best_label,
            distance=best_dist,
            distances=distances,
            confidence=confidence,
        )