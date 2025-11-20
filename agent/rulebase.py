from dataclasses import dataclass
import pandas as pd
from typing import Literal, TypedDict

DiagnosisLabel = Literal["normal", "external_interference", "congestion", "unknown"]


class Diagnosis(TypedDict):
    root_cause: DiagnosisLabel
    confidence: float
    features: dict


@dataclass
class RuleThresholds:
    sinr_drop_external: float = 7.0
    noise_rise_external: float = 5.0
    prb_util_congestion: float = 75.0
    sinr_drop_congestion: float = 2.0
    bler_congestion: float = 2.0


def _window_stats(df: pd.DataFrame, lookback: int = 60) -> dict:
    tail = df.tail(lookback)
    head = df.head(lookback)

    return {
        "sinr_now": tail["sinr"].mean(),
        "sinr_before": head["sinr"].mean(),
        "noise_now": tail["pusch_noise"].mean(),
        "noise_before": head["pusch_noise"].mean(),
        "prb_util_now": tail["prb_util"].mean(),
        "bler_now": tail["bler"].mean(),
        "rsrq_now": tail["rsrq"].mean(),
        "ho_fail_rate_now": tail["ho_failures"].mean(),
    }


def diagnose_window(
    df: pd.DataFrame,
    thresholds: RuleThresholds | None = None,
) -> Diagnosis:
    """
    Apply RAN RF-style rules to a KPI window and output a diagnosis.
    """
    if thresholds is None:
        thresholds = RuleThresholds()

    stats = _window_stats(df)
    sinr_drop = stats["sinr_before"] - stats["sinr_now"]
    noise_rise = stats["noise_now"] - stats["noise_before"]

    # Rule 1: external interference
    if sinr_drop > thresholds.sinr_drop_external and noise_rise > thresholds.noise_rise_external:
        return Diagnosis(
            root_cause="external_interference",
            confidence=0.9,
            features={**stats, "sinr_drop": sinr_drop, "noise_rise": noise_rise},
        )

    # Rule 2: congestion
    if (
        stats["prb_util_now"] > thresholds.prb_util_congestion
        and sinr_drop > thresholds.sinr_drop_congestion
        and stats["bler_now"] > thresholds.bler_congestion
    ):
        return Diagnosis(
            root_cause="congestion",
            confidence=0.85,
            features={**stats, "sinr_drop": sinr_drop, "noise_rise": noise_rise},
        )

    # Rule 3: normal-ish
    if sinr_drop < 2.0 and stats["prb_util_now"] < 70 and stats["bler_now"] < 2.0:
        return Diagnosis(
            root_cause="normal",
            confidence=0.8,
            features={**stats, "sinr_drop": sinr_drop, "noise_rise": noise_rise},
        )

    # Fallback
    return Diagnosis(
        root_cause="unknown",
        confidence=0.5,
        features={**stats, "sinr_drop": sinr_drop, "noise_rise": noise_rise},
    )

@dataclass
class RuleThresholds:
    sinr_drop_external: float = 5.0   # was 7.0
    noise_rise_external: float = 3.0  # was 5.0
    prb_util_congestion: float = 70.0 # was 75.0
    sinr_drop_congestion: float = 1.5 # was 2.0
    bler_congestion: float = 1.5      # was 2.0
