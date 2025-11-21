from __future__ import annotations
from typing import Tuple
import pandas as pd

from .actions import ActionPlan


def _apply_dl_tilt(df: pd.DataFrame, delta_tilt_deg: float) -> pd.DataFrame:
    """
    Simulate effect of a small downlink tilt change on SINR / noise / BLER.

    Assumption:
      - Tilting down (negative delta) reduces overshoot + external interference.
      - Improves SINR a bit and reduces UL noise in the affected window.
    """
    improved = df.copy()

    # Focus changes on the last 40% of samples (where the issue usually is)
    n = len(improved)
    tail_start = int(n * 0.6)
    tail_idx = improved.index[tail_start:]

    # simple linear model for effect
    sinr_gain = abs(delta_tilt_deg) * 2.0   # -1.5° → +3 dB SINR
    noise_reduction = abs(delta_tilt_deg) * 1.5
    bler_factor = 0.8

    improved.loc[tail_idx, "sinr"] = improved.loc[tail_idx, "sinr"] + sinr_gain
    improved.loc[tail_idx, "pusch_noise"] = improved.loc[tail_idx, "pusch_noise"] - noise_reduction
    improved.loc[tail_idx, "bler"] = improved.loc[tail_idx, "bler"] * bler_factor

    return improved


def _apply_load_balance(df: pd.DataFrame, total_offload: float | None = None) -> pd.DataFrame:    
    """
    Simulate effect of SON / load balancing:
      - Reduce PRB utilization on this cell
      - Slightly reduce BLER
    """
    improved = df.copy()
    n = len(improved)
    tail_start = int(n * 0.6)
    tail_idx = improved.index[tail_start:]

    prb_factor = 0.75   # 25% less load
    bler_factor = 0.9

    if total_offload is not None and total_offload > 0:
        prb_factor = max(0.4, 1 - total_offload / 100)

    improved.loc[tail_idx, "prb_util"] = improved.loc[tail_idx, "prb_util"] * prb_factor
    improved.loc[tail_idx, "bler"] = improved.loc[tail_idx, "bler"] * bler_factor

    return improved


def apply_action_to_kpis(
    df: pd.DataFrame,
    action: ActionPlan,
) -> Tuple[pd.DataFrame, str]:
    """
    Apply the agent's recommended action to the KPI time-series and return:

      - mitigated_df: new DataFrame with adjusted KPIs
      - note: short description of what was simulated
    """
    intent = action["intent"]
    params = action.get("parameters", {})

    if intent == "ADJUST_DL_TILT":
        delta_tilt_deg = float(params.get("delta_tilt_deg", -1.5))
        mitigated = _apply_dl_tilt(df, delta_tilt_deg)
        note = (
            f"Applied downlink tilt change of {delta_tilt_deg}° to the last 40% of samples, "
            "increasing SINR and reducing UL noise / BLER."
        )
        return mitigated, note

    if intent == "BALANCE_LOAD":
        plan = params.get("rebalancing_plan", [])
        total_offload = sum(move[2] for move in plan) if plan else None
        mitigated = _apply_load_balance(df, total_offload=total_offload)
        note = (
            "Applied load-balancing in the last 40% of samples, reducing PRB utilization "
            "and slightly improving BLER."
        )
        if total_offload:
            note += f" Offloaded approximately {total_offload:.1f} traffic units via optimizer."
        return mitigated, note

    # For NO_ACTION or intents we don't simulate yet, just return original
    return df.copy(), "No mitigation applied (intent was NO_ACTION or unsupported)."
