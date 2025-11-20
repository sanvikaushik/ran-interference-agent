from dataclasses import dataclass
import pandas as pd
from typing import TypedDict

from .rulebase import diagnose_window, Diagnosis
from .actions import propose_action, ActionPlan


class AgentOutput(TypedDict):
    diagnosis: Diagnosis
    action: ActionPlan
    explanation: str


@dataclass
class AgentConfig:
    lookback: int = 60
    cell_id: str = "Cell-1"


class InterferenceAgent:
    def __init__(self, cfg: AgentConfig | None = None):
        self.cfg = cfg or AgentConfig()

    def run_on_sequence(self, df: pd.DataFrame) -> AgentOutput:
        """
        Run the agent on a full KPI time-series and focus on the trailing window.
        """
        if len(df) < self.cfg.lookback * 2:
            raise ValueError("Time-series too short for before/after comparison.")

        diagnosis = diagnose_window(df, thresholds=None)
        action = propose_action(diagnosis, cell_id=self.cfg.cell_id)
        explanation = self._build_explanation(diagnosis, action)

        return AgentOutput(
            diagnosis=diagnosis,
            action=action,
            explanation=explanation,
        )

    def _build_explanation(self, diagnosis: Diagnosis, action: ActionPlan) -> str:
        features = diagnosis["features"]
        cause = diagnosis["root_cause"]
        conf = diagnosis["confidence"]

        base = [
            f"Root cause: {cause} (confidence: {conf:.2f})",
            f"Current SINR: {features['sinr_now']:.2f} dB (drop of {features['sinr_before'] - features['sinr_now']:.2f} dB vs earlier).",
            f"Current PRB utilization: {features['prb_util_now']:.2f}%.",
            f"Current PUSCH noise: {features['noise_now']:.2f} dBm.",
            f"Current BLER: {features['bler_now']:.2f}%.",
        ]

        if cause == "external_interference":
            base.append(
                "Pattern shows a strong SINR drop with significant noise rise, "
                "while utilization is not extremely high – consistent with external interference."
            )
        elif cause == "congestion":
            base.append(
                "Pattern shows very high PRB utilization with elevated BLER and some SINR degradation – "
                "consistent with cell congestion."
            )
        elif cause == "normal":
            base.append(
                "KPI levels look stable with no major degradation – traffic appears within normal bounds."
            )
        else:
            base.append(
                "The KPIs do not clearly match a known pattern, so the agent remains conservative."
            )

        base.append(
            f"Suggested action intent: {action['intent']} – {action['description']}"
        )

        return " ".join(base)
