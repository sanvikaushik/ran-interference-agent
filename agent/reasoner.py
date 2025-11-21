import pandas as pd
from typing import TypedDict

from .rulebase import _window_stats, diagnose_window, Diagnosis
from .actions import propose_action, ActionPlan
from .dtw_signature_engine import DTWSignatureEngine, SignatureMatch
from .graph_optimizer import optimize_load_shift, LoadShiftPlan

class AgentOutput(TypedDict):
    diagnosis: Diagnosis
    action: ActionPlan
    explanation: str
    dtw: SignatureMatch
    load_plan: list[tuple[str, str, float]]

class AgentConfig:
    lookback: int = 60
    cell_id: str = "Cell-1"

class AgentConfig:
    def __init__(self, lookback: int = 60, cell_id: str = "Cell-1"):
        self.lookback = lookback
        self.cell_id = cell_id

class InterferenceAgent:
    def __init__(
        self,
        cfg: AgentConfig | None = None,
        matcher: DTWSignatureEngine | None = None,
    ):
        self.cfg = cfg or AgentConfig()
        self.matcher = matcher or DTWSignatureEngine.from_synthetic(
            timesteps=self.cfg.lookback
        )

    def run_on_sequence(self, df: pd.DataFrame) -> AgentOutput:
        """
        Run the agent on a full KPI time-series and focus on the trailing window.
        """
        if len(df) < self.cfg.lookback * 2:
            raise ValueError("Time-series too short for before/after comparison.")

        window = df.tail(self.cfg.lookback)
        signature = self.matcher.match(window)

        # Keep rule-based stats for interpretability but use DTW label
        stats = _window_stats(df, lookback=self.cfg.lookback)
        diagnosis = Diagnosis(
            root_cause=signature.label,  # type: ignore
            confidence=signature.confidence,
            features=stats,
        )

        load_plan, cost, projection = self._maybe_plan_load_shift(diagnosis)
        action = propose_action(
            diagnosis,
            cell_id=self.cfg.cell_id,
            rebalancing_plan=load_plan,
            plan_cost=cost,
        )
        explanation = self._build_explanation(diagnosis, action, signature, load_plan, projection)

        return AgentOutput(
            diagnosis=diagnosis,
            action=action,
            explanation=explanation,
            dtw=signature,
            load_plan=load_plan,
        )
    
    def _maybe_plan_load_shift(self, diagnosis: Diagnosis) -> tuple[list[tuple[str, str, float]], float | None, LoadShiftPlan | None]:
        if diagnosis["root_cause"] != "congestion":
            return [], None, None

        prb_now = diagnosis["features"].get("prb_util_now", 70.0)
        cell_loads = {
            self.cfg.cell_id: prb_now,
            "Cell-2": max(45.0, prb_now - 10),
            "Cell-3": 55.0,
            "Cell-4": 60.0,
        }
        capacities = {cid: 100.0 for cid in cell_loads}
        neighbor_costs = {
            (self.cfg.cell_id, "Cell-2"): 1.0,
            (self.cfg.cell_id, "Cell-3"): 1.5,
            (self.cfg.cell_id, "Cell-4"): 2.0,
            ("Cell-2", self.cfg.cell_id): 1.2,
            ("Cell-3", self.cfg.cell_id): 1.4,
            ("Cell-2", "Cell-3"): 0.8,
            ("Cell-3", "Cell-4"): 1.1,
        }
        plan_result = optimize_load_shift(
            focal_cell=self.cfg.cell_id,
            cell_loads=cell_loads,
            capacities=capacities,
            neighbor_costs=neighbor_costs,
            target_utilization=70.0,
        )
        return plan_result.moves, plan_result.total_cost, plan_result

    def _build_explanation(
        self,
        diagnosis: Diagnosis,
        action: ActionPlan,
        signature: SignatureMatch,
        load_plan: list[tuple[str, str, float]],
        projection: LoadShiftPlan | None,
    ) -> str:
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
            if load_plan:
                moves = "; ".join([f"{a}->{b}: {amt:.1f}" for a, b, amt in load_plan])
                base.append(f"Graph optimizer proposes shifts: {moves}.")
            if projection:
                base.append(
                    f"Projected PRB on focal cell after shift: {projection.projected_loads['focal']:.1f}% "
                    f"(relief {projection.relieved_prb:.1f} p.p.)."
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

        base.append(
            f"DTW distances → normal: {signature.distances['normal']:.1f}, "
            f"congestion: {signature.distances['congestion']:.1f}, "
            f"external_interference: {signature.distances['external_interference']:.1f}."
        )

        return " ".join(base)
