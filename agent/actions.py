from typing import TypedDict, Literal

from .rulebase import Diagnosis, DiagnosisLabel


IntentType = Literal[
    "ADJUST_DL_TILT",
    "REDUCE_TX_POWER",
    "BALANCE_LOAD",
    "NO_ACTION",
]


class ActionPlan(TypedDict):
    intent: IntentType
    description: str
    parameters: dict
    optimization_cost: float | None

def propose_action(
    diagnosis: Diagnosis,
    cell_id: str = "Cell-1",
    rebalancing_plan: list[tuple[str, str, float]] | None = None,
    plan_cost: float | None = None,
) -> ActionPlan:
    cause: DiagnosisLabel = diagnosis["root_cause"]

    if cause == "external_interference":
        return ActionPlan(
            intent="ADJUST_DL_TILT",
            description=(
                "Suggest slight downlink tilt change and channel reassessment "
                "to mitigate suspected external interference."
            ),
            parameters={
                "cell_id": cell_id,
                "delta_tilt_deg": -1.5,
                "recheck_interval_s": 300,
            },
            optimization_cost=None,
        )

    if cause == "congestion":
        return ActionPlan(
            intent="BALANCE_LOAD",
            description=(
                "Suggest load-balancing: adjust handover offsets or QCI scheduling "
                "to relieve congestion on the current cell."
            ),
            parameters={
                "cell_id": cell_id,
                "max_prb_util_target": 70,
                "offload_neighbors": ["Cell-2", "Cell-3"],
                "proposed_plan": rebalancing_plan or [],
            },
            optimization_cost=plan_cost,
        )

    if cause == "normal":
        return ActionPlan(
            intent="NO_ACTION",
            description="KPIs are within expected ranges. No immediate RAN changes suggested.",
            parameters={"cell_id": cell_id},
            optimization_cost=None,
        )

    # unknown
    return ActionPlan(
        intent="NO_ACTION",
        description=(
            "Pattern is not clearly mapped to known issues. Recommend deeper "
            "offline analysis before applying configuration changes."
        ),
        parameters={"cell_id": cell_id},
        optimization_cost=None,
    )
