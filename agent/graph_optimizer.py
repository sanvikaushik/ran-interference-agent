"""Graph-inspired load mitigation without external dependencies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class CellLoad:
    cell_id: str
    load: float
    capacity: float


@dataclass
class EdgeCost:
    src: str
    dst: str
    cost: float

@dataclass
class LoadShiftPlan:
    """Detailed result of the graph optimization step."""

    moves: list[tuple[str, str, float]]
    total_cost: float
    projected_loads: Dict[str, float]
    relieved_prb: float

    @property
    def relief_percent(self) -> float:
        """Percentage relief for the focal cell relative to its pre-plan load."""
        initial = self.relieved_prb + self.projected_loads.get("focal", 0)
        if initial == 0:
            return 0.0
        return (self.relieved_prb / initial) * 100


def _build_edges(neighbor_costs: Dict[Tuple[str, str], float]) -> List[EdgeCost]:
    return [EdgeCost(src=u, dst=v, cost=c) for (u, v), c in neighbor_costs.items()]

def _project_loads(
    cell_loads: Dict[str, float],
    capacities: Dict[str, float],
    plan: List[tuple[str, str, float]],
) -> Dict[str, float]:
    """Compute projected loads after applying a shift plan.

    Loads are clamped to [0, capacity] for safety.
    """
    projected = dict(cell_loads)
    for src, dst, amount in plan:
        projected[src] = max(0.0, projected.get(src, 0.0) - amount)
        projected[dst] = min(capacities.get(dst, projected.get(dst, 0.0)), projected.get(dst, 0.0) + amount)
    return projected


def plan_load_shift(
    focal_cell: str,
    cell_loads: Dict[str, float],
    capacities: Dict[str, float],
    neighbor_costs: Dict[Tuple[str, str], float],
    target_utilization: float = 70.0,
    max_transfer: float = 30.0,
) -> Tuple[List[tuple[str, str, float]], float]:
    """Greedy min-cost pairing of overloaded cells to neighbors.

    Returns a list of (from_cell, to_cell, amount) and total cost.
    """
    edges = _build_edges(neighbor_costs)
    default_cost = 5.0

    deltas: Dict[str, float] = {}
    for cell, load in cell_loads.items():
        desired = min(capacities[cell], target_utilization)
        deltas[cell] = load - desired

    overloads = {c: d for c, d in deltas.items() if d > 0}
    underloads = {c: -d for c, d in deltas.items() if d < 0}

    plan: List[tuple[str, str, float]] = []
    total_cost = 0.0

    for src, supply in overloads.items():
        remaining = supply
        # prioritize cheapest edges first
        candidates = [e for e in edges if e.src == src and underloads.get(e.dst, 0) > 0]
        candidates.sort(key=lambda e: e.cost)

        # fallback to any underloaded neighbor with default cost
        if not candidates:
            candidates = [
                EdgeCost(src=src, dst=dst, cost=default_cost)
                for dst in underloads
                if underloads[dst] > 0
            ]

        for edge in candidates:
            if remaining <= 0:
                break
            spare = underloads.get(edge.dst, 0)
            if spare <= 0:
                continue
            move = min(remaining, spare, max_transfer)
            if move <= 0:
                continue
            plan.append((edge.src, edge.dst, move))
            remaining -= move
            underloads[edge.dst] = spare - move
            total_cost += move * edge.cost

    # Keep plan focused on relieving the focal cell but retain spillover if needed
    filtered_plan = [p for p in plan if p[0] == focal_cell or p[1] == focal_cell]
    return filtered_plan or plan, total_cost

def optimize_load_shift(
    focal_cell: str,
    cell_loads: Dict[str, float],
    capacities: Dict[str, float],
    neighbor_costs: Dict[Tuple[str, str], float],
    target_utilization: float = 70.0,
    max_transfer: float = 30.0,
) -> LoadShiftPlan:
    """Run the greedy planner and provide projected relief details."""
    plan, cost = plan_load_shift(
        focal_cell=focal_cell,
        cell_loads=cell_loads,
        capacities=capacities,
        neighbor_costs=neighbor_costs,
        target_utilization=target_utilization,
        max_transfer=max_transfer,
    )
    projected = _project_loads(cell_loads, capacities, plan)
    relieved = max(0.0, cell_loads.get(focal_cell, 0.0) - projected.get(focal_cell, cell_loads.get(focal_cell, 0.0)))
    projected_with_marker = {"focal": projected.get(focal_cell, 0.0), **projected}
    return LoadShiftPlan(
        moves=plan,
        total_cost=cost,
        projected_loads=projected_with_marker,
        relieved_prb=relieved,
    )
