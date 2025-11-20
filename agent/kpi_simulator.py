import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal

Scenario = Literal["normal", "external_interference", "congestion"]


@dataclass
class KpiConfig:
    timesteps: int = 300
    seed: int | None = None


def _base_series(cfg: KpiConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    t = np.arange(cfg.timesteps)

    sinr = 20 + rng.normal(0, 1, cfg.timesteps)
    rsrq = -8 + rng.normal(0, 0.5, cfg.timesteps)
    prb_util = 40 + rng.normal(0, 3, cfg.timesteps)
    pusch_noise = -95 + rng.normal(0, 1, cfg.timesteps)
    bler = 1 + rng.normal(0, 0.3, cfg.timesteps)
    ho_fail = rng.poisson(1, cfg.timesteps)

    return pd.DataFrame(
        {
            "time": t,
            "sinr": sinr,
            "rsrq": rsrq,
            "prb_util": prb_util,
            "pusch_noise": pusch_noise,
            "bler": bler,
            "ho_failures": ho_fail,
        }
    )


def generate_kpi_series(
    scenario: Scenario = "normal",
    cfg: KpiConfig | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic RAN KPI time-series for one cell under a given scenario.
    """
    if cfg is None:
        cfg = KpiConfig()

    df = _base_series(cfg)
    rng = np.random.default_rng(cfg.seed)

    if scenario == "external_interference":
        w_end = cfg.timesteps
        w_start = w_end - cfg.timesteps // 4  # last quarter
        win = slice(w_start, w_end)

        length = w_end - w_start

        df.iloc[win, df.columns.get_loc("sinr")] -= np.linspace(5, 15, length)
        df.iloc[win, df.columns.get_loc("pusch_noise")] += np.linspace(3, 12, length)
        df.iloc[win, df.columns.get_loc("rsrq")] -= np.linspace(1, 4, length)
        df.iloc[win, df.columns.get_loc("ho_failures")] += rng.poisson(3, length)
        df.iloc[win, df.columns.get_loc("bler")] += np.linspace(1, 3, length)

    elif scenario == "congestion":
        w_end = cfg.timesteps
        w_start = w_end - cfg.timesteps // 3  # last third
        win = slice(w_start, w_end)

        length = w_end - w_start

        df.iloc[win, df.columns.get_loc("prb_util")] += np.linspace(20, 40, length)
        df.iloc[win, df.columns.get_loc("bler")] += np.linspace(1, 2.5, length)
        df.iloc[win, df.columns.get_loc("sinr")] -= np.linspace(1, 4, length)

    return df




def generate_dataset(
    n_per_scenario: int = 20,
    cfg: KpiConfig | None = None,
) -> list[dict]:
    """
    Generate a small dataset of labeled sequences for evaluation.

    Returns a list of entries: { "scenario": str, "df": DataFrame }
    """
    if cfg is None:
        cfg = KpiConfig()

    data: list[dict] = []
    scenarios: list[Scenario] = ["normal", "external_interference", "congestion"]

    seed_base = cfg.seed or 42
    for i, scenario in enumerate(scenarios):
        for j in range(n_per_scenario):
            local_cfg = KpiConfig(
                timesteps=cfg.timesteps,
                seed=seed_base + i * 100 + j,
            )
            df = generate_kpi_series(scenario=scenario, cfg=local_cfg)
            data.append({"scenario": scenario, "df": df})

    return data
