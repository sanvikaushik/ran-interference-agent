# tests/test_kpi_simulator.py
import pandas as pd

from agent.kpi_simulator import generate_kpi_series, KpiConfig


def test_kpi_simulator_shape_and_columns():
    cfg = KpiConfig(timesteps=200, seed=1)
    df = generate_kpi_series("normal", cfg=cfg)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 200
    for col in ["time", "sinr", "rsrq", "prb_util", "pusch_noise", "bler", "ho_failures"]:
        assert col in df.columns


def test_external_interference_pattern():
    cfg = KpiConfig(timesteps=300, seed=2)
    df = generate_kpi_series("external_interference", cfg=cfg)

    head = df.head(60)["sinr"].mean()
    tail = df.tail(60)["sinr"].mean()
    sinr_drop = head - tail

    assert sinr_drop > 5.0  # verify the interference actually hurts SINR
