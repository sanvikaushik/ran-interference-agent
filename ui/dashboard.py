# ui/dashboard.py
import sys, os

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import streamlit as st

from agent.kpi_simulator import KpiConfig, generate_kpi_series
from agent.reasoner import InterferenceAgent
from agent.what_if_simulator import apply_action_to_kpis


def main():
    st.set_page_config(page_title="RAN Interference-Hunting Agent", layout="wide")

    st.title("RAN Interference-Hunting Agent – What-if Simulator")

    # Sidebar controls
    st.sidebar.header("Scenario configuration")
    scenario = st.sidebar.selectbox(
        "Scenario",
        ["normal", "external_interference", "congestion"],
        index=1,
    )
    seed = st.sidebar.number_input("Random seed", value=123, step=1)
    timesteps = st.sidebar.slider("Timesteps", min_value=150, max_value=600, value=300, step=50)

    cfg = KpiConfig(timesteps=int(timesteps), seed=int(seed))
    df = generate_kpi_series(scenario, cfg=cfg)

    agent = InterferenceAgent()
    out = agent.run_on_sequence(df)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Original KPIs")
        st.caption("Showing SINR, PRB utilization, PUSCH noise, and BLER over time.")
        st.line_chart(df.set_index("time")[["sinr", "prb_util", "pusch_noise", "bler"]])

    with col2:
        st.subheader("Agent Diagnosis")
        st.write(f"**Scenario:** `{scenario}`")
        st.write(f"**Root cause:** `{out['diagnosis']['root_cause']}`")
        st.write(f"**Confidence:** `{out['diagnosis']['confidence']:.2f}`")
        st.write(f"**Action intent:** `{out['action']['intent']}`")
        st.write(out["explanation"])

    st.markdown("---")

    mitigated_df = None
    note = ""

    if out["action"]["intent"] != "NO_ACTION":
        if st.button("Apply recommended mitigation"):
            mitigated_df, note = apply_action_to_kpis(df, out["action"])

    if mitigated_df is not None:
        st.subheader("Effect of Mitigation")

        st.caption(note)

        c1, c2 = st.columns(2)

        # SINR before vs after
        with c1:
            st.markdown("**SINR Before vs After**")
            sinr_orig = df[["time", "sinr"]].copy()
            sinr_orig["series"] = "original"

            sinr_mit = mitigated_df[["time", "sinr"]].copy()
            sinr_mit["series"] = "mitigated"

            plot_df = pd.concat([sinr_orig, sinr_mit], ignore_index=True)
            plot_df = plot_df.set_index("time")
            st.line_chart(plot_df.pivot(columns="series", values="sinr"))

        # PRB util before vs after (for congestion cases)
        with c2:
            st.markdown("**PRB Utilization Before vs After**")
            prb_orig = df[["time", "prb_util"]].copy()
            prb_orig["series"] = "original"

            prb_mit = mitigated_df[["time", "prb_util"]].copy()
            prb_mit["series"] = "mitigated"

            plot_prb = pd.concat([prb_orig, prb_mit], ignore_index=True)
            plot_prb = plot_prb.set_index("time")
            st.line_chart(plot_prb.pivot(columns="series", values="prb_util"))

        st.markdown("**All KPIs After Mitigation**")
        st.line_chart(mitigated_df.set_index("time")[["sinr", "prb_util", "pusch_noise", "bler"]])
    else:
        st.info("Click 'Apply recommended mitigation' to simulate and visualize the change.")


if __name__ == "__main__":
    main()
