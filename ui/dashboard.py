import sys, os

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import streamlit as st
import textwrap

from agent.kpi_simulator import KpiConfig, generate_kpi_series
from agent.reasoner import InterferenceAgent
from agent.what_if_simulator import apply_action_to_kpis


def main():
    st.set_page_config(page_title="RAN Interference Algorithms Lab", layout="wide")
    st.title("RAN Interference Algorithms Lab")
    
    st.caption(
        "Dynamic Time Warping (DTW) signature detection + graph-based min-cost flow mitigation"
    )
    st.info(
        "Start by choosing or adjusting a scenario, then generate KPIs and let the agent "
        "diagnose the issue. Review the recommended mitigation before applying it to see "
        "the projected impact."
    )

    with st.expander("How to use this dashboard", expanded=True):
        st.markdown(
            "1. Select a scenario and seed from the sidebar, then choose the number of timesteps.\n"
            "2. Inspect the generated KPIs to understand the baseline behavior.\n"
            "3. Review the agent's diagnosis summary and open the details tab for full context.\n"
            "4. Apply the recommended mitigation to visualize the expected improvement."
        )

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

    core1, core2 = st.columns([1.5, 1.5])

    with core1:
        st.subheader("DTW Signature Engine")
        st.caption("Multivariate DTW distances to learned signatures (lower is better)")
        dist_df = pd.DataFrame(
            [
                {
                    "class": label.replace("_", " "),
                    "distance": value,
                }
                for label, value in out["dtw"].distances.items()
            ]
        ).sort_values("distance")
        st.dataframe(dist_df, hide_index=True, use_container_width=True)
        st.metric(
            "Predicted class",
            out["diagnosis"]["root_cause"],
            f"confidence {out['diagnosis']['confidence']:.2f}",
        )

    with core2:
        st.subheader("Graph Optimizer (Min-Cost Flow)")
        st.caption("Optimal offload plan when congestion is detected")
        if out["load_plan"]:
            plan_df = pd.DataFrame(
                out["load_plan"], columns=["from", "to", "amount"]
            )
            st.dataframe(plan_df, hide_index=True, use_container_width=True)
            cost = out["action"].get("optimization_cost")
            if cost is not None:
                st.metric("Optimization cost", f"{cost:.2f}")
        else:
            st.info("No congestion detected → optimizer idle.")

    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Original KPIs")
        st.caption("Showing SINR, PRB utilization, PUSCH noise, and BLER over time.")
        st.line_chart(df.set_index("time")[["sinr", "prb_util", "pusch_noise", "bler"]])

    with col2:
        st.subheader("Agent Diagnosis + Action")
        st.write(f"**Scenario:** `{scenario}`")

        root_cause = out["diagnosis"]["root_cause"]
        truncated_root_cause = textwrap.shorten(root_cause, width=60, placeholder=" …")

        summary_tab, details_tab = st.tabs(["Summary", "Details"])

        with summary_tab:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Root cause", truncated_root_cause)
                st.metric("Confidence", f"{out['diagnosis']['confidence']:.2f}")
            with c2:
                st.metric("Action intent", out["action"]["intent"])

        with details_tab:
            st.markdown(f"**Full root cause:** `{root_cause}`")
            st.markdown(out["explanation"])

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
