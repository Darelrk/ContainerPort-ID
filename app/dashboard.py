"""ContainerPort-ID dashboard: forecasts, anomalies, and port benchmarking.

Run: streamlit run app/dashboard.py
Reads artifacts produced by run_forecasting.py + reports/anomaly_log.csv.
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="ContainerPort-ID", layout="wide", page_icon="🚢")

# ---------------------------------------------------------------- data
@st.cache_data
def load_all():
    weekly = pd.read_parquet(ROOT / "data/processed/weekly_top10.parquet")
    summary = pd.read_csv(ROOT / "data/processed/model_summary.csv")
    res_calls = pd.read_csv(ROOT / "data/processed/forecast_results_calls.csv")
    proj = pd.read_csv(ROOT / "data/processed/projections_8w.csv")
    anom = pd.read_csv(ROOT / "reports/anomaly_log.csv")
    return weekly, summary, res_calls, proj, anom

weekly, summary, res_calls, proj, anom = load_all()

# ---------------------------------------------------------------- header
st.title("🚢 ContainerPort-ID")
st.caption("Weekly container-activity forecasting & early warning across Indonesia's top-10 container ports · data: IMF PortWatch (via HDX), 2019–2026")

tab1, tab2, tab3 = st.tabs(["Forecast", "Anomalies", "Benchmark"])

# ---------------------------------------------------------------- forecast tab
with tab1:
    c1, c2 = st.columns([1, 3])
    with c1:
        port = st.selectbox("Port", sorted(weekly.portname.unique()))
        port_sum = summary[(summary.portname == port) & (summary.target == "portcalls_container")]
        best_row = port_sum.sort_values("sMAPE").iloc[0]
        st.metric("Best model (sMAPE)", best_row.model)
        st.metric("sMAPE %", f"{best_row.sMAPE:.1f}")
        st.metric("Skill vs naive", f"{best_row.skill*100:+.0f}%")
    with c2:
        s = weekly[weekly.portname == port].sort_values("year_week")
        preds = res_calls[(res_calls.portname == port) & (res_calls.model == best_row.model)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s.year_week.astype(str), y=s.portcalls_container,
                                 mode="lines", name="Actual", line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(x=preds.year_week.astype(str), y=preds.y_pred,
                                 mode="lines", name=f"{best_row.model} walk-forward", line=dict(color="#ff7f0e")))
        pj = proj[proj.portname == port].sort_values("year_week")
        fig.add_trace(go.Scatter(x=pj.year_week.astype(str), y=pj.y_pred,
                                 mode="lines", name="8-week projection", line=dict(color="#2ca02c", dash="dot")))
        fig.update_layout(title=f"{port} — weekly container calls", height=420,
                          xaxis_title="ISO week", yaxis_title="calls")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(port_sum[["model", "MAE", "RMSE", "sMAPE", "skill"]].round(2), hide_index=True)

# ---------------------------------------------------------------- anomaly tab
with tab2:
    st.subheader("Early-warning log")
    a = anom.copy()
    a["date"] = a.start_yw.astype(str)
    st.dataframe(
        a.sort_values("max_abs_z", ascending=False).head(30)[["portname", "start_yw", "end_yw", "dir", "max_abs_z", "n_weeks"]].rename(
            columns={"start_yw": "week (start)", "end_yw": "week (end)", "dir": "direction", "max_abs_z": "|z|", "n_weeks": "weeks"}),
        hide_index=True,
    )
    fig = px.scatter(a, x="start_yw", y="portname", color="dir", size="max_abs_z",
                     color_discrete_map={"drop": "#d62728", "spike": "#2ca02c"},
                     labels={"start_yw": "ISO week", "portname": ""},
                     title="Anomaly episodes (green=spike, red=drop; size=|z|)")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""**Detected events include:** Lebaran shutdowns (e.g. 2024-W15 drop across ports), the May-2021 Delta wave and Sept-2021 PPKM restrictions (Priok/Surabaya), the 2020 post-lockdown rebound spike, and mid-2023 freight-recession lows (Teluk Bayur, Makassar).""")

# ---------------------------------------------------------------- benchmark tab
with tab3:
    st.subheader("Port benchmarking (container calls, weekly)")
    tot = weekly.groupby("portname")[["portcalls_container", "import_container", "export_container"]].sum()
    tot["share_%"] = (tot.portcalls_container / tot.portcalls_container.sum() * 100).round(1)
    st.dataframe(tot.sort_values("portcalls_container", ascending=False).round(0))
    fig = px.bar(tot.sort_values("portcalls_container", ascending=False).reset_index(),
                 x="portname", y="portcalls_container", title="Total container calls by port (2019–2026)")
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)
    pivot = weekly.pivot_table(index="year_week", columns="portname", values="portcalls_container")
    fig2 = px.area(pivot, title="Weekly container calls by port — hub & feeder structure")
    fig2.update_layout(height=420)
    st.plotly_chart(fig2, use_container_width=True)
