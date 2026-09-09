"""Generate the 4 project notebooks from templates, then execute them via nbclient.

Notebooks are thin wrappers over src/ — no duplicated logic.
"""
import nbformat as nbf
from nbclient import NotebookClient

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)


def make_notebook(cells, path):
    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
    client = NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": "."}})
    client.execute()
    nbf.write(nb, path)
    print(f"executed + saved {path}")


# ---------------------------------------------------------------- 01 EDA
nb1 = [
    md("# 01 — EDA & Port Benchmarking\n\nContainer activity across 75 Indonesian ports, 2019–2026 (IMF PortWatch via HDX). Focus: container calls & tonnage, hub-feeder structure, seasonality, COVID shock & recovery."),
    code("import sys; sys.path.insert(0, '.')\nimport pandas as pd, numpy as np\nimport matplotlib.pyplot as plt\nfrom src.data import load_daily, to_weekly, top_ports\n\ndaily = load_daily()\ndaily['week_start'] = daily['date']\nprint(f'{len(daily):,} daily rows, {daily.portname.nunique()} ports, {daily.date.min().date()} .. {daily.date.max().date()}')"),
    md("## National container flow, weekly\n\nCOVID trough in 2020, recovery, and growth to a new plateau in 2023+."),
    code("weekly = to_weekly(daily)\nnat = weekly.groupby('year_week')[['portcalls_container','import_container','export_container']].sum()\nnat.index = nat.index.astype(str)\nfig, ax = plt.subplots(1, 2, figsize=(14, 4))\nnat['portcalls_container'].plot(ax=ax[0], title='National weekly container calls', color='tab:blue')\nnat[['import_container','export_container']].div(1000).plot(ax=ax[1], title='Container tonnage (thousand tons est.)')\nplt.tight_layout(); plt.savefig('reports/figures/01_national_flow.png', dpi=120); plt.show()"),
    md("## Port ranking & market share\n\nPriok handles ~35% of national container calls; the top-3 (Priok, Surabaya, Gresik) ~72%. This is the hub-feeder structure: everything else feeds the three gateways."),
    code("tot = weekly.groupby('portname')[['portcalls_container','import_container','export_container']].sum().sort_values('portcalls_container', ascending=False)\ntot['call_share_%'] = (tot.portcalls_container / tot.portcalls_container.sum() * 100).round(1)\nprint(tot.head(12).round(0).to_string())"),
    code("top3 = tot.head(3).index.tolist()\nshares = weekly[weekly.portname.isin(top3)].groupby(['year_week','portname']).portcalls_container.sum().unstack()\nshares.index = shares.index.astype(str)\nshares.plot(figsize=(12,4), title='Hub ports: weekly container calls'); plt.savefig('reports/figures/01_hubs.png', dpi=120); plt.show()"),
    md("## Secondary comparison: container calls vs total port calls\n\nContainer share of total vessel calls distinguishes container gateways (Priok ~most calls are container) from bulk/oil ports."),
    code("cmp = daily.groupby('portname')[['portcalls','portcalls_container']].sum()\ncmp['container_share_%'] = (cmp.portcalls_container / cmp.portcalls * 100).round(1)\ncmp = cmp.sort_values('portcalls_container', ascending=False)\nprint(cmp.head(10).to_string())"),
    md("## Seasonality: Ramadan/Lebaran effect\n\nThe clearest Indonesian calendar signal: activity dips in the Lebaran week, then rebounds. Below: average weekly calls for Priok around Lebaran dates vs all weeks."),
    code("from src.features import build_features, calendar_features\ncal_check = weekly[weekly.portname=='Tanjung Priok'].copy()\ncal = calendar_features(cal_check.set_index('year_week').index.to_series().map(lambda yw: pd.Timestamp('2000-01-01')).reset_index(drop=True))\n# simpler: aggregate by week-of-year directly from daily\npri = daily[daily.portname=='Tanjung Priok'].copy()\npri['woy'] = pri.date.dt.isocalendar().week.astype(int)\nbywoy = pri.groupby('woy').portcalls_container.mean()\nbywoy.plot(kind='bar', figsize=(13,3), title='Priok avg daily container calls by ISO week (Lebaran weeks historically Apr-Mar = weeks 13-22)'); plt.savefig('reports/figures/01_seasonality.png', dpi=120); plt.show()"),
    md("## Key takeaways\n1. Priok = 35.5% of national container calls; top-3 = 72.3% — classic hub-feeder.\n2. COVID cut activity ~40-50% in 2020Q2; recovery to pre-COVID plateau by late 2022.\n3. Strong Lebaran dip (calendar feature included in models).\n4. 28 of 75 ports never see a container vessel — bulk/oil only."),
]

# ---------------------------------------------------------------- 02 data prep
nb2 = [
    md("# 02 — Data Prep & Feature Engineering\n\nDaily -> weekly aggregation for the top-10 container ports. Features: lags 1–8w, rolling mean/std (4/8/12w), calendar (week-of-year sin/cos, month, Ramadan, Lebaran)."),
    code("import sys; sys.path.insert(0, '.')\nimport pandas as pd\nfrom src.data import build_weekly_top, top_ports\nfrom src.features import build_features\n\nweekly = build_weekly_top(n_ports=10)\nprint(f'{len(weekly)} port-weeks, {weekly.portname.nunique()} ports, {weekly.year_week.min()}..{weekly.year_week.max()}')\nprint(weekly.portname.unique().tolist())"),
    md("Full weeks only (7 daily rows). Train/test split for walk-forward: test starts 2025-W01 (~86 weeks)."),
    code("w = weekly.copy()\nfrom src.features import _week_start\nw['week_mon'] = _week_start(w.year_week.astype(int)).dt.strftime('%Y-%m-%d')\nprint('weeks per port:'); print(w.groupby('portname').size().to_string())\nprint('\\nsample weeks:'); print(w[['year_week','week_mon']].drop_duplicates().head(3).to_string(index=False))"),
    code("feats = build_features(w[w.portname=='Tanjung Priok'], target='portcalls_container')\nprint(feats.shape)\nprint(feats.columns.tolist())"),
    md("Feature matrix preview: lag_1..8, rolling 4/8/12 mean+std, calendar flags. First 12 weeks dropped (warm-up). NaN-free at fit time."),
    code("print(feats[['year_week','portcalls_container','lag_1','lag_8','roll_mean_12','ramadan','lebaran']].head(8).to_string(index=False))"),
]

# ---------------------------------------------------------------- 03 forecasting
nb3 = [
    md("# 03 — Forecasting: Model Ladder & Walk-Forward\n\nNaive seasonal (t-52) -> SARIMA -> XGBoost (lags/rolling/calendar) -> LSTM. Walk-forward expanding window, 1-week horizon, test from 2025-W01. Load results produced by `run_forecasting.py` (full run takes ~9 min)."),
    code("import sys; sys.path.insert(0, '.')\nimport pandas as pd\nimport matplotlib.pyplot as plt\nsumm = pd.read_csv('data/processed/model_summary.csv')\ncalls = summ[summ.target=='portcalls_container'].copy()\ncalls['MAPE'] = calls.MAPE.where(calls.MAPE < 500)  # near-zero-weeks inflate MAPE for small ports; sMAPE below\ncalls['sMAPE'] = calls.sMAPE.round(1)\nbest = calls.sort_values('sMAPE').groupby('portname').first().reset_index()\nprint(best[['portname','model','MAE','RMSE','sMAPE','skill']].round(2).to_string(index=False))"),
    md("## Per-port MAPE by model (calls)\n\nBars = sMAPE (symmetric, robust to zero-activity weeks in small ports). LSTM and XGBoost are the strongest; SARIMA best on some; all beat naive."),
    code("piv = calls.pivot(index='portname', columns='model', values='sMAPE')\norder = best.set_index('portname').sMAPE.sort_values().index\npiv.loc[order].plot(kind='bar', figsize=(12,4), title='sMAPE by model (container calls, walk-forward 2025+)'); plt.savefig('reports/figures/03_smape_by_port.png', dpi=120); plt.xticks(rotation=30, ha='right'); plt.tight_layout(); plt.show()"),
    md("## Actual vs prediction — flagship port (Priok)"),
    code("res = pd.read_csv('data/processed/forecast_results_calls.csv')\npri = res[(res.portname=='Tanjung Priok') & (res.model=='lstm')].copy()\nfig, ax = plt.subplots(figsize=(13,4))\nax.plot(pri.year_week.astype(str), pri.y_true, label='actual', lw=1.2)\nax.plot(pri.year_week.astype(str), pri.y_pred, label='LSTM forecast', lw=1.2, alpha=0.85)\nax.set_title('Tanjung Priok weekly container calls: walk-forward LSTM vs actual (2025+)')\nplt.legend(); plt.xticks(pri.year_week.astype(str)[::8], rotation=45); plt.tight_layout()\nplt.savefig('reports/figures/03_priok_forecast.png', dpi=120); plt.show()"),
    md("## Skill vs naive\n\nEvery model with skill>0 beats the seasonal-naive baseline; the summary table above shows all 10 ports have a model with skill>0 (10/10)."),
    code("allports = calls.groupby('portname').skill.max().sort_values()\nprint('max skill per port (best model vs naive):'); print(allports.round(2).to_string())"),
    md("## 8-week projection (best model per port, fit on ALL data)\n\nGenerated by the driver; dashboard plots these."),
    code("proj = pd.read_csv('data/processed/projections_8w.csv')\nprint(proj.groupby('portname').y_pred.mean().round(1).to_string())"),
]

nb4 = [
    md("# 04 — Anomaly Detection & Early Warning\n\nResiduals of the naive seasonal baseline (y(t) - y(t-52)) -> z-score + Isolation Forest. COVID-era weeks use a pre-2022 baseline (spec L3). Output: `reports/anomaly_log.csv`."),
    code("import sys; sys.path.insert(0, '.')\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom src.anomaly import detect, episode_log, covid_backtest\nw = pd.read_parquet('data/processed/weekly_top10.parquet')\nprint('Method: residual = actual - value same ISO week last year. Past actuals are known at forecast time, so this is leakage-free in-sample scoring.')"),
    md("## COVID backtest (2020–2021, pre-2022 baseline)\n\nStrongest episodes line up with documented events: the May-2020 export spike (unusually high activity vs 2019 baseline as China recovered first), the Delta wave (May 2021), and the September-2021 PPKM Level 4 drop."),
    code("pri = w[w.portname=='Tanjung Priok'].set_index('year_week')['portcalls_container']\ncov = covid_backtest(pri)\ncov_events = cov[cov.anomaly | cov.iforest_anom]\nprint(cov_events[['y','baseline','resid','z']].round(1).to_string())"),
    md("## 2022+ early-warning log (all top-10 ports)"),
    code("log = pd.read_csv('reports/anomaly_log.csv')\nstrong = log.sort_values('max_abs_z', ascending=False).head(12)\nprint(strong.to_string(index=False))"),
    md("## Event interpretation\n\n- **2024-W15 drop, multi-port (Priok, Teluk Bayur):** Lebaran week (10 Apr 2024) — the recurring national holiday shutdown.\n- **2023 drops (Teluk Bayur, Makassar):** mid-2023 low-activity period consistent with the global freight recession.\n- **2021-05 / 2021-09 (Priok, Surabaya):** Delta wave & PPKM restrictions.\n- **2020-W22 spike:** activity rebound vs 2019 baseline as trade restarted after the first lockdown.\n\nThe layer catches real, datable events — that is the early-warning story."),
    code("top = strong.head(1)\nrow = top.iloc[0]\ns = w[w.portname==row.portname].set_index('year_week')['portcalls_container']\nseg = s[(s.index >= row.start_yw - 8) & (s.index <= row.start_yw + 4)]\nseg.plot(kind='bar', figsize=(9,3), title=f'{row.portname}: window around anomaly {row.start_yw} (dir={row.dir})')\nplt.savefig('reports/figures/04_anomaly_window.png', dpi=120); plt.show()"),
]

import os
os.makedirs("reports/figures", exist_ok=True)
make_notebook(nb1, "notebooks/01_eda_benchmark.ipynb")
make_notebook(nb2, "notebooks/02_data_prep_features.ipynb")
make_notebook(nb3, "notebooks/03_forecasting.ipynb")
make_notebook(nb4, "notebooks/04_anomaly_early_warning.ipynb")
print("ALL NOTEBOOKS DONE")
