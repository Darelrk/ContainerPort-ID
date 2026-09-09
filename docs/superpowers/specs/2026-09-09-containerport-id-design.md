# ContainerPort-ID — Design Spec

**Date:** 2026-09-09
**Owner:** Darrell Rafif Kenzie (Darelrk)
**Status:** Approved (with amendment: portcalls total as secondary comparison in EDA)
**Goal:** Flagship data-science portfolio project targeting internships at Samudera Indonesia, Pelindo, and maritime-logistics companies.

## 1. Problem & Positioning

Indonesian container ports (Tanjung Priok, Surabaya, Gresik, etc.) operate on berth planning, crane scheduling, and truck-flow management. Terminal operators need short-horizon visibility into: (a) how many container vessels will call next week, (b) how much import/export tonnage will flow, and (c) early warning when activity deviates abnormally (disruption, policy shifts, seasonal shocks).

This project builds that visibility from public data:

- **Forecast** weekly container port calls and container tonnage for top Indonesian ports.
- **Detect anomalies** in container-flow residuals as an early-warning layer.
- **Benchmark** ports and characterize the hub-feeder structure of the national container network.

Narrative link to prior work: KKI2026 (autonomous surface vessel + mission dashboard) covers the maritime engineering side; this project covers the maritime data/economics side.

## 2. Data (verified, downloaded, quality-checked)

**Source:** IMF PortWatch via HDX (OCHA), dataset `indonesia-daily-port-activity-data-and-shipment-estimates`, updated weekly by the `hdx-scraper-portwatch` pipeline.

- **Local file:** `data/raw/portwatch_indonesia_daily.parquet` (262,088 rows × 29 cols) — already saved.
- **Coverage:** 75 Indonesian ports, daily, 2019-01-01 → 2026-08-28 (fresh as of this week).
- **License:** public/open (IMF PortWatch / World Bank alternative-data program). Cite in README.
- **Columns used:** `date, portname, portcalls, portcalls_container, import, export, import_container, export_container` (+ by-type splits as needed).
- **Verified quality:**
  - Top container ports: Tanjung Priok (avg 11.66 container calls/day, ~47% national share), Surabaya (7.40), Gresik (4.72), Makassar (2.31), Belawan (1.32).
  - 47/75 ports have container activity; 28 are bulk/oil-only (excluded from container modeling).
  - Weekly CV (2022+): Priok 0.09, Surabaya 0.13, Gresik 0.17, Makassar 0.19, Belawan 0.32 — clean signal for forecasting.
  - No missing days 2024+, 0% zero-call days for Priok 2024+.

**Modeling universe:** top 10 container ports by average daily container calls (fallback: top 8 if tail CV too high — decision made in notebook 02, documented there).

**Granularity:** daily data, aggregated to **weekly** for modeling (smoother signal, standard practice for port planning), daily retained for anomaly layer where useful.

## 3. Analysis Layers

### L1 — EDA & Benchmarking (notebook 01)
- National container flow 2019–2026: calls, import/export tonnage, COVID trough & recovery timeline.
- Port ranking & market share (container calls, container tonnage).
- **Secondary comparison (amendment):** total portcalls (all vessel types) vs container calls per port — shows container share of activity and contrasts container hubs vs bulk ports.
- Hub-feeder structure: activity distribution, concentration (e.g., HHI), Priok dominance.
- Seasonality: weekly cycle, Ramadan/Lebaran effect (Indonesian calendar), monsoon months.

### L2 — Forecasting (notebook 03, the head of the project)
- **Targets:** weekly `portcalls_container` (primary) and weekly `import_container`/`export_container` tonnage (secondary) per top port.
- **Feature set:** lags (1–8 weeks), rolling mean/std (4, 8, 12), calendar features (week-of-year, month, Ramadan/Lebaran window from `holidays` ID + manual windows), port size class.
- **Model ladder (honest, walked in order):**
  1. Naive seasonal (baseline)
  2. SARIMA (statsmodels)
  3. XGBoost with engineered features
  4. LSTM (PyTorch) — global model across ports
- **Validation:** walk-forward (expanding window), no leakage; metrics MAE, RMSE, MAPE per port; skill score vs naive baseline.
- **Reporting:** per-port model ranking table + forecast fan chart for the flagship port (Priok).

### L3 — Anomaly / Early Warning (notebook 04)
- Residuals from the best per-port forecast → z-score & Isolation Forest on daily/weekly residuals.
- Backtest narrative: which known events do we detect? (COVID lockdown 2020, export moratoria 2022–2023 waves, Ramadan shifts, weather disruptions). Documented with dates + magnitude.
- Output: list of detected anomaly episodes (port, date, direction, severity) — the "early warning log".

### L4 — Executive Summary (report)
- 1-page operator-language summary: per-port sensitivity (what drives each port), lead time of signals, what a terminal planner should do with the forecast.

## 4. Deliverables

1. **GitHub repo** (this repo, `D:/portofolio`): modular `src/` + notebooks + README (English primary).
2. **Streamlit dashboard** (`app/`): port selector, actual vs forecast chart, anomaly timeline, benchmark view. Self-contained, runs from repo.
3. **Case-study report** (`reports/case_study.md` + figures): CV/LinkedIn-ready writeup.

## 5. Repo Structure

```
portofolio/                      (repo root, git initialized)
├── data/
│   ├── raw/portwatch_indonesia_daily.parquet   # exists
│   └── processed/                              # engineered weekly frames
├── notebooks/
│   ├── 01_eda_benchmark.ipynb
│   ├── 02_data_prep_features.ipynb
│   ├── 03_forecasting.ipynb
│   └── 04_anomaly_early_warning.ipynb
├── src/
│   ├── data.py        # load, clean, aggregate daily->weekly
│   ├── features.py    # lags, rolling, calendar (Ramadan/Lebaran)
│   ├── models.py      # naive, SARIMA, XGBoost, LSTM train/predict
│   ├── anomaly.py     # residual scoring, isolation forest
│   └── evaluate.py    # walk-forward, metrics
├── app/
│   └── dashboard.py   # streamlit entry
├── reports/
│   ├── case_study.md
│   └── figures/
├── tests/
│   └── test_pipeline.py  # minimal smoke: load->features->forecast shape invariants
├── requirements.txt
└── README.md
```

## 6. Tech Stack

Python 3.13 (system). pandas, numpy, matplotlib, statsmodels, xgboost, scikit-learn, torch (LSTM), holidays, streamlit, jupyter. No new exotic dependencies.

## 7. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Container tonnage units are AIS-derived estimates | Frame as "estimated shipment volume" in all outputs; check scale in EDA before over-claiming |
| Small ports (Belawan CV 0.32) may forecast poorly | Restrict modeling to top ports; report honest per-port MAPE; keep Belawan only if it beats naive |
| LSTM may not beat XGBoost on weekly aggregates | That's a finding, not a failure — report the ladder honestly |
| Lebaran/Ramadan calendar windows wrong | Cross-check 2019–2026 dates manually (public table), hardcode with source comment |
| COVID period distorts models | Primary training from 2022+ (post-pandemic); include full history only in EDA narrative; robustness check with 2020–2021 included vs excluded |

## 8. Out of Scope (deliberate)

- AIS trajectory-level modeling (NOAA 43GB, no labels — verified dead end on this machine)
- Per-vessel ETA prediction (Kaggle candidates verified unusable: 1-day snapshot, capped target, 50 labeled visits)
- Full web dashboard (React) — web skills already shown in Amourea; this repo stays data-science focused
- Real-time ingestion pipeline — weekly manual refresh documented in README is enough for a portfolio project

## 9. Success Criteria

- [ ] EDA notebook runs top-to-bottom; all figures reproducible from `src/` functions
- [ ] Forecasting: XGBoost (or best) beats naive seasonal baseline on MAPE for ≥8 of top 10 ports
- [ ] Walk-forward validation implemented with zero leakage (asserted in test)
- [ ] Anomaly layer detects ≥2 documented historical events with dates
- [ ] Streamlit dashboard runs locally via `streamlit run app/dashboard.py`
- [ ] README explains data source, method, results, and limitations honestly
- [ ] Case study report is 1-page and copy-pasteable to CV
