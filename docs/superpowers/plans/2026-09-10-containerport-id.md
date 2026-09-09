# ContainerPort-ID Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Forecast weekly container activity (calls + tonnage) for top Indonesian ports with walk-forward validation, plus an anomaly/early-warning layer and Streamlit dashboard — portfolio-ready by 08:00 WIB.

**Architecture:** Single Python repo. `src/` = importable modules (data → features → models → evaluate → anomaly). Notebooks call those functions (no logic duplication). Streamlit app reads saved artifacts (parquet/csv under `data/processed/` + `reports/figures/`). Weekly granularity for forecasting; daily retained for anomaly.

**Tech Stack:** Python 3.14.4 — pandas 3.0.3, statsmodels 0.15, xgboost 3.4.1, torch 2.12.1+cpu, scikit-learn 1.9, holidays 0.104, streamlit 1.63, pyarrow 24. (All verified installed.)

**Deadline constraint (user):** portfolio finished by 08:00 WIB Sep 10 2026. Started 02:29 WIB. Timebox per phase in todo. Prefer boring, working code over perfection. Notebooks get light prose (bilingual-lite: English headings, short English narrative).

---

## Data contract (verified against parquet)

Columns: `date (str), year, month, day, portid, portname, country, ISO3, portcalls_container, portcalls_dry_bulk, portcalls_general_cargo, portcalls_roro, portcalls_tanker, portcalls_cargo, portcalls, import_container, ..., export_container, ..., import, export`.

- Week aggregation: ISO week via `date.isocalendar()` — `year_week = iso_year*100 + iso_week`, weekly sum of daily counts/tonnage. Weeks with <7 daily rows flagged `n_days` (partial final week excluded from modeling).
- Top-10 ports by total container calls 2019–2026: Tanjung Priok, Surabaya, Gresik, Makassar, Belawan, Bitung, + next 4 determined at runtime (Panjang, Pontianak, Semarang, Teluk Bayur candidates).
- Train split: primary models 2022+ (post-COVID); robustness check includes 2020–21. Walk-forward: expanding window, refit each step, horizon 1 week, test starts 2025-01-01 (≈ 80+ test weeks).
- Naive seasonal baseline: ŷ(t) = y(t−52).

## Files & responsibility

| File | Responsibility |
|---|---|
| `src/data.py` | load parquet → clean → weekly aggregate → top-N filter; save `data/processed/weekly_top10.parquet` |
| `src/features.py` | build feature matrix: lags 1–8, rolling mean/std (4,8,12), calendar (week-of-year sin/cos, month, Ramadan/Lebaran window flags) |
| `src/models.py` | Naive, SARIMA, XGBoost, LSTM wrappers with common `fit_predict(train, test)` shape |
| `src/evaluate.py` | walk-forward loop, MAE/RMSE/MAPE, skill vs naive, results DataFrame |
| `src/anomaly.py` | residuals → z-score + IsolationForest, event backtest log |
| `app/dashboard.py` | Streamlit: port selector, actual vs forecast, anomaly timeline, benchmark |
| `tests/test_pipeline.py` | smoke invariants: weekly agg correctness, no-leakage (train max date < test date), feature shape |
| `notebooks/01–04` | EDA, prep, forecasting, anomaly — thin wrappers over `src/` |
| `reports/case_study.md` + figures | CV-ready summary |

## Tasks (checkboxes tracked in todo, executed inline tonight)

- [ ] Task 1: `src/data.py` + test (weekly agg sums correct, top-10 selection) → commit
- [ ] Task 2: `src/features.py` + test (lag alignment, no NaN in feature matrix at fit time, Ramadan flags present) → commit
- [ ] Task 3: `src/models.py` naive + SARIMA + test (naive == y(t−52)) → commit
- [ ] Task 4: `src/models.py` XGBoost + LSTM + test (predict shape, determinism seed) → commit
- [ ] Task 5: `src/evaluate.py` walk-forward + test (expanding window order, no leakage) → commit
- [ ] Task 6: `run_forecasting.py` driver: top-10 ports × 2 targets (calls, tonnage) → `data/processed/forecast_results.csv` + per-port best model → commit
- [ ] Task 7: `src/anomaly.py` + backtest (COVID 2020 via pre-2022 baseline; post-2022 events via main residuals) → `reports/anomaly_log.csv` → commit
- [ ] Task 8: notebooks 01–04 executed with outputs saved → commit
- [ ] Task 9: `app/dashboard.py` + smoke test (runs, renders port data) → commit
- [ ] Task 10: README (source, method, results, limitations) + `reports/case_study.md` → commit
- [ ] Task 11: final verify — pytest green, `streamlit run` boots, all success criteria from spec checked → final commit

## Ramadan/Lebaran windows (hardcode w/ source comment; cross-check holidays pkg `holidays.ID()` where possible)

2022: Apr 2–May 1 (R), Lebaran Apr 3+; 2023: Mar 22–Apr 20 (R), Lebaran Apr 22; 2024: Mar 11–Apr 9 (R), Lebaran Apr 10; 2025: Mar 1–Mar 29 (R), Lebaran Mar 31; 2026: Feb 18–Mar 19 (R), Lebaran Mar 20. (Source: Kemenag calendar, cross-checked `holidays`.)

## Success criteria (from spec §9 — verify at end)

1. XGBoost-or-best beats naive MAPE for ≥8/10 ports
2. Zero leakage asserted in test
3. Anomaly layer detects ≥2 documented events
4. Dashboard runs
5. README honest, case study 1-page
