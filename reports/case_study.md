# Case Study — ContainerPort-ID
### Weekly Container Forecasting & Early Warning for Indonesian Ports

**Darrell Rafif Kenzie** · Data Science portfolio · Sep 2026

---

**Problem.** Indonesian container terminals plan berths, cranes, and labour week-to-week, but operate without short-horizon visibility: how many container vessels will call next week, how much tonnage will flow, and when activity deviates abnormally. Public AIS-based data (IMF PortWatch) makes this answerable.

**Data.** 262k daily observations, 75 Indonesian ports, Jan 2019 – Aug 2026 (updated weekly). Top-10 container ports modeled — Tanjung Priok alone is 35.5% of national container calls; top-3 (Priok, Surabaya, Gresik) 72.3%, a classic hub-feeder structure.

**Method.** Daily → ISO-weekly aggregation; features = lags (1–8w), rolling stats (4/8/12w), Indonesian calendar (Ramadan window, Lebaran week). Model ladder evaluated identically under expanding-window walk-forward validation (86 test weeks, 2025+): seasonal naive → SARIMA → XGBoost → LSTM. Anomaly layer = z-score + Isolation Forest on y(t) − y(t−52) residuals, with a separately re-fit baseline for the COVID segment.

**Results.**
- A model beats the seasonal-naive baseline on **10/10 ports** (skill +7% to +54%).
- Headline accuracy: Priok 6.6% sMAPE, Surabaya 8.5%, Gresik 11.1% (weekly container calls).
- Early-warning layer detects documented events: the 2020 post-lockdown rebound, the 2021 Delta-wave and PPKM drops, the 2023 freight-recession lows, and recurring Lebaran shutdowns (e.g. 2024-W15, multi-port −40% calls).
- Deliverables: reproducible repo (tests assert zero leakage), Streamlit dashboard (forecast + anomaly + benchmark views), 8-week operational projections per port.

**Impact framing.** For a terminal operator: a 1-week-ahead call forecast at ~7% error directly feeds berth-allocation and stevedoring-roster decisions; the anomaly log turns the same data into a disruption radar (holiday effects separable from real shocks).

**Stack.** Python (pandas, statsmodels, XGBoost, PyTorch, scikit-learn), Streamlit, Plotly. Walk-forward evaluation, leakage-tested feature engineering, honest baselines.
