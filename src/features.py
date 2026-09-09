"""Feature engineering: lags, rolling stats, Indonesian calendar (Ramadan/Lebaran).

Ramadan/Lebaran windows below are from the Kemenag (Ministry of Religious Affairs)
calendar, cross-checked against the `holidays` package ID holidays for Lebaran dates.
"""
import numpy as np
import pandas as pd

# (start, end, label) inclusive; Ramadan month window + Lebaran week window
# Source: Kemenag hijri calendar 2019-2026
RAMADAN_WINDOWS = [
    ("2019-05-06", "2019-06-03"),
    ("2020-04-24", "2020-05-23"),
    ("2021-04-13", "2021-05-12"),
    ("2022-04-02", "2022-05-01"),
    ("2023-03-22", "2023-04-20"),
    ("2024-03-11", "2024-04-09"),
    ("2025-03-01", "2025-03-30"),
    ("2026-02-18", "2026-03-19"),
]
LEBARAN_DATES = [  # Eid al-Fitr day 1
    "2019-06-05", "2020-05-24", "2021-05-13", "2022-05-02",
    "2023-04-22", "2024-04-10", "2025-03-31", "2026-03-20",
]

LAGS = list(range(1, 9))            # 1-8 weeks
ROLLING_WINDOWS = [4, 8, 12]        # weeks


def _week_start(year_week: pd.Series) -> pd.Series:
    """Recover the true Monday of an ISO year_week.

    ISO week 1 of year Y always contains Jan 4, so Monday of week 1 is the
    Monday of the week containing Jan 4 (works across year boundaries).
    """
    iso_year = (year_week // 100).astype(int)
    iso_week = (year_week % 100).astype(int)
    jan4 = pd.to_datetime({"year": iso_year, "month": 1, "day": 4})
    week1_mon = jan4 - pd.to_timedelta(jan4.dt.weekday.to_numpy(), unit="D")
    return week1_mon + pd.to_timedelta((iso_week - 1) * 7, unit="D")


def calendar_features(week_start: pd.Series) -> pd.DataFrame:
    ws = pd.Series(pd.to_datetime(week_start)).reset_index(drop=True)
    feats = pd.DataFrame({
        "week_of_year": ws.dt.isocalendar().week.astype(int),
        "month": ws.dt.month,
        "woy_sin": np.sin(2 * np.pi * ws.dt.isocalendar().week / 52),
        "woy_cos": np.cos(2 * np.pi * ws.dt.isocalendar().week / 52),
    })
    ram = np.zeros(len(ws), dtype=int)
    for s, e in RAMADAN_WINDOWS:
        ram |= ((ws >= s) & (ws <= e)).to_numpy()
    feats["ramadan"] = ram
    # Lebaran: flag weeks within 7 days after Eid
    leb = np.zeros(len(ws), dtype=int)
    for d in LEBARAN_DATES:
        leb |= ((ws >= d) & (ws <= pd.Timestamp(d) + pd.Timedelta(days=7))).to_numpy()
    feats["lebaran"] = leb
    return feats


def build_features(weekly_port: pd.DataFrame, target: str = "portcalls_container") -> pd.DataFrame:
    """One-port weekly frame (columns: year_week, target[, others]) -> feature matrix.

    Output has: year_week, target, lag_1..lag_8, roll_mean_4/8/12, roll_std_4/8/12,
    + calendar features. Rows with any NaN lag (first 12 weeks) are dropped.
    """
    df = weekly_port.sort_values("year_week").reset_index(drop=True).copy()
    y = df[target]
    for lag in LAGS:
        df[f"lag_{lag}"] = y.shift(lag)
    for w in ROLLING_WINDOWS:
        df[f"roll_mean_{w}"] = y.shift(1).rolling(w).mean()
        df[f"roll_std_{w}"] = y.shift(1).rolling(w).std()
    cal = calendar_features(df["week_start"])
    for c in cal.columns:
        df[c] = cal[c].to_numpy()
    feature_cols = [f"lag_{l}" for l in LAGS] \
        + [f"roll_mean_{w}" for w in ROLLING_WINDOWS] + [f"roll_std_{w}" for w in ROLLING_WINDOWS] \
        + ["week_of_year", "month", "woy_sin", "woy_cos", "ramadan", "lebaran"]
    df = df.dropna(subset=[f"lag_{l}" for l in LAGS] + [f"roll_mean_{w}" for w in ROLLING_WINDOWS]).reset_index(drop=True)
    df.attrs["feature_cols"] = feature_cols
    return df
