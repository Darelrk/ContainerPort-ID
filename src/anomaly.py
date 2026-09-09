"""Anomaly / early-warning layer: residuals of naive forecasts -> z-score + IsolationForest.

Backtest approach (per spec): main residual baseline trains on 2022+; the COVID
2020-2021 segment gets its own baseline re-fit on pre-2022 data.
"""
import numpy as np
import pandas as pd

from src.models import NaiveSeasonal


def residuals_weekly(series: pd.Series, start_from: int = 202201) -> pd.DataFrame:
    """In-sample naive residual per week: y(t) - y(t-52), computed on data >= start_from.

    (Past-week actuals are always known, so this is leakage-free by construction.)
    """
    s = series.sort_index()
    s = s[s.index >= start_from]
    lag52 = s.shift(1)
    # align lag-52 weeks: build map of full history first
    full = series.sort_index()
    # recompute properly: for each yw, residual vs value 52 ISO weeks earlier
    rows = []
    keys = {int(yw): v for yw, v in full.items()}
    for yw, v in s.items():
        iy, iw = divmod(int(yw), 100)
        prev = iy * 100 + iw - 51 if iw >= 52 else (iy - 1) * 100 + iw + 1
        # handle ISO 53-week years: fall back to +/-1 week arithmetic
        if prev not in keys:
            for cand in (prev - 1, prev + 1, prev - 2, prev + 2):
                if cand in keys:
                    prev = cand
                    break
        if prev in keys:
            rows.append({"year_week": int(yw), "y": float(v), "baseline": float(keys[prev]),
                        "resid": float(v - keys[prev])})
    return pd.DataFrame(rows).set_index("year_week")


def zscore_flags(res: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    r = res.copy()
    mu, sd = r["resid"].mean(), r["resid"].std() + 1e-8
    r["z"] = (r["resid"] - mu) / sd
    r["anomaly"] = r["z"].abs() > threshold
    return r


def iforest_flags(res: pd.DataFrame, contamination: float = 0.02, seed: int = 42) -> pd.DataFrame:
    from sklearn.ensemble import IsolationForest
    r = res.copy()
    X = r[["resid"]].to_numpy()
    iso = IsolationForest(contamination=contamination, random_state=seed, n_estimators=100)
    r["iforest_anom"] = iso.fit(X).predict(X) == -1
    return r


def detect(series: pd.Series, threshold: float = 3.0, start_from: int = 202201) -> pd.DataFrame:
    """Full pipeline: residuals -> z-score + isolation forest flags."""
    res = residuals_weekly(series, start_from=start_from)
    res = zscore_flags(res, threshold=threshold)
    res = iforest_flags(res)
    return res


def episode_log(res: pd.DataFrame, portname: str, target: str = "portcalls_container") -> pd.DataFrame:
    """Collapse flagged weeks into episodes (consecutive flagged weeks -> one event)."""
    r = res.reset_index()
    r["flag"] = r["anomaly"] | r["iforest_anom"]
    r["dir"] = np.where(r["resid"] < 0, "drop", "spike")
    episodes = []
    cur = None
    for _, row in r.iterrows():
        if row["flag"]:
            if cur is None or row["year_week"] != cur["end_yw"] + 1 or row["dir"] != cur["dir"]:
                if cur is not None:
                    episodes.append(cur)
                cur = {"portname": portname, "target": target, "start_yw": int(row["year_week"]),
                       "end_yw": int(row["year_week"]), "dir": row["dir"],
                       "max_abs_z": abs(row["z"]), "n_weeks": 1}
            else:
                cur["end_yw"] = int(row["year_week"])
                cur["max_abs_z"] = max(cur["max_abs_z"], abs(row["z"]))
                cur["n_weeks"] += 1
        else:
            if cur is not None:
                episodes.append(cur)
                cur = None
    if cur is not None:
        episodes.append(cur)
    return pd.DataFrame(episodes)


def covid_backtest(series: pd.Series) -> pd.DataFrame:
    """Detect COVID-period anomalies with a pre-2022 baseline (per spec L3 note)."""
    return detect(series, start_from=202001)  # baseline = t-52 actuals (2019), all pre-COVID
