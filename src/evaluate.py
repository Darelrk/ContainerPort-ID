"""Walk-forward evaluation and metrics for per-port weekly forecasting."""
import numpy as np
import pandas as pd

from src.models import MODEL_REGISTRY


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8)) * 100)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


RETRAIN_EVERY = {"naive": 1, "sarima": 26, "xgboost": 4, "lstm": 4}


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE — bounded [0,200], robust to near-zero actuals."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(denom, 1e-8)) * 100)


def walk_forward(series: pd.Series, model_name: str, test_start: int = 202501,
                 retrain_every: int | None = None, start_from: int | None = None) -> pd.DataFrame:
    """Expanding-window walk-forward, 1-week horizon. Model refit every `retrain_every` weeks.

    Returns DataFrame(year_week, y_true, y_pred) over the test span. No leakage:
    each prediction uses only weeks strictly before the predicted week.
    """
    series = series.sort_index()
    if start_from is not None:
        series = series[series.index >= start_from]
    train = series[series.index < test_start]
    test = series[series.index >= test_start]
    if len(train) < 60 or len(test) == 0:
        raise ValueError(f"insufficient data: train={len(train)}, test={len(test)}")
    if retrain_every is None:
        retrain_every = RETRAIN_EVERY.get(model_name, 4)
    preds = []
    fitted = None
    for i, tw in enumerate(test.index):
        if fitted is None or i % retrain_every == 0:
            fitted = MODEL_REGISTRY[model_name]().fit(train)
        pred = fitted.predict(pd.Index([tw]))[0]
        preds.append(float(pred))
        # append ACTUAL to train after prediction (oracle expanding window)
        train = pd.concat([train, test.loc[[tw]]])
    return pd.DataFrame({
        "year_week": test.index.astype(int),
        "y_true": test.to_numpy(),
        "y_pred": preds,
    })


def skill_vs_naive(result: pd.DataFrame) -> float:
    """1 - MAPE_model/MAPE_naive. >0 means model beats seasonal naive."""
    y, p = result["y_true"].to_numpy(), result["y_pred"].to_numpy()
    from src.models import NaiveSeasonal
    # naive on same result frame: use y(t-52) from history — approximate with lag-52 of y_true where available
    hist = result["year_week"].to_numpy()
    naive_pred = []
    y_map = dict(zip(result["year_week"], result["y_true"]))
    for tw in hist:
        iy, iw = divmod(int(tw), 100)
        prev_yw = iy * 100 + iw - 51 if iw >= 52 else (iy - 1) * 100 + iw + 1
        naive_pred.append(y_map.get(prev_yw, np.nan))
    naive_pred = np.array(naive_pred, dtype=float)
    mask = ~np.isnan(naive_pred)
    if mask.sum() < 10:
        return float("nan")
    # NOTE: this naive uses actual test values from 52w earlier within the test span,
    # which is legitimate (past actuals are known at forecast time).
    m_naive = mape(y[mask], naive_pred[mask])
    m_model = mape(y, p)
    return 1 - m_model / m_naive if m_naive > 0 else float("nan")


def evaluate_all(weekly: pd.DataFrame, target: str = "portcalls_container",
                 test_start: int = 202501, models=("naive", "sarima", "xgboost", "lstm"),
                 start_from: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all models for every port. Returns (results_long, summary).

    results_long: portname, model, year_week, y_true, y_pred
    summary: portname, model, MAE, RMSE, MAPE, skill (vs naive)
    """
    rows, summaries = [], []
    for port in weekly["portname"].unique():
        s = weekly[weekly.portname == port].set_index("year_week")[target].sort_index()
        port_res = {}
        for m in models:
            try:
                r = walk_forward(s, m, test_start=test_start, start_from=start_from)
            except ValueError as e:
                print(f"skip {port}/{m}: {e}")
                continue
            r = r.assign(portname=port, model=m, target=target)
            rows.append(r)
            port_res[m] = r
            y, p = r["y_true"].to_numpy(), r["y_pred"].to_numpy()
            summaries.append({
                "portname": port, "model": m, "target": target,
                "MAE": mae(y, p), "RMSE": rmse(y, p), "MAPE": mape(y, p),
                "n_test": len(r),
            })
        if "naive" in port_res:
            yn, pn = port_res["naive"]["y_true"].to_numpy(), port_res["naive"]["y_pred"].to_numpy()
            m_naive = mape(yn, pn)
            for m, r in port_res.items():
                m_model = mape(r["y_true"].to_numpy(), r["y_pred"].to_numpy())
                for srow in summaries:
                    if srow["portname"] == port and srow["model"] == m and srow["target"] == target:
                        srow["skill"] = 1 - m_model / m_naive if m_naive > 0 else float("nan")
    return pd.concat(rows, ignore_index=True), pd.DataFrame(summaries)
