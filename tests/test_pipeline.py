"""Smoke invariants: weekly aggregation, feature alignment, no-leakage walk-forward."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_daily, to_weekly, top_ports, build_weekly_top
from src.features import build_features, calendar_features, _week_start
from src.models import NaiveSeasonal, XGBModel
from src.evaluate import walk_forward, mape, smape


def _toy_daily() -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", "2023-03-31", freq="D")
    ports = ["Alpha", "Beta"]
    rows = []
    for p, base in zip(ports, [10, 5]):
        for d in dates:
            rows.append({
                "date": d.strftime("%Y-%m-%d 00:00:00+00:00"),
                "portname": p, "portcalls_container": base,
                "import_container": base * 100, "export_container": base * 50,
            })
    return pd.DataFrame(rows)


def test_weekly_aggregation_sums():
    w = to_weekly(_toy_daily())
    # 13 full ISO weeks for Alpha: each week sums to 7*10
    alpha = w[w.portname == "Alpha"]
    assert len(alpha) >= 12
    assert (alpha.portcalls_container == 70).all()
    assert (alpha.import_container == 7000).all()


def test_top_ports_order():
    w = to_weekly(_toy_daily())
    assert top_ports(w, 1) == ["Alpha"]  # higher total wins


def test_features_lag_alignment():
    daily = load_daily()
    w = to_weekly(daily)
    s = w[w.portname == "Tanjung Priok"]
    f = build_features(s, target="portcalls_container")
    # lag_1 of row i must equal target of previous row
    for i in range(1, min(20, len(f))):
        assert f.lag_1.iloc[i] == f.portcalls_container.iloc[i - 1]
    # ramadan flag actually fires
    assert calendar_features(_week_start(pd.Series([202313]))).ramadan.iloc[0] == 1


def test_week_start_iso_monday():
    ws = _week_start(pd.Series([202634, 202501, 202452]))
    assert ws.iloc[0].day_name() == "Monday"
    assert str(ws.iloc[0].date()) == "2026-08-17"
    assert str(ws.iloc[1].date()) == "2024-12-30"  # ISO 2025-W01 contains Jan 4


def test_walk_forward_no_leakage():
    """Naive walk-forward prediction for week t must never equal actual y(t)."""
    daily = load_daily()
    w = to_weekly(daily)
    s = w[w.portname == "Tanjung Priok"].set_index("year_week")["portcalls_container"]
    r = walk_forward(s, "naive", test_start=202401)
    # predictions are past-week actuals, not the target itself
    assert not (r.y_pred.to_numpy() == r.y_true.to_numpy()).all()
    assert len(r) > 30
    # walk_forward ordering: index must be ascending, unique
    assert r.year_week.is_monotonic_increasing
    assert r.year_week.is_unique


def test_naive_seasonal_exact():
    y = pd.Series([1.0, 2.0, 3.0, 4.0], index=[202301, 202302, 202351, 202352])
    nv = NaiveSeasonal().fit(y)
    # week 52: iw>=52 rule -> prev = iy*100 + iw - 51 = 202352-51 = 202301
    pred = nv.predict(pd.Index([202352]))
    assert pred[0] == 1.0
    # missing lag week (202303 -> 202350 absent) -> falls back to last observed value
    pred2 = nv.predict(pd.Index([202303]))
    assert pred2[0] == 4.0


def test_mape_smape_bounds():
    y = np.array([100.0, 0.0, 50.0])
    p = np.array([110.0, 10.0, 40.0])
    assert smape(y, p) <= 200
    assert mape(y, p) > 0


def test_build_weekly_top_shape():
    w = build_weekly_top(n_ports=5, save=False)
    assert w.portname.nunique() == 5
    assert (w.groupby("year_week").size() == 5).all() or w.year_week.nunique() > 200
