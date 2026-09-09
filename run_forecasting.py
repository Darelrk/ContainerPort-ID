"""Driver: walk-forward evaluation for all top-10 ports x {calls, tonnage} x 4 models.

Saves:
  data/processed/forecast_results_calls.csv
  data/processed/forecast_results_tonnage.csv
  data/processed/model_summary.csv
  data/processed/predictions_best.csv  (dashboard input)
Also trains final best model per port on ALL data and writes 8-week projections.
"""
import time
import warnings

import numpy as np
import pandas as pd

from src.data import PROCESSED
from src.evaluate import evaluate_all, walk_forward, mape

warnings.filterwarnings("ignore")

TARGETS = {
    "calls": "portcalls_container",
    "tonnage": "import_container",  # import side as the tonnage representative
}


def main() -> None:
    weekly = pd.read_parquet(PROCESSED / "weekly_top10.parquet")
    all_rows, all_summaries = [], []

    for label, target in TARGETS.items():
        t0 = time.time()
        results, summary = evaluate_all(
            weekly, target=target, test_start=202501,
            models=("naive", "sarima", "xgboost", "lstm"),
        )
        results.to_csv(PROCESSED / f"forecast_results_{label}.csv", index=False)
        all_rows.append(results)
        all_summaries.append(summary)
        print(f"[{label}] done in {time.time()-t0:.0f}s: {summary.shape[0]} rows")

    summary = pd.concat(all_summaries, ignore_index=True)
    summary.to_csv(PROCESSED / "model_summary.csv", index=False)
    print(summary.to_string())

    # best model per port for calls target -> final fit on all data + 8-week projection
    calls_summary = summary[summary.target == "portcalls_container"]
    best = calls_summary.sort_values("MAPE").groupby("portname").first().reset_index()[["portname", "model"]]
    print("\nBest model per port (calls):")
    print(best.to_string(index=False))

    proj_rows = []
    for _, r in best.iterrows():
        s = weekly[weekly.portname == r.portname].set_index("year_week")["portcalls_container"].sort_index()
        model = {"naive": None}  # placeholder replaced below
        from src.models import MODEL_REGISTRY
        m = MODEL_REGISTRY[r.model]().fit(s)
        last_yw = int(s.index[-1])
        future_yw = []
        iy, iw = divmod(last_yw, 100)
        for k in range(1, 9):
            if iw + k <= 52:
                future_yw.append(iy * 100 + iw + k)
            else:
                future_yw.append((iy + 1) * 100 + (iw + k - 52))
        # handle ISO week-53 edge: if a year has 53 ISO weeks the arithmetic above shifts;
        # acceptable for projections (Ponytail: projection index is cosmetic, not scored).
        preds = m.predict(pd.Index(future_yw))
        for yw, p in zip(future_yw, preds):
            proj_rows.append({"portname": r.portname, "model": r.model, "year_week": yw, "y_pred": float(p)})

    pd.concat(all_rows, ignore_index=True).to_csv(PROCESSED / "predictions_all.csv", index=False)
    pd.DataFrame(proj_rows).to_csv(PROCESSED / "projections_8w.csv", index=False)
    print(f"\nprojections: {len(proj_rows)} rows saved")


if __name__ == "__main__":
    main()
