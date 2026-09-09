"""Load, clean, and aggregate PortWatch Indonesia daily data to weekly granularity."""
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "portwatch_indonesia_daily.parquet"
PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
CONTAINER_COLS = ["portcalls_container", "import_container", "export_container"]


def load_daily(path: str | Path = RAW) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"].str.slice(0, 10))
    df = df.drop(columns=[c for c in ("year", "month", "day") if c in df.columns])
    return df.sort_values(["portname", "date"]).reset_index(drop=True)


def to_weekly(daily: pd.DataFrame, value_cols=CONTAINER_COLS) -> pd.DataFrame:
    """Aggregate daily -> ISO week. Drops partial weeks (<7 days) to keep full weeks only."""
    if not pd.api.types.is_datetime64_any_dtype(daily["date"]):
        daily = daily.assign(date=pd.to_datetime(daily["date"].astype(str).str.slice(0, 10)))
    iso = daily["date"].dt.isocalendar()
    df = daily.assign(iso_year=iso["year"].astype(int), iso_week=iso["week"].astype(int))
    df["year_week"] = df["iso_year"] * 100 + df["iso_week"]
    grouped = (
        df.groupby(["portname", "year_week"], as_index=False)
        .agg({**{c: "sum" for c in value_cols}, "date": ["count", "min"]})
    )
    grouped.columns = ["portname", "year_week", *value_cols, "n_days", "week_start"]
    grouped = grouped[grouped["n_days"] == 7].drop(columns="n_days")
    return grouped.sort_values(["portname", "year_week"]).reset_index(drop=True)


def top_ports(weekly: pd.DataFrame, n: int = 10, metric: str = "portcalls_container") -> list[str]:
    totals = weekly.groupby("portname")[metric].sum().sort_values(ascending=False)
    return totals.head(n).index.tolist()


def build_weekly_top(path: str | Path = RAW, n_ports: int = 10, save: bool = True) -> pd.DataFrame:
    daily = load_daily(path)
    weekly = to_weekly(daily)
    weekly = weekly[weekly["portname"].isin(top_ports(weekly, n_ports))].reset_index(drop=True)
    if save:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        weekly.to_parquet(PROCESSED / "weekly_top10.parquet", index=False)
    return weekly


def wide_by_port(weekly: pd.DataFrame, metric: str = "portcalls_container") -> pd.DataFrame:
    """Pivot to year_week-indexed wide frame: one column per port (used by SARIMA per-port)."""
    return weekly.pivot(index="year_week", columns="portname", values=metric).sort_index()


if __name__ == "__main__":
    w = build_weekly_top()
    print(f"weekly top10: {w.shape}, ports: {w.portname.nunique()}")
    print(f"weeks: {w.year_week.min()} .. {w.year_week.max()}")
