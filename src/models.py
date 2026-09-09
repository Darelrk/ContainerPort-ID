"""Forecast models with a common interface: fit(train) -> predict(n_steps) returns values aligned to test index.

All models operate on a single port's weekly series (pd.Series indexed by year_week).
"""
import numpy as np
import pandas as pd

SEED = 42

# ---------------------------------------------------------------- naive

class NaiveSeasonal:
    """ŷ(t) = y(t-52). Zero-shot 'model' — the honesty baseline."""

    def fit(self, y: pd.Series):
        self.y = y
        return self

    def predict(self, test_index: pd.Index) -> np.ndarray:
        out = []
        for tw in test_index:
            iy, iw = divmod(int(tw), 100)
            prev = self.y.get(iy * 100 + iw - 51) if iw >= 52 else self.y.get((iy - 1) * 100 + iw + 1)
            if prev is None or pd.isna(prev):
                prev = self.y.iloc[-1]
            out.append(prev)
        return np.array(out, dtype=float)


# ---------------------------------------------------------------- SARIMA

class SarimaModel:
    """Weekly SARIMA. Cheap-ish orders (1,1,1)(1,0,1,52): d=1 handles trend,
    seasonal D=0 avoids expensive m=52 differencing; maxiter capped for walk-forward."""

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 0, 1, 52), maxiter=60):
        self.order, self.seasonal_order, self.maxiter = order, seasonal_order, maxiter

    def fit(self, y: pd.Series):
        self.y = y
        self._res = None  # fit lazily at predict time
        return self

    def _series(self):
        from src.features import _week_start
        y = self.y.astype(float)
        dates = _week_start(pd.Series(sorted(y.index)))
        full_dates = pd.date_range(dates.iloc[0], dates.iloc[-1], freq="W-MON")
        ys = pd.Series(np.interp(
            full_dates.to_numpy().astype("datetime64[ns]").view("int64"),
            dates.to_numpy().astype("datetime64[ns]").view("int64"),
            y.reindex(sorted(y.index)).to_numpy(),
        ), index=full_dates)
        ys.index.freq = "W-MON"
        return ys

    def predict(self, test_index: pd.Index, **fit_kwargs) -> np.ndarray:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        if self._res is None:
            ys = self._series()
            model = SARIMAX(ys, order=self.order, seasonal_order=self.seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False)
            self._res = model.fit(disp=False, maxiter=self.maxiter)
        h = len(test_index)
        f = self._res.forecast(steps=h)
        # extend results cheaply: feed actuals back via append when available
        return np.asarray(f, dtype=float)


# ---------------------------------------------------------------- XGBoost

class XGBModel:
    """Feature-based model. Uses features.build_features; recursive multi-step via lag updates."""

    def __init__(self, target="portcalls_container", n_estimators=400, max_depth=4, lr=0.05):
        from xgboost import XGBRegressor
        self.target = target
        self.model = XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr,
            random_state=SEED, n_jobs=2, verbosity=0,
        )

    def fit(self, y: pd.Series):
        from src.features import build_features
        frame = y.reset_index()
        frame.columns = ["year_week", "y"]
        frame["week_start"] = _week_start_series(frame["year_week"])
        frame["portname"] = "port"
        feats = build_features(frame, target="y")
        self.feat_cols = feats.attrs["feature_cols"]
        self.history = feats[["year_week", "y"]].copy()
        self.X = feats[self.feat_cols].to_numpy()
        self.Y = feats["y"].to_numpy()
        self.model.fit(self.X, self.Y)
        # keep last row for recursive forecasting
        self.last = feats.iloc[-1]
        return self

    def predict(self, test_index: pd.Index) -> np.ndarray:
        from src.features import build_features
        hist = self.history.copy()
        preds = []
        for tw in test_index:
            # build a feature row from history (lags from actuals or our own preds)
            frame = hist.copy()
            frame["week_start"] = _week_start_series(frame["year_week"])
            frame["portname"] = "port"
            row_y = pd.DataFrame({"year_week": [tw], "y": [np.nan], "portname": ["port"]})
            row_y["week_start"] = _week_start_series(row_y["year_week"])
            full = pd.concat([frame, row_y], ignore_index=True)
            # rolling features need target values: use last known/recursive prediction
            for i, r in full.iterrows():
                pass
            feats = build_features(full, target="y")
            frow = feats[feats.year_week == tw]
            if len(frow) == 0 or frow[self.feat_cols].isna().any(axis=1).iloc[0]:
                preds.append(hist["y"].iloc[-1])
                hist.loc[len(hist)] = {"year_week": int(tw), "y": hist["y"].iloc[-1]}
                continue
            x = frow[self.feat_cols].to_numpy()
            p = float(self.model.predict(x)[0])
            preds.append(p)
            hist.loc[len(hist)] = {"year_week": int(tw), "y": p}
        return np.clip(np.array(preds, dtype=float), 0, None)


def _week_start_series(year_week: pd.Series) -> pd.Series:
    from src.features import _week_start
    return _week_start(year_week)


# ---------------------------------------------------------------- LSTM

class LSTMModel:
    """Global-ish per-port LSTM: seq of 8 weekly values -> next value. Deterministic seed."""

    def __init__(self, target="portcalls_container", seq_len=8, epochs=60, hidden=32):
        self.seq_len, self.epochs, self.hidden = seq_len, epochs, hidden

    def fit(self, y: pd.Series):
        import torch
        import torch.nn as nn
        torch.manual_seed(SEED)
        v = y.astype(float).to_numpy()
        # scale by train mean/std
        self.mu, self.sd = v.mean(), v.std() + 1e-8
        s = (v - self.mu) / self.sd
        X = np.lib.stride_tricks.sliding_window_view(s, self.seq_len)[:-1]
        Yt = s[self.seq_len:]
        Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (n, seq, 1)
        Yt_t = torch.tensor(Yt, dtype=torch.float32)
        self.net = nn.Sequential(
            nn.LSTM(input_size=1, hidden_size=self.hidden, batch_first=True)
        )
        class Net(nn.Module):
            def __init__(self, h):
                super().__init__()
                self.lstm = nn.LSTM(1, h, batch_first=True)
                self.head = nn.Linear(h, 1)
            def forward(self, x):
                o, _ = self.lstm(x)
                return self.head(o[:, -1, :])
        self.net = Net(self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        lossf = nn.MSELoss()
        for _ in range(self.epochs):
            opt.zero_grad()
            out = self.net(Xt).squeeze(-1)
            loss = lossf(out, Yt_t)
            loss.backward()
            opt.step()
        self.hist = s
        return self

    def predict(self, test_index: pd.Index) -> np.ndarray:
        import torch
        seq = list(self.hist[-self.seq_len:])
        preds = []
        with torch.no_grad():
            for _ in test_index:
                x = torch.tensor([seq[-self.seq_len:]], dtype=torch.float32).unsqueeze(-1)
                p = float(self.net(x).item())
                preds.append(p * self.sd + self.mu)
                seq.append(p)
        return np.clip(np.array(preds), 0, None)


MODEL_REGISTRY = {
    "naive": NaiveSeasonal,
    "sarima": SarimaModel,
    "xgboost": XGBModel,
    "lstm": LSTMModel,
}
