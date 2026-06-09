from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .regimes import model_quality_row, proba_frame

try:  # optional dependency for Project 19
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


def hmm_proba_frame(model, x, index: Sequence[pd.Timestamp] | pd.Index | None = None, prefix: str = "state") -> pd.DataFrame:
    p = model.predict_proba(x)
    if index is None:
        index = getattr(x, "index", pd.RangeIndex(len(p)))
    return proba_frame(p, index=index, prefix=prefix)


def _gaussian_hmm_n_params(model, n_features: int) -> int:
    n_states = int(getattr(model, "n_components", 1))
    cov_type = str(getattr(model, "covariance_type", "diag"))
    start_params = n_states - 1
    transition_params = n_states * (n_states - 1)
    mean_params = n_states * n_features
    if cov_type == "full":
        cov_params = int(n_states * n_features * (n_features + 1) / 2)
    elif cov_type == "tied":
        cov_params = int(n_features * (n_features + 1) / 2)
    elif cov_type == "spherical":
        cov_params = n_states
    else:
        cov_params = n_states * n_features
    return start_params + transition_params + mean_params + cov_params


def hmm_quality_row(
    name: str,
    model,
    x,
    *,
    outcomes: pd.DataFrame | pd.Series | None = None,
    labels: pd.Series | Sequence[int] | None = None,
    proba: pd.DataFrame | np.ndarray | None = None,
    n_params: int | None = None,
) -> dict[str, float | str]:
    X = np.asarray(x, dtype=float)
    n = len(X)
    if labels is None:
        labels = model.predict(X)
    if proba is None:
        proba = model.predict_proba(X)
    loglike = float(model.score(X))
    if n_params is None:
        n_params = _gaussian_hmm_n_params(model, X.shape[1] if X.ndim == 2 else 1)
    aic = 2.0 * n_params - 2.0 * loglike
    bic = np.log(max(n, 1)) * n_params - 2.0 * loglike
    return model_quality_row(
        name,
        x,
        labels,
        proba=proba,
        outcomes=outcomes,
        loglike=loglike,
        aic=aic,
        bic=bic,
    )


def pca_hmm_inputs(
    x: pd.DataFrame | np.ndarray,
    n_components: int = 5,
    *,
    random_state: int = 42,
    scaler: StandardScaler | None = None,
    pca: PCA | None = None,
) -> tuple[pd.DataFrame | np.ndarray, StandardScaler, PCA]:
    X = x.replace([np.inf, -np.inf], np.nan).dropna() if isinstance(x, pd.DataFrame) else np.asarray(x, dtype=float)
    n_comp = int(max(1, min(int(n_components), min(np.asarray(X).shape))))
    scaler = StandardScaler() if scaler is None else scaler
    pca = PCA(n_components=n_comp, random_state=int(random_state)) if pca is None else pca
    if not hasattr(scaler, "mean_"):
        z = scaler.fit_transform(X)
    else:
        z = scaler.transform(X)
    if not hasattr(pca, "components_"):
        arr = pca.fit_transform(z)
    else:
        arr = pca.transform(z)
    if isinstance(X, pd.DataFrame):
        cols = [f"PC{i + 1}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, index=X.index, columns=cols), scaler, pca
    return arr, scaler, pca


def align_state_probabilities(
    proba: pd.DataFrame,
    order: Sequence[int],
    prefix: str = "state",
) -> pd.DataFrame:
    cols = list(proba.columns)
    ordered_cols = [cols[int(i)] for i in order if int(i) < len(cols)]
    out = proba.reindex(columns=ordered_cols).copy()
    out.columns = [f"{prefix}_{i}" for i in range(out.shape[1])]
    row_sum = out.sum(axis=1).replace(0.0, np.nan)
    return out.div(row_sum, axis=0).fillna(0.0)


def walkforward_hmm_probabilities(
    model_factory,
    x: pd.DataFrame,
    rebalance_dates: Sequence[pd.Timestamp],
    *,
    train_days: int = 1260,
    pca_components: int | None = None,
) -> pd.DataFrame:
    rows = {}
    for dt in pd.to_datetime(rebalance_dates):
        hist = x.loc[:dt].tail(int(train_days))
        if len(hist) < max(250, int(train_days) // 2):
            continue
        if pca_components is None:
            scaler = StandardScaler().fit(hist)
            z_train = scaler.transform(hist)
            z_last = scaler.transform(hist.iloc[[-1]])
        else:
            z_train, scaler, pca = pca_hmm_inputs(hist, n_components=int(pca_components))
            z_last, _, _ = pca_hmm_inputs(hist.iloc[[-1]], n_components=int(pca_components), scaler=scaler, pca=pca)
        model = model_factory()
        model.fit(z_train)
        rows[pd.Timestamp(dt)] = model.predict_proba(z_last)[0]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame.from_dict(rows, orient="index").sort_index().rename(columns=lambda i: f"state_{i}")


def torch_available() -> bool:
    return torch is not None and nn is not None


def auto_device():
    if not torch_available():  # pragma: no cover
        raise ImportError("PyTorch is required for sequence models.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _require_torch():
    if not torch_available():  # pragma: no cover
        raise ImportError("PyTorch is required for Project 19 neural models.")


def _ordered_quantiles(out):
    q50 = out[:, 1]
    q10 = q50 - torch.nn.functional.softplus(out[:, 0])
    q90 = q50 + torch.nn.functional.softplus(out[:, 2])
    return torch.stack([q10, q50, q90], dim=1)


if torch_available():

    class MlpForecast(nn.Module):
        def __init__(
            self,
            *,
            n_features: int,
            n_assets: int,
            embedding_dim: int = 4,
            hidden_sizes: Sequence[int] = (64, 32),
            output_size: int = 1,
            dropout: float = 0.10,
            ordered_quantiles: bool = False,
        ):
            super().__init__()
            self.embedding = nn.Embedding(int(n_assets), int(embedding_dim))
            layers: list[nn.Module] = []
            in_dim = int(n_features) + int(embedding_dim)
            for hidden in hidden_sizes:
                layers.extend(
                    [
                        nn.Linear(in_dim, int(hidden)),
                        nn.LayerNorm(int(hidden)),
                        nn.SiLU(),
                        nn.Dropout(float(dropout)),
                    ]
                )
                in_dim = int(hidden)
            layers.append(nn.Linear(in_dim, int(output_size)))
            self.net = nn.Sequential(*layers)
            self.ordered_quantiles = bool(ordered_quantiles and int(output_size) == 3)

        def forward(self, x, asset_id):
            emb = self.embedding(asset_id.long())
            out = self.net(torch.cat([x.float(), emb], dim=1))
            return _ordered_quantiles(out) if self.ordered_quantiles else out


    class LstmForecast(nn.Module):
        def __init__(
            self,
            *,
            n_features: int,
            n_assets: int,
            embedding_dim: int = 4,
            hidden_size: int = 48,
            num_layers: int = 1,
            output_size: int = 1,
            dropout: float = 0.10,
            ordered_quantiles: bool = False,
        ):
            super().__init__()
            self.embedding = nn.Embedding(int(n_assets), int(embedding_dim))
            self.lstm = nn.LSTM(
                int(n_features) + int(embedding_dim),
                int(hidden_size),
                num_layers=int(num_layers),
                batch_first=True,
                dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(int(hidden_size)),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_size), int(output_size)),
            )
            self.ordered_quantiles = bool(ordered_quantiles and int(output_size) == 3)

        def forward(self, x, asset_id):
            emb = self.embedding(asset_id.long()).unsqueeze(1).expand(-1, x.shape[1], -1)
            seq = torch.cat([x.float(), emb], dim=2)
            out, _ = self.lstm(seq)
            pred = self.head(out[:, -1, :])
            return _ordered_quantiles(pred) if self.ordered_quantiles else pred


    class _Chomp1d(nn.Module):
        def __init__(self, chomp_size: int):
            super().__init__()
            self.chomp_size = int(chomp_size)

        def forward(self, x):
            return x[:, :, : -self.chomp_size].contiguous() if self.chomp_size > 0 else x


    class _TemporalBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
            super().__init__()
            padding = (int(kernel_size) - 1) * int(dilation)
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
                _Chomp1d(padding),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
                _Chomp1d(padding),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
            )
            self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

        def forward(self, x):
            out = self.net(x)
            res = x if self.downsample is None else self.downsample(x)
            return out + res


    class TcnForecast(nn.Module):
        def __init__(
            self,
            *,
            n_features: int,
            n_assets: int,
            embedding_dim: int = 4,
            channels: Sequence[int] = (32, 32, 32),
            kernel_size: int = 3,
            output_size: int = 1,
            dropout: float = 0.10,
            ordered_quantiles: bool = False,
        ):
            super().__init__()
            self.embedding = nn.Embedding(int(n_assets), int(embedding_dim))
            layers: list[nn.Module] = []
            in_ch = int(n_features) + int(embedding_dim)
            for i, ch in enumerate(channels):
                layers.append(
                    _TemporalBlock(
                        in_ch,
                        int(ch),
                        kernel_size=int(kernel_size),
                        dilation=2**i,
                        dropout=float(dropout),
                    )
                )
                in_ch = int(ch)
            self.tcn = nn.Sequential(*layers)
            self.head = nn.Sequential(
                nn.LayerNorm(in_ch),
                nn.Dropout(float(dropout)),
                nn.Linear(in_ch, int(output_size)),
            )
            self.ordered_quantiles = bool(ordered_quantiles and int(output_size) == 3)

        def forward(self, x, asset_id):
            emb = self.embedding(asset_id.long()).unsqueeze(1).expand(-1, x.shape[1], -1)
            seq = torch.cat([x.float(), emb], dim=2).transpose(1, 2)
            out = self.tcn(seq)[:, :, -1]
            pred = self.head(out)
            return _ordered_quantiles(pred) if self.ordered_quantiles else pred

else:  # pragma: no cover

    class MlpForecast:  # type: ignore[no-redef]
        def __init__(self, *_, **__):
            _require_torch()

    class LstmForecast:  # type: ignore[no-redef]
        def __init__(self, *_, **__):
            _require_torch()

    class TcnForecast:  # type: ignore[no-redef]
        def __init__(self, *_, **__):
            _require_torch()


def pinball_loss_torch(pred, target, quantiles: Sequence[float]):
    q = torch.as_tensor(list(quantiles), dtype=pred.dtype, device=pred.device)
    err = target.view(-1, 1) - pred
    return torch.maximum(q * err, (q - 1.0) * err).mean()


def gaussian_nll_loss_torch(pred, target):
    mean = pred[:, 0]
    log_var = pred[:, 1].clamp(-8.0, 6.0)
    var = torch.exp(log_var)
    return 0.5 * (log_var + torch.square(target - mean) / var).mean()


def build_sequence_arrays(
    data: pd.DataFrame,
    *,
    features: Sequence[str],
    target: str | None = None,
    asset_col: str = "asset_id",
    date_col: str = "date",
    lookback: int = 21,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, pd.Index]:
    """Build per-asset rolling sequence arrays aligned to original row index."""
    df = pd.DataFrame(data).copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([asset_col, date_col])
    feature_cols = list(features)
    x_list: list[np.ndarray] = []
    a_list: list[int] = []
    y_list: list[float] = []
    idx_list: list[object] = []
    for _, group in df.groupby(asset_col, sort=False):
        g = group.sort_values(date_col)
        arr = (
            g[feature_cols]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        y_arr = None
        if target is not None:
            y_arr = pd.to_numeric(g[target], errors="coerce").to_numpy(dtype=np.float32)
        asset_vals = pd.to_numeric(g[asset_col], errors="coerce").astype(int).to_numpy()
        for i in range(int(lookback) - 1, len(g)):
            if y_arr is not None and not np.isfinite(y_arr[i]):
                continue
            window = arr[i - int(lookback) + 1 : i + 1]
            if not np.isfinite(window).all():
                continue
            x_list.append(window.astype(np.float32))
            a_list.append(int(asset_vals[i]))
            if y_arr is not None:
                y_list.append(float(y_arr[i]))
            idx_list.append(g.index[i])
    X = np.asarray(x_list, dtype=np.float32)
    A = np.asarray(a_list, dtype=np.int64)
    Y = np.asarray(y_list, dtype=np.float32) if target is not None else None
    return X, A, Y, pd.Index(idx_list)


def _loss_for(name: str, pred, target, quantiles: Sequence[float] | None):
    key = str(name).lower()
    if key in {"mse", "l2"}:
        return torch.nn.functional.mse_loss(pred.view(-1), target)
    if key in {"huber", "smooth_l1"}:
        return torch.nn.functional.smooth_l1_loss(pred.view(-1), target)
    if key in {"pinball", "quantile"}:
        return pinball_loss_torch(pred, target, quantiles or (0.10, 0.50, 0.90))
    if key in {"gaussian", "nll", "gaussian_nll"}:
        return gaussian_nll_loss_torch(pred, target)
    return torch.nn.functional.mse_loss(pred.view(-1), target)


def _point_prediction_array(pred: np.ndarray) -> np.ndarray:
    arr = np.asarray(pred, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.shape[1] >= 3:
        return arr[:, 1]
    return arr[:, 0]


def _spearman_array(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 3 or x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return float("nan")
    return float(x.corr(y, method="spearman"))


def _forecast_validation_score(
    pred: np.ndarray,
    target: np.ndarray,
    dates: Sequence[pd.Timestamp | str] | np.ndarray,
    asset_id: Sequence[int] | np.ndarray,
    *,
    top_frac: float = 0.25,
    turnover_penalty: float = 0.02,
) -> dict[str, float]:
    p = _point_prediction_array(pred)
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "asset": np.asarray(asset_id).astype(str),
            "prediction": p,
            "target": np.asarray(target, dtype=float),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return {
            "validation_score": np.nan,
            "rank_ic": np.nan,
            "bucket_spread": np.nan,
            "top_k_hit_rate": np.nan,
            "turnover": np.nan,
        }

    rank_ic = []
    spread = []
    hit = []
    turnover = []
    prev_top: set[str] | None = None
    for _, group in frame.groupby("date", sort=True):
        if len(group) < 4:
            continue
        rank_ic.append(_spearman_array(group["prediction"], group["target"]))
        n = max(1, int(np.ceil(len(group) * float(top_frac))))
        ordered = group.sort_values("prediction")
        low = ordered.head(n)["target"].mean()
        high = ordered.tail(n)["target"].mean()
        spread.append(float(high - low))
        pred_top = set(ordered.tail(n)["asset"])
        actual_top = set(group.sort_values("target").tail(n)["asset"])
        hit.append(float(len(pred_top & actual_top) / max(1, n)))
        if prev_top is not None:
            union = len(pred_top | prev_top)
            turnover.append(float(1.0 - len(pred_top & prev_top) / union) if union else 0.0)
        prev_top = pred_top

    rank_ic_mean = float(pd.Series(rank_ic).mean()) if rank_ic else np.nan
    spread_mean = float(pd.Series(spread).mean()) if spread else np.nan
    hit_mean = float(pd.Series(hit).mean()) if hit else np.nan
    turnover_mean = float(pd.Series(turnover).mean()) if turnover else 0.0
    score = (
        0.50 * (rank_ic_mean if np.isfinite(rank_ic_mean) else 0.0)
        + 0.35 * (spread_mean if np.isfinite(spread_mean) else 0.0)
        + 0.15 * ((hit_mean - float(top_frac)) if np.isfinite(hit_mean) else 0.0)
        - float(turnover_penalty) * (turnover_mean if np.isfinite(turnover_mean) else 0.0)
    )
    return {
        "validation_score": float(score),
        "rank_ic": rank_ic_mean,
        "bucket_spread": spread_mean,
        "top_k_hit_rate": hit_mean,
        "turnover": turnover_mean,
    }


def train_torch_model(
    model,
    *,
    x: pd.DataFrame | np.ndarray | None = None,
    asset_id: pd.Series | np.ndarray | None = None,
    y: pd.Series | np.ndarray | None = None,
    data: pd.DataFrame | None = None,
    features: Sequence[str] | None = None,
    target: str = "z_21",
    asset_col: str = "asset_id",
    date_col: str = "date",
    lookback: int = 21,
    valid_fraction: float = 0.20,
    dates: Sequence[pd.Timestamp | str] | pd.Series | np.ndarray | None = None,
    train_mask: Sequence[bool] | np.ndarray | pd.Series | None = None,
    valid_mask: Sequence[bool] | np.ndarray | pd.Series | None = None,
    train_end: pd.Timestamp | str | None = None,
    valid_start: pd.Timestamp | str | None = None,
    valid_end: pd.Timestamp | str | None = None,
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 12,
    loss_name: str = "huber",
    quantiles: Sequence[float] | None = None,
    early_stop_metric: str = "composite",
    top_frac: float = 0.25,
    turnover_penalty: float = 0.02,
    device=None,
):
    _require_torch()
    device = auto_device() if device is None else device
    model = model.to(device)
    if data is not None:
        if features is None:
            raise ValueError("features is required when training from data.")
        X, A, Y, seq_index = build_sequence_arrays(
            data,
            features=features,
            target=target,
            asset_col=asset_col,
            date_col=date_col,
            lookback=lookback,
        )
        seq_dates = pd.to_datetime(pd.DataFrame(data).loc[seq_index, date_col]).to_numpy()
    else:
        if x is None or asset_id is None or y is None:
            raise ValueError("x, asset_id, and y are required for tabular training.")
        X = np.asarray(x, dtype=np.float32)
        A = np.asarray(asset_id, dtype=np.int64).reshape(-1)
        Y = np.asarray(y, dtype=np.float32).reshape(-1)
        seq_dates = pd.to_datetime(dates).to_numpy() if dates is not None else None
    n = len(X)
    if n < 20:
        raise ValueError("Not enough observations to train the model.")

    if train_mask is not None or valid_mask is not None:
        tr = np.asarray(train_mask if train_mask is not None else np.zeros(n, dtype=bool), dtype=bool).reshape(-1)
        va = np.asarray(valid_mask if valid_mask is not None else np.zeros(n, dtype=bool), dtype=bool).reshape(-1)
        if len(tr) != n or len(va) != n:
            raise ValueError("train_mask and valid_mask must match the number of observations.")
        train_idx = np.where(tr & ~va)[0]
        valid_idx = np.where(va)[0]
    elif seq_dates is not None and (train_end is not None or valid_start is not None or valid_end is not None):
        d = pd.to_datetime(seq_dates)
        ve = pd.Timestamp(valid_end) if valid_end is not None else pd.Timestamp(train_end)
        vs = pd.Timestamp(valid_start) if valid_start is not None else None
        if vs is None:
            if train_end is None:
                raise ValueError("train_end is required when valid_start is not supplied.")
            cutoff = pd.Timestamp(train_end)
            split_point = cutoff - pd.tseries.offsets.BDay(max(21, int(round(n * float(valid_fraction) / max(1, len(np.unique(d)))))))
            train_idx = np.where(d <= split_point)[0]
            valid_idx = np.where((d > split_point) & (d <= cutoff))[0]
        else:
            train_idx = np.where(d < vs)[0]
            valid_idx = np.where((d >= vs) & (d <= ve))[0]
    else:
        split = max(1, min(n - 1, int(n * (1.0 - float(valid_fraction)))))
        train_idx = np.arange(0, split)
        valid_idx = np.arange(split, n)
    if len(train_idx) < 10 or len(valid_idx) < 1:
        split = max(1, min(n - 1, int(n * (1.0 - float(valid_fraction)))))
        train_idx = np.arange(0, split)
        valid_idx = np.arange(split, n)

    train_ds = TensorDataset(
        torch.as_tensor(X[train_idx], dtype=torch.float32),
        torch.as_tensor(A[train_idx], dtype=torch.long),
        torch.as_tensor(Y[train_idx], dtype=torch.float32),
    )
    valid_ds = TensorDataset(
        torch.as_tensor(X[valid_idx], dtype=torch.float32),
        torch.as_tensor(A[valid_idx], dtype=torch.long),
        torch.as_tensor(Y[valid_idx], dtype=torch.float32),
    )
    pin_memory = bool(torch.cuda.is_available() and str(device).startswith("cuda"))
    loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_ds, batch_size=int(batch_size), shuffle=False, pin_memory=pin_memory)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="max",
        factor=0.5,
        patience=max(5, int(patience) // 4),
        min_lr=1e-5,
    )
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_value = -np.inf
    bad = 0
    rows = []
    valid_dates_for_score = pd.to_datetime(seq_dates[valid_idx]) if seq_dates is not None else None
    valid_assets_for_score = A[valid_idx]
    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_losses = []
        for xb, ab, yb in loader:
            xb, ab, yb = xb.to(device), ab.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = _loss_for(loss_name, model(xb, ab), yb, quantiles)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        valid_losses = []
        valid_preds = []
        valid_targets = []
        with torch.no_grad():
            for xb, ab, yb in valid_loader:
                xb, ab, yb = xb.to(device), ab.to(device), yb.to(device)
                out = model(xb, ab)
                valid_losses.append(float(_loss_for(loss_name, out, yb, quantiles).detach().cpu()))
                valid_preds.append(out.detach().cpu().numpy())
                valid_targets.append(yb.detach().cpu().numpy())
        train_loss = float(np.mean(train_losses))
        valid_loss = float(np.mean(valid_losses)) if valid_losses else train_loss
        score_stats = {
            "validation_score": np.nan,
            "rank_ic": np.nan,
            "bucket_spread": np.nan,
            "top_k_hit_rate": np.nan,
            "turnover": np.nan,
        }
        if (
            str(early_stop_metric).lower() in {"composite", "rank", "selection"}
            and valid_dates_for_score is not None
            and valid_preds
        ):
            score_stats = _forecast_validation_score(
                np.vstack(valid_preds),
                np.concatenate(valid_targets),
                valid_dates_for_score,
                valid_assets_for_score,
                top_frac=top_frac,
                turnover_penalty=turnover_penalty,
            )
        early_value = (
            score_stats["validation_score"]
            if np.isfinite(score_stats["validation_score"])
            else -valid_loss
        )
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "validation_score": score_stats["validation_score"],
                "valid_rank_ic": score_stats["rank_ic"],
                "valid_bucket_spread": score_stats["bucket_spread"],
                "valid_top_k_hit_rate": score_stats["top_k_hit_rate"],
                "valid_turnover": score_stats["turnover"],
                "early_stop_value": early_value,
            }
        )
        scheduler.step(float(early_value))
        if early_value > best_value + 1e-5:
            best_value = float(early_value)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= int(patience):
            break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(rows)


def torch_predictions(
    model,
    *,
    x: pd.DataFrame | np.ndarray | None = None,
    asset_id: pd.Series | np.ndarray | None = None,
    data: pd.DataFrame | None = None,
    features: Sequence[str] | None = None,
    asset_col: str = "asset_id",
    date_col: str = "date",
    lookback: int = 21,
    batch_size: int = 512,
    device=None,
):
    _require_torch()
    device = auto_device() if device is None else device
    model = model.to(device)
    model.eval()
    if data is not None:
        if features is None:
            raise ValueError("features is required when predicting from data.")
        X, A, _, index = build_sequence_arrays(
            data,
            features=features,
            target=None,
            asset_col=asset_col,
            date_col=date_col,
            lookback=lookback,
        )
    else:
        if x is None or asset_id is None:
            raise ValueError("x and asset_id are required for tabular predictions.")
        X = np.asarray(x, dtype=np.float32)
        A = np.asarray(asset_id, dtype=np.int64).reshape(-1)
        index = getattr(x, "index", pd.RangeIndex(len(X)))
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(A, dtype=torch.long))
    pin_memory = bool(torch.cuda.is_available() and str(device).startswith("cuda"))
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False, pin_memory=pin_memory)
    preds = []
    with torch.no_grad():
        for xb, ab in loader:
            out = model(xb.to(device), ab.to(device)).detach().cpu().numpy()
            preds.append(out)
    arr = np.vstack(preds) if preds else np.empty((0, 1))
    if arr.shape[1] == 1:
        return pd.Series(arr[:, 0], index=index, name="prediction")
    cols = [f"prediction_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, index=index, columns=cols)


__all__ = [
    "align_state_probabilities",
    "auto_device",
    "build_sequence_arrays",
    "hmm_proba_frame",
    "hmm_quality_row",
    "gaussian_nll_loss_torch",
    "LstmForecast",
    "MlpForecast",
    "pca_hmm_inputs",
    "pinball_loss_torch",
    "torch_available",
    "torch_predictions",
    "TcnForecast",
    "train_torch_model",
    "walkforward_hmm_probabilities",
]
