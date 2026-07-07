from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantfinlab.ml.environment import (
    StateTables,
    portfolio_state_vector,
    portfolio_step_path_return,
    portfolio_step_return,
)
from quantfinlab.ml.rewards import DifferentialSharpe, DifferentialSortino, reward_components


@dataclass
class TrainingResult:
    """Container for policy training or evaluation outputs.

    Attributes
    ----------
    name : str
        Training or evaluation run name.
    history : pandas.DataFrame
        Epoch-level training history. Empty for pure evaluation results.
    validation : dict of str to float
        Validation metrics such as return, volatility, Sharpe, drawdown, reward, and
        exposure diagnostics.
    weights : pandas.DataFrame
        Policy weights over the validation or evaluation period.
    returns : pandas.Series
        Realized policy returns.
    components : list of dict
        Per-step reward component dictionaries.
    model_path : pathlib.Path, optional
        Checkpoint path used for saving or loading the policy.
    """
    name: str
    history: pd.DataFrame
    validation: dict[str, float]
    weights: pd.DataFrame
    returns: pd.Series
    components: list[dict[str, float]]
    model_path: Path | None = None


def _torch_device(device=None):
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device


def _period_rf(settings: Mapping[str, Any], n_days: int = 5) -> float:
    rf_daily = float(settings.get("rf_daily", 0.0))
    return float((1.0 + rf_daily) ** max(int(n_days), 1) - 1.0)


def _step_returns(
    state: StateTables,
    j: int,
    weights: np.ndarray,
    w_prev: np.ndarray,
    *,
    cost_bps: float,
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    if state.daily_windows_np:
        return portfolio_step_path_return(weights, state.daily_windows_np[int(j)], w_prev=w_prev, cost_bps=cost_bps)
    if state.daily_windows:
        return portfolio_step_path_return(weights, state.daily_windows[int(j)], w_prev=w_prev, cost_bps=cost_bps)
    ret, turnover, cost = portfolio_step_return(
        weights,
        state.returns_period.iloc[int(j)].to_numpy(dtype=float),
        w_prev=w_prev,
        cost_bps=cost_bps,
    )
    return ret, turnover, cost, np.asarray(weights, dtype=float), np.asarray([ret], dtype=float), np.asarray([1.0 + ret], dtype=float)


def _benchmark_step(
    state: StateTables,
    j: int,
    settings: Mapping[str, Any],
    bench_prev: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    name = str(settings.get("active_benchmark", state.active_benchmark or ""))
    frame = state.prior_weight_frames.get(name)
    if frame is None or frame.empty:
        n = len(state.daily_windows[int(j)]) if state.daily_windows else 1
        return 0.0, bench_prev, np.zeros(max(int(n), 1), dtype=float)
    W = frame.reindex(index=state.dates, columns=state.columns).ffill().fillna(0.0)
    w = W.iloc[int(j)].to_numpy(dtype=float)
    ret, _, _, w_end, daily_net, _ = _step_returns(
        state,
        int(j),
        w,
        bench_prev,
        cost_bps=float(settings.get("cost_bps", 10.0)),
    )
    return float(ret), w_end, np.asarray(daily_net, dtype=float)


def _fit_length(values: Sequence[float] | np.ndarray, n: int, *, fill_value: float = 0.0) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    n = max(int(n), 1)
    if arr.size == n:
        return arr
    if arr.size == 0:
        return np.full(n, float(fill_value), dtype=float)
    if arr.size > n:
        return arr[:n]
    return np.r_[arr, np.full(n - arr.size, float(arr[-1]), dtype=float)]


def _reference_daily_returns(state: StateTables, j: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    if state.daily_windows:
        R = pd.DataFrame(state.daily_windows[int(j)]).reindex(columns=state.columns).fillna(0.0)
        risky = R.reindex(columns=state.assets).fillna(0.0)
        equal_daily = risky.mean(axis=1).to_numpy(dtype=float)
        if "SPY" in R.columns:
            market_daily = R["SPY"].to_numpy(dtype=float)
        else:
            market_daily = equal_daily.copy()
    else:
        row = state.returns_period.iloc[int(j)].reindex(state.columns).fillna(0.0)
        risky_values = row.reindex(state.assets).to_numpy(dtype=float)
        equal_daily = np.asarray([float(np.mean(risky_values))], dtype=float)
        market_daily = np.asarray([float(row["SPY"] if "SPY" in row.index else equal_daily[0])], dtype=float)
    return _fit_length(equal_daily, n), _fit_length(market_daily, n)


def _active_path_penalties(
    *,
    daily_returns_hist: Sequence[float],
    active_daily_hist: Sequence[float],
    equal_daily_hist: Sequence[float],
    market_daily_hist: Sequence[float],
    daily_net: np.ndarray,
    bench_daily: np.ndarray,
    equal_daily: np.ndarray,
    market_daily: np.ndarray,
    settings: Mapping[str, Any],
) -> dict[str, float]:
    lookback = int(settings.get("reward_lookback_days", 63))
    floor = float(settings.get("tracking_vol_floor", 0.06))
    active_path = np.asarray([*active_daily_hist[-lookback:], *(daily_net - bench_daily)], dtype=float)
    if active_path.size > 20 and float(np.std(active_path, ddof=1)) > 0:
        tracking_vol = float(np.std(active_path, ddof=1) * np.sqrt(252.0))
    else:
        tracking_vol = floor

    port_path = np.asarray([*daily_returns_hist[-lookback:], *daily_net], dtype=float)
    market_path = np.asarray([*market_daily_hist[-lookback:], *market_daily], dtype=float)
    equal_path = np.asarray([*equal_daily_hist[-lookback:], *equal_daily], dtype=float)
    target_beta = float(settings.get("target_beta", 0.85))
    if port_path.size > 20 and market_path.size == port_path.size and float(np.var(market_path)) > 1e-12:
        beta = float(np.cov(port_path, market_path, ddof=1)[0, 1] / np.var(market_path, ddof=1))
    else:
        beta = target_beta
    if port_path.size > 20 and equal_path.size == port_path.size and float(np.std(port_path)) > 1e-12 and float(np.std(equal_path)) > 1e-12:
        corr = float(np.corrcoef(port_path, equal_path)[0, 1])
    else:
        corr = 0.0
    corr_cap = float(settings.get("corr_cap", 0.92))
    return {
        "tracking_vol": tracking_vol,
        "beta": beta,
        "corr_to_equal": corr,
        "beta_penalty": float((beta - target_beta) ** 2),
        "corr_penalty": float(max(0.0, corr - corr_cap) ** 2),
    }


def _inputs_for_step(
    state: StateTables,
    j: int,
    w_prev: np.ndarray,
    *,
    previous_turnover: float,
    nav: float,
    peak: float,
    returns_hist: Sequence[float],
    daily_returns_hist: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    asset_x = np.array(state.asset_state[j], copy=True)
    if "previous_weight" in state.asset_feature_names:
        asset_x[:, state.asset_feature_names.index("previous_weight")] = w_prev[: state.n_assets]
    if daily_returns_hist is not None and len(daily_returns_hist) > 20:
        rolling_vol = float(np.std(daily_returns_hist[-63:], ddof=1) * np.sqrt(252.0))
    else:
        rolling_vol = float(np.std(returns_hist[-13:], ddof=1) * np.sqrt(52.0)) if len(returns_hist) > 2 else 0.0
    portfolio_x = portfolio_state_vector(
        w_prev,
        previous_turnover=previous_turnover,
        current_drawdown=nav / peak - 1.0,
        rolling_portfolio_vol=rolling_vol,
        recent_portfolio_return=returns_hist[-1] if returns_hist else 0.0,
    )
    return asset_x.astype(np.float32), state.global_state[j].astype(np.float32), portfolio_x


def _score_returns(r: pd.Series, *, rf_daily: float = 0.0, annualization: float = 252.0, nav: pd.Series | None = None) -> dict[str, float]:
    x = pd.Series(r, dtype=float).dropna()
    if len(x) < 3:
        return {"return": np.nan, "vol": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}
    nav_s = pd.Series(nav, dtype=float).dropna() if nav is not None else (1.0 + x).cumprod()
    vol = float(x.std(ddof=1) * np.sqrt(float(annualization)))
    rf_period = float((1.0 + rf_daily) ** (252.0 / float(annualization)) - 1.0)
    sharpe = float((x - rf_period).mean() / x.std(ddof=1) * np.sqrt(float(annualization))) if x.std(ddof=1) > 0 else np.nan
    dd = nav_s / nav_s.cummax() - 1.0
    total = float(nav_s.iloc[-1] / nav_s.iloc[0] - 1.0) if len(nav_s) else float((1.0 + x).prod() - 1.0)
    return {"return": total, "vol": vol, "sharpe": sharpe, "max_drawdown": float(dd.min())}


def _returns_slice_for_indices(state: StateTables, indices: Sequence[int]) -> pd.DataFrame | None:
    if state.returns_daily is None or len(indices) == 0:
        return None
    idx = pd.DatetimeIndex(state.returns_daily.index)
    first_decision = pd.Timestamp(state.dates[int(indices[0])])
    last_j = int(indices[-1])
    start_pos = int(idx.searchsorted(first_decision, side="right"))
    if last_j + 1 < state.n_dates:
        end_pos = int(idx.searchsorted(pd.Timestamp(state.dates[last_j + 1]), side="right"))
    else:
        end_pos = min(int(idx.searchsorted(pd.Timestamp(state.dates[last_j]), side="right")) + 5, len(idx))
    if end_pos <= start_pos:
        return None
    return state.returns_daily.iloc[start_pos:end_pos].reindex(columns=state.columns).fillna(0.0)


def _collect_rollout(
    policy,
    state: StateTables,
    indices: Sequence[int],
    settings: Mapping[str, Any],
    *,
    device=None,
    deterministic: bool = False,
    recurrent: bool = False,
    recurrent_reset_every: int | None = None,
) -> dict[str, Any]:
    import torch

    dev = _torch_device(device)
    policy.to(dev)
    reward_mode = str(settings.get("reward_mode", "active_te")).lower().replace("-", "_")
    dsr_mode = reward_mode in {"dsr", "differential_sharpe", "sortino", "ddr"}
    if dsr_mode:
        eta = float(settings.get("dsr_eta", 0.01))
        if reward_mode in {"sortino", "ddr"}:
            dsr_state: Any = DifferentialSortino(eta=eta, rf_daily=float(settings.get("rf_daily", 0.0)))
        else:
            dsr_state = DifferentialSharpe(eta=eta)
    w_prev = np.zeros(state.n_assets + 1, dtype=float)
    bench_prev = np.zeros(state.n_assets + 1, dtype=float)
    previous_turnover = 0.0
    nav = 1.0
    peak = 1.0
    returns_hist: list[float] = []
    daily_returns_hist: list[float] = []
    active_daily_hist: list[float] = []
    equal_daily_hist: list[float] = []
    market_daily_hist: list[float] = []
    hidden = None
    rows: dict[str, list[Any]] = {
        "asset": [],
        "global": [],
        "portfolio": [],
        "action": [],
        "log_prob": [],
        "value": [],
        "entropy": [],
        "reward": [],
        "return": [],
        "daily_returns": [],
        "weights": [],
        "components": [],
        "date": [],
    }
    for pos, j_raw in enumerate(indices):
        j = int(j_raw)
        if recurrent and recurrent_reset_every is not None and int(recurrent_reset_every) > 0 and pos % int(recurrent_reset_every) == 0:
            hidden = None
        asset_x, global_x, portfolio_x = _inputs_for_step(
            state,
            j,
            w_prev,
            previous_turnover=previous_turnover,
            nav=nav,
            peak=peak,
            returns_hist=returns_hist,
            daily_returns_hist=daily_returns_hist,
        )
        ax = torch.as_tensor(asset_x[None], dtype=torch.float32, device=dev)
        gx = torch.as_tensor(global_x[None], dtype=torch.float32, device=dev)
        px = torch.as_tensor(portfolio_x[None], dtype=torch.float32, device=dev)
        with torch.no_grad():
            if recurrent:
                raw, log_prob, entropy, value, weights_t, hidden = policy.act(
                    ax,
                    gx,
                    px,
                    hidden=hidden,
                    deterministic=deterministic,
                )
                hidden = tuple(h.detach() for h in hidden)
            elif hasattr(policy, "sample_action"):
                raw, log_prob, entropy, weights_t = policy.sample_action(ax, gx, px, deterministic=deterministic)
                value = torch.zeros_like(log_prob)
            else:
                raw, log_prob, entropy, value, weights_t = policy.act(ax, gx, px, deterministic=deterministic)
        weights = weights_t.detach().cpu().numpy().reshape(-1)
        step_return, turnover, cost, w_end, daily_net, nav_path = _step_returns(
            state,
            j,
            weights,
            w_prev,
            cost_bps=float(settings.get("cost_bps", 10.0)),
        )
        daily_net = np.asarray(daily_net, dtype=float)
        realized_vol = (
            float(np.std([*daily_returns_hist[-63:], *daily_net], ddof=1) * np.sqrt(252.0))
            if len(daily_returns_hist) + len(daily_net) > 20
            else 0.0
        )
        nav_path_abs = nav * np.asarray(nav_path, dtype=float)
        nav_next = float(nav_path_abs[-1]) if len(nav_path_abs) else nav * (1.0 + step_return)
        peak_next = max(peak, float(np.nanmax(nav_path_abs)) if len(nav_path_abs) else nav_next)
        drawdown = nav_next / peak_next - 1.0
        if dsr_mode:
            equal_daily, _ = _reference_daily_returns(state, j, len(daily_net))
            dsr_value = 0.0
            for daily_r in daily_net:
                dsr_value = dsr_state.update(float(daily_r))
            decorr = 0.0
            hist_p = [*daily_returns_hist[-63:], *daily_net]
            hist_e = [*equal_daily_hist[-63:], *equal_daily]
            if len(hist_p) > 20 and len(hist_p) == len(hist_e):
                sp = float(np.std(hist_p))
                se = float(np.std(hist_e))
                if sp > 1e-9 and se > 1e-9:
                    corr = float(np.corrcoef(hist_p, hist_e)[0, 1])
                    decorr = max(0.0, corr - float(settings.get("decorr_cap", 0.88)))
            comp = reward_components(
                portfolio_return=step_return,
                turnover=turnover,
                cost=cost,
                weights=weights,
                realized_vol=realized_vol,
                drawdown=drawdown,
                reward_mode=reward_mode,
                dsr=float(dsr_value),
                dsr_scale=float(settings.get("dsr_scale", 4.0)),
                target_vol=float(settings.get("target_vol", 0.18)),
                drawdown_floor=float(settings.get("drawdown_floor", -0.18)),
                lambda_cost=float(settings.get("lambda_cost", 0.10)),
                turnover_budget=float(settings.get("turnover_budget", 0.12)),
                lambda_turnover_extra=float(settings.get("lambda_turnover_extra", 0.0)),
                lambda_vol=float(settings.get("lambda_vol", 0.25)),
                lambda_drawdown=float(settings.get("lambda_drawdown", 1.0)),
                lambda_conc=float(settings.get("lambda_conc", 0.10)),
                hhi_target=float(settings.get("hhi_target", 0.30)),
                cost_bps=float(settings.get("cost_bps", 10.0)),
                cash_weight=float(weights[-1]) if len(weights) else 0.0,
                cash_cap=float(settings.get("cash_cap", 0.15)),
                lambda_cash=float(settings.get("lambda_cash", 1.0)),
                decorr_penalty=float(decorr),
                lambda_decorr=float(settings.get("lambda_decorr", 0.5)),
            )
            equal_daily_hist.extend([float(x) for x in equal_daily])
        else:
            bench, bench_prev, bench_daily = _benchmark_step(state, j, settings, bench_prev)
            bench_daily = _fit_length(bench_daily, len(daily_net), fill_value=float(bench))
            equal_daily, market_daily = _reference_daily_returns(state, j, len(daily_net))
            active_stats = _active_path_penalties(
                daily_returns_hist=daily_returns_hist,
                active_daily_hist=active_daily_hist,
                equal_daily_hist=equal_daily_hist,
                market_daily_hist=market_daily_hist,
                daily_net=daily_net,
                bench_daily=bench_daily,
                equal_daily=equal_daily,
                market_daily=market_daily,
                settings=settings,
            )
            comp = reward_components(
                portfolio_return=step_return,
                benchmark_return=bench,
                turnover=turnover,
                cost=cost,
                weights=weights,
                rf_period=_period_rf(settings, n_days=max(len(daily_net), 1)),
                realized_vol=realized_vol,
                drawdown=drawdown,
                beta_active=float(settings.get("beta_active", 0.75)),
                vol_low=float(settings.get("vol_low", 0.09)),
                vol_high=float(settings.get("vol_high", 0.16)),
                lambda_cost=float(settings.get("lambda_cost", 1.25)),
                turnover_budget=float(settings.get("turnover_budget", 0.12)),
                lambda_turnover_extra=float(settings.get("lambda_turnover_extra", 10.0)),
                lambda_vol=float(settings.get("lambda_vol", 1.0)),
                lambda_drawdown=float(settings.get("lambda_drawdown", 1.0)),
                lambda_conc=float(settings.get("lambda_conc", 10.0)),
                hhi_target=float(settings.get("hhi_target", 0.18)),
                cost_bps=float(settings.get("cost_bps", 10.0)),
                reward_mode=reward_mode,
                tracking_vol=float(active_stats["tracking_vol"]),
                tracking_vol_floor=float(settings.get("tracking_vol_floor", 0.06)),
                active_reward_scale=float(settings.get("active_reward_scale", 100.0)),
                target_vol=float(settings.get("target_vol", settings.get("vol_high", 0.15))),
                drawdown_floor=float(settings.get("drawdown_floor", -0.10)),
                beta_penalty=float(active_stats["beta_penalty"]),
                corr_penalty=float(active_stats["corr_penalty"]),
                lambda_beta=float(settings.get("lambda_beta", 1.0)),
                lambda_corr=float(settings.get("lambda_corr", 1.0)),
            )
            comp["beta"] = float(active_stats["beta"])
            comp["corr_to_equal"] = float(active_stats["corr_to_equal"])
            active_daily_hist.extend([float(x) for x in daily_net - bench_daily])
            equal_daily_hist.extend([float(x) for x in equal_daily])
            market_daily_hist.extend([float(x) for x in market_daily])
        rows["asset"].append(asset_x)
        rows["global"].append(global_x)
        rows["portfolio"].append(portfolio_x)
        rows["action"].append(raw.detach().cpu().numpy().reshape(-1))
        rows["log_prob"].append(float(log_prob.detach().cpu().reshape(-1)[0]))
        rows["value"].append(float(value.detach().cpu().reshape(-1)[0]))
        rows["entropy"].append(float(entropy.detach().cpu().reshape(-1)[0]))
        rows["reward"].append(float(comp["reward"]))
        rows["return"].append(float(step_return))
        rows["daily_returns"].append(np.asarray(daily_net, dtype=float))
        rows["weights"].append(weights)
        rows["components"].append(comp)
        rows["date"].append(state.dates[j])
        returns_hist.append(float(step_return))
        daily_returns_hist.extend([float(x) for x in daily_net])
        nav, peak = nav_next, peak_next
        previous_turnover = turnover
        w_prev = w_end
    return rows


def _gae(rewards: np.ndarray, values: np.ndarray, *, gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    adv = np.zeros_like(rewards, dtype=np.float32)
    last = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + float(gamma) * next_value - values[t]
        last = delta + float(gamma) * float(gae_lambda) * last
        adv[t] = last
    ret = adv + values[: len(rewards)]
    return adv, ret.astype(np.float32)


def _sequence_ranges(n: int, seq_len: int) -> list[tuple[int, int]]:
    out = []
    step = max(1, int(seq_len))
    for start in range(0, int(n), step):
        end = min(start + step, int(n))
        if end - start >= 2:
            out.append((start, end))
    return out


def _sequence_gae(rewards: np.ndarray, values: np.ndarray, ranges: Sequence[tuple[int, int]], *, gamma: float, gae_lambda: float):
    adv = np.zeros_like(rewards, dtype=np.float32)
    ret = np.zeros_like(rewards, dtype=np.float32)
    for start, end in ranges:
        a, r = _gae(rewards[start:end], values[start:end], gamma=gamma, gae_lambda=gae_lambda)
        adv[start:end] = a
        ret[start:end] = r
    return adv, ret


def evaluate_policy(
    *,
    policy,
    state: StateTables,
    period: tuple[str | pd.Timestamp, str | pd.Timestamp] | None = None,
    reward_settings: Mapping[str, Any] | None = None,
    device=None,
) -> TrainingResult:
    """Evaluate a policy on a state period and return validation artifacts.

    The function rolls the policy through the requested state period, computes
    policy returns and reward components, and, when daily return windows are
    available, replays the resulting weights through the standard weight backtest
    engine for execution-consistent net returns.

    Parameters
    ----------
    policy : object
        Policy object exposing the expected action/evaluation interface.
    state : StateTables
        State object.
    period : tuple, optional
        Evaluation period as ``(start, end)``. If omitted, all state dates are used.
    reward_settings : mapping, optional
        Reward and cost settings.
    device : optional
        Torch device.

    Returns
    -------
    TrainingResult
        Evaluation result with validation metrics, weights, realized returns, and
        reward components.

    Notes
    -----
    If the portfolio backtest replay fails, the function falls back to the rollout
    period returns. When replay succeeds, reported performance uses daily net
    returns and an annualization factor appropriate for daily data.
    """
    from quantfinlab.backtest.portfolio import run_weights_backtest

    settings = dict(reward_settings or {})
    idx = state.period_indices(period)
    recurrent = policy.__class__.__name__.lower().startswith("recurrent")
    rows = _collect_rollout(policy, state, idx, settings, device=device, deterministic=True, recurrent=recurrent)
    period_returns = pd.Series(rows["return"], index=pd.DatetimeIndex(rows["date"]), name="policy_return")
    weights = pd.DataFrame(rows["weights"], index=pd.DatetimeIndex(rows["date"]), columns=state.columns)

    returns_out = period_returns
    stats = _score_returns(period_returns, rf_daily=float(settings.get("rf_daily", 0.0)), annualization=52.0)
    R_eval = _returns_slice_for_indices(state, idx)
    if R_eval is not None and not weights.empty:
        try:
            bt = run_weights_backtest(
                returns=R_eval,
                weights=weights,
                cost_bps=float(settings.get("cost_bps", 10.0)),
                rf_daily=float(settings.get("rf_daily", 0.0)),
                w_min=0.0,
                w_max=1.0,
                long_only=True,
                normalize=True,
                weight_timing="next_close",
            )
            returns_out = bt.net_returns
            stats = _score_returns(
                bt.net_returns,
                rf_daily=float(settings.get("rf_daily", 0.0)),
                annualization=252.0,
                nav=bt.net_values,
            )
        except Exception:
            pass
    stats["total_reward"] = float(np.sum(rows["reward"])) if rows["reward"] else np.nan
    stats["mean_reward"] = float(np.mean(rows["reward"])) if rows["reward"] else np.nan
    stats["avg_exposure"] = float(weights[state.assets].sum(axis=1).mean()) if not weights.empty else np.nan
    return TrainingResult(
        name="evaluation",
        history=pd.DataFrame(),
        validation=stats,
        weights=weights,
        returns=returns_out,
        components=list(rows["components"]),
    )


def _save_best(policy, path: Path | None, best_state: Mapping[str, Any] | None):
    if path is None or best_state is None:
        return
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, path)


def _load_if_available(policy, path: Path | None, device) -> bool:
    if path is None or not path.exists():
        return False
    import torch

    try:
        policy.load_state_dict(torch.load(path, map_location=device))
    except Exception:
        return False
    return True


def train_ppo(
    *,
    policy,
    state: StateTables,
    returns: pd.DataFrame | None = None,
    train_period: tuple[str | pd.Timestamp, str | pd.Timestamp],
    valid_period: tuple[str | pd.Timestamp, str | pd.Timestamp],
    reward_settings: Mapping[str, Any],
    epochs: int = 120,
    rollout_length: int = 96,
    minibatch_size: int = 128,
    ppo_epochs: int = 5,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_ratio: float = 0.20,
    lr: float = 3e-4,
    value_weight: float = 0.50,
    entropy_weight: float = 0.005,
    model_path: str | Path | None = None,
    device=None,
) -> TrainingResult:
    """Train or load a PPO portfolio policy.

    The function trains a policy with clipped Proximal Policy Optimization on
    rollouts sampled from the training period. It computes generalized advantage
    estimates, updates policy and value networks over minibatches, periodically
    evaluates on the validation period, keeps the best validation checkpoint, and
    returns the final validation artifacts.

    Parameters
    ----------
    policy : object
        PPO-compatible policy exposing ``evaluate_actions`` and rollout action
        methods.
    state : StateTables
        State tensors and return windows.
    returns : pandas.DataFrame, optional
        Reserved for compatibility with training workflows.
    train_period : tuple
        Training period.
    valid_period : tuple
        Validation period.
    reward_settings : mapping
        Reward, cost, and evaluation settings.
    epochs : int, default=120
        Number of training epochs.
    rollout_length : int, default=96
        Number of decision steps sampled per rollout.
    minibatch_size : int, default=128
        PPO minibatch size.
    ppo_epochs : int, default=5
        Number of optimization passes over each rollout.
    gamma : float, default=0.99
        Discount factor.
    gae_lambda : float, default=0.95
        GAE smoothing parameter.
    clip_ratio : float, default=0.20
        PPO policy-ratio clipping width.
    lr : float, default=3e-4
        AdamW learning rate.
    value_weight : float, default=0.50
        Value loss multiplier.
    entropy_weight : float, default=0.005
        Entropy bonus multiplier.
    model_path : str or pathlib.Path, optional
        Checkpoint path. If a compatible checkpoint exists, it is loaded and the
        policy is evaluated without retraining.
    device : optional
        Torch device.

    Returns
    -------
    TrainingResult
        Trained or loaded PPO result containing training history, validation
        metrics, validation weights, returns, reward components, and checkpoint path.

    Notes
    -----
    The routine samples contiguous training rollouts rather than independent rows,
    which preserves path-dependent state variables such as previous weights,
    turnover, drawdown, and recurrent hidden state when applicable.
    """
    import torch

    dev = _torch_device(device)
    path = Path(model_path) if model_path is not None else None
    policy.to(dev)
    if _load_if_available(policy, path, dev):
        ev = evaluate_policy(policy=policy, state=state, period=valid_period, reward_settings=reward_settings, device=dev)
        ev.name = "PPO"
        ev.model_path = path
        return ev

    opt = torch.optim.AdamW(policy.parameters(), lr=float(lr), weight_decay=1e-5)
    train_idx = state.period_indices(train_period)
    hist = []
    best_score = -np.inf
    best_state = None
    eval_every = max(1, min(10, int(epochs) // 20 if int(epochs) >= 20 else 1))
    for epoch in range(1, int(epochs) + 1):
        if len(train_idx) > int(rollout_length):
            start = int(np.random.randint(0, max(1, len(train_idx) // 5))) if np.random.rand() < 0.4 else int(np.random.randint(0, max(1, len(train_idx) - int(rollout_length))))
            use_idx = train_idx[start : start + int(rollout_length)]
        else:
            use_idx = train_idx
        roll = _collect_rollout(policy, state, use_idx, reward_settings, device=dev)
        rewards_np = np.asarray(roll["reward"], dtype=np.float32)
        values_np = np.asarray(roll["value"], dtype=np.float32)
        adv_np, ret_np = _gae(rewards_np, values_np, gamma=gamma, gae_lambda=gae_lambda)
        adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)

        asset_t = torch.as_tensor(np.asarray(roll["asset"]), dtype=torch.float32, device=dev)
        global_t = torch.as_tensor(np.asarray(roll["global"]), dtype=torch.float32, device=dev)
        port_t = torch.as_tensor(np.asarray(roll["portfolio"]), dtype=torch.float32, device=dev)
        action_t = torch.as_tensor(np.asarray(roll["action"]), dtype=torch.float32, device=dev)
        old_log_t = torch.as_tensor(np.asarray(roll["log_prob"]), dtype=torch.float32, device=dev)
        adv_t = torch.as_tensor(adv_np, dtype=torch.float32, device=dev)
        ret_t = torch.as_tensor(ret_np, dtype=torch.float32, device=dev)
        n = len(rewards_np)
        last_losses = {"policy_loss": np.nan, "value_loss": np.nan, "entropy": np.nan}
        for _ in range(int(ppo_epochs)):
            perm = torch.randperm(n, device=dev)
            for start in range(0, n, int(minibatch_size)):
                mb = perm[start : start + int(minibatch_size)]
                logp, entropy, value, _ = policy.evaluate_actions(asset_t[mb], global_t[mb], port_t[mb], action_t[mb])
                ratio = torch.exp(logp - old_log_t[mb])
                unclipped = ratio * adv_t[mb]
                clipped = torch.clamp(ratio, 1.0 - float(clip_ratio), 1.0 + float(clip_ratio)) * adv_t[mb]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = torch.nn.functional.mse_loss(value, ret_t[mb])
                entropy_loss = entropy.mean()
                loss = policy_loss + float(value_weight) * value_loss - float(entropy_weight) * entropy_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 2.0)
                opt.step()
                last_losses = {
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "value_loss": float(value_loss.detach().cpu()),
                    "entropy": float(entropy_loss.detach().cpu()),
                }
        valid_stats = {}
        if epoch % eval_every == 0 or epoch == int(epochs):
            ev = evaluate_policy(policy=policy, state=state, period=valid_period, reward_settings=reward_settings, device=dev)
            valid_stats = ev.validation
            score = float(valid_stats.get("sharpe", np.nan))
            if not np.isfinite(score):
                score = float(valid_stats.get("total_reward", -np.inf))
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
        hist.append(
            {
                "epoch": epoch,
                "train_reward": float(rewards_np.sum()),
                "mean_train_reward": float(rewards_np.mean()) if len(rewards_np) else np.nan,
                "avg_exposure": float(np.mean([np.sum(w[: state.n_assets]) for w in roll["weights"]])) if roll["weights"] else np.nan,
                **last_losses,
                "validation_reward": valid_stats.get("total_reward", np.nan),
                "validation_sharpe": valid_stats.get("sharpe", np.nan),
            }
        )
    if best_state is not None:
        policy.load_state_dict(best_state)
    _save_best(policy, path, best_state)
    ev = evaluate_policy(policy=policy, state=state, period=valid_period, reward_settings=reward_settings, device=dev)
    return TrainingResult("PPO", pd.DataFrame(hist), ev.validation, ev.weights, ev.returns, ev.components, path)


def train_recurrent_ppo(
    *,
    policy,
    state: StateTables,
    returns: pd.DataFrame | None = None,
    train_period: tuple[str | pd.Timestamp, str | pd.Timestamp],
    valid_period: tuple[str | pd.Timestamp, str | pd.Timestamp],
    reward_settings: Mapping[str, Any],
    epochs: int = 140,
    rollout_length: int = 104,
    minibatch_size: int = 128,
    ppo_epochs: int = 5,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_ratio: float = 0.20,
    lr: float = 2.5e-4,
    value_weight: float = 0.50,
    entropy_weight: float = 0.004,
    model_path: str | Path | None = None,
    device=None,
) -> TrainingResult:
    import torch

    dev = _torch_device(device)
    path = Path(model_path) if model_path is not None else None
    policy.to(dev)
    if _load_if_available(policy, path, dev):
        ev = evaluate_policy(policy=policy, state=state, period=valid_period, reward_settings=reward_settings, device=dev)
        ev.name = "Recurrent PPO"
        ev.model_path = path
        return ev

    opt = torch.optim.AdamW(policy.parameters(), lr=float(lr), weight_decay=1e-5)
    train_idx = state.period_indices(train_period)
    seq_len = int(getattr(policy, "sequence_length", 26))
    rollout_length = max(seq_len, int(rollout_length))
    hist = []
    best_score = -np.inf
    best_state = None
    eval_every = max(1, min(10, int(epochs) // 20 if int(epochs) >= 20 else 1))
    for epoch in range(1, int(epochs) + 1):
        if len(train_idx) > rollout_length:
            start = int(np.random.randint(0, max(1, len(train_idx) // 5))) if np.random.rand() < 0.4 else int(np.random.randint(0, max(1, len(train_idx) - rollout_length)))
            use_idx = train_idx[start : start + rollout_length]
        else:
            use_idx = train_idx
        roll = _collect_rollout(
            policy,
            state,
            use_idx,
            reward_settings,
            device=dev,
            recurrent=True,
            recurrent_reset_every=seq_len,
        )
        rewards_np = np.asarray(roll["reward"], dtype=np.float32)
        values_np = np.asarray(roll["value"], dtype=np.float32)
        ranges = _sequence_ranges(len(rewards_np), seq_len)
        adv_np, ret_np = _sequence_gae(rewards_np, values_np, ranges, gamma=gamma, gae_lambda=gae_lambda)
        used = np.zeros(len(rewards_np), dtype=bool)
        for start, end in ranges:
            used[start:end] = True
        adv_mean = adv_np[used].mean() if used.any() else adv_np.mean()
        adv_std = adv_np[used].std() if used.any() else adv_np.std()
        adv_np = (adv_np - adv_mean) / (adv_std + 1e-8)

        asset_np = np.asarray(roll["asset"])
        global_np = np.asarray(roll["global"])
        port_np = np.asarray(roll["portfolio"])
        action_np = np.asarray(roll["action"])
        old_log_np = np.asarray(roll["log_prob"], dtype=np.float32)
        seq_order = np.arange(len(ranges))
        seq_batch = max(1, int(minibatch_size) // max(1, seq_len))
        last_losses = {"policy_loss": np.nan, "value_loss": np.nan, "entropy": np.nan}
        for _ in range(int(ppo_epochs)):
            np.random.shuffle(seq_order)
            for b0 in range(0, len(seq_order), seq_batch):
                picked = [ranges[i] for i in seq_order[b0 : b0 + seq_batch]]
                B = len(picked)
                L = max(end - start for start, end in picked)
                ax_b = np.zeros((B, L, asset_np.shape[1], asset_np.shape[2]), dtype=np.float32)
                gx_b = np.zeros((B, L, global_np.shape[1]), dtype=np.float32)
                px_b = np.zeros((B, L, port_np.shape[1]), dtype=np.float32)
                ac_b = np.zeros((B, L, action_np.shape[1]), dtype=np.float32)
                old_b = np.zeros((B, L), dtype=np.float32)
                adv_b = np.zeros((B, L), dtype=np.float32)
                ret_b = np.zeros((B, L), dtype=np.float32)
                mask_b = np.zeros((B, L), dtype=np.float32)
                for bi, (start, end) in enumerate(picked):
                    length = end - start
                    ax_b[bi, :length] = asset_np[start:end]
                    gx_b[bi, :length] = global_np[start:end]
                    px_b[bi, :length] = port_np[start:end]
                    ac_b[bi, :length] = action_np[start:end]
                    old_b[bi, :length] = old_log_np[start:end]
                    adv_b[bi, :length] = adv_np[start:end]
                    ret_b[bi, :length] = ret_np[start:end]
                    mask_b[bi, :length] = 1.0
                ax_t = torch.as_tensor(ax_b, dtype=torch.float32, device=dev)
                gx_t = torch.as_tensor(gx_b, dtype=torch.float32, device=dev)
                px_t = torch.as_tensor(px_b, dtype=torch.float32, device=dev)
                ac_t = torch.as_tensor(ac_b, dtype=torch.float32, device=dev)
                old_t = torch.as_tensor(old_b, dtype=torch.float32, device=dev)
                adv_t = torch.as_tensor(adv_b, dtype=torch.float32, device=dev)
                ret_t = torch.as_tensor(ret_b, dtype=torch.float32, device=dev)
                mask_t = torch.as_tensor(mask_b, dtype=torch.float32, device=dev)
                denom = mask_t.sum().clamp_min(1.0)
                logp, entropy, value, _ = policy.evaluate_sequence_actions(ax_t, gx_t, px_t, ac_t)
                ratio = torch.exp(logp - old_t)
                unclipped = ratio * adv_t
                clipped = torch.clamp(ratio, 1.0 - float(clip_ratio), 1.0 + float(clip_ratio)) * adv_t
                policy_loss = -(torch.min(unclipped, clipped) * mask_t).sum() / denom
                value_loss = (((value - ret_t) ** 2) * mask_t).sum() / denom
                entropy_loss = (entropy * mask_t).sum() / denom
                loss = policy_loss + float(value_weight) * value_loss - float(entropy_weight) * entropy_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 2.0)
                opt.step()
                last_losses = {
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "value_loss": float(value_loss.detach().cpu()),
                    "entropy": float(entropy_loss.detach().cpu()),
                }
        valid_stats = {}
        if epoch % eval_every == 0 or epoch == int(epochs):
            ev = evaluate_policy(policy=policy, state=state, period=valid_period, reward_settings=reward_settings, device=dev)
            valid_stats = ev.validation
            score = float(valid_stats.get("sharpe", np.nan))
            if not np.isfinite(score):
                score = float(valid_stats.get("total_reward", -np.inf))
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
        hist.append(
            {
                "epoch": epoch,
                "train_reward": float(rewards_np.sum()),
                "mean_train_reward": float(rewards_np.mean()) if len(rewards_np) else np.nan,
                "avg_exposure": float(np.mean([np.sum(w[: state.n_assets]) for w in roll["weights"]])) if roll["weights"] else np.nan,
                **last_losses,
                "validation_reward": valid_stats.get("total_reward", np.nan),
                "validation_sharpe": valid_stats.get("sharpe", np.nan),
            }
        )
    if best_state is not None:
        policy.load_state_dict(best_state)
    _save_best(policy, path, best_state)
    ev = evaluate_policy(policy=policy, state=state, period=valid_period, reward_settings=reward_settings, device=dev)
    return TrainingResult("Recurrent PPO", pd.DataFrame(hist), ev.validation, ev.weights, ev.returns, ev.components, path)


def _soft_update(target, source, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters(), strict=False):
        tp.data.mul_(1.0 - float(tau)).add_(sp.data, alpha=float(tau))


def train_sac(
    *,
    policy,
    state: StateTables,
    returns: pd.DataFrame | None = None,
    train_period: tuple[str | pd.Timestamp, str | pd.Timestamp],
    valid_period: tuple[str | pd.Timestamp, str | pd.Timestamp],
    reward_settings: Mapping[str, Any],
    epochs: int = 140,
    batch_size: int = 128,
    replay_size: int = 100000,
    rollout_length: int = 156,
    gamma: float = 0.97,
    tau: float = 0.005,
    lr_actor: float = 3e-4,
    lr_critic: float = 3e-4,
    alpha: float = 0.05,
    updates_per_step: int = 2,
    policy_delay: int = 2,
    model_path: str | Path | None = None,
    device=None,
) -> TrainingResult:
    """Train or load a Soft Actor-Critic style portfolio policy.

    The function trains an off-policy actor-critic policy using a replay buffer,
    twin critics, entropy regularization, target critic updates, and delayed actor
    updates. It periodically evaluates on the validation period, keeps the best
    checkpoint, and returns validation artifacts.

    Parameters
    ----------
    policy : object
        SAC-compatible policy exposing actor, critic, sampling, and Q-value methods.
    state : StateTables
        State tensors and return windows.
    returns : pandas.DataFrame, optional
        Reserved for compatibility with training workflows.
    train_period : tuple
        Training period.
    valid_period : tuple
        Validation period.
    reward_settings : mapping
        Reward, entropy, and scaling settings.
    epochs : int, default=140
        Number of training epochs.
    batch_size : int, default=128
        Replay minibatch size.
    replay_size : int, default=100000
        Maximum replay-buffer length.
    rollout_length : int, default=156
        Number of decision steps collected per rollout.
    gamma : float, default=0.97
        Discount factor.
    tau : float, default=0.005
        Target critic soft-update rate.
    lr_actor : float, default=3e-4
        Actor learning rate.
    lr_critic : float, default=3e-4
        Critic learning rate.
    alpha : float, default=0.05
        Initial entropy temperature.
    updates_per_step : int, default=2
        Multiplier controlling critic/actor updates per epoch.
    policy_delay : int, default=2
        Actor/temperature update delay relative to critic updates.
    model_path : str or pathlib.Path, optional
        Checkpoint path. If present, the policy is loaded and evaluated.
    device : optional
        Torch device.

    Returns
    -------
    TrainingResult
        Trained or loaded SAC result containing history, validation metrics,
        validation weights, returns, components, and checkpoint path.

    Notes
    -----
    Rewards are optionally rescaled using ``sac_reward_scale`` from
    ``reward_settings``. The learned entropy temperature is optimized toward
    ``target_entropy`` when supplied, otherwise a dimension-based target is used.
    """
    import torch

    dev = _torch_device(device)
    path = Path(model_path) if model_path is not None else None
    policy.to(dev)
    if _load_if_available(policy, path, dev):
        ev = evaluate_policy(policy=policy, state=state, period=valid_period, reward_settings=reward_settings, device=dev)
        ev.name = "SAC"
        ev.model_path = path
        return ev

    target_critic = copy.deepcopy(policy.critic).to(dev)
    actor_params = (
        list(policy.actor_encoder.parameters())
        + list(policy.actor_score_head.parameters())
        + list(policy.actor_exposure_head.parameters())
        + list(policy.actor_log_std_head.parameters())
        + ([policy.alpha_gain] if getattr(policy, "alpha_gain", None) is not None else [])
    )
    critic_params = list(policy.critic.parameters())
    actor_opt = torch.optim.AdamW(actor_params, lr=float(lr_actor), weight_decay=1e-5)
    critic_opt = torch.optim.AdamW(critic_params, lr=float(lr_critic), weight_decay=1e-5)
    log_alpha = torch.tensor(np.log(max(float(alpha), 1e-6)), dtype=torch.float32, device=dev, requires_grad=True)
    alpha_opt = torch.optim.AdamW([log_alpha], lr=float(lr_actor))
    target_entropy = float(reward_settings.get("target_entropy", -0.75 * (int(policy.n_assets) + 1)))
    reward_scale = float(reward_settings.get("sac_reward_scale", 0.01))
    train_idx = state.period_indices(train_period)
    replay: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, float]] = []
    hist = []
    best_score = -np.inf
    best_state = None
    eval_every = max(1, min(10, int(epochs) // 20 if int(epochs) >= 20 else 1))
    update_count = 0

    for epoch in range(1, int(epochs) + 1):
        if len(train_idx) > int(rollout_length):
            start = int(np.random.randint(0, max(1, len(train_idx) // 5))) if np.random.rand() < 0.4 else int(np.random.randint(0, max(1, len(train_idx) - int(rollout_length))))
            use_idx = train_idx[start : start + int(rollout_length)]
        else:
            use_idx = train_idx
        roll = _collect_rollout(policy, state, use_idx, reward_settings, device=dev, deterministic=False)
        for k in range(len(roll["reward"])):
            done = 1.0 if k == len(roll["reward"]) - 1 else 0.0
            nk = min(k + 1, len(roll["reward"]) - 1)
            replay.append(
                (
                    roll["asset"][k],
                    roll["global"][k],
                    roll["portfolio"][k],
                    roll["action"][k],
                    float(roll["reward"][k]),
                    roll["asset"][nk],
                    roll["global"][nk],
                    roll["portfolio"][nk],
                    done,
                )
            )
        if len(replay) > int(replay_size):
            replay = replay[-int(replay_size) :]

        critic_losses = []
        actor_losses = []
        alpha_losses = []
        q_vals = []
        n_updates = max(1, int(updates_per_step)) * max(1, len(train_idx) // max(1, int(batch_size)))
        for _ in range(n_updates):
            if len(replay) < max(8, int(batch_size)):
                break
            batch_i = np.random.randint(0, len(replay), size=int(batch_size))
            batch = [replay[i] for i in batch_i]
            asset = torch.as_tensor(np.asarray([b[0] for b in batch]), dtype=torch.float32, device=dev)
            global_x = torch.as_tensor(np.asarray([b[1] for b in batch]), dtype=torch.float32, device=dev)
            port = torch.as_tensor(np.asarray([b[2] for b in batch]), dtype=torch.float32, device=dev)
            action = torch.as_tensor(np.asarray([b[3] for b in batch]), dtype=torch.float32, device=dev)
            reward = torch.as_tensor(np.asarray([b[4] for b in batch]), dtype=torch.float32, device=dev) * reward_scale
            next_asset = torch.as_tensor(np.asarray([b[5] for b in batch]), dtype=torch.float32, device=dev)
            next_global = torch.as_tensor(np.asarray([b[6] for b in batch]), dtype=torch.float32, device=dev)
            next_port = torch.as_tensor(np.asarray([b[7] for b in batch]), dtype=torch.float32, device=dev)
            done = torch.as_tensor(np.asarray([b[8] for b in batch]), dtype=torch.float32, device=dev)

            alpha_value = log_alpha.exp().detach()
            with torch.no_grad():
                next_action, next_logp, _, _ = policy.sample_action(next_asset, next_global, next_port)
                tq1, tq2 = target_critic(next_asset, next_global, next_port, next_action)
                target_q = torch.min(tq1, tq2) - alpha_value * next_logp
                y = reward + float(gamma) * (1.0 - done) * target_q
            q1, q2 = policy.q_values(asset, global_x, port, action)
            critic_loss = torch.nn.functional.mse_loss(q1, y) + torch.nn.functional.mse_loss(q2, y)
            critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic_params, 2.0)
            critic_opt.step()
            critic_losses.append(float(critic_loss.detach().cpu()))

            update_count += 1
            if update_count % max(1, int(policy_delay)) == 0:
                for p in policy.critic.parameters():
                    p.requires_grad_(False)
                pi_action, logp, entropy, _ = policy.sample_action(asset, global_x, port)
                q1_pi, q2_pi = policy.q_values(asset, global_x, port, pi_action)
                q_pi = torch.min(q1_pi, q2_pi)
                alpha_live = log_alpha.exp()
                actor_loss = (alpha_live.detach() * logp - q_pi).mean()
                actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_params, 2.0)
                actor_opt.step()
                alpha_loss = -(log_alpha * (logp.detach() + target_entropy)).mean()
                alpha_opt.zero_grad(set_to_none=True)
                alpha_loss.backward()
                alpha_opt.step()
                for p in policy.critic.parameters():
                    p.requires_grad_(True)
                _soft_update(target_critic, policy.critic, tau)
                actor_losses.append(float(actor_loss.detach().cpu()))
                alpha_losses.append(float(alpha_loss.detach().cpu()))
                q_vals.append(float(q_pi.mean().detach().cpu()))

        valid_stats = {}
        if epoch % eval_every == 0 or epoch == int(epochs):
            ev = evaluate_policy(policy=policy, state=state, period=valid_period, reward_settings=reward_settings, device=dev)
            valid_stats = ev.validation
            score = float(valid_stats.get("sharpe", np.nan))
            if not np.isfinite(score):
                score = float(valid_stats.get("total_reward", -np.inf))
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
        hist.append(
            {
                "epoch": epoch,
                "train_reward": float(np.sum(roll["reward"])),
                "mean_train_reward": float(np.mean(roll["reward"])) if roll["reward"] else np.nan,
                "critic_loss": float(np.mean(critic_losses)) if critic_losses else np.nan,
                "actor_loss": float(np.mean(actor_losses)) if actor_losses else np.nan,
                "alpha_loss": float(np.mean(alpha_losses)) if alpha_losses else np.nan,
                "alpha": float(log_alpha.exp().detach().cpu()),
                "q_value": float(np.mean(q_vals)) if q_vals else np.nan,
                "entropy": float(np.mean(roll["entropy"])) if roll["entropy"] else np.nan,
                "avg_exposure": float(np.mean([np.sum(w[: state.n_assets]) for w in roll["weights"]])) if roll["weights"] else np.nan,
                "validation_reward": valid_stats.get("total_reward", np.nan),
                "validation_sharpe": valid_stats.get("sharpe", np.nan),
            }
        )
    if best_state is not None:
        policy.load_state_dict(best_state)
    _save_best(policy, path, best_state)
    ev = evaluate_policy(policy=policy, state=state, period=valid_period, reward_settings=reward_settings, device=dev)
    return TrainingResult("SAC", pd.DataFrame(hist), ev.validation, ev.weights, ev.returns, ev.components, path)


def validation_policy_table(
    results: Mapping[str, TrainingResult],
    *,
    benchmark_name: str = "Forecast-Gated MaxSharpe",
) -> pd.DataFrame:
    """Collect validation metrics from several training results.

    Parameters
    ----------
    results : mapping of str to TrainingResult
        Policy training or evaluation results.
    benchmark_name : str, default="Forecast-Gated MaxSharpe"
        Reserved compatibility label for benchmark-aware reporting.

    Returns
    -------
    pandas.DataFrame
        Table indexed by policy name containing validation metrics and checkpoint
        path when available.
    """
    rows = []
    for name, result in results.items():
        row = dict(result.validation)
        row["Policy"] = str(name)
        row["checkpoint"] = str(result.model_path) if result.model_path is not None else ""
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.set_index("Policy") if "Policy" in out.columns else out


def policy_backtest(
    *,
    policy,
    state: StateTables,
    returns: pd.DataFrame,
    period: tuple[str | pd.Timestamp, str | pd.Timestamp],
    cost_bps: float = 10.0,
    device=None,
):
    """Backtest a policy by first converting it to a weight frame.

    Parameters
    ----------
    policy : object
        Trained policy.
    state : StateTables
        State object.
    returns : pandas.DataFrame
        Daily return panel used by the backtest.
    period : tuple
        Backtest period.
    cost_bps : float, default=10.0
        Transaction cost in basis points.
    device : optional
        Torch device.

    Returns
    -------
    BacktestResult
        Standard weight-backtest result using ``weight_timing="next_close"``.

    Notes
    -----
    This function bridges learned policies and the standard portfolio backtest
    engine, making RL results comparable to deterministic allocation strategies.
    """
    from quantfinlab.backtest.portfolio import run_weights_backtest
    from quantfinlab.ml.environment import policy_weight_frame

    weights = policy_weight_frame(policy=policy, state=state, period=period, device=device)
    return run_weights_backtest(
        returns=returns,
        weights=weights,
        cost_bps=cost_bps,
        weight_timing="next_close",
        normalize=True,
        long_only=True,
    )


def policy_checkpoints(model_dir: str | Path) -> pd.DataFrame:
    root = Path(model_dir)
    rows = []
    for path in sorted(root.glob("*.pt")) if root.exists() else []:
        rows.append({"checkpoint": path.name, "path": str(path), "bytes": path.stat().st_size})
    return pd.DataFrame(rows)


__all__ = [
    "TrainingResult",
    "evaluate_policy",
    "policy_backtest",
    "policy_checkpoints",
    "train_ppo",
    "train_recurrent_ppo",
    "train_sac",
    "validation_policy_table",
]
