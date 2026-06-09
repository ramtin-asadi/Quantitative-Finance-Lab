from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


class DifferentialSharpe:
    """Online Differential Sharpe Ratio (Moody & Saffell, 1998).

    Maintains exponentially weighted moving averages ``A`` (returns) and ``B``
    (squared returns) and returns the marginal contribution of each new return to
    the running Sharpe ratio. Optimising the cumulative DSR is equivalent to
    optimising risk-adjusted return directly, without tracking any benchmark.
    """

    def __init__(self, eta: float = 0.01):
        self.eta = float(eta)
        self.A = 0.0
        self.B = 0.0
        self.initialized = False

    def update(self, r: float) -> float:
        r = float(r)
        if not self.initialized:
            self.A = r
            self.B = r * r
            self.initialized = True
            return 0.0
        delta_a = r - self.A
        delta_b = r * r - self.B
        denom = max((self.B - self.A * self.A) ** 1.5, 1e-8)
        dsr = (self.B * delta_a - 0.5 * self.A * delta_b) / denom
        self.A += self.eta * delta_a
        self.B += self.eta * delta_b
        return float(np.clip(dsr, -10.0, 10.0))

    def reset(self) -> None:
        self.A = 0.0
        self.B = 0.0
        self.initialized = False


class DifferentialSortino:
    """Differential Downside Risk reward: a DSR variant penalising only downside.

    The variance term ``B`` accumulates squared *negative* excess returns, so the
    signal rewards upside while only charging for downside deviation.
    """

    def __init__(self, eta: float = 0.01, rf_daily: float = 0.0):
        self.eta = float(eta)
        self.rf = float(rf_daily)
        self.A = 0.0
        self.B = 0.0
        self.initialized = False

    def update(self, r: float) -> float:
        r = float(r)
        downside = min(r - self.rf, 0.0)
        if not self.initialized:
            self.A = r
            self.B = downside * downside
            self.initialized = True
            return 0.0
        delta_a = r - self.A
        delta_b = downside * downside - self.B
        denom = max(self.B ** 1.5, 1e-8)
        ddr = (self.B * delta_a - 0.5 * self.A * delta_b) / denom
        self.A += self.eta * delta_a
        self.B += self.eta * delta_b
        return float(np.clip(ddr, -10.0, 10.0))

    def reset(self) -> None:
        self.A = 0.0
        self.B = 0.0
        self.initialized = False


def vol_band_penalty(realized_vol: float, *, vol_low: float = 0.09, vol_high: float = 0.16) -> float:
    """Annualized volatility-band breach in bps-equivalent squared units."""
    vol = float(realized_vol)
    if not np.isfinite(vol):
        return 0.0
    gap = 0.0
    if vol < float(vol_low):
        gap = float(vol_low) - vol
    elif vol > float(vol_high):
        gap = vol - float(vol_high)
    return float(10000.0 * gap * gap)


def drawdown_penalty(drawdown: float, *, drawdown_floor: float = -0.10) -> float:
    """Drawdown breach in bps-equivalent squared units."""
    dd = float(drawdown)
    if not np.isfinite(dd):
        return 0.0
    gap = max(0.0, float(drawdown_floor) - dd)
    return float(10000.0 * gap * gap)


def turnover_penalty(turnover: float, *, turnover_budget: float = 0.12, lambda_turnover_extra: float = 10.0) -> float:
    excess = max(0.0, float(turnover) - float(turnover_budget))
    return float(lambda_turnover_extra) * excess


def concentration_penalty(weights: Sequence[float], *, hhi_target: float = 0.18) -> float:
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size == 0:
        return 0.0
    risky = np.clip(w[:-1] if w.size > 1 else w, 0.0, 1.0)
    hhi = float(np.square(risky).sum())
    return float(max(0.0, hhi - float(hhi_target)))


def active_reward(portfolio_return: float, benchmark_return: float, *, beta_active: float = 0.75) -> float:
    return float(beta_active) * 10000.0 * float(portfolio_return - benchmark_return)


def portfolio_reward(
    *,
    portfolio_return: float,
    benchmark_return: float = 0.0,
    turnover: float = 0.0,
    cost: float = 0.0,
    weights: Sequence[float] | None = None,
    rf_period: float = 0.0,
    realized_vol: float = 0.0,
    drawdown: float = 0.0,
    beta_active: float = 0.75,
    vol_low: float = 0.09,
    vol_high: float = 0.16,
    lambda_cost: float = 1.25,
    turnover_budget: float = 0.12,
    lambda_turnover_extra: float = 10.0,
    lambda_vol: float = 1.0,
    lambda_drawdown: float = 1.0,
    lambda_conc: float = 10.0,
    hhi_target: float = 0.18,
    cost_bps: float | None = None,
    **legacy_kwargs,
) -> float:
    comp = reward_components(
        portfolio_return=portfolio_return,
        benchmark_return=benchmark_return,
        turnover=turnover,
        cost=cost,
        weights=weights,
        rf_period=rf_period,
        realized_vol=realized_vol,
        drawdown=drawdown,
        beta_active=beta_active,
        vol_low=vol_low,
        vol_high=vol_high,
        lambda_cost=lambda_cost,
        turnover_budget=turnover_budget,
        lambda_turnover_extra=lambda_turnover_extra,
        lambda_vol=lambda_vol,
        lambda_drawdown=lambda_drawdown,
        lambda_conc=lambda_conc,
        hhi_target=hhi_target,
        cost_bps=cost_bps,
        **legacy_kwargs,
    )
    return float(comp["reward"])


def reward_components(
    *,
    portfolio_return: float,
    benchmark_return: float = 0.0,
    turnover: float = 0.0,
    cost: float = 0.0,
    weights: Sequence[float] | None = None,
    rf_period: float = 0.0,
    realized_vol: float = 0.0,
    drawdown: float = 0.0,
    beta_active: float = 0.75,
    vol_low: float = 0.09,
    vol_high: float = 0.16,
    lambda_cost: float = 1.25,
    turnover_budget: float = 0.12,
    lambda_turnover_extra: float = 10.0,
    lambda_vol: float = 1.0,
    lambda_drawdown: float = 1.0,
    lambda_conc: float = 10.0,
    hhi_target: float = 0.18,
    cost_bps: float | None = None,
    lambda_turnover: float | None = None,
    lambda_concentration: float | None = None,
    max_weight: float | None = None,
    reward_mode: str = "active_te",
    tracking_vol: float | None = None,
    tracking_vol_floor: float = 0.06,
    active_reward_scale: float = 100.0,
    target_vol: float = 0.15,
    drawdown_floor: float = -0.10,
    beta_penalty: float = 0.0,
    corr_penalty: float = 0.0,
    lambda_beta: float = 1.0,
    lambda_corr: float = 1.0,
    dsr: float = 0.0,
    dsr_scale: float = 100.0,
    cash_weight: float = 0.0,
    cash_cap: float = 0.30,
    lambda_cash: float = 1.0,
    decorr_penalty: float = 0.0,
    lambda_decorr: float = 0.0,
) -> dict[str, float]:
    """Basis-point reward with real costs and bounded extra risk penalties."""
    safe_return = max(float(portfolio_return), -0.999)
    safe_rf = max(float(rf_period), -0.999)
    log_excess_return = float(np.log1p(safe_return) - np.log1p(safe_rf))
    active_return = float(portfolio_return - benchmark_return)
    log_excess_bps = 10000.0 * log_excess_return
    active_bps = 10000.0 * active_return

    if cost_bps is not None:
        cost_bps_realized = float(cost_bps) * float(turnover)
    else:
        cost_bps_realized = 10000.0 * float(cost)

    vol_pen_bps = vol_band_penalty(realized_vol, vol_low=vol_low, vol_high=vol_high)
    dd_pen_bps = drawdown_penalty(drawdown, drawdown_floor=drawdown_floor)
    turnover_extra = max(0.0, float(turnover) - float(turnover_budget))
    turnover_extra_penalty_bps = float(lambda_turnover_extra) * turnover_extra
    conc_excess = concentration_penalty(weights if weights is not None else [], hhi_target=hhi_target)
    conc_penalty_bps = float(lambda_conc) * conc_excess
    hhi = (
        float(np.square(np.clip(np.asarray(weights if weights is not None else [], dtype=float)[:-1], 0.0, 1.0)).sum())
        if weights is not None and len(weights) > 1
        else 0.0
    )

    mode = str(reward_mode).lower().replace("-", "_")
    tracking_vol_value = float(tracking_vol) if tracking_vol is not None and np.isfinite(float(tracking_vol)) else float(tracking_vol_floor)
    tracking_vol_value = max(tracking_vol_value, float(tracking_vol_floor), 1e-8)
    tracking_week = max(tracking_vol_value / np.sqrt(52.0), 1e-8)
    active_ir_score = float(active_reward_scale) * active_return / tracking_week
    vol_excess = max(0.0, float(realized_vol) - float(target_vol))
    vol_target_penalty_bps = 10000.0 * vol_excess * vol_excess
    drawdown_excess = max(0.0, float(drawdown_floor) - float(drawdown))
    drawdown_target_penalty_bps = 10000.0 * drawdown_excess * drawdown_excess

    if mode in {"dsr", "differential_sharpe", "sortino", "ddr"}:
        # Everything is kept on an O(1) scale so the Differential Sharpe primary
        # signal drives learning instead of being swamped by crisis-period penalties.
        primary = float(dsr_scale) * float(dsr)
        dd_pen = float(lambda_drawdown) * 100.0 * drawdown_excess * drawdown_excess
        vol_pen = float(lambda_vol) * 100.0 * vol_excess * vol_excess
        cost_pen = float(lambda_cost) * cost_bps_realized
        conc_pen = float(lambda_conc) * conc_excess
        cash_excess = max(0.0, float(cash_weight) - float(cash_cap))
        cash_pen = float(lambda_cash) * cash_excess
        decorr_pen = float(lambda_decorr) * 100.0 * float(decorr_penalty) * float(decorr_penalty)
        reward = primary - dd_pen - vol_pen - cost_pen - conc_pen - cash_pen - decorr_pen
        return {
            "reward": float(reward),
            "dsr": float(dsr),
            "primary_reward": float(primary),
            "log_excess_return": log_excess_return,
            "portfolio_return": float(portfolio_return),
            "active_return": active_return,
            "cost": float(cost),
            "cost_bps_realized": float(cost_bps_realized),
            "cost_penalty": float(cost_pen),
            "turnover": float(turnover),
            "turnover_extra": float(turnover_extra),
            "vol_penalty": float(vol_pen),
            "vol_excess": float(vol_excess),
            "drawdown_penalty": float(dd_pen),
            "drawdown_excess": float(drawdown_excess),
            "concentration_penalty": float(conc_pen),
            "cash_weight": float(cash_weight),
            "cash_penalty": float(cash_pen),
            "decorr_penalty": float(decorr_pen),
            "hhi": float(hhi),
            "realized_vol": float(realized_vol),
            "drawdown": float(drawdown),
        }

    if mode in {"active_te", "active_tracking", "tracking_error"}:
        reward = (
            active_ir_score
            - float(lambda_cost) * cost_bps_realized
            - turnover_extra_penalty_bps
            - float(lambda_vol) * vol_target_penalty_bps
            - float(lambda_drawdown) * drawdown_target_penalty_bps
            - float(lambda_conc) * conc_excess
            - float(lambda_beta) * float(beta_penalty)
            - float(lambda_corr) * float(corr_penalty)
        )
        return {
            "reward": float(reward),
            "log_excess_return": log_excess_return,
            "log_excess_bps": float(log_excess_bps),
            "active_return": active_return,
            "active_bps": float(active_bps),
            "active_ir_score": float(active_ir_score),
            "tracking_vol": float(tracking_vol_value),
            "cost": float(cost),
            "cost_bps_realized": float(cost_bps_realized),
            "turnover": float(turnover),
            "turnover_budget": float(turnover_budget),
            "turnover_extra": float(turnover_extra),
            "turnover_extra_penalty_bps": float(turnover_extra_penalty_bps),
            "vol_band_penalty": float(vol_target_penalty_bps),
            "vol_band_penalty_bps": float(vol_target_penalty_bps),
            "vol_excess": float(vol_excess),
            "drawdown_penalty": float(drawdown_target_penalty_bps),
            "drawdown_penalty_bps": float(drawdown_target_penalty_bps),
            "drawdown_excess": float(drawdown_excess),
            "concentration_penalty": float(conc_excess),
            "concentration_penalty_bps": float(conc_penalty_bps),
            "beta_penalty": float(beta_penalty),
            "corr_penalty": float(corr_penalty),
            "hhi": float(hhi),
            "realized_vol": float(realized_vol),
            "drawdown": float(drawdown),
        }

    reward = (
        log_excess_bps
        + float(beta_active) * active_bps
        - float(lambda_cost) * cost_bps_realized
        - turnover_extra_penalty_bps
        - float(lambda_vol) * vol_pen_bps
        - float(lambda_drawdown) * dd_pen_bps
        - conc_penalty_bps
    )
    return {
        "reward": float(reward),
        "log_excess_return": log_excess_return,
        "log_excess_bps": float(log_excess_bps),
        "active_return": active_return,
        "active_bps": float(active_bps),
        "cost": float(cost),
        "cost_bps_realized": float(cost_bps_realized),
        "turnover": float(turnover),
        "turnover_budget": float(turnover_budget),
        "turnover_extra": float(turnover_extra),
        "turnover_extra_penalty_bps": float(turnover_extra_penalty_bps),
        "vol_band_penalty": float(vol_pen_bps),
        "vol_band_penalty_bps": float(vol_pen_bps),
        "drawdown_penalty": float(dd_pen_bps),
        "drawdown_penalty_bps": float(dd_pen_bps),
        "concentration_penalty": float(conc_excess),
        "concentration_penalty_bps": float(conc_penalty_bps),
        "active_ir_score": float(active_ir_score),
        "tracking_vol": float(tracking_vol_value),
        "vol_excess": float(vol_excess),
        "drawdown_excess": float(drawdown_excess),
        "beta_penalty": float(beta_penalty),
        "corr_penalty": float(corr_penalty),
        "hhi": float(hhi),
        "realized_vol": float(realized_vol),
        "drawdown": float(drawdown),
    }


def reward_component_table(
    results: Mapping[str, object] | None = None,
    *,
    component_rows: Mapping[str, Sequence[Mapping[str, float]]] | None = None,
) -> pd.DataFrame:
    rows = []
    if component_rows is not None:
        for name, comps in component_rows.items():
            frame = pd.DataFrame(list(comps))
            if frame.empty:
                continue
            row = frame.mean(numeric_only=True).to_dict()
            row["Strategy"] = str(name)
            rows.append(row)
    if results is not None:
        for name, result in results.items():
            comps = getattr(result, "components", None)
            if comps is None and isinstance(result, Mapping):
                comps = result.get("components")
            if comps is None:
                continue
            frame = pd.DataFrame(comps)
            if frame.empty:
                continue
            row = frame.mean(numeric_only=True).to_dict()
            row["Strategy"] = str(name)
            rows.append(row)
    out = pd.DataFrame(rows)
    return out.set_index("Strategy") if "Strategy" in out.columns else out


__all__ = [
    "DifferentialSharpe",
    "DifferentialSortino",
    "active_reward",
    "concentration_penalty",
    "drawdown_penalty",
    "portfolio_reward",
    "reward_component_table",
    "reward_components",
    "turnover_penalty",
    "vol_band_penalty",
]
