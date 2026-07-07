from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


class DifferentialSharpe:
    """Online Differential Sharpe reward.

    The object maintains exponentially weighted estimates of the first and second
    moments of returns and returns the marginal contribution of each new return to
    the running Sharpe ratio. It is intended for reinforcement-learning reward
    signals where the objective should reflect risk-adjusted performance rather
    than raw return.

    Parameters
    ----------
    eta : float, default=0.01
        Exponential update rate for running moments.

    Attributes
    ----------
    eta : float
        Update rate.
    A : float
        Running mean return estimate.
    B : float
        Running second-moment estimate.
    initialized : bool
        Whether the first observation has initialized the state.
    """

    def __init__(self, eta: float = 0.01):
        self.eta = float(eta)
        self.A = 0.0
        self.B = 0.0
        self.initialized = False

    def update(self, r: float) -> float:
        """Update the Differential Sharpe state and return the marginal reward.

        Parameters
        ----------
        r : float
            New portfolio return.

        Returns
        -------
        float
            Clipped Differential Sharpe reward contribution.

        Notes
        -----
        The first update initializes the running moments and returns zero.
        """
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
        """Reset the running Differential Sharpe state.

        Returns
        -------
        None
            The running moments are reset to zero and the object is marked
            uninitialized.
        """
        self.A = 0.0
        self.B = 0.0
        self.initialized = False


class DifferentialSortino:
    """Online Differential Sortino-style reward.

    This object is a downside-risk variant of Differential Sharpe. The first moment
    tracks returns, while the second moment tracks squared negative excess returns.
    The reward therefore charges downside deviation rather than total variance.

    Parameters
    ----------
    eta : float, default=0.01
        Exponential update rate.
    rf_daily : float, default=0.0
        One-period risk-free rate used to define downside excess returns.
    """

    def __init__(self, eta: float = 0.01, rf_daily: float = 0.0):
        self.eta = float(eta)
        self.rf = float(rf_daily)
        self.A = 0.0
        self.B = 0.0
        self.initialized = False

    def update(self, r: float) -> float:
        """Update the Differential Sortino state and return the marginal reward.

        Parameters
        ----------
        r : float
            New portfolio return.

        Returns
        -------
        float
            Clipped Differential downside-risk reward contribution.

        Notes
        -----
        Upside returns contribute to the running mean without increasing the downside
        second moment.
        """
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
        """Reset the running Differential Sortino state.

        Returns
        -------
        None
            The running mean, downside moment, and initialized flag are reset.
        """
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
    """Return the scalar portfolio reward from detailed reward components.

    This is a convenience wrapper around ``reward_components``. It accepts the same
    economic inputs and penalty settings, computes the component dictionary, and
    returns only the final scalar reward.

    Parameters
    ----------
    portfolio_return : float
        Realized portfolio return for the period.
    benchmark_return : float, default=0.0
        Benchmark return for active-reward modes.
    turnover : float, default=0.0
        One-way turnover for the period.
    cost : float, default=0.0
        Realized cost as a return drag.
    weights : sequence of float, optional
        Portfolio weights used for concentration penalties.
    rf_period : float, default=0.0
        Risk-free return for the period.
    realized_vol : float, default=0.0
        Realized portfolio volatility used in volatility penalties.
    drawdown : float, default=0.0
        Current portfolio drawdown.
    beta_active, vol_low, vol_high, lambda_cost, turnover_budget,
    lambda_turnover_extra, lambda_vol, lambda_drawdown, lambda_conc, hhi_target,
    cost_bps
        Reward-shaping parameters passed to ``reward_components``.
    **legacy_kwargs
        Additional compatibility parameters passed through.

    Returns
    -------
    float
        Final scalar reward.
    """
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
    """Compute portfolio reward and diagnostic reward components.

    The function supports several reward modes, including log-excess plus active
    return, active-return scaled by tracking error, and Differential Sharpe/Sortino
    modes. It reports the final reward along with the economic quantities and
    penalties that produced it.

    Parameters
    ----------
    portfolio_return : float
        Realized portfolio return.
    benchmark_return : float, default=0.0
        Benchmark return.
    turnover : float, default=0.0
        One-way portfolio turnover.
    cost : float, default=0.0
        Realized transaction cost as a return drag.
    weights : sequence of float, optional
        Portfolio weights used for concentration diagnostics.
    rf_period : float, default=0.0
        Risk-free return over the reward period.
    realized_vol : float, default=0.0
        Realized annualized volatility estimate.
    drawdown : float, default=0.0
        Current drawdown.
    beta_active : float, default=0.75
        Weight applied to active return in the default reward mode.
    vol_low, vol_high : float
        Volatility band used in the default volatility penalty.
    lambda_cost : float, default=1.25
        Cost penalty multiplier.
    turnover_budget : float, default=0.12
        Turnover budget before extra turnover penalty applies.
    lambda_turnover_extra : float, default=10.0
        Extra turnover penalty multiplier.
    lambda_vol : float, default=1.0
        Volatility penalty multiplier.
    lambda_drawdown : float, default=1.0
        Drawdown penalty multiplier.
    lambda_conc : float, default=10.0
        Concentration penalty multiplier.
    hhi_target : float, default=0.18
        Target Herfindahl-Hirschman concentration.
    cost_bps : float, optional
        If supplied, realized cost is computed as ``cost_bps * turnover`` in basis
        points.
    lambda_turnover, lambda_concentration, max_weight
        Legacy compatibility parameters.
    reward_mode : str, default="active_te"
        Reward mode. Supported families include active tracking-error modes,
        differential Sharpe/Sortino modes, and the default log-excess mode.
    tracking_vol : float, optional
        Tracking-volatility estimate used in active-tracking reward mode.
    tracking_vol_floor : float, default=0.06
        Lower bound for tracking volatility.
    active_reward_scale : float, default=100.0
        Scaling applied to active tracking reward.
    target_vol : float, default=0.15
        Target volatility used in active-tracking and DSR modes.
    drawdown_floor : float, default=-0.10
        Drawdown threshold below which drawdown penalties apply.
    beta_penalty, corr_penalty : float
        Additional exposure/correlation penalties.
    lambda_beta, lambda_corr : float
        Multipliers for beta and correlation penalties.
    dsr : float, default=0.0
        Differential Sharpe or Sortino reward input.
    dsr_scale : float, default=100.0
        Scaling applied to ``dsr`` in DSR modes.
    cash_weight : float, default=0.0
        Cash weight used in DSR cash penalties.
    cash_cap : float, default=0.30
        Cash cap before cash penalty applies.
    lambda_cash : float, default=1.0
        Cash penalty multiplier.
    decorr_penalty : float, default=0.0
        Decorrelation penalty input.
    lambda_decorr : float, default=0.0
        Decorrelation penalty multiplier.

    Returns
    -------
    dict
        Dictionary containing ``reward`` and the relevant component diagnostics for
        the selected reward mode.

    Notes
    -----
    Most non-DSR modes are expressed in basis-point-like units. DSR modes are kept
    closer to an order-one scale so the differential risk-adjusted signal is not
    overwhelmed by rare crisis penalties.
    """
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
    """Average reward components across strategies or result objects.

    Parameters
    ----------
    results : mapping, optional
        Mapping from strategy name to objects containing a ``components`` attribute
        or mapping key.
    component_rows : mapping, optional
        Mapping from strategy name to an explicit sequence of component dictionaries.

    Returns
    -------
    pandas.DataFrame
        Strategy-indexed table of mean numeric reward components.

    Notes
    -----
    At least one of ``results`` or ``component_rows`` should be supplied.
    """
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
