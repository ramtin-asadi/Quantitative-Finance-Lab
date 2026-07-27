"""Cross-sectional fundamental scoring and stock selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from quantfinlab.ml.evaluation import forecast_buckets, rank_metrics

corporate_score = {
    "family": "corporate",
    "blocks": {
        "profitability": {
            "gross_profitability_assets": 1.0,
            "roa": 1.0,
            "roe": 0.8,
            "roic_proxy": 1.0,
            "operating_margin": 0.8,
            "fcf_margin": 0.8,
        },
        "cash_quality": {
            "cfo_assets": 1.0,
            "fcf_assets": 1.0,
            "cfo_net_income": 0.8,
            "fcf_conversion": 0.8,
            "total_accruals": 1.0,
            "positive_cfo_frequency": 0.7,
            "positive_fcf_frequency": 0.7,
        },
        "growth": {
            "revenue_growth": 1.0,
            "gross_profit_growth": 0.7,
            "operating_income_growth": 0.8,
            "cfo_growth": 0.8,
            "fcf_growth": 0.8,
            "eps_growth": 0.8,
            "operating_margin_change": 0.7,
            "roa_change": 0.7,
            "roic_change": 0.7,
            "cash_conversion_change": 0.5,
        },
        "strength": {
            "net_debt_assets": 1.0,
            "debt_assets": 0.8,
            "liabilities_assets": 0.8,
            "interest_coverage": 0.8,
            "cfo_debt": 0.8,
            "current_ratio": 0.6,
            "cash_assets": 0.7,
            "tangible_equity_assets": 0.8,
        },
        "efficiency": {
            "asset_turnover": 1.0,
            "receivable_turnover": 0.7,
            "inventory_turnover": 0.7,
            "cash_conversion_cycle": 0.7,
        },
        "capital_allocation": {
            "net_shareholder_yield": 1.0,
            "share_count_dilution": 1.0,
            "dividend_coverage_cfo": 0.6,
            "repurchase_yield": 0.7,
            "per_share_growth_spread": 0.7,
        },
        "valuation": {
            "earnings_yield": 1.0,
            "fcf_yield": 1.0,
            "ebit_ev": 0.9,
            "sales_ev": 0.6,
            "book_to_market": 0.7,
            "tangible_book_to_market": 0.6,
        },
    },
    "block_weights": {
        "profitability": 0.35,
        "cash_quality": 0.20,
        "growth": 0.15,
        "strength": 0.10,
        "efficiency": 0.05,
        "capital_allocation": 0.05,
        "valuation": 0.10,
    },
    "lower_is_better": (
        "total_accruals",
        "net_debt_assets",
        "debt_assets",
        "liabilities_assets",
        "cash_conversion_cycle",
        "share_count_dilution",
    ),
    "minimum_metrics": 12,
    "minimum_blocks": 4,
    "required_blocks": ("profitability",),
    "one_of_blocks": ("cash_quality", "strength"),
}


financial_score = {
    "family": "financial",
    "blocks": {
        "profitability": {
            "fin_roa": 1.0,
            "fin_roe": 1.0,
            "fin_pretax_assets": 0.8,
            "fin_net_margin": 0.6,
        },
        "capital_strength": {
            "fin_equity_assets": 1.0,
            "fin_tangible_equity_assets": 1.0,
            "fin_liabilities_assets": 0.8,
            "fin_assets_equity": 0.8,
        },
        "growth": {
            "fin_revenue_growth": 0.8,
            "fin_net_income_growth": 0.8,
            "fin_equity_growth": 0.8,
            "fin_bvps_growth": 1.0,
            "fin_tbvps_growth": 0.8,
            "fin_roa_change": 0.7,
            "fin_roe_change": 0.7,
            "fin_equity_assets_change": 0.8,
        },
        "stability": {
            "fin_roa_variability": 1.0,
            "fin_roe_variability": 0.8,
            "fin_net_income_variability": 0.8,
            "fin_positive_earnings_frequency": 1.0,
        },
        "efficiency": {
            "fin_operating_expense_ratio": 1.0,
            "fin_pretax_margin": 0.8,
            "fin_revenue_assets": 0.8,
        },
        "valuation_return": {
            "fin_earnings_yield": 1.0,
            "fin_book_to_market": 1.0,
            "fin_tangible_book_to_market": 0.8,
            "fin_revenue_market_cap": 0.6,
            "fin_net_payout_yield": 0.8,
            "fin_share_dilution": 0.8,
        },
    },
    "block_weights": {
        "profitability": 0.20,
        "capital_strength": 0.15,
        "growth": 0.05,
        "stability": 0.15,
        "efficiency": 0.15,
        "valuation_return": 0.30,
    },
    "lower_is_better": (
        "fin_liabilities_assets",
        "fin_assets_equity",
        "fin_roa_variability",
        "fin_roe_variability",
        "fin_net_income_variability",
        "fin_operating_expense_ratio",
        "fin_share_dilution",
    ),
    "minimum_metrics": 8,
    "minimum_blocks": 4,
    "required_blocks": ("profitability", "capital_strength"),
    "one_of_blocks": (),
}


def metric_percentile_scores(
    frame: pd.DataFrame,
    metrics: Sequence[str],
    *,
    lower_is_better: Sequence[str] = (),
    date_column: str = "decision_date",
    family_column: str = "score_family",
    peer_column: str = "industry",
    peer_weight: float = 0.70,
    minimum_peers: int = 10,
    minimum_family: int = 10,
    winsor_limits: tuple[float, float] = (0.01, 0.99),
    minimum_winsor_count: int = 20,
) -> pd.DataFrame:
    """Turn raw metrics into peer- and family-relative percentile scores."""

    metric_names = list(dict.fromkeys(metrics))
    values = frame[metric_names].apply(pd.to_numeric, errors="coerce")
    family_keys = [frame[date_column], frame[family_column]]
    peer_keys = [*family_keys, frame[peer_column]]

    family_groups = values.groupby(family_keys, sort=False)
    family_count = family_groups.transform("count")
    lower = family_groups.transform("quantile", q=float(winsor_limits[0]))
    upper = family_groups.transform("quantile", q=float(winsor_limits[1]))
    clipped = values.clip(lower=lower, upper=upper).where(
        family_count.ge(int(minimum_winsor_count)),
        values,
    )

    family_percentile = clipped.groupby(family_keys, sort=False).rank(pct=True)
    peer_groups = clipped.groupby(peer_keys, sort=False)
    peer_count = peer_groups.transform("count")
    peer_percentile = peer_groups.rank(pct=True).where(
        peer_count.ge(int(minimum_peers))
    )

    family_weight = 1.0 - float(peer_weight)
    scores = (
        float(peer_weight) * peer_percentile
        + family_weight * family_percentile
    ).fillna(family_percentile)
    scores = (100.0 * scores).where(family_count.ge(int(minimum_family)))

    negative = [name for name in lower_is_better if name in scores]
    scores[negative] = 100.0 - scores[negative]
    return scores.add_suffix("_score")


def block_scores(
    frame: pd.DataFrame,
    blocks: Mapping[str, Mapping[str, float]],
    *,
    prefix: str,
) -> pd.DataFrame:
    """Combine available metric scores within each scoring block."""

    weights = pd.DataFrame(blocks, dtype=float).fillna(0.0)
    metric_values = frame[
        [f"{metric}_score" for metric in weights.index]
    ].copy()
    metric_values.columns = weights.index
    numerator = metric_values.fillna(0.0).dot(weights)
    denominator = metric_values.notna().astype(float).dot(weights)
    scores = numerator.div(denominator).where(denominator.gt(0))
    scores.columns = [
        f"{prefix}_{block}_score" for block in scores.columns
    ]
    return scores


def family_composite_score(
    frame: pd.DataFrame,
    block_values: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    family_column: str = "score_family",
) -> pd.DataFrame:
    """Apply one score family's block weights and minimum-coverage rules."""

    family = str(config["family"])
    blocks = config["blocks"]
    block_weights = config["block_weights"]
    if not isinstance(blocks, Mapping) or not isinstance(block_weights, Mapping):
        raise TypeError("blocks and block_weights must be mappings")

    block_columns = [
        f"{family}_{block}_score" for block in block_weights
    ]
    weights = pd.Series(
        {
            f"{family}_{block}_score": float(weight)
            for block, weight in block_weights.items()
        }
    )
    values = block_values[block_columns]
    numerator = values.mul(weights, axis=1).sum(axis=1, min_count=1)
    denominator = values.notna().mul(weights, axis=1).sum(axis=1)

    metric_names = list(dict.fromkeys(
        metric
        for block in blocks.values()
        for metric in block
    ))
    metric_count = frame[
        [f"{metric}_score" for metric in metric_names]
    ].notna().sum(axis=1)
    block_count = values.notna().sum(axis=1)
    family_rows = frame[family_column].eq(family)

    eligible = (
        family_rows
        & metric_count.ge(int(config["minimum_metrics"]))
        & block_count.ge(int(config.get("minimum_blocks", 1)))
    )
    for block in config.get("required_blocks", ()):
        eligible &= values[f"{family}_{block}_score"].notna()

    one_of = tuple(config.get("one_of_blocks", ()))
    if one_of:
        eligible &= values[
            [f"{family}_{block}_score" for block in one_of]
        ].notna().any(axis=1)

    return pd.DataFrame(
        {
            f"{family}_base_score": (
                numerator / denominator
            ).where(eligible & denominator.gt(0)),
            f"{family}_valid_metrics": metric_count.where(family_rows),
            f"{family}_valid_blocks": block_count.where(family_rows),
        },
        index=frame.index,
    )


def piotroski_penalty(
    frame: pd.DataFrame,
    *,
    score_column: str = "piotroski_f_score",
    family_column: str = "score_family",
    corporate_family: str = "corporate",
    threshold: float = 5.0,
    penalty_per_point: float = 0.50,
) -> pd.Series:
    """Penalize corporate Piotroski scores below five."""

    penalty = (
        float(threshold) - pd.to_numeric(frame[score_column], errors="coerce")
    ).clip(lower=0.0).fillna(0.0)
    penalty = penalty * float(penalty_per_point)
    return penalty.where(
        frame[family_column].eq(corporate_family),
        0.0,
    ).rename("piotroski_penalty")


def red_flag_penalty(
    frame: pd.DataFrame,
    *,
    warning_column: str = "warning_penalty",
    scale: float = 0.10,
) -> pd.Series:
    """Scale the weighted accounting-warning total."""

    return (
        float(scale)
        * pd.to_numeric(frame[warning_column], errors="coerce").fillna(0.0)
    ).rename("red_flag_penalty")


def price_signal_frame(
    prices: pd.DataFrame,
    *,
    decision_dates: Sequence[pd.Timestamp | str] | None = None,
    horizons: Sequence[int] = (1, 3, 6, 12),
    momentum_horizons: Sequence[int] = (3, 6, 12),
    skip_recent_months: int = 1,
) -> pd.DataFrame:
    """Build monthly momentum and forward-return signals from adjusted prices."""

    monthly = prices.copy()
    monthly.index = pd.to_datetime(monthly.index)
    monthly = monthly.sort_index()
    if decision_dates is not None:
        monthly = monthly.reindex(pd.DatetimeIndex(pd.to_datetime(decision_dates)))

    signals: dict[str, pd.Series] = {}
    momentum_months = {int(months) for months in momentum_horizons}
    for months_value in horizons:
        months = int(months_value)
        if months in momentum_months:
            signals[f"momentum_{months}_{int(skip_recent_months)}"] = (
                monthly.shift(int(skip_recent_months))
                / monthly.shift(months)
                - 1.0
            ).stack()
        signals[f"forward_{months}m"] = (
            monthly.shift(-months) / monthly - 1.0
        ).stack()

    result = pd.concat(signals, axis=1)
    result.index.names = ["decision_date", "ticker"]
    return result


def _monthly_block_ic(
    frame: pd.DataFrame,
    block_columns: Sequence[str],
    *,
    return_column: str,
    date_column: str,
) -> pd.DataFrame:
    values = {}
    columns = list(block_columns)
    for date, month in frame[
        [date_column, return_column, *columns]
    ].groupby(date_column):
        values[date] = month[columns].corrwith(
            month[return_column],
            method="spearman",
        )
    return pd.DataFrame(values).T.sort_index()


def _prior_block_weights(
    block_columns: Sequence[str],
    prior_weights: Mapping[str, float] | pd.Series,
    family: str | None,
) -> pd.Series:
    supplied = pd.Series(prior_weights, dtype=float)
    values = {}
    for column in block_columns:
        short_name = str(column)
        if family and short_name.startswith(f"{family}_"):
            short_name = short_name[len(family) + 1 :]
        if short_name.endswith("_score"):
            short_name = short_name[:-6]
        key = column if column in supplied.index else short_name
        values[column] = supplied[key]
    result = pd.Series(values, dtype=float)
    return result / result.sum()


def _cap_block_weights(row: pd.Series, weight_cap: float) -> pd.Series:
    weights = row.copy()
    if float(weight_cap) * len(weights) < 1.0 - 1e-12:
        raise ValueError("weight_cap is too small for the number of blocks")
    for _ in range(len(weights)):
        excess = (weights - float(weight_cap)).clip(lower=0.0).sum()
        weights = weights.clip(upper=float(weight_cap))
        if excess <= 1e-12:
            break
        room = (float(weight_cap) - weights).clip(lower=0.0)
        weights = weights + excess * room / room.sum()
    return weights / weights.sum()


def walkforward_block_weights(
    frame: pd.DataFrame,
    block_columns: Sequence[str],
    prior_weights: Mapping[str, float] | pd.Series,
    *,
    family: str | None = None,
    family_column: str = "score_family",
    date_column: str = "decision_date",
    horizons: Sequence[int] = (3, 6, 12),
    horizon_weights: Sequence[float] = (0.25, 0.50, 0.25),
    window: int = 36,
    minimum_periods: int = 12,
    adaptive_share: float = 0.75,
    weight_cap: float = 0.35,
) -> pd.DataFrame:
    """Estimate fully lagged, capped block weights from observed rank IC."""

    columns = list(block_columns)
    research = frame
    if family is not None:
        research = frame[frame[family_column].eq(family)]

    horizon_values = [int(value) for value in horizons]
    blend_values = np.asarray(horizon_weights, dtype=float)
    if len(horizon_values) != len(blend_values):
        raise ValueError("horizons and horizon_weights must have the same length")
    blend_values = blend_values / blend_values.sum()

    observed = None
    for months, share in zip(horizon_values, blend_values, strict=True):
        monthly_ic = _monthly_block_ic(
            research,
            columns,
            return_column=f"forward_{months}m",
            date_column=date_column,
        ).shift(months)
        contribution = float(share) * monthly_ic
        observed = contribution if observed is None else observed + contribution

    if observed is None:
        return pd.DataFrame(columns=columns, dtype=float)

    strength = observed.rolling(
        int(window),
        min_periods=int(minimum_periods),
    ).mean().clip(lower=0.0)
    learned = strength.div(strength.sum(axis=1), axis=0)
    prior = _prior_block_weights(columns, prior_weights, family)
    fallback = pd.DataFrame(
        np.tile(prior.to_numpy(), (len(learned), 1)),
        index=learned.index,
        columns=columns,
    )
    learned = learned.fillna(fallback)

    weights = (
        (1.0 - float(adaptive_share)) * prior
        + float(adaptive_share) * learned
    )
    weights = weights.div(weights.sum(axis=1), axis=0)
    return weights.apply(
        _cap_block_weights,
        axis=1,
        weight_cap=float(weight_cap),
    )


def adaptive_family_score(
    frame: pd.DataFrame,
    block_values: pd.DataFrame,
    weight_history: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    penalties: pd.Series | None = None,
    date_column: str = "decision_date",
    family_column: str = "score_family",
) -> pd.DataFrame:
    """Apply dated block weights and rank the resulting family score."""

    family = str(config["family"])
    block_columns = [
        f"{family}_{block}_score"
        for block in config["block_weights"]
    ]
    family_rows = frame[family_column].eq(family)
    eligibility_column = f"{family}_base_score"
    if eligibility_column in frame:
        family_rows &= frame[eligibility_column].notna()
    dates = frame.loc[family_rows, date_column]
    row_weights = weight_history.reindex(dates).set_axis(frame.index[family_rows])
    values = block_values.loc[family_rows, block_columns]
    numerator = values.mul(row_weights).sum(axis=1, min_count=1)
    denominator = values.notna().mul(row_weights).sum(axis=1)

    adaptive = pd.Series(np.nan, index=frame.index, dtype=float)
    adaptive.loc[family_rows] = numerator.div(denominator).where(
        denominator.gt(0)
    )
    penalty = (
        pd.Series(0.0, index=frame.index)
        if penalties is None
        else pd.Series(penalties, index=frame.index).fillna(0.0)
    )
    uncapped = adaptive - penalty
    family_keys = [frame[date_column], frame[family_column]]
    final = uncapped.groupby(family_keys).rank(pct=True) * 100.0
    return pd.DataFrame(
        {
            "adaptive_base_score": adaptive,
            "uncapped_score": uncapped,
            "final_score": final,
        },
        index=frame.index,
    )


def definitive_selection_score(
    frame: pd.DataFrame,
    *,
    fundamental_column: str = "final_score",
    momentum_column: str = "momentum_6_1",
    date_column: str = "decision_date",
    family_column: str = "score_family",
    momentum_weight: float = 0.10,
) -> pd.DataFrame:
    """Normalize momentum within score families and form the final selection score."""

    family_keys = [frame[date_column], frame[family_column]]
    momentum_score = (
        pd.to_numeric(frame[momentum_column], errors="coerce")
        .groupby(family_keys)
        .rank(pct=True)
        * 100.0
    )
    fundamental_weight = 1.0 - float(momentum_weight)
    selection_score = (
        fundamental_weight * frame[fundamental_column]
        + float(momentum_weight) * momentum_score
    )
    return pd.DataFrame(
        {
            "momentum_score": momentum_score,
            "selection_score": selection_score,
        },
        index=frame.index,
    )


def rank_ic_table(
    frame: pd.DataFrame,
    *,
    score_columns: Sequence[str],
    horizons: Sequence[int] = (1, 3, 6, 12),
    date_column: str = "decision_date",
    ticker_column: str = "ticker",
    top_fraction: float = 0.20,
) -> pd.DataFrame:
    """Evaluate cross-sectional score ranks at each forward horizon."""

    tables = []
    for months_value in horizons:
        months = int(months_value)
        table = rank_metrics(
            frame,
            date_col=date_column,
            asset_col=ticker_column,
            y_col=f"forward_{months}m",
            prediction_cols=list(score_columns),
            top_frac=float(top_fraction),
        ).reset_index()
        table["horizon_months"] = months
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def bucket_return_table(
    frame: pd.DataFrame,
    *,
    score_column: str = "selection_score",
    horizons: Sequence[int] = (1, 3, 6, 12),
    buckets: int = 5,
    date_column: str = "decision_date",
) -> pd.DataFrame:
    """Summarize forward returns by within-date score bucket."""

    tables = []
    for months_value in horizons:
        months = int(months_value)
        table = forecast_buckets(
            frame,
            date_col=date_column,
            y_col=f"forward_{months}m",
            score_col=score_column,
            n_buckets=int(buckets),
        ).reset_index()
        table["horizon_months"] = months
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def select_stocks(
    scores: pd.DataFrame,
    *,
    top_n: int | Sequence[int] = (15, 50, 100),
    score_column: str = "selection_score",
    date_column: str = "decision_date",
    start: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Select the highest-scoring stocks independently at each decision date."""

    breadths = [int(top_n)] if isinstance(top_n, int) else [int(value) for value in top_n]
    scoreable = scores[scores[score_column].notna()].copy()
    if start is not None:
        scoreable = scoreable[
            pd.to_datetime(scoreable[date_column]).ge(pd.Timestamp(start))
        ]
    scoreable = scoreable.sort_values(
        [date_column, score_column],
        ascending=[True, False],
    )

    selections = []
    for breadth in breadths:
        selected = scoreable.groupby(
            date_column,
            group_keys=False,
        ).head(breadth).copy()
        selected["top_n"] = breadth
        selected["selection_rank"] = selected.groupby(date_column)[
            score_column
        ].rank(ascending=False, method="first")
        selections.append(selected)
    if not selections:
        return scoreable.assign(
            top_n=pd.Series(dtype=int),
            selection_rank=pd.Series(dtype=float),
        )
    return pd.concat(selections, ignore_index=True)


def investment_universes(
    monthly_universe: pd.DataFrame,
    stock_selections: pd.DataFrame,
    returns: pd.DataFrame,
    date_map: pd.DataFrame,
    *,
    top_n: Sequence[int] = (15, 50, 100),
    lookback: int = 189,
    minimum_observations: int = 177,
    minimum_assets: int = 10,
    date_column: str = "decision_date",
    execution_column: str = "execution_date",
    ticker_column: str = "ticker",
    prices: pd.DataFrame | None = None,
) -> dict[str, dict[pd.Timestamp, dict[str, list[str]]]]:
    """Build seasoned full and selected universes keyed by execution date.

    When filled prices are supplied, seasoning follows the portfolio notebook's
    exact rule: ``lookback + 1`` price observations for a ``lookback``-return
    covariance window. Without prices, the original return-count behavior is
    preserved.
    """

    breadths = [int(value) for value in top_n]
    universes: dict[str, dict[pd.Timestamp, dict[str, list[str]]]] = {
        "full": {},
        **{f"top{breadth}": {} for breadth in breadths},
    }
    decision_to_execution = (
        date_map[[date_column, execution_column]]
        .drop_duplicates(date_column, keep="last")
        .set_index(date_column)[execution_column]
    )
    decision_to_execution.index = pd.to_datetime(decision_to_execution.index)
    decision_dates = pd.DatetimeIndex(
        sorted(
            set(pd.to_datetime(stock_selections[date_column]))
            & set(decision_to_execution.index)
        )
    )
    if decision_dates.empty:
        return universes

    if prices is None:
        observations = returns.notna().rolling(int(lookback), min_periods=1).sum()
    else:
        observations = prices.notna().rolling(int(lookback) + 1, min_periods=1).sum()
    seasoned = observations.reindex(decision_dates, method="ffill")
    full_source = {
        pd.Timestamp(date): list(dict.fromkeys(group[ticker_column].astype(str)))
        for date, group in monthly_universe[
            monthly_universe[date_column].isin(decision_dates)
        ].groupby(date_column)
    }
    selection_source = {
        breadth: {
            pd.Timestamp(date): list(dict.fromkeys(group[ticker_column].astype(str)))
            for date, group in stock_selections[
                stock_selections["top_n"].eq(breadth)
                & stock_selections[date_column].isin(decision_dates)
            ].sort_values(
                [date_column, "selection_rank"]
                if "selection_rank" in stock_selections
                else [date_column]
            ).groupby(date_column)
        }
        for breadth in breadths
    }

    for decision_date in decision_dates:
        execution_date = pd.Timestamp(decision_to_execution.loc[decision_date])
        observations = seasoned.loc[decision_date]
        sources = {
            "full": full_source.get(decision_date, []),
            **{
                f"top{breadth}": selection_source[breadth].get(decision_date, [])
                for breadth in breadths
            },
        }
        for label, candidates in sources.items():
            names = [
                ticker
                for ticker in candidates
                if ticker in observations.index
                and observations[ticker] >= int(minimum_observations)
            ]
            if len(names) >= int(minimum_assets):
                universes[label][execution_date] = {"tickers": names}
    return universes


__all__ = [
    "adaptive_family_score",
    "block_scores",
    "bucket_return_table",
    "corporate_score",
    "definitive_selection_score",
    "family_composite_score",
    "financial_score",
    "investment_universes",
    "metric_percentile_scores",
    "piotroski_penalty",
    "price_signal_frame",
    "rank_ic_table",
    "red_flag_penalty",
    "select_stocks",
    "walkforward_block_weights",
]
