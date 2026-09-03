from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import numpy as np
import pandas as pd


class FactorAttributionResult(NamedTuple):
    """Tables produced by :func:`factor_attribution`."""

    exposures: pd.DataFrame
    inference: pd.DataFrame
    attribution: pd.DataFrame


class FamaMacBethResult(NamedTuple):
    """Tables produced by :func:`fama_macbeth`."""

    summary: pd.DataFrame
    monthly_coefficients: pd.DataFrame
    diagnostics: pd.DataFrame


def month_end_prices(prices):
    return pd.DataFrame(prices).sort_index().resample("ME").last().dropna(how="all")


def excess_returns(returns, rf):
    r = pd.DataFrame(returns).astype(float)
    rf_s = pd.Series(rf, dtype=float).reindex(r.index)
    return r.subtract(rf_s, axis=0)


def _ols_hac(y, x, maxlags=3):
    y_values = np.asarray(y, dtype=float).reshape(-1)
    x_values = np.asarray(x, dtype=float)
    if x_values.ndim != 2 or len(y_values) != len(x_values):
        raise ValueError("y and x must contain the same number of observations.")
    if len(y_values) <= x_values.shape[1]:
        raise ValueError("OLS requires more observations than regressors.")
    if np.linalg.matrix_rank(x_values) < x_values.shape[1]:
        raise ValueError("OLS design matrix is not full rank.")

    params = np.linalg.lstsq(x_values, y_values, rcond=None)[0]
    residuals = y_values - x_values @ params
    bread = np.linalg.pinv(x_values.T @ x_values)
    score = x_values * residuals[:, None]
    meat = score.T @ score
    lag_count = min(max(int(maxlags), 0), len(score) - 1)
    for lag in range(1, lag_count + 1):
        weight = 1.0 - lag / (lag_count + 1.0)
        lag_covariance = score[lag:].T @ score[:-lag]
        meat = meat + weight * (lag_covariance + lag_covariance.T)
    covariance = bread @ meat @ bread
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    t_values = np.divide(
        params,
        standard_errors,
        out=np.full_like(params, np.nan),
        where=standard_errors > 0.0,
    )
    centered = y_values - y_values.mean()
    total_variation = float(centered @ centered)
    r_squared = (
        1.0 - float(residuals @ residuals) / total_variation
        if total_variation > 0.0
        else np.nan
    )
    return params, standard_errors, t_values, r_squared


def factor_attribution(
    portfolio_returns,
    factor_returns,
    risk_free=None,
    *,
    factor_columns=None,
    rf_column="RF",
    hac_lags=3,
    periods_per_year=12.0,
):
    """Estimate portfolio factor exposures, HAC inference, and return attribution.

    Parameters
    ----------
    portfolio_returns : pandas.DataFrame
        Portfolio returns with one portfolio per column.
    factor_returns : pandas.DataFrame
        Factor returns indexed on the same frequency as the portfolios. The
        risk-free return may be included in ``rf_column``.
    risk_free : pandas.Series, scalar, or None, optional
        Risk-free return. When omitted, ``rf_column`` is read from
        ``factor_returns``; if that column is absent, zero is used.
    factor_columns : sequence of str or None, optional
        Factor columns to include. By default all columns except ``rf_column``
        are used.
    rf_column : str, default "RF"
        Risk-free column name in ``factor_returns``.
    hac_lags : int, default 3
        Newey-West maximum lag used for coefficient inference.
    periods_per_year : float, default 12
        Annualization factor for alpha, factor contributions, and returns.

    Returns
    -------
    FactorAttributionResult
        Exposure, HAC-inference, and annualized attribution tables indexed by
        portfolio.
    """

    portfolios = pd.DataFrame(portfolio_returns).astype(float).sort_index()
    factors = pd.DataFrame(factor_returns).astype(float).sort_index()
    factor_names = list(
        factor_columns
        if factor_columns is not None
        else [column for column in factors if column != rf_column]
    )
    missing = [column for column in factor_names if column not in factors]
    if missing:
        raise KeyError(f"Missing factor columns: {missing}")
    if not factor_names:
        raise ValueError("At least one factor column is required.")

    if risk_free is None:
        risk_free_returns = (
            factors[rf_column]
            if rf_column in factors
            else pd.Series(0.0, index=factors.index, name=rf_column)
        )
    elif np.isscalar(risk_free):
        risk_free_returns = pd.Series(float(risk_free), index=factors.index, name=rf_column)
    else:
        risk_free_returns = pd.Series(risk_free, dtype=float, name=rf_column)

    common_index = portfolios.index.intersection(factors.index)
    factor_sample = pd.concat(
        [
            factors.loc[common_index, factor_names],
            risk_free_returns.reindex(common_index).rename("__risk_free"),
        ],
        axis=1,
    ).dropna()
    annual_factor_premia = factor_sample[factor_names].mean() * float(periods_per_year)
    exposure_rows = []
    inference_rows = []
    attribution_rows = []

    for portfolio_name in portfolios:
        sample = pd.concat(
            [
                portfolios[portfolio_name].rename("__portfolio_return"),
                factors[factor_names],
                risk_free_returns.rename("__risk_free"),
            ],
            axis=1,
        ).dropna()
        design = np.column_stack(
            [np.ones(len(sample)), sample[factor_names].to_numpy(dtype=float)]
        )
        if len(sample) <= design.shape[1]:
            continue
        portfolio_excess = (
            sample["__portfolio_return"] - sample["__risk_free"]
        ).to_numpy(dtype=float)
        params, _, t_values, r_squared = _ols_hac(
            portfolio_excess,
            design,
            maxlags=hac_lags,
        )
        coefficients = pd.Series(params, index=["const", *factor_names])
        coefficient_t = pd.Series(t_values, index=["const", *factor_names])
        contributions = coefficients[factor_names] * annual_factor_premia
        annual_alpha = coefficients["const"] * float(periods_per_year)
        exposure_rows.append({
            "portfolio": portfolio_name,
            "months": len(sample),
            "annual_alpha": annual_alpha,
            "alpha_hac_t": coefficient_t["const"],
            "r_squared": r_squared,
            **{factor: coefficients[factor] for factor in factor_names},
        })
        inference_rows.append({
            "portfolio": portfolio_name,
            "alpha_hac_t": coefficient_t["const"],
            **{f"{factor}_hac_t": coefficient_t[factor] for factor in factor_names},
        })
        attribution_rows.append({
            "portfolio": portfolio_name,
            "alpha": annual_alpha,
            **{factor: contributions[factor] for factor in factor_names},
            "fitted_excess_return": annual_alpha + contributions.sum(),
            "realized_excess_return": portfolio_excess.mean() * float(periods_per_year),
        })

    exposure_columns = [
        "months", "annual_alpha", "alpha_hac_t", "r_squared", *factor_names,
    ]
    inference_columns = [
        "alpha_hac_t", *[f"{factor}_hac_t" for factor in factor_names],
    ]
    attribution_columns = [
        "alpha", *factor_names, "fitted_excess_return", "realized_excess_return",
    ]

    def result_table(rows, columns):
        if not rows:
            return pd.DataFrame(columns=columns).rename_axis("portfolio")
        return pd.DataFrame(rows).set_index("portfolio").reindex(columns=columns)

    return FactorAttributionResult(
        exposures=result_table(exposure_rows, exposure_columns),
        inference=result_table(inference_rows, inference_columns),
        attribution=result_table(attribution_rows, attribution_columns),
    )


def point_in_time_market_return(
    returns,
    issuers,
    date_map,
    *,
    date_column="decision_date",
    execution_column="execution_date",
    asset_column="ticker",
):
    """Build an equal-weight market return using point-in-time membership."""

    return_panel = pd.DataFrame(returns).astype(float).sort_index()
    issuer_frame = pd.DataFrame(issuers)
    mapping = pd.DataFrame(date_map).sort_values(date_column).reset_index(drop=True)
    members = {
        pd.Timestamp(date): group[asset_column].drop_duplicates().tolist()
        for date, group in issuer_frame.groupby(date_column, sort=True)
    }
    market_return = pd.Series(
        np.nan,
        index=return_panel.index,
        name="eligible_market_return",
    )
    for position, row in mapping.iterrows():
        start = pd.Timestamp(row[execution_column])
        stop = (
            pd.Timestamp(mapping.loc[position + 1, execution_column])
            if position + 1 < len(mapping)
            else return_panel.index.max() + pd.Timedelta(days=1)
        )
        assets = return_panel.columns.intersection(
            members.get(pd.Timestamp(row[date_column]), [])
        )
        dates = return_panel.index[
            (return_panel.index >= start) & (return_panel.index < stop)
        ]
        if len(dates) and len(assets):
            market_return.loc[dates] = return_panel.loc[dates, assets].mean(axis=1)
    return market_return


def fama_macbeth(
    data,
    *,
    date_column,
    return_column,
    characteristics,
    industry_column=None,
    start=None,
    min_cross_section=120,
    min_industry_size=5,
    winsor_limits=(0.01, 0.99),
    minimum_residual_df=20,
    hac_lags=3,
    periods_per_year=12.0,
):
    """Run monthly Fama-MacBeth cross-sectional regressions with HAC inference.

    Each monthly regression winsorizes returns and characteristics, standardizes
    the characteristics cross-sectionally, and optionally includes industry
    fixed effects. The time-series mean of each characteristic slope is then
    reported with Newey-West inference.

    Parameters
    ----------
    data : pandas.DataFrame
        Long-form stock panel.
    date_column : str
        Decision-date column.
    return_column : str
        One-period-ahead return column.
    characteristics : sequence of str
        Cross-sectional characteristics whose premia are estimated.
    industry_column : str or None, optional
        Categorical industry control. Rare groups are pooled into ``Other``.
    start : date-like or None, optional
        Earliest decision date included.
    min_cross_section : int, default 120
        Minimum complete observations required before fitting a month.
    min_industry_size : int, default 5
        Minimum observations retained as a separate industry category.
    winsor_limits : tuple of float or None, default (0.01, 0.99)
        Cross-sectional clipping quantiles for returns and characteristics.
    minimum_residual_df : int, default 20
        Required observations beyond the number of regression columns.
    hac_lags : int, default 3
        Newey-West maximum lag for the time-series mean slopes.
    periods_per_year : float, default 12
        Annualization factor for premia and standard errors.

    Returns
    -------
    FamaMacBethResult
        Annualized characteristic-premium summary, monthly coefficients, and
        monthly regression diagnostics.
    """

    frame = pd.DataFrame(data).copy()
    characteristic_names = list(characteristics)
    required = [date_column, return_column, *characteristic_names]
    if industry_column is not None:
        required.append(industry_column)
    missing = [column for column in required if column not in frame]
    if missing:
        raise KeyError(f"Missing Fama-MacBeth columns: {missing}")
    if not characteristic_names:
        raise ValueError("At least one characteristic is required.")

    frame[date_column] = pd.to_datetime(frame[date_column])
    if start is not None:
        frame = frame[frame[date_column].ge(pd.Timestamp(start))]

    monthly_rows = []
    diagnostic_rows = []
    for decision_date, month in frame.groupby(date_column, sort=True):
        cross_section = month[required[1:]].dropna().copy()
        if len(cross_section) < int(min_cross_section):
            continue
        numeric_columns = [return_column, *characteristic_names]
        cross_section[numeric_columns] = cross_section[numeric_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        cross_section = cross_section.dropna(subset=numeric_columns)
        if winsor_limits is not None:
            lower_q, upper_q = winsor_limits
            for column in numeric_columns:
                lower, upper = cross_section[column].quantile([lower_q, upper_q])
                cross_section[column] = cross_section[column].clip(lower, upper)

        standardized = cross_section[characteristic_names]
        standardized = standardized.subtract(standardized.mean()).divide(
            standardized.std(ddof=0).replace(0.0, np.nan)
        )
        controls = pd.DataFrame(index=cross_section.index)
        if industry_column is not None:
            industry_counts = cross_section[industry_column].value_counts()
            industry_group = cross_section[industry_column].where(
                cross_section[industry_column].map(industry_counts).ge(
                    int(min_industry_size)
                ),
                "Other",
            )
            controls = pd.get_dummies(
                industry_group,
                prefix="industry",
                drop_first=True,
                dtype=float,
            )
        regression = pd.concat(
            [cross_section[[return_column]], standardized, controls],
            axis=1,
        ).dropna()
        design_columns = [*characteristic_names, *controls.columns]
        design = np.column_stack(
            [np.ones(len(regression)), regression[design_columns].to_numpy(dtype=float)]
        )
        design_rank = np.linalg.matrix_rank(design)
        if (
            len(regression) <= design.shape[1] + int(minimum_residual_df)
            or design_rank < design.shape[1]
        ):
            continue
        coefficients = np.linalg.lstsq(
            design,
            regression[return_column].to_numpy(dtype=float),
            rcond=None,
        )[0]
        monthly_rows.append(
            pd.Series(
                coefficients,
                index=["const", *design_columns],
                name=pd.Timestamp(decision_date),
            )
        )
        diagnostic_rows.append({
            date_column: pd.Timestamp(decision_date),
            "cross_section": len(regression),
            "industry_controls": len(controls.columns),
            "design_rank": design_rank,
            "design_columns": design.shape[1],
        })

    monthly_coefficients = pd.DataFrame(monthly_rows).sort_index()
    diagnostics = (
        pd.DataFrame(diagnostic_rows).set_index(date_column).sort_index()
        if diagnostic_rows
        else pd.DataFrame(
            columns=[
                "cross_section", "industry_controls", "design_rank", "design_columns",
            ]
        ).rename_axis(date_column)
    )
    summary_rows = []
    for characteristic in characteristic_names:
        slopes = monthly_coefficients.get(
            characteristic,
            pd.Series(dtype=float),
        ).dropna()
        if len(slopes) <= 1:
            continue
        params, standard_errors, t_values, _ = _ols_hac(
            slopes.to_numpy(dtype=float),
            np.ones((len(slopes), 1)),
            maxlags=hac_lags,
        )
        annualized_premium = params[0] * float(periods_per_year)
        annualized_standard_error = standard_errors[0] * float(periods_per_year)
        summary_rows.append({
            "characteristic": characteristic,
            "annualized_premium": annualized_premium,
            "annualized_hac_se": annualized_standard_error,
            "hac_t": t_values[0],
            "ci_95_low": annualized_premium - 1.96 * annualized_standard_error,
            "ci_95_high": annualized_premium + 1.96 * annualized_standard_error,
            "positive_months": slopes.gt(0.0).mean(),
            "months": len(slopes),
            "average_cross_section": diagnostics["cross_section"].mean(),
            "average_industry_controls": diagnostics["industry_controls"].mean(),
        })
    summary_columns = [
        "annualized_premium", "annualized_hac_se", "hac_t", "ci_95_low",
        "ci_95_high", "positive_months", "months", "average_cross_section",
        "average_industry_controls",
    ]
    summary = (
        pd.DataFrame(summary_rows)
        .set_index("characteristic")
        .reindex(columns=summary_columns)
        if summary_rows
        else pd.DataFrame(columns=summary_columns).rename_axis("characteristic")
    )
    return FamaMacBethResult(
        summary=summary,
        monthly_coefficients=monthly_coefficients,
        diagnostics=diagnostics,
    )


def rolling_factor_fit(y, x, window=60):
    y = pd.DataFrame(y).astype(float)
    x = pd.DataFrame(x).astype(float)
    idx = y.index.intersection(x.index)
    y = y.loc[idx]
    x = x.loc[idx]

    assets = list(y.columns)
    factors = list(x.columns)
    beta_cols = pd.MultiIndex.from_product([assets, factors], names=["asset", "factor"])

    alpha = pd.DataFrame(np.nan, index=idx, columns=assets)
    beta = pd.DataFrame(np.nan, index=idx, columns=beta_cols)
    r2 = pd.DataFrame(np.nan, index=idx, columns=assets)
    eps = pd.DataFrame(np.nan, index=idx, columns=assets)

    for i in range(int(window) - 1, len(idx)):
        x_win = x.iloc[i - int(window) + 1 : i + 1]
        X_full = np.column_stack([np.ones(len(x_win)), x_win.to_numpy(dtype=float)])
        for asset in assets:
            y_win = y[asset].iloc[i - int(window) + 1 : i + 1]
            mask = np.isfinite(y_win.to_numpy(dtype=float)) & np.isfinite(X_full).all(axis=1)
            if mask.sum() < len(factors) + 3:
                continue
            X = X_full[mask]
            yy = y_win.to_numpy(dtype=float)[mask]
            coef = np.linalg.lstsq(X, yy, rcond=None)[0]
            fitted = X @ coef
            resid = yy - fitted
            sse = float(np.sum(resid**2))
            sst = float(np.sum((yy - yy.mean()) ** 2))
            date = idx[i]
            alpha.loc[date, asset] = coef[0]
            beta.loc[date, pd.IndexSlice[asset, :]] = coef[1:]
            r2.loc[date, asset] = 1.0 - sse / sst if sst > 0 else np.nan
            row_x = X_full[-1]
            if np.isfinite(row_x).all() and np.isfinite(y[asset].iloc[i]):
                eps.loc[date, asset] = float(y[asset].iloc[i] - row_x @ coef)

    return alpha, beta, r2, eps


def factor_state(factors, short_window=6, long_window=12, vol_window=36, skip=1, clip=2.0):
    """Build standardized factor timing states from factor returns.

    The function combines short-horizon and skip-window long-horizon factor
    momentum, scales the result by rolling volatility, standardizes it using a
    rolling z-score, and clips extreme values.

    Parameters
    ----------
    factors : pandas.DataFrame or array-like
        Factor return panel.
    short_window : int, default=6
        Rolling window used for short-horizon cumulative return.
    long_window : int, default=12
        Rolling window used for longer-horizon cumulative return.
    vol_window : int, default=36
        Rolling window used for volatility scaling and z-score standardization.
    skip : int, default=1
        Number of periods skipped for the long-horizon momentum leg.
    clip : float, default=2.0
        Absolute clipping bound applied to standardized states.

    Returns
    -------
    pandas.DataFrame
        Standardized and clipped factor-state panel.

    Notes
    -----
    The function is designed for monthly factor data. The volatility scaling uses
    ``sqrt(12)`` internally.
    """

    f = pd.DataFrame(factors).astype(float)
    short = (1.0 + f).rolling(int(short_window)).apply(np.prod, raw=True) - 1.0
    long_len = max(int(long_window) - int(skip), 1)
    long = (1.0 + f.shift(int(skip))).rolling(long_len).apply(np.prod, raw=True) - 1.0
    vol = f.rolling(int(vol_window)).std().replace(0.0, np.nan) * np.sqrt(12.0)
    raw = (0.45 * short + 0.55 * long) / vol
    mean = raw.rolling(int(vol_window)).mean()
    sd = raw.rolling(int(vol_window)).std().replace(0.0, np.nan)
    return ((raw - mean) / sd).clip(-float(clip), float(clip))


def factor_proxy_spreads(returns, benchmark_ticker="SPY", cash_ticker="SHY"):
    r = pd.DataFrame(returns).astype(float)
    out = {}
    if benchmark_ticker in r.columns and cash_ticker in r.columns:
        out["Mkt-RF"] = r[benchmark_ticker] - r[cash_ticker]
    elif benchmark_ticker in r.columns:
        out["Mkt-RF"] = r[benchmark_ticker]
    if "IWM" in r.columns and benchmark_ticker in r.columns:
        out["SMB"] = r["IWM"] - r[benchmark_ticker]
    elif "RSP" in r.columns and benchmark_ticker in r.columns:
        out["SMB"] = r["RSP"] - r[benchmark_ticker]
    if "IWD" in r.columns and "IWF" in r.columns:
        out["HML"] = r["IWD"] - r["IWF"]
    elif "VLUE" in r.columns and benchmark_ticker in r.columns:
        out["HML"] = r["VLUE"] - r[benchmark_ticker]
    if "QUAL" in r.columns and benchmark_ticker in r.columns:
        out["RMW"] = r["QUAL"] - r[benchmark_ticker]
    if "USMV" in r.columns and benchmark_ticker in r.columns:
        out["CMA"] = r["USMV"] - r[benchmark_ticker]
    if "MTUM" in r.columns and benchmark_ticker in r.columns:
        out["MOM"] = r["MTUM"] - r[benchmark_ticker]
    return pd.DataFrame(out).sort_index()


def blend_factor_states(academic_state, tradable_state, academic_weight=0.70, disagreement_scale=0.50):
    """Blend academic and tradable factor-state estimates.

    The function forms a weighted average of two aligned state panels and
    haircuts the blended signal when the two sources disagree in sign.

    Parameters
    ----------
    academic_state : pandas.DataFrame
        Factor-state panel from academic factor returns.
    tradable_state : pandas.DataFrame
        Factor-state panel from tradable proxy returns.
    academic_weight : float, default=0.70
        Weight assigned to the academic state.
    disagreement_scale : float, default=0.50
        Multiplicative haircut applied where academic and tradable signs
        disagree.

    Returns
    -------
    pandas.DataFrame
        Blended factor-state panel indexed and columned like the academic input
        where possible.

    Notes
    -----
    Only overlapping dates and columns are blended. Non-overlapping entries
    retain the academic-state values.
    """

    a = pd.DataFrame(academic_state).astype(float)
    t = pd.DataFrame(tradable_state).astype(float)
    idx = a.index.intersection(t.index)
    out = a.copy()
    cols = a.columns.intersection(t.columns)
    if len(idx) == 0 or len(cols) == 0:
        return out
    aw = float(academic_weight)
    blended = aw * a.loc[idx, cols] + (1.0 - aw) * t.loc[idx, cols]
    agree = np.sign(a.loc[idx, cols]) * np.sign(t.loc[idx, cols]) >= 0
    out.loc[idx, cols] = blended.where(agree, blended * float(disagreement_scale))
    return out


def cross_section_z(values):
    v = pd.Series(values, dtype=float) if isinstance(values, pd.Series) else pd.DataFrame(values).astype(float)
    mean = v.mean(axis=1) if isinstance(v, pd.DataFrame) else v.mean()
    sd = v.std(axis=1, ddof=0) if isinstance(v, pd.DataFrame) else v.std(ddof=0)
    if isinstance(v, pd.Series):
        return (v - mean) / sd if sd and np.isfinite(sd) else v * np.nan
    return v.subtract(mean, axis=0).divide(sd.replace(0.0, np.nan), axis=0)


def factor_scores(beta, z_factor):
    """Project factor states through asset factor betas.

    For each date, asset-level scores are computed as the dot product between
    asset factor exposures and current factor states.

    Parameters
    ----------
    beta : pandas.DataFrame
        Factor beta panel indexed by date. Columns should be a MultiIndex with
        levels that include asset and factor identifiers.
    z_factor : pandas.DataFrame
        Factor-state panel indexed by date with factor columns.

    Returns
    -------
    pandas.DataFrame
        Asset score panel with dates in rows and assets in columns.

    Notes
    -----
    Only factors present in both the beta slice and the factor-state table are
    used on each date. Missing dates or factors produce missing scores.
    """

    z = pd.DataFrame(z_factor).astype(float)
    dates = beta.index.intersection(z.index)
    assets = beta.columns.get_level_values("asset").unique()
    out = pd.DataFrame(np.nan, index=dates, columns=assets)
    for date in dates:
        b = beta.loc[date].unstack("factor")
        factors = b.columns.intersection(z.columns)
        if len(factors) == 0:
            continue
        out.loc[date, b.index] = b[factors].dot(z.loc[date, factors])
    return out


def residual_strength(eps, window=6):
    return pd.DataFrame(eps).astype(float).rolling(int(window)).sum()


def trend_strength(returns, window=12, skip=1):
    r = pd.DataFrame(returns).astype(float)
    length = max(int(window) - int(skip), 1)
    return (1.0 + r.shift(int(skip))).rolling(length).apply(np.prod, raw=True) - 1.0


def risk_penalty(returns, vol_window=36, drawdown_window=12):
    r = pd.DataFrame(returns).astype(float)
    vol = r.rolling(int(vol_window)).std() * np.sqrt(12.0)
    nav = (1.0 + r.fillna(0.0)).cumprod()
    peak = nav.rolling(int(drawdown_window), min_periods=int(drawdown_window)).max()
    drawdown = (nav / peak - 1.0).clip(upper=0.0)
    return vol + drawdown.abs()


def combine_scores(scores, weights):
    frames = {name: pd.DataFrame(value).astype(float) for name, value in scores.items()}
    if not frames:
        return pd.DataFrame()
    idx = None
    cols = None
    for frame in frames.values():
        idx = frame.index if idx is None else idx.intersection(frame.index)
        cols = frame.columns if cols is None else cols.intersection(frame.columns)
    out = pd.DataFrame(0.0, index=idx, columns=cols)
    for name, frame in frames.items():
        out = out + float(weights.get(name, 0.0)) * frame.reindex(index=idx, columns=cols)
    return out


def equal_weight_schedule(returns):
    r = pd.DataFrame(returns)
    out = pd.DataFrame(0.0, index=r.index, columns=r.columns)
    if len(r.columns):
        out.loc[:, :] = 1.0 / len(r.columns)
    return out


def benchmark_weight_schedule(returns, benchmark_ticker):
    r = pd.DataFrame(returns)
    out = pd.DataFrame(0.0, index=r.index, columns=r.columns)
    if benchmark_ticker not in out.columns:
        out[benchmark_ticker] = 0.0
    out.loc[:, benchmark_ticker] = 1.0
    return out


def top_score_weights(scores, vol, cash_returns, top_n=4, max_weight=0.35, cash_ticker="SHY"):
    """Convert positive cross-sectional scores into top-N inverse-volatility weights.

    At each date, the function selects the highest positive scores, scales them
    by inverse volatility, caps individual weights, and allocates residual
    capital to a cash ticker.

    Parameters
    ----------
    scores : pandas.DataFrame
        Asset score panel.
    vol : pandas.DataFrame
        Asset volatility panel aligned to ``scores``.
    cash_returns : pandas.Series
        Cash return series used to define the date index when available.
    top_n : int, default=4
        Maximum number of risky assets selected per date.
    max_weight : float, default=0.35
        Maximum risky-asset weight before residual cash allocation.
    cash_ticker : str, default="SHY"
        Cash or defensive asset column receiving residual capital.

    Returns
    -------
    pandas.DataFrame
        Weight panel with asset columns plus the cash ticker.

    Notes
    -----
    If no positive score is available on a date, the allocation is fully assigned
    to the cash ticker.
    """

    s = pd.DataFrame(scores).astype(float)
    v = pd.DataFrame(vol).astype(float).reindex(index=s.index, columns=s.columns)
    cash = pd.Series(cash_returns, dtype=float)
    dates = s.index.intersection(cash.index) if len(cash.index) else s.index
    cols = list(s.columns)
    out = pd.DataFrame(0.0, index=dates, columns=[*cols, cash_ticker])

    for date in dates:
        row = s.loc[date].dropna().sort_values(ascending=False)
        row = row[row > 0].head(int(top_n))
        if row.empty:
            out.loc[date, cash_ticker] = 1.0
            continue
        inv_vol = 1.0 / v.loc[date, row.index].replace(0.0, np.nan)
        raw = (row * inv_vol).replace([np.inf, -np.inf], np.nan).dropna()
        raw = raw[raw > 0]
        if raw.empty:
            out.loc[date, cash_ticker] = 1.0
            continue
        weights = (raw / raw.sum()).clip(upper=float(max_weight))
        out.loc[date, weights.index] = weights
        out.loc[date, cash_ticker] = max(0.0, 1.0 - float(weights.sum()))
    return out


def soft_active_weights(
    scores,
    returns,
    cash_returns=None,
    risk_score=None,
    active_budget=0.35,
    min_weight=0.02,
    max_weight=0.25,
    turnover_limit=0.25,
    cash_ticker="SHY",
    cash_weight_max=0.25,
    risk_window=36,
    risk_quantile=0.70,
):
    s = pd.DataFrame(scores).astype(float)
    r = pd.DataFrame(returns).astype(float)
    assets = list(s.columns.intersection(r.columns))
    idx = s.index.intersection(r.index)
    cash = pd.Series(cash_returns, dtype=float) if cash_returns is not None else None
    if cash is not None:
        idx = idx.intersection(cash.index)
    out_cols = [*assets, cash_ticker] if cash is not None else assets
    out = pd.DataFrame(0.0, index=idx, columns=out_cols)
    if not assets:
        return out

    base = pd.Series(1.0 / len(assets), index=assets, dtype=float)
    prev = None
    broad = None
    threshold = None
    if risk_score is not None:
        risk = pd.DataFrame(risk_score).astype(float)
        broad = risk.reindex(index=idx, columns=assets).mean(axis=1)
        threshold = broad.rolling(int(risk_window), min_periods=max(12, int(risk_window) // 2)).quantile(float(risk_quantile))

    for date in idx:
        row = s.loc[date, assets].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        row = row - row.mean()
        denom = float(row.abs().sum())
        if denom > 1e-12:
            target = base + float(active_budget) * row / denom
            target = target.clip(lower=float(min_weight), upper=float(max_weight))
            target = target / float(target.sum())
        else:
            target = base.copy()

        cash_weight = 0.0
        if broad is not None and threshold is not None:
            b = broad.loc[date]
            th = threshold.loc[date]
            if np.isfinite(b) and np.isfinite(th) and b > th:
                cash_weight = float(cash_weight_max)
        if cash is not None:
            target = target * (1.0 - cash_weight)
            target = pd.concat([target, pd.Series({cash_ticker: cash_weight})])

        if prev is not None and float(turnover_limit) > 0:
            target = target.reindex(out_cols).fillna(0.0)
            delta = target - prev.reindex(out_cols).fillna(0.0)
            turnover = 0.5 * float(delta.abs().sum())
            if turnover > float(turnover_limit):
                scale = float(turnover_limit) / turnover
                target = prev.reindex(out_cols).fillna(0.0) + scale * delta

        target = target.reindex(out_cols).fillna(0.0)
        total = float(target.sum())
        if total > 1e-12:
            target = target / total
        out.loc[date] = target
        prev = target
    return out


def _rank_ic_series(score, future_returns):
    vals = []
    for date in score.index.intersection(future_returns.index):
        a = score.loc[date]
        b = future_returns.loc[date].reindex(a.index)
        valid = a.notna() & b.notna()
        if valid.sum() < 3:
            vals.append((date, np.nan))
        else:
            vals.append((date, a[valid].rank().corr(b[valid].rank())))
    return pd.Series(dict(vals), name="rank_ic").sort_index()


def _top_active_series(score, future_returns, top_n=4):
    score = pd.DataFrame(score).astype(float)
    future = pd.DataFrame(future_returns).astype(float)
    vals = []
    for date in score.index.intersection(future.index):
        s = score.loc[date].dropna().sort_values(ascending=False)
        r = future.loc[date].reindex(s.index)
        if len(s) < int(top_n):
            vals.append((date, np.nan))
            continue
        top = r.reindex(s.head(int(top_n)).index).mean()
        avg = r.mean()
        vals.append((date, float(top - avg) if np.isfinite(top) and np.isfinite(avg) else np.nan))
    return pd.Series(dict(vals), name="top_active").sort_index()


def _top_return_series(score, future_returns, top_n=4):
    score = pd.DataFrame(score).astype(float)
    future = pd.DataFrame(future_returns).astype(float)
    vals = []
    for date in score.index.intersection(future.index):
        s = score.loc[date].dropna().sort_values(ascending=False)
        r = future.loc[date].reindex(s.index)
        if len(s) < int(top_n):
            vals.append((date, np.nan))
            continue
        top = r.reindex(s.head(int(top_n)).index).mean()
        vals.append((date, float(top) if np.isfinite(top) else np.nan))
    return pd.Series(dict(vals), name="top_return").sort_index()


def _max_drawdown_from_returns(returns):
    r = pd.Series(returns, dtype=float).dropna()
    if r.empty:
        return np.nan
    nav = (1.0 + r).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


def _component_weights(raw_weights, base_weights=None, validation_strength=1.0, max_component_weights=None):
    vals = pd.Series(raw_weights, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    if float(vals.sum()) <= 1e-12:
        vals.loc[:] = 1.0 / len(vals)
    else:
        vals = vals / float(vals.sum())
    if base_weights is not None:
        base = pd.Series(base_weights, dtype=float).reindex(vals.index).fillna(0.0).clip(lower=0.0)
        if float(base.sum()) > 1e-12:
            base = base / float(base.sum())
            strength = float(np.clip(validation_strength, 0.0, 1.0))
            vals = (1.0 - strength) * base + strength * vals
            vals = vals / float(vals.sum())
    if max_component_weights is not None:
        caps = pd.Series(max_component_weights, dtype=float).reindex(vals.index).fillna(1.0)
        caps = caps.clip(lower=0.0, upper=1.0)
        for _ in range(len(vals) + 2):
            vals = vals.clip(upper=caps)
            deficit = 1.0 - float(vals.sum())
            if deficit <= 1e-12:
                break
            room = (caps - vals).clip(lower=0.0)
            if float(room.sum()) <= 1e-12:
                break
            vals = vals + deficit * room / float(room.sum())
        if float(vals.sum()) > 1e-12:
            vals = vals / float(vals.sum())
    return vals.to_dict()


def validation_weighted_score(
    scores,
    future_returns,
    risk=None,
    window=72,
    min_periods=36,
    top_n=4,
    risk_weight=0.20,
    base_weights=None,
    validation_strength=1.0,
    max_component_weights=None,
):
    """Blend multiple score panels using rolling validation evidence.

    The function evaluates each score component over a historical validation
    window using rank information coefficient and top-N active return evidence,
    then forms a date-specific weighted blend. An optional risk score can be
    subtracted when it improves historical active performance or drawdown.

    Parameters
    ----------
    scores : mapping
        Mapping from component name to score panel.
    future_returns : pandas.DataFrame
        Forward return panel used for validation.
    risk : pandas.DataFrame, optional
        Risk penalty panel aligned with the score panels.
    window : int, default=72
        Number of historical periods used for validation.
    min_periods : int, default=36
        Minimum number of validation observations required before producing
        scores.
    top_n : int, default=4
        Number of top-ranked assets used when evaluating active return.
    risk_weight : float, default=0.20
        Candidate risk penalty weight.
    base_weights : mapping, optional
        Baseline component weights blended with validation weights.
    validation_strength : float, default=1.0
        Strength of validation-driven reweighting.
    max_component_weights : mapping, optional
        Component-specific maximum weights.

    Returns
    -------
    scores : pandas.DataFrame
        Validation-weighted score panel.
    weights : pandas.DataFrame
        Component weight panel, including the realized risk-penalty weight.

    Notes
    -----
    The function does not look ahead from each decision date; it uses only prior
    validation observations when constructing the current blend.
    """

    frames = {name: pd.DataFrame(value).astype(float) for name, value in scores.items()}
    future = pd.DataFrame(future_returns).astype(float)
    idx = future.index
    cols = future.columns
    for frame in frames.values():
        idx = idx.intersection(frame.index)
        cols = cols.intersection(frame.columns)
    if risk is not None:
        risk_frame = pd.DataFrame(risk).astype(float)
        idx = idx.intersection(risk_frame.index)
        cols = cols.intersection(risk_frame.columns)
    else:
        risk_frame = None

    idx = pd.DatetimeIndex(idx).sort_values()
    cols = pd.Index(cols)
    out = pd.DataFrame(np.nan, index=idx, columns=cols)
    weight_cols = [*frames.keys(), "risk"]
    weights = pd.DataFrame(np.nan, index=idx, columns=weight_cols)

    ic = {name: _rank_ic_series(frame.reindex(index=idx, columns=cols), future.reindex(index=idx, columns=cols)) for name, frame in frames.items()}
    active = {name: _top_active_series(frame.reindex(index=idx, columns=cols), future.reindex(index=idx, columns=cols), top_n=top_n) for name, frame in frames.items()}

    for i, date in enumerate(idx):
        hist = idx[max(0, i - int(window)) : i]
        if len(hist) < int(min_periods):
            continue
        edges = {}
        for name in frames:
            ic_mean = ic[name].reindex(hist).mean()
            active_mean = active[name].reindex(hist).mean()
            edges[name] = max(0.0, float(ic_mean) if np.isfinite(ic_mean) else 0.0) * max(0.0, float(active_mean) if np.isfinite(active_mean) else 0.0)
        total_edge = float(sum(edges.values()))
        raw_weights = (
            {name: 1.0 / len(frames) for name in frames}
            if total_edge <= 1e-12
            else {name: edge / total_edge for name, edge in edges.items()}
        )
        score_weights = _component_weights(
            raw_weights,
            base_weights=base_weights,
            validation_strength=validation_strength,
            max_component_weights=max_component_weights,
        )

        base_score = pd.Series(0.0, index=cols, dtype=float)
        hist_score = pd.DataFrame(0.0, index=hist, columns=cols)
        for name, frame in frames.items():
            w = float(score_weights[name])
            base_score = base_score + w * frame.loc[date, cols]
            hist_score = hist_score + w * frame.loc[hist, cols]

        risk_use = 0.0
        if risk_frame is not None:
            candidate_hist = hist_score - float(risk_weight) * risk_frame.loc[hist, cols]
            base_active = _top_active_series(hist_score, future.reindex(index=hist, columns=cols), top_n=top_n).mean()
            candidate_active = _top_active_series(candidate_hist, future.reindex(index=hist, columns=cols), top_n=top_n).mean()
            base_dd = _max_drawdown_from_returns(_top_return_series(hist_score, future.reindex(index=hist, columns=cols), top_n=top_n))
            candidate_dd = _max_drawdown_from_returns(_top_return_series(candidate_hist, future.reindex(index=hist, columns=cols), top_n=top_n))
            active_ok = np.isfinite(candidate_active) and np.isfinite(base_active) and candidate_active >= base_active
            drawdown_ok = np.isfinite(candidate_dd) and np.isfinite(base_dd) and candidate_dd >= base_dd
            if active_ok or drawdown_ok:
                risk_use = float(risk_weight)
                base_score = base_score - risk_use * risk_frame.loc[date, cols]

        out.loc[date] = base_score
        for name, w in score_weights.items():
            weights.loc[date, name] = w
        weights.loc[date, "risk"] = risk_use
    return out, weights


def rank_ic_table(scores, future_returns):
    """Compute rank information-coefficient diagnostics for one or more signals.

    Parameters
    ----------
    scores : pandas.DataFrame or mapping
        Score panel, or mapping from signal name to score panel.
    future_returns : pandas.DataFrame
        Forward return panel aligned with the scores.

    Returns
    -------
    pandas.DataFrame
        Signal-indexed table containing average rank IC, t-statistic, positive
        IC hit rate, and observation count.

    Notes
    -----
    Rank IC is computed cross-sectionally by date and then summarized through
    time.
    """

    score_map = scores if isinstance(scores, Mapping) else {"score": scores}
    future = pd.DataFrame(future_returns).astype(float)
    rows = []
    for name, score in score_map.items():
        ic = _rank_ic_series(pd.DataFrame(score).astype(float), future).dropna()
        rows.append(
            {
                "signal": str(name),
                "rank_ic": float(ic.mean()) if len(ic) else np.nan,
                "rank_ic_t": float(ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic))) if len(ic) > 2 and ic.std(ddof=1) > 0 else np.nan,
                "hit_rate": float((ic > 0).mean()) if len(ic) else np.nan,
                "n": len(ic),
            }
        )
    return pd.DataFrame(rows).set_index("signal")


def top_bottom_table(scores, future_returns, top_n=4, bottom_n=4):
    score_map = scores if isinstance(scores, Mapping) else {"score": scores}
    future = pd.DataFrame(future_returns).astype(float)
    rows = []
    for name, score in score_map.items():
        score = pd.DataFrame(score).astype(float)
        spreads = []
        tops = []
        bottoms = []
        for date in score.index.intersection(future.index):
            s = score.loc[date].dropna().sort_values(ascending=False)
            r = future.loc[date].reindex(s.index)
            if len(s) < max(int(top_n), int(bottom_n)):
                continue
            top = r.reindex(s.head(int(top_n)).index).mean()
            bottom = r.reindex(s.tail(int(bottom_n)).index).mean()
            if np.isfinite(top) and np.isfinite(bottom):
                tops.append(float(top))
                bottoms.append(float(bottom))
                spreads.append(float(top - bottom))
        spreads_s = pd.Series(spreads, dtype=float)
        rows.append(
            {
                "signal": str(name),
                "top": float(np.mean(tops)) if tops else np.nan,
                "bottom": float(np.mean(bottoms)) if bottoms else np.nan,
                "top_minus_bottom": float(spreads_s.mean()) if len(spreads_s) else np.nan,
                "hit_rate": float((spreads_s > 0).mean()) if len(spreads_s) else np.nan,
                "n": len(spreads_s),
            }
        )
    return pd.DataFrame(rows).set_index("signal")


def signal_decay_table(scores, returns, horizons=(1, 3, 6), top_n=4):
    score_map = scores if isinstance(scores, Mapping) else {"score": scores}
    r = pd.DataFrame(returns).astype(float)
    rows = []
    for h in horizons:
        future = (1.0 + r).rolling(int(h)).apply(np.prod, raw=True).shift(-int(h)) - 1.0
        universe = future.mean(axis=1)
        for name, score in score_map.items():
            score = pd.DataFrame(score).astype(float)
            vals = []
            avg = []
            for date in score.index.intersection(future.index):
                s = score.loc[date].dropna().sort_values(ascending=False)
                if len(s) < int(top_n):
                    continue
                ret = future.loc[date, s.head(int(top_n)).index].mean()
                base = universe.loc[date]
                if np.isfinite(ret) and np.isfinite(base):
                    vals.append(float(ret))
                    avg.append(float(base))
            vals_s = pd.Series(vals, dtype=float)
            avg_s = pd.Series(avg, dtype=float)
            active = vals_s - avg_s
            rows.append(
                {
                    "signal": str(name),
                    "horizon": int(h),
                    "top_return": float(vals_s.mean()) if len(vals_s) else np.nan,
                    "universe_return": float(avg_s.mean()) if len(avg_s) else np.nan,
                    "active_return": float(active.mean()) if len(active) else np.nan,
                    "hit_rate": float((active > 0).mean()) if len(active) else np.nan,
                    "n": len(active),
                }
            )
    return pd.DataFrame(rows).set_index(["signal", "horizon"])


def portfolio_factor_exposure(weights, beta):
    W = pd.DataFrame(weights).astype(float)
    assets = beta.columns.get_level_values("asset").unique()
    factors = beta.columns.get_level_values("factor").unique()
    out = pd.DataFrame(np.nan, index=W.index, columns=factors)
    for date in W.index:
        b_hist = beta.loc[:date]
        if b_hist.empty:
            continue
        b = b_hist.iloc[-1].unstack("factor")
        w = W.loc[date].reindex(assets).fillna(0.0)
        common = b.index.intersection(w.index)
        if len(common):
            out.loc[date] = w.reindex(common).dot(b.reindex(common))
    return out


__all__ = [
    "FactorAttributionResult",
    "FamaMacBethResult",
    "benchmark_weight_schedule",
    "blend_factor_states",
    "combine_scores",
    "cross_section_z",
    "equal_weight_schedule",
    "excess_returns",
    "factor_attribution",
    "factor_proxy_spreads",
    "factor_scores",
    "factor_state",
    "fama_macbeth",
    "month_end_prices",
    "point_in_time_market_return",
    "portfolio_factor_exposure",
    "rank_ic_table",
    "residual_strength",
    "risk_penalty",
    "rolling_factor_fit",
    "signal_decay_table",
    "soft_active_weights",
    "top_bottom_table",
    "top_score_weights",
    "trend_strength",
    "validation_weighted_score",
]
