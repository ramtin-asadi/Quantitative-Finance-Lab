"""Credit figures; functions respect the caller's style and never fit models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


def _axis(ax):
    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
    return ax


def plot_default_capture(data, target, models, *, ax=None, title="Out-of-sample bankruptcy capture"):
    """Cumulative event capture as issuer observations are reviewed in risk order."""
    ax = _axis(ax)
    for label, column in models.items():
        x = data[[target, column]].dropna().sort_values(column, ascending=False)
        ax.plot(np.arange(1, len(x) + 1) / len(x), x[target].cumsum() / x[target].sum(), label=label)
    ax.plot([0, 1], [0, 1], color="0.45", linestyle="--", linewidth=1, label="Random")
    ax.set(xlim=(0, 0.5), ylim=(0, 1.02), title=title,
           xlabel="Share of observations reviewed", ylabel="Share of events captured")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="lower right")
    return ax


def plot_pd_calibration(data, target, models, *, bins=10, ax=None, title="12-month PD calibration"):
    """Quantile-binned predicted probability against observed event frequency."""
    ax, limit = _axis(ax), 0.0
    for label, column in models.items():
        sample = data[[target, column]].dropna()
        groups = pd.qcut(sample[column], bins, duplicates="drop")
        curve = sample.groupby(groups, observed=True).agg(predicted=(column, "mean"), realized=(target, "mean"))
        ax.plot(curve["predicted"], curve["realized"], marker="o", label=label)
        limit = max(limit, curve.max().max())
    ax.plot([0, limit * 1.08], [0, limit * 1.08], color="0.45", linestyle="--", linewidth=1)
    ax.set(title=title, xlabel="Predicted PD", ylabel="Observed event rate")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend()
    return ax


def plot_structural_ranks(data, *, accounting="pd12", structural="pd_merton", ax=None):
    """Cross-sectional accounting and structural risk ranks on the same issuer set."""
    ax = _axis(ax)
    x = data[[accounting, structural]].dropna().rank(pct=True)
    ax.scatter(x[accounting], x[structural], s=16, alpha=0.55)
    ax.plot([0, 1], [0, 1], color="0.45", linestyle="--", linewidth=1)
    ax.set(title="Accounting and Merton risk ranks", xlabel="Accounting percentile", ylabel="Merton percentile")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    return ax


def plot_credit_premium(data, *, estimate="public_excess_credit_premium", benchmark="ebp", ax=None,
                         scale=1.0, unit="Percentage points"):
    """Public non-default premium and external EBP benchmark on common dates."""
    ax = _axis(ax)
    ax.plot(data.index, scale * data[estimate], label="Accounting estimate")
    ax.plot(data.index, scale * data[benchmark], label="Fed EBP", linestyle="--")
    ax.axhline(0, color="0.45", linewidth=0.7)
    ax.set(title="Public excess credit premium", xlabel="", ylabel=unit)
    ax.legend()
    return ax


def plot_cds_curves(data, *, names=None, ax=None):
    """Issuer synthetic CDS term structures, with quoted spreads in basis points."""
    ax = _axis(ax)
    names = data["ticker"].unique() if names is None else names
    for name in names:
        x = data[data["ticker"].eq(name)].sort_values("maturity")
        ax.plot(x["maturity"], x["spread_bp"], marker="o", label=name)
    ax.set(title="Issuer fair-value CDS curves", xlabel="Maturity (years)", ylabel="Par spread (bp)")
    ax.legend()
    return ax


def plot_credit_losses(losses: dict, *, notional=1.0, ax=None):
    """Empirical survival functions keep sparse credit-loss tails visible."""
    ax = _axis(ax)
    for label, values in losses.items():
        x, counts = np.unique(np.asarray(values) / notional, return_counts=True)
        survival = 1 - counts.cumsum() / counts.sum()
        ax.step(x, survival, where="post", label=label)
    ax.set(yscale="log", ylim=(1e-4, 1), title="Portfolio loss exceedance", xlabel="Portfolio loss", ylabel="Probability of exceeding loss")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="upper right")
    return ax


def plot_tranche_losses(data, *, ax=None):
    """Expected tranche loss by maturity, as fractions of each tranche's notional."""
    ax = _axis(ax)
    for name, x in data.groupby("tranche", sort=False):
        x = x.sort_values("year")
        ax.plot(x["year"], x["expected_loss"], marker="o", label=name)
    ax.set(title="Expected tranche losses", xlabel="Horizon (years)", ylabel="Tranche notional lost")
    ax.set_ylim(bottom=0)
    ax.set_xticks(sorted(data["year"].unique()))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend()
    return ax


def plot_structured_prices(prices: pd.DataFrame, *, ax=None):
    """Actual structured-credit prices, retaining the caller's rating/vintage columns."""
    ax = _axis(ax)
    for name in prices:
        label = " · ".join(map(str, name)) if isinstance(name, tuple) else str(name)
        ax.plot(prices.index, prices[name], label=label)
    ax.set(title="FINRA CBO/CDO/CLO prices", xlabel="", ylabel="Reported average price")
    ax.legend()
    return ax


__all__ = ["plot_default_capture", "plot_pd_calibration", "plot_structural_ranks", "plot_credit_premium",
           "plot_cds_curves", "plot_credit_losses", "plot_tranche_losses", "plot_structured_prices"]
