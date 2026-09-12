"""Tranche loss allocation and structured-credit market comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd


def tranche_loss(loss, A: float, D: float) -> np.ndarray:
    """Loss as a fraction of tranche notional for portfolio loss fractions."""
    if not 0 <= A < D <= 1:
        raise ValueError("Require 0 <= attachment < detachment <= 1.")
    return np.clip(np.asarray(loss) - A, 0, D - A) / (D - A)


def tranche_summary(losses, tranches) -> pd.DataFrame:
    """Expected loss, impairment and wipeout frequencies for named tranches."""
    rows = []
    for name, A, D in tranches:
        loss = tranche_loss(losses, A, D)
        rows.append({"tranche": name, "attachment": A, "detachment": D,
                     "expected_loss": loss.mean(), "impairment": np.mean(loss > 0),
                     "wipeout": np.mean(loss >= 1)})
    return pd.DataFrame(rows).set_index("tranche")


def vintage_prices(data: pd.DataFrame, *, rating="rating", vintage="vintage",
                    date="date", price="price") -> pd.DataFrame:
    """Keep rating and collateral vintage distinct when comparing reported prices."""
    return data.groupby([date, rating, vintage], dropna=False)[price].mean().unstack([rating, vintage])


def stress_comparison(prices: pd.DataFrame, windows: dict) -> pd.DataFrame:
    """Price changes and trough drawdowns over explicitly supplied stress windows."""
    rows = []
    for name, (start, end) in windows.items():
        for column in prices:
            x = prices.loc[start:end, column].dropna()
            if len(x) >= 2:
                rows.append({"window": name, "series": column, "observations": len(x),
                             "change": x.iloc[-1] / x.iloc[0] - 1,
                             "drawdown": (x / x.cummax() - 1).min()})
    return pd.DataFrame(rows)


def structured_categories(data: pd.DataFrame) -> pd.DataFrame:
    """Decode the categories FINRA actually publishes, preserving collateral vintage."""
    data = data.copy()
    text = (data["rating_group"].fillna("") + " | " + data["column_label"].fillna("")).str.upper()
    category = pd.Series("aggregate", index=data.index, dtype="string")
    category[text.str.contains("NON-INVESTMENT GRADE")] = "non-investment grade"
    category[text.str.contains("NON-AAA IG")] = "non-AAA investment grade"
    category[text.str.contains("AAA") & ~text.str.contains("NON-AAA")] = "AAA"
    data["category"] = category
    data["vintage"] = "aggregate"
    for label in ["pre-2022", "2022-2025", "pre-2023", "2023-2026"]:
        data.loc[text.str.contains(label.upper()), "vintage"] = label
    data["metric"] = data["row_label"].where(data["row_label"].str.len().gt(0), data["metric_group"]).str.upper()
    return data


def trading_activity(data: pd.DataFrame) -> pd.DataFrame:
    """FINRA structured trading totals by month, grade and published metric."""
    data = data.copy()
    text = data["column_label"].str.upper()
    data["grade"] = np.where(text.str.contains("NON-INVESTMENT"), "non-investment grade", "investment grade")
    data["metric"] = np.select([text.str.contains(r"\$ TRADES"), text.str.contains(r"TRADE \| COUNT")], ["volume", "trades"], default="other")
    data["date"] = pd.to_datetime(data["report_date"]).dt.to_period("M").dt.to_timestamp("M")
    return data[data["metric"].ne("other")].groupby(["date", "grade", "metric"])["value"].sum(min_count=1).unstack(["grade", "metric"])
