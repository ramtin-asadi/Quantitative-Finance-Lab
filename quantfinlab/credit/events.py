"""Filing-derived issuer histories, censoring, and forecast labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

filing_flags = {"is_financial_obligation_trigger": "item_204",
                "is_exit_or_disposal": "exit_disposal", "is_material_impairment": "impairment_filing",
                "is_delisting_or_listing_failure": "delisting", "is_nonreliance": "nonreliance",
                "is_late_filing": "late_filing", "is_deregistration": "deregistration"}


def risk_set(filings: pd.DataFrame, *, start="2012-01-01", minimum_filings=5,
              minimum_years=1, strict="is_registrant_bankruptcy_event",
              broad="is_financial_obligation_trigger") -> pd.DataFrame:
    """Issuer cohort, first events and right-censoring dates from reporting histories.

    Cohort eligibility uses the reporting history supplied by the caller, as in
    the notebook. ``eligible_at`` additionally records when the minimum history
    became observable, for prospective universe construction.
    """
    periodic = filings[filings["form_type"].isin(["10-K", "10-K/A", "10-Q", "10-Q/A"])].sort_values("accepted_at")
    history = periodic.groupby("cik").agg(
        first_periodic=("accepted_at", "min"), last_periodic=("accepted_at", "max"),
        n_periodic=("accession", "size"), entity_name=("entity_name", "last"),
        ticker=("ticker", "last"), sec_sic=("sec_sic", "last"), sec_industry=("sec_industry", "last"))
    annual = periodic[periodic["form_type"].isin(["10-K", "10-K/A"])]
    history["n_10k"] = annual.groupby("cik").size()
    history["last_observed"] = filings.groupby("cik")["accepted_at"].max()
    history["strict_date"] = filings[filings[strict]].groupby("cik")["accepted_at"].min()
    history["broad_date"] = filings[filings[strict] | filings[broad]].groupby("cik")["accepted_at"].min()
    history["history_years"] = (history["last_periodic"] - history["first_periodic"]).dt.days / 365.25
    nth = periodic.groupby("cik").nth(minimum_filings - 1).set_index("cik")["accepted_at"]
    first_annual = annual.groupby("cik")["accepted_at"].min()
    history["eligible_at"] = pd.concat([
        nth, first_annual, history["first_periodic"] + pd.to_timedelta(365.25 * minimum_years, unit="D")
    ], axis=1).max(axis=1)
    history = history[history["n_10k"].ge(1) & history["n_periodic"].ge(minimum_filings)
                      & history["history_years"].ge(minimum_years)].copy()
    history["active_start"] = history["first_periodic"].clip(lower=pd.Timestamp(start))
    history["active_end"] = history[["last_observed", "strict_date"]].min(axis=1)
    return history.reset_index()


def issuer_months(history: pd.DataFrame, dates, *, statements=False,
                   prospective=False) -> pd.DataFrame:
    """Monthly issuer observations before bankruptcy and the relevant censoring date."""
    result = pd.MultiIndex.from_product([dates, history["cik"]], names=["decision_date", "cik"]).to_frame(index=False)
    result = result.merge(history, on="cik", validate="many_to_one")
    end = result[["last_periodic" if statements else "last_observed", "strict_date"]].min(axis=1)
    entry = result["eligible_at"] if prospective else result["first_periodic"]
    return result[result["decision_date"].gt(entry) & result["decision_date"].lt(end)].reset_index(drop=True)


def event_labels(data: pd.DataFrame, *, horizons=(1, 3, 6, 12, 24),
                  events=None, origin="decision_date", censor="last_observed") -> pd.DataFrame:
    """Forward labels plus observability masks; censored non-events remain unobserved."""
    events = events or {"event": "strict_date", "broad": "broad_date"}
    result = pd.DataFrame(index=data.index)
    for h in horizons:
        end = data[origin] + pd.offsets.MonthEnd(h)
        for name, column in events.items():
            hit = data[column].gt(data[origin]) & data[column].le(end)
            result[f"{name}_{h}m"] = hit
            result[f"observed_{name}_{h}m"] = hit | data[censor].ge(end)
    return result


def distress_states(data: pd.DataFrame) -> pd.Series:
    """State known at each origin: healthy, distressed or bankrupt."""
    date = data["decision_date"]
    return pd.Series(np.select([data["strict_date"].le(date), data["broad_date"].le(date)],
                                ["bankrupt", "distressed"], default="healthy"), index=data.index)


def filing_counts(filings: pd.DataFrame, observations: pd.DataFrame, *, months=12,
                   flags=None) -> pd.DataFrame:
    """Trailing monthly filing counts on the supplied complete issuer-month grid.

    Dates denote completed calendar months, matching the source notebook.
    Intramonth origins should use an explicit timestamp-filtered event query.
    """
    flags = flags or filing_flags
    source = filings.assign(decision_date=filings["accepted_at"].dt.to_period("M").dt.to_timestamp("M"))
    monthly = source.groupby(["decision_date", "cik"])[list(flags)].sum().rename(columns=flags)
    result = observations[["decision_date", "cik"]].join(monthly, on=["decision_date", "cik"])
    names = list(flags.values())
    result[names] = result[names].fillna(0)
    result = result.sort_values(["cik", "decision_date"])
    for name in names:
        result[f"{name}_{months}m"] = result.groupby("cik")[name].transform(
            lambda x: x.rolling(months, min_periods=1).sum())
    return result.reindex(observations.index)


def statement_sample(accounting: pd.DataFrame, history: pd.DataFrame, dates) -> pd.DataFrame:
    """Carry available statements through the monthly risk set until event/censoring."""
    result = issuer_months(history, dates)[["decision_date", "cik"]]
    result = result.merge(accounting, on=["decision_date", "cik"], how="left", validate="one_to_one")
    columns = [name for name in accounting if name not in {"decision_date", "cik"}]
    result[columns] = result.groupby("cik", sort=False)[columns].ffill()
    return result.merge(history, on="cik", validate="many_to_one").sort_values(
        ["cik", "decision_date"]).reset_index(drop=True)


def market_members(market: pd.DataFrame, mappings: pd.DataFrame) -> pd.DataFrame:
    """Notebook PIT member-month selection and date-valid issuer/ticker mapping."""
    monthly = market.assign(decision_date=market["date"].dt.to_period("M").dt.to_timestamp("M"))
    monthly = monthly[monthly["is_sp500_member"]].sort_values("date").groupby(
        ["decision_date", "ticker"], as_index=False).tail(1)
    mapping = mappings[["ticker", "cik", "mapping_valid_from", "mapping_valid_to"]].dropna().drop_duplicates()
    monthly = monthly.merge(mapping, on="ticker", how="inner")
    valid = monthly["decision_date"].ge(monthly["mapping_valid_from"]) & monthly["decision_date"].le(monthly["mapping_valid_to"])
    return monthly[valid].sort_values(["decision_date", "cik", "market_cap"]).drop_duplicates(["decision_date", "cik"], keep="last")
