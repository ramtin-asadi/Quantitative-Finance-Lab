import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from quantfinlab.common.cache import cache_key

from .schemas import ContextSnapshot, utc


def finite(values: dict) -> dict:
    return {str(key): round(float(value), 6) for key, value in values.items()
            if isinstance(value, (int, float, np.number)) and np.isfinite(value)}


def close_time(date) -> datetime:
    return pd.Timestamp(date).to_pydatetime().replace(hour=18, minute=0, second=0,
        tzinfo=ZoneInfo("America/New_York")).astimezone(timezone.utc)


def observed_at(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)

def context_dependencies(root, name):
    files = {
        "market": ["core_cross_asset_etfs.csv"], "risk": ["core_cross_asset_etfs.csv"],
        "cross_asset": ["core_cross_asset_etfs.csv"],
        "volatility": ["core_cross_asset_etfs.csv", "macro_high_frequency.parquet"],
        "rates": ["us_treasury_yields.csv", "acm_term_premium.csv"],
        "factors": ["factor_proxy_etfs.csv", "sp500_market_data.parquet"],
        "financial_conditions": ["us_macro_factors.csv", "nfci.csv"],
        "fundamentals": ["sp500_fundamentals.parquet", "sec_credit.parquet",
                         "sp500_fundamentals/cache/ticker_cik_mapping.parquet", "sp500_market_data.parquet"],
        "credit": ["fed_credit.parquet", "nyfed_cmdi.parquet", "sec_credit.parquet", "finra_credit_market.parquet"],
        "macro": ["alfred_realtime.parquet", "gdpnow_forecasts.parquet", "atlanta_mpt.parquet"],
    }
    return [Path(root) / "data" / name for name in files[name]]

def load_market_prices(root, cutoff, name="core_cross_asset_etfs.csv"):
    from quantfinlab.dataio import load_yfinance_panel

    prices = load_yfinance_panel(Path(root) / "data" / name, fields=["close"], lowercase=False)["close"]
    prices = prices.loc[[close_time(date) <= cutoff for date in prices.index]].tail(800)
    return prices

def context_snapshot(root, name, cutoff, latest, available, measures, paths, notes=(), max_age=7):
    age = (cutoff - latest).days if latest is not None else None
    status = "unavailable" if not measures else "stale" if age is None or age > max_age else "current"
    return ContextSnapshot(name=name, as_of=cutoff, latest_data_at=latest, available_at=available,
        freshness=status, measures=measures, dependencies=[str(p.relative_to(Path(root))) for p in paths],
        notes=list(notes), version="context-v2")

def market_context(root, cutoff, ticker=None):
    prices = load_market_prices(root, cutoff)
    assets = [x for x in ["SPY", "QQQ", "IWM", "HYG", "LQD", "IEF", "TLT", "GLD", "DBC", "UUP"] if x in prices]
    if prices.empty:
        return context_snapshot(root, "market", cutoff, None, None, {}, context_dependencies(root, "market"))
    from .market import market_moves, snapshot_records

    moves = market_moves(prices[assets])
    result = snapshot_records(moves)
    for asset in result:
        result[asset]["data_at"] = close_time(moves.loc[asset, "data_date"]).isoformat()
    latest = close_time(prices.index[-1])
    return context_snapshot(root, "market", cutoff, latest, latest, result, context_dependencies(root, "market"),
        ["Returns use the existing adjusted-price panel; prices are not a vendor-vintage archive.",
         "Daily observations become eligible at 18:00 America/New_York; no intraday attribution."])

def risk_context(root, cutoff, ticker=None):
    from .market import risk_measures, snapshot_records

    prices = load_market_prices(root, cutoff)
    if prices.empty or "SPY" not in prices:
        return context_snapshot(root, "risk", cutoff, None, None, {}, context_dependencies(root, "risk"))
    assets = [asset for asset in ["SPY", "QQQ", "IWM", "HYG", "TLT", "GLD"] if asset in prices]
    result = snapshot_records(risk_measures(prices[assets]))
    latest = close_time(prices.index[-1])
    return context_snapshot(root, "risk", cutoff, latest, latest, result, context_dependencies(root, "risk"))

def cross_asset_context(root, cutoff, ticker=None):
    from quantfinlab.ml.features import build_cross_asset_feature_block

    prices = load_market_prices(root, cutoff)
    if prices.empty:
        return context_snapshot(root, "cross_asset", cutoff, None, None, {}, context_dependencies(root, "cross_asset"))
    assets = [x for x in ["SPY", "QQQ", "IWM", "HYG", "LQD", "IEF", "TLT", "GLD", "DBC", "SHY"] if x in prices]
    features = build_cross_asset_feature_block(prices[assets], assets=assets)
    latest = close_time(prices.index[-1])
    measures = finite(features.iloc[-1].to_dict())
    from .market import relative_moves, snapshot_records

    measures["relative_moves"] = snapshot_records(relative_moves(prices))
    return context_snapshot(root, "cross_asset", cutoff, latest, latest, measures,
                          context_dependencies(root, "cross_asset"), ["Breadth refers to the selected ETF universe, not all listed stocks."])

def volatility_context(root, cutoff, ticker=None):
    from quantfinlab.ml.features import realized_vol

    prices = load_market_prices(root, cutoff)
    if prices.empty:
        return context_snapshot(root, "volatility", cutoff, None, None, {}, context_dependencies(root, "volatility"))
    returns = prices.SPY.pct_change(fill_method=None)
    vol = realized_vol(returns, 21)
    measures = finite({"spy_realized_vol_21d": vol.iloc[-1], "vol_change_5d": vol.diff(5).iloc[-1]})
    notes = ["Options-based variance risk premium and heavy forecasts require a separately refreshed snapshot."]
    path = Path(root) / "data/macro_high_frequency.parquet"
    if path.exists():
        vix = pd.read_parquet(path, filters=[("series_id", "==", "VIXCLS")])
        vix = vix.loc[vix.date.map(close_time).le(cutoff)].sort_values("date")
        if len(vix) > 20 and (cutoff - close_time(vix.iloc[-1].date)).days <= 7:
            values = vix.value
            measures.update(finite({"vix": values.iloc[-1], "vix_change": values.diff().iloc[-1],
                "vix_z": (values.iloc[-1] - values.tail(252).mean()) / values.tail(252).std()}))
            notes.append("VIX data at " + close_time(vix.iloc[-1].date).isoformat())
        else:
            notes.append("VIX unavailable or stale at this cutoff.")
    latest = close_time(prices.index[-1])
    return context_snapshot(root, "volatility", cutoff, latest, latest, measures, context_dependencies(root, "volatility"), notes)

def rates_context(root, cutoff, ticker=None):
    from quantfinlab.dataio import load_par_yield_curve

    path = Path(root) / "data/us_treasury_yields.csv"
    rates = load_par_yield_curve(path, percent=True)
    rates = rates.loc[[close_time(date) <= cutoff for date in rates.index]].tail(504)
    if len(rates) < 6:
        return context_snapshot(root, "rates", cutoff, None, None, {}, context_dependencies(root, "rates"))
    from .market import curve_moves, curve_shape, snapshot_records

    measures = snapshot_records(curve_moves(rates))
    measures["curve"] = finite(curve_shape(rates).iloc[-1].to_dict())
    latest = close_time(rates.index[-1])
    acm_path = Path(root) / "data/acm_term_premium.csv"
    notes = ["Curve measures use Treasury par yields; curvature is the 2x5Y-2Y-10Y butterfly in basis points."]
    if acm_path.exists() and observed_at(acm_path) <= cutoff:
        acm = pd.read_csv(acm_path, index_col=0, parse_dates=True)
        acm = acm.loc[[close_time(date) <= cutoff for date in acm.index]]
        if len(acm) and (cutoff - close_time(acm.index[-1])).days <= 7:
            columns = [c for c in acm if re.search(r"ACMTP(02|05|10)$", c, re.I)]
            measures["acm_term_premium"] = finite(acm[columns].iloc[-1].to_dict())
            notes.append("ACM current revision; available from acquisition " + observed_at(acm_path).isoformat())
        else:
            notes.append("ACM term premium is stale and omitted.")
    else:
        notes.append("ACM current revisions are ineligible before acquisition.")
    available = max(latest, observed_at(acm_path)) if "acm_term_premium" in measures else latest
    return context_snapshot(root, "rates", cutoff, latest, available, measures, context_dependencies(root, "rates"),
        notes)

def factors_context(root, cutoff, ticker=None):
    from quantfinlab.portfolio.factors import factor_proxy_spreads, rolling_factor_fit

    prices = load_market_prices(root, cutoff, "factor_proxy_etfs.csv")
    if len(prices) < 64:
        return context_snapshot(root, "factors", cutoff, None, None, {}, context_dependencies(root, "factors"))
    spreads = factor_proxy_spreads(prices.pct_change(fill_method=None))
    measures = {"proxy_spread_21d": finite(spreads.tail(21).sum().to_dict())}
    notes = ["Tradable daily factor-proxy spreads from Project 15, not academic factor realizations."]
    path = Path(root) / "data/sp500_market_data.parquet"
    if ticker and path.exists():
        issuer = pd.read_parquet(path, columns=["date", "ticker", "adj_close"], filters=[("ticker", "==", ticker.upper())])
        issuer = issuer.loc[issuer.date.map(close_time).le(cutoff)].sort_values("date").drop_duplicates("date")
        issuer_returns = issuer.set_index("date").adj_close.pct_change(fill_method=None).rename(ticker).to_frame()
        common = spreads.index.intersection(issuer_returns.index)
        if len(common) >= 147:
            sample = common[-147:]
            alpha, beta, r2, residual = rolling_factor_fit(issuer_returns.loc[sample], spreads.loc[sample], window=126)
            measures["issuer"] = {"ticker": ticker, "data_at": close_time(sample[-1]).isoformat(),
                "beta": finite(beta[ticker].iloc[-1].to_dict()),
                "fit": finite({"r_squared": r2[ticker].iloc[-1], "alpha_daily": alpha[ticker].iloc[-1],
                               "residual_return_21d": residual[ticker].tail(21).sum()})}
        else:
            notes.append("Insufficient aligned issuer/factor history for a 126-day exposure fit.")
    latest = close_time(prices.index[-1])
    return context_snapshot(root, "factors", cutoff, latest, latest, measures, context_dependencies(root, "factors"),
        notes)

def financial_conditions_context(root, cutoff, ticker=None):
    from quantfinlab.dataio import load_macro_factors, load_nfci
    from quantfinlab.ml.features import build_fci_feature_block

    paths = context_dependencies(root, "financial_conditions")
    availability = max(observed_at(path) for path in paths)
    if availability > cutoff:
        return context_snapshot(root, "financial_conditions", cutoff, None, None, {}, paths,
            ["Current revised macro/FCI files were acquired after cutoff; historical use is disabled."])
    macro = load_macro_factors(paths[0])
    nfci = load_nfci(paths[1])
    features = build_fci_feature_block(macro, nfci)
    features = features.loc[[close_time(date) <= cutoff for date in features.index]]
    feature_dates = {column: str(features[column].last_valid_index()) for column in features}
    features = features.ffill()
    latest = close_time(features.index[-1])
    measures = finite(features.iloc[-1].to_dict())
    if "nfci_level" in features:
        measures["nfci_percentile"] = float(features.nfci_level.rank(pct=True).iloc[-1])
    measures["feature_dates"] = feature_dates
    notes = ["Snapshot of current source revisions."]
    fci_date = feature_dates.get("fci_level")
    if fci_date and fci_date != "None" and (cutoff - close_time(pd.Timestamp(fci_date))).days > 45:
        notes.append("Macro-derived FCI components are stale; their last signal date is " + fci_date + ".")
    for before, after in [("fci_change_21", "fci_change_1_month"), ("fci_change_63", "fci_change_3_months"),
                          ("nfci_change_21", "nfci_change_1_observation"), ("nfci_change_63", "nfci_change_3_observations")]:
        if before in measures:
            measures[after] = measures.pop(before)
    return context_snapshot(root, "financial_conditions", cutoff, latest, availability,
        measures, paths, notes, max_age=45)

def fundamentals_context(root, cutoff, ticker=None):
    from quantfinlab.dataio import read_sec_facts
    from quantfinlab.fundamentals import analysis as ratios, statements

    from .sec import accepted_utc, local_filings, resolve_ticker

    paths = context_dependencies(root, "fundamentals")
    if not ticker:
        return context_snapshot(root, "fundamentals", cutoff, None, None, {}, paths, ["A ticker is required."])
    cik, entity = resolve_ticker(Path(root), ticker, as_of=cutoff)
    filings = local_filings(Path(root), cik)
    filings["available_at"] = filings.accepted_at.map(accepted_utc)
    facts = read_sec_facts(paths[0], ciks=[cik])
    facts = facts.merge(filings[["accession", "available_at"]], on="accession", how="inner", validate="many_to_one")
    facts = facts[facts.available_at.le(cutoff)].copy()
    if facts.empty:
        return context_snapshot(root, "fundamentals", cutoff, None, None, {}, paths, ["No accepted facts before cutoff."])
    facts["filed_date"] = pd.to_datetime(facts.available_at, utc=True).dt.tz_localize(None)
    classified = statements.classify_duration_facts(facts)
    selected = statements.select_filing_facts(classified)
    quarters, _ = statements.reconstruct_quarters(selected)
    duration = statements.duration_statement_values(quarters, selected).set_index("cik")
    instant = statements.instant_statement_values(selected).set_index("cik")
    frame = duration.join(instant, rsuffix="_instant")
    def get(name):
        return frame[name] if name in frame else pd.Series(np.nan, index=frame.index)
    revenue, operating, income = get("revenue_ttm"), get("operating_income_ttm"), get("net_income_ttm")
    fcf = ratios.free_cash_flow(get("cfo_ttm"), get("capex_ttm"))
    debt = ratios.total_debt(get("total_debt_reported"), get("long_term_debt_noncurrent"), get("debt_current"), get("short_term_borrowings"))
    metrics = {"gross_margin_ttm": ratios.gross_margin(get("gross_profit_ttm"), revenue),
        "operating_margin_ttm": ratios.operating_margin(operating, revenue),
        "net_margin_ttm": ratios.net_margin(income, revenue), "fcf_margin_ttm": ratios.fcf_margin(fcf, revenue),
        "cfo_net_income": ratios.cfo_to_net_income(get("cfo_ttm"), income),
        "debt_assets": ratios.debt_to_assets(debt, get("total_assets")),
        "current_ratio": ratios.current_ratio(get("current_assets"), get("current_liabilities")),
        "interest_coverage": ratios.interest_coverage(operating, get("interest_expense_ttm")),
        "capex_revenue": ratios.capex_to_revenue(get("capex_ttm"), revenue)}
    assets = ratios.average_balance(get("total_assets"), get("total_assets_prior"))
    equity = ratios.average_balance(get("common_equity"), get("common_equity_prior"))
    cash = get("cash")
    metrics.update({"roa": ratios.return_on_assets(income, assets), "roe": ratios.return_on_equity(income, equity),
        "accruals_assets": ratios.total_accruals(income, get("cfo_ttm"), assets),
        "asset_turnover": ratios.asset_turnover(revenue, assets), "cfo_debt": ratios.cfo_to_debt(get("cfo_ttm"), debt),
        "net_debt_assets": ratios.net_debt_to_assets(ratios.net_debt(debt, cash), get("total_assets")),
        "cash_ratio": ratios.cash_ratio(cash, get("current_liabilities")),
        "working_capital_revenue": ratios.working_capital_to_revenue(
            ratios.working_capital(get("current_assets"), get("current_liabilities")), revenue),
        "dividend_payout": ratios.dividend_payout_ratio(get("dividends_ttm"), income)})
    measures = {"entity": entity, "ticker": ticker, "metrics": finite({k: v.iloc[-1] for k, v in metrics.items()}),
        "growth": finite({c: frame[c].iloc[-1] for c in frame if c.endswith(("_qoq", "_q_yoy")) and c.split('_')[0] in {'revenue', 'operating', 'net', 'cfo', 'capex'}})}
    quarterly = quarters.sort_values("filed_date").drop_duplicates(["field", "period_end"], keep="last")
    quarterly = quarterly.pivot(index="period_end", columns="field", values="value").sort_index().tail(8)
    trends = {}
    for name, field, function in [("operating_margin", "operating_income", ratios.operating_margin),
                                  ("gross_margin", "gross_profit", ratios.gross_margin),
                                  ("net_margin", "net_income", ratios.net_margin)]:
        if field in quarterly and "revenue" in quarterly:
            values = function(quarterly[field], quarterly.revenue).dropna()
            if len(values):
                trends[name] = {"quarters": {str(date.date()): round(float(value), 6) for date, value in values.tail(5).items()},
                    "changes": finite({"quarter_change_pp": values.diff().iloc[-1] * 100,
                                       "year_change_pp": values.diff(4).iloc[-1] * 100})}
    measures["margin_trends"] = trends
    measures["capital_allocation"] = finite({"repurchases_ttm": get("repurchases_ttm").iloc[-1],
        "dividends_ttm": get("dividends_ttm").iloc[-1], "issuance_ttm": get("share_issuance_ttm").iloc[-1],
        "shares_outstanding": get("shares_outstanding").iloc[-1],
        "shares_change_percent": (get("shares_outstanding").iloc[-1] / get("shares_outstanding_prior").iloc[-1] - 1) * 100})
    market_path = Path(root) / "data/sp500_market_data.parquet"
    notes = ["Ratios and quarter reconstruction reuse Project 21; facts without acceptance metadata are excluded.",
             "Quarter trends use the latest revisions available at the cutoff. Share change is versus the prior reported instant."]
    if market_path.exists():
        prices = pd.read_parquet(market_path, columns=["date", "ticker", "market_cap", "industry"],
                                 filters=[("ticker", "==", ticker.upper())])
        prices = prices.loc[prices.date.map(close_time).le(cutoff)].sort_values("date")
        if len(prices) and (cutoff - close_time(prices.iloc[-1].date)).days <= 7:
            market_cap = pd.Series(prices.iloc[-1].market_cap, index=frame.index)
            ev = ratios.enterprise_value(market_cap, debt, cash)
            measures["valuation"] = finite({"price_earnings": ratios.price_to_earnings(market_cap, income).iloc[-1],
                "price_sales": ratios.price_to_sales(market_cap, revenue).iloc[-1],
                "fcf_yield": ratios.fcf_yield(fcf, market_cap).iloc[-1],
                "ev_ebit": ratios.enterprise_value_to_ebit(ev, operating).iloc[-1]})
            notes.append("Valuation price date " + str(prices.iloc[-1].date.date()))
        else:
            notes.append("Issuer prices are stale; current valuation is omitted.")
    notes.append("Peer percentiles require a separately refreshed, comparable peer snapshot.")
    latest = pd.Timestamp(facts.period_end.max()).to_pydatetime().replace(tzinfo=timezone.utc)
    return context_snapshot(root, "fundamentals", cutoff, latest, max(facts.available_at), measures, paths,
        notes, max_age=180)

def credit_context(root, cutoff, ticker=None):
    from quantfinlab.dataio import read_credit_market

    paths = context_dependencies(root, "credit")
    measures, times, notes = {}, [], []
    for path in paths[:2]:
        availability = observed_at(path)
        if availability > cutoff:
            notes.append(f"{path.name}: revised snapshot acquired after cutoff.")
            continue
        data = read_credit_market(path).sort_values("date")
        data = data.loc[data.date.map(close_time).le(cutoff)]
        if data.empty:
            notes.append(f"{path.name}: no observations before cutoff.")
            continue
        latest = data.iloc[-1]
        measures[path.stem] = finite(latest.to_dict())
        measures[path.stem]["data_at"] = close_time(latest.date).isoformat()
        measures[path.stem]["changes"] = finite(data.select_dtypes("number").diff().iloc[-1].to_dict())
        times.append((close_time(latest.date), availability))
    if ticker:
        from .sec import local_filings, resolve_ticker

        cik, _ = resolve_ticker(Path(root), ticker, as_of=cutoff)
        filings = local_filings(Path(root), cik)
        filings = filings[pd.to_datetime(filings.accepted_at, utc=True).le(cutoff)]
        recent = filings[pd.to_datetime(filings.accepted_at, utc=True).ge(cutoff - timedelta(days=365))]
        flags = [c for c in recent if c.startswith("is_")]
        measures["issuer_filing_flags"] = {c: int(recent[c].fillna(False).sum()) for c in flags}
        if len(filings):
            stamp = pd.Timestamp(filings.accepted_at.max(), tz="UTC").to_pydatetime()
            times.append((stamp, stamp))
    notes.append("Filing flags are triage evidence, not default probabilities. Heavy P22 models are not rerun.")
    return context_snapshot(root, "credit", cutoff, max((x[0] for x in times), default=None),
        max((x[1] for x in times), default=None), measures, paths, notes, max_age=45)

def macro_context(root, cutoff, ticker=None):
    from quantfinlab.dataio.realtime import read_alfred
    from quantfinlab.macro.realtime import vintage_asof

    paths = context_dependencies(root, "macro")
    vintage = read_alfred(paths[0], series=["PAYEMS", "UNRATE", "CPIAUCSL", "PCEPILFE", "FEDFUNDS"])
    vintage["available_at"] = vintage.available_at.map(lambda value:
        pd.Timestamp(value).to_pydatetime().replace(hour=23, minute=59, second=59,
            tzinfo=ZoneInfo("America/New_York")).astimezone(timezone.utc))
    if "valid_to" in vintage:
        vintage["valid_to"] = pd.to_datetime(vintage.valid_to, utc=True)
    known = vintage_asof(vintage, cutoff).sort_values(["series_id", "observation_date"])
    measures, latest, availability = {}, None, None
    for series, group in known.groupby("series_id"):
        row = group.iloc[-1]
        measures[series] = {"value": float(row.value), "period": str(row.observation_date.date()),
                           "available_at": row.available_at.isoformat(),
                           "unit": "thousands of persons" if series == "PAYEMS" else "percent" if series in {"UNRATE", "FEDFUNDS"} else "index"}
        values = group.value.astype(float)
        measures[series]["change"] = finite({"previous_value": values.iloc[-2] if len(values) > 1 else np.nan,
            "change": values.diff().iloc[-1], "change_percent": values.pct_change(fill_method=None).iloc[-1] * 100,
            "year_change_percent": values.pct_change(12, fill_method=None).iloc[-1] * 100})
        if series in {"UNRATE", "FEDFUNDS"}:
            measures[series]["change"].pop("change_percent", None)
            measures[series]["change"].pop("year_change_percent", None)
        if series == "PAYEMS" and len(values) > 1:
            measures[series]["payroll_change_persons"] = int(round(values.diff().iloc[-1] * 1000))
        latest = max(latest or row.available_at, row.available_at)
        availability = max(availability or row.available_at, row.available_at)
    notes = ["ALFRED vintages are eligible after the end of their release day; no invented intraday release time.",
             "No survey surprise is inferred without an explicit expectation."]
    if paths[1].exists() and observed_at(paths[1]) <= cutoff:
        nowcasts = pd.read_parquet(paths[1])
        nowcasts = nowcasts.loc[nowcasts.forecast_date.map(close_time).le(cutoff)]
        headline = nowcasts[nowcasts.component.str.contains(r"^GDP$|Real GDP", case=False, regex=True)]
        if len(headline):
            row = headline.sort_values("forecast_date").iloc[-1]
            if (cutoff - close_time(row.forecast_date)).days <= 45:
                measures["gdpnow"] = {"growth_percent_saar": round(float(row.forecast_value), 2),
                    "forecast_date": str(row.forecast_date.date()), "target_quarter": str(row.target_quarter.date()),
                    "available_at": observed_at(paths[1]).isoformat()}
                availability = max(availability or observed_at(paths[1]), observed_at(paths[1]))
    notes.append("Policy distributions and macro-market gaps are consumed only as separately timestamped snapshots.")
    return context_snapshot(root, "macro", cutoff, latest, availability, measures, paths,
        notes, max_age=45)

class ContextRegistry:
    def __init__(self, root):
        self.root = Path(root)
        self.builders = {"market": market_context, "risk": risk_context, "volatility": volatility_context,
                         "rates": rates_context, "financial_conditions": financial_conditions_context,
                         "factors": factors_context, "cross_asset": cross_asset_context,
                         "fundamentals": fundamentals_context, "credit": credit_context, "macro": macro_context}

    def dependencies(self, name):
        return context_dependencies(self.root, name)

    def build(self, name: str, *, as_of, ticker=None, use_cache=True) -> ContextSnapshot:
        if name not in self.builders:
            raise KeyError(name)
        cutoff = utc(as_of)
        paths = self.dependencies(name)
        present = [path for path in paths if path.exists()]
        supplemental = self.root / "workspace/financial_analyst/inputs" / f"{name}-{ticker or 'market'}.json"
        if supplemental.exists():
            present.append(supplemental)
        key = cache_key(present, {"builder": name, "version": "context-v2", "ticker": ticker,
                                 "as_of": cutoff.isoformat(), "missing": [str(p) for p in paths if not p.exists()]})
        cache = self.root / "workspace/financial_analyst/snapshots" / name / f"{key}.json"
        if use_cache and cache.exists():
            return ContextSnapshot.model_validate_json(cache.read_text())
        try:
            result = self.builders[name](self.root, cutoff, ticker)
        except FileNotFoundError as error:
            result = context_snapshot(self.root, name, cutoff, None, None, {}, paths, [str(error)])
        if supplemental.exists():
            supplied = ContextSnapshot.model_validate_json(supplemental.read_text())
            if supplied.as_of <= cutoff and supplied.available_at is not None and supplied.available_at <= cutoff:
                result.measures["supplemental"] = supplied.model_dump(mode="json")
                result.available_at = max(result.available_at or supplied.available_at, supplied.available_at)
                result.notes.append("Additional separately refreshed snapshot retains its own availability and freshness.")
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".tmp")
        temporary.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(cache)
        return result
