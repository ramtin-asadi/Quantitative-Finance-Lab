from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

data_dir = Path(__file__).resolve().parents[1]
cache_dir = Path(__file__).resolve().parent / "cache"
us_output = data_dir / "us_macro_factors.csv"
ca_output = data_dir / "canada_macro_factors.csv"
summary_output = data_dir / "macro_factor_summary.csv"
issues_output = data_dir / "macro_download_issues.csv"

default_start = "1990-01-01"
default_end = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")

us_fred_series = {
    "cpi_all_items": "CPIAUCSL",
    "cpc_core_cpi": "CPILFESL",
    "cpc_pce_price": "PCEPI",
    "cpc_core_pce": "PCEPILFE",
    "ppi_all_commodities": "PPIACO",
    "ppi_final_demand": "WPSFD49207",
    "ipp_import_price_all": "IR",
    "ipp_export_price_all": "IQ",
    "rmp_wti_oil": "MCOILWTICO",
    "inrt_fed_funds": "FEDFUNDS",
    "fdph_fed_funds_daily": "DFF",
    "fdrh_target_upper": "DFEDTARU",
    "fdrh_target_lower": "DFEDTARL",
    "gvbg_3m": "DGS3MO",
    "gvbg_2y": "DGS2",
    "gvbg_10y": "DGS10",
    "gvbg_30y": "DGS30",
    "gvbg_10y2y": "T10Y2Y",
    "gvbg_10y3m": "T10Y3M",
    "gdp_real": "GDPC1",
    "gdp_real_growth": "A191RL1Q225SBEA",
    "inp_industrial_production": "INDPRO",
    "inp_manufacturing_production": "IPMAN",
    "rsa_retail_sales": "RSAFS",
    "rsl_real_retail_sales": "RRSFS",
    "pmmn_durable_orders": "DGORDER",
    "pmmn_total_manufacturing_orders": "AMTMNO",
    "pmsr_manufacturing_shipments": "AMTMVS",
    "ivp_business_inventories": "BUSINV",
    "ivp_inventory_sales_ratio": "ISRATIO",
    "clin_oecd_cli_us": "USALOLITOAASTSAM",
    "unrt_unemployment": "UNRATE",
    "unrt_u6": "U6RATE",
    "injc_initial_claims": "ICSA",
    "ctcl_continued_claims": "CCSA",
    "nfp_total_payrolls": "PAYEMS",
    "nfp_private_payrolls": "USPRIV",
    "adpe_adp_private": "ADPWNUSNERSA",
    "avwh_avg_hourly_earnings": "CES0500000003",
    "whs_avg_weekly_hours": "AWHNONAG",
    "emci_employment_cost": "ECIWAG",
    "hsp_housing_starts": "HOUST",
    "hbp_building_permits": "PERMIT",
    "hon_new_home_sales": "HSN1F",
    "hoe_existing_home_sales": "EXHOSLUSM495S",
    "hop_case_shiller": "CSUSHPISA",
    "s20_case_shiller_20city": "SPCS20RSA",
    "psi_mortgage_rate": "MORTGAGE30US",
    "exp_exports_goods_services": "BOPXGS",
    "imp_imports_goods_services": "BOPMGS",
    "trbn_trade_balance": "BOPGSTB",
    "crab_current_account": "IEABC",
    "trbn_trade_weighted_dollar": "DTWEXBGS",
    "exp_usd_cad": "DEXCAUS",
    "umcc_michigan_sentiment": "UMCSENT",
    "cnci_oecd_consumer_confidence": "CSCICP03USM665S",
    "bsi_oecd_business_confidence": "BSCICP03USM665S",
}

ca_fred_series = {
    "clin_oecd_cli_ca": "CANLOLITOAASTSAM",
    "cnci_oecd_consumer_confidence": "CSCICP03CAM665S",
}

ca_statcan_vectors = {
    "cpi_all_items": 41690973,
    "cpc_core_cpi": 41691233,
    "ppi_industrial_products": 1230995983,
    "rmp_raw_materials": 1230998135,
    "rmp_crude_energy": 1230998136,
    "gdp_real": 65201210,
    "gdp_goods_industries": 65201211,
    "rsa_retail_sales": 1446859483,
    "pmmn_manufacturing_shipments": 800450,
    "pmmn_manufacturing_orders": 800913,
    "ivp_manufacturing_inventories": 803227,
    "ivp_inventory_sales_ratio": 803313,
    "unrt_unemployment": 2062815,
    "unrt_unemployed": 2062814,
    "emci_employment": 2062811,
    "avhe_hourly_earnings": 54027308,
    "avwh_weekly_hours": 54027310,
    "whs_weekly_earnings": 54027306,
    "hos_housing_starts": 52300157,
    "hbp_building_permits": 121293395,
    "psi_mortgage_rate": 122497,
    "exp_exports_goods": 87008955,
    "imp_imports_goods": 87008839,
    "trbn_trade_balance": 87008984,
    "crab_current_account": 61915304,
}

ca_boc_series = {
    "inrt_target_overnight": "V39079",
    "fdph_bank_rate": "V39078",
    "gvbg_3m": "TB.CDN.90D.MID",
    "gvbg_2y": "BD.CDN.2YR.DQ.YLD",
    "gvbg_10y": "BD.CDN.10YR.DQ.YLD",
    "gvbg_long": "BD.CDN.LONG.DQ.YLD",
}

known_issues = [
    {
        "country": "us",
        "source": "fred",
        "series": "whs_avg_weekly_hours",
        "requested_id": "CES0500000002",
        "used_id": "AWHNONAG",
        "message": "FRED returned 404 for CES0500000002, so AWHNONAG is used.",
    },
    {
        "country": "ca",
        "source": "fred",
        "series": "bsi_oecd_business_confidence",
        "requested_id": "BSCICP03CAM665S",
        "used_id": "",
        "message": "FRED returned 404 for the Canada OECD business confidence id.",
    },
]


def clean_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))


def read_cached_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def download_fred_series(
    series: dict[str, str],
    *,
    country: str,
    start: str,
    end: str,
    force: bool,
    offline: bool,
    issues: list[dict[str, str]],
) -> pd.DataFrame:
    frames = []
    fred_dir = cache_dir / "fred"
    fred_dir.mkdir(parents=True, exist_ok=True)
    for name, series_id in series.items():
        cache_path = fred_dir / f"{clean_name(series_id)}.csv"
        raw = None if force else read_cached_csv(cache_path)
        if raw is None and not offline:
            url = (
                "https://fred.stlouisfed.org/graph/fredgraph.csv?"
                f"id={series_id}&cosd={start}&coed={end}"
            )
            try:
                raw = pd.read_csv(url)
                raw.to_csv(cache_path, index=False)
            except Exception as exc:
                issues.append(
                    {
                        "country": country,
                        "source": "fred",
                        "series": name,
                        "requested_id": series_id,
                        "used_id": "",
                        "message": str(exc),
                    }
                )
                continue
        if raw is None:
            issues.append(
                {
                    "country": country,
                    "source": "fred",
                    "series": name,
                    "requested_id": series_id,
                    "used_id": "",
                    "message": "Missing cache and offline mode is enabled.",
                }
            )
            continue
        out = raw.iloc[:, :2].copy()
        out.columns = ["date", name]
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out[name] = pd.to_numeric(out[name], errors="coerce")
        s = out.dropna(subset=["date"]).set_index("date")[name].replace([np.inf, -np.inf], np.nan)
        frames.append(s.rename(name))
    return pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()


def download_statcan_vectors(
    vectors: dict[str, int],
    *,
    latest_n: int,
    force: bool,
    offline: bool,
    issues: list[dict[str, str]],
) -> pd.DataFrame:
    cache_path = cache_dir / "statcan_vectors.json"
    data = None
    if cache_path.exists() and not force:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    if data is None and not offline:
        payload = [{"vectorId": int(vector_id), "latestN": int(latest_n)} for vector_id in vectors.values()]
        try:
            response = requests.post(
                "https://www150.statcan.gc.ca/t1/wds/rest/getDataFromVectorsAndLatestNPeriods",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(data), encoding="utf-8")
        except Exception as exc:
            issues.append(
                {
                    "country": "ca",
                    "source": "statcan",
                    "series": "all_statcan_vectors",
                    "requested_id": ",".join(str(x) for x in vectors.values()),
                    "used_id": "",
                    "message": str(exc),
                }
            )
            return pd.DataFrame()
    if data is None:
        issues.append(
            {
                "country": "ca",
                "source": "statcan",
                "series": "all_statcan_vectors",
                "requested_id": ",".join(str(x) for x in vectors.values()),
                "used_id": "",
                "message": "Missing cache and offline mode is enabled.",
            }
        )
        return pd.DataFrame()
    vector_to_name = {int(vector_id): name for name, vector_id in vectors.items()}
    series_list = []
    for item in data:
        obj = item.get("object", {}) if isinstance(item, dict) else {}
        vector_id = int(obj.get("vectorId", 0) or 0)
        name = vector_to_name.get(vector_id)
        points = obj.get("vectorDataPoint", [])
        if name is None or not points:
            continue
        frame = pd.DataFrame(points)
        frame["date"] = pd.to_datetime(frame["refPer"], errors="coerce")
        frame[name] = pd.to_numeric(frame["value"], errors="coerce")
        s = frame.dropna(subset=["date"]).set_index("date")[name].replace([np.inf, -np.inf], np.nan)
        series_list.append(s.rename(name))
    return pd.concat(series_list, axis=1).sort_index() if series_list else pd.DataFrame()


def download_boc_series(
    series: dict[str, str],
    *,
    start: str,
    end: str,
    force: bool,
    offline: bool,
    issues: list[dict[str, str]],
) -> pd.DataFrame:
    cache_path = cache_dir / "boc_valet.json"
    data = None
    if cache_path.exists() and not force:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    if data is None and not offline:
        series_ids = ",".join(series.values())
        url = (
            "https://www.bankofcanada.ca/valet/observations/"
            f"{series_ids}/json?start_date={start}&end_date={end}"
        )
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            data = response.json()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(data), encoding="utf-8")
        except Exception as exc:
            issues.append(
                {
                    "country": "ca",
                    "source": "boc",
                    "series": "all_boc_series",
                    "requested_id": series_ids,
                    "used_id": "",
                    "message": str(exc),
                }
            )
            return pd.DataFrame()
    if data is None:
        issues.append(
            {
                "country": "ca",
                "source": "boc",
                "series": "all_boc_series",
                "requested_id": ",".join(series.values()),
                "used_id": "",
                "message": "Missing cache and offline mode is enabled.",
            }
        )
        return pd.DataFrame()
    reverse = {sid: name for name, sid in series.items()}
    rows = []
    for obs in data.get("observations", []):
        row = {"date": pd.to_datetime(obs.get("d"), errors="coerce")}
        for sid, name in reverse.items():
            row[name] = pd.to_numeric(obs.get(sid, {}).get("v"), errors="coerce")
        rows.append(row)
    return pd.DataFrame(rows).dropna(subset=["date"]).set_index("date").sort_index() if rows else pd.DataFrame()


def monthly_table(frames: list[pd.DataFrame], *, start: str, end: str, ffill_limit: int = 2) -> pd.DataFrame:
    valid = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid:
        return pd.DataFrame()
    raw = pd.concat(valid, axis=1).sort_index()
    monthly = raw.resample("ME").last()
    monthly = monthly.loc[pd.Timestamp(start) : pd.Timestamp(end)]
    monthly = monthly.ffill(limit=int(ffill_limit))
    index = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="ME")
    return monthly.reindex(index).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def trim_by_coverage(table: pd.DataFrame, min_series: int) -> pd.DataFrame:
    coverage = table.notna().sum(axis=1)
    keep = coverage >= int(min_series)
    if bool(keep.any()):
        return table.loc[keep[keep].index[0] :]
    return table


def availability_summary(country: str, factors: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "country": country,
            "factor": "all_factors",
            "start": factors.index.min(),
            "end": factors.index.max(),
            "observations": len(factors),
            "available_share": float(factors.notna().mean().mean()) if len(factors) else np.nan,
        }
    ]
    for column in factors.columns:
        series = factors[column]
        rows.append(
            {
                "country": country,
                "factor": column,
                "start": series.first_valid_index(),
                "end": series.last_valid_index(),
                "observations": int(series.notna().sum()),
                "available_share": float(series.notna().mean()) if len(series) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def add_low_coverage_issues(
    country: str,
    factors: pd.DataFrame,
    issues: list[dict[str, str]],
    *,
    min_observations: int = 60,
) -> None:
    for column in factors.columns:
        observations = int(factors[column].notna().sum())
        if observations < int(min_observations):
            issues.append(
                {
                    "country": country,
                    "source": "coverage",
                    "series": column,
                    "requested_id": "",
                    "used_id": "",
                    "message": f"Only {observations} monthly observations after alignment.",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=default_start)
    parser.add_argument("--end", default=default_end)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--statcan-latest-n", type=int, default=1000)
    args = parser.parse_args()

    cache_dir.mkdir(parents=True, exist_ok=True)
    issues = list(known_issues)

    us_fred = download_fred_series(
        us_fred_series,
        country="us",
        start=args.start,
        end=args.end,
        force=args.force_download,
        offline=args.offline,
        issues=issues,
    )
    ca_fred = download_fred_series(
        ca_fred_series,
        country="ca",
        start=args.start,
        end=args.end,
        force=args.force_download,
        offline=args.offline,
        issues=issues,
    )
    ca_statcan = download_statcan_vectors(
        ca_statcan_vectors,
        latest_n=args.statcan_latest_n,
        force=args.force_download,
        offline=args.offline,
        issues=issues,
    )
    ca_boc = download_boc_series(
        ca_boc_series,
        start=args.start,
        end=args.end,
        force=args.force_download,
        offline=args.offline,
        issues=issues,
    )

    us_factors = trim_by_coverage(monthly_table([us_fred], start=args.start, end=args.end), min_series=20)
    ca_factors = trim_by_coverage(
        monthly_table([ca_statcan, ca_boc, ca_fred], start=args.start, end=args.end),
        min_series=12,
    )
    add_low_coverage_issues("us", us_factors, issues)
    add_low_coverage_issues("ca", ca_factors, issues)

    us_factors.to_csv(us_output, index_label="date")
    ca_factors.to_csv(ca_output, index_label="date")
    pd.concat(
        [availability_summary("us", us_factors), availability_summary("ca", ca_factors)],
        axis=0,
        ignore_index=True,
    ).to_csv(summary_output, index=False)
    pd.DataFrame(issues).to_csv(issues_output, index=False)

    print(f"saved {us_output}")
    print(f"us rows: {len(us_factors):,}, range: {us_factors.index.min()} to {us_factors.index.max()}")
    print(f"saved {ca_output}")
    print(f"ca rows: {len(ca_factors):,}, range: {ca_factors.index.min()} to {ca_factors.index.max()}")
    print(f"saved {summary_output}")
    print(f"saved {issues_output}")


if __name__ == "__main__":
    main()
