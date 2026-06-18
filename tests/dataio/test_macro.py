from __future__ import annotations

import pandas as pd

from quantfinlab.dataio.macro import (
    clean_monthly_index,
    load_acm_term_premium,
    load_macro_factors,
    load_nfci,
    macro_availability_table,
)


def test_macro_loaders_collapse_to_month_end_and_filter_start(tmp_path) -> None:
    macro_path = tmp_path / "macro.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-15", "2024-01-31", "2024-02-15"],
            "growth": [1.0, 2.0, 3.0],
            "inflation": ["4.0", "bad", "5.0"],
        }
    ).to_csv(macro_path, index=False)

    nfci_path = tmp_path / "nfci.csv"
    pd.DataFrame(
        {
            "Friday_of_Week": ["2024-01-05", "2024-01-26", "2024-02-02"],
            "NFCI": [-0.2, -0.1, 0.1],
        }
    ).to_csv(nfci_path, index=False)

    acm_path = tmp_path / "acm_term_premium.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-31", "2024-02-29"],
            "ACMTP10": [0.10, 0.12, 0.20],
        }
    ).to_csv(acm_path, index=False)

    macro = load_macro_factors(macro_path, start="2024-02-01")
    nfci = load_nfci(tmp_path)
    acm = load_acm_term_premium(tmp_path, monthly=True)

    assert macro.index.tolist() == [pd.Timestamp("2024-02-29")]
    assert nfci.loc[pd.Timestamp("2024-01-31"), "NFCI"] == -0.1
    assert acm.loc[pd.Timestamp("2024-01-31"), "ACMTP10"] == 0.12

    cleaned = clean_monthly_index(pd.DataFrame({"x": [1, 2]}, index=["2024-03-01", "2024-03-15"]))
    assert cleaned.loc[pd.Timestamp("2024-03-31"), "x"] == 2
    availability = macro_availability_table(acm)
    assert availability.loc["ACMTP10", "observations"] == 2
