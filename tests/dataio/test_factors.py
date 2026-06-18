from __future__ import annotations

import pandas as pd

from quantfinlab.dataio.factors import load_ff_factors, load_ff_industries, load_ff_momentum


def test_fama_french_reader_uses_first_monthly_table_and_decimal_returns(tmp_path) -> None:
    path = tmp_path / "ff.csv"
    path.write_text(
        "\n".join(
            [
                "This is a synthetic header line",
                ",Mkt-RF,SMB,HML,RF",
                "202401,1.20,-0.50,0.30,0.40",
                "202402,-99.99,0.10,0.20,0.35",
                "Annual Factors: January-December",
                "2024,12.0,1.0,2.0,4.0",
            ]
        ),
        encoding="utf-8",
    )

    factors = load_ff_factors(path, columns=["Mkt-RF", "RF"])
    momentum = load_ff_momentum(path, name="MOM")
    industries = load_ff_industries(path)

    assert factors.index[0] == pd.Timestamp("2024-01-31")
    assert factors.loc[pd.Timestamp("2024-01-31"), "Mkt-RF"] == 0.012
    assert pd.isna(factors.loc[pd.Timestamp("2024-02-29"), "Mkt-RF"])
    assert momentum.name == "MOM"
    assert industries.shape == (2, 4)
