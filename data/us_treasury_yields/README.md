# Us Treasury Yields

    Source: FRED fredgraph CSV endpoints for Treasury constant-maturity rates.

    Script: `download.py`

    Final output files:
    - `data/us_treasury_yields.csv`

    Source and download links:
    - https://fred.stlouisfed.org/series/DGS10
- https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10

    Notes:
    Values are kept in percent units because the notebooks and dataio rate loader expect percent-style par-yield inputs.

    Rebuild from the repository root:

    ```bash
    python data/us_treasury_yields/download.py
    ```

    Raw/manual files, when required, belong in this folder's `raw/` directory and should not be committed.
