# Us Treasury Yields

    Source: U.S. Treasury daily par yield curve CSV.

    Script: `download.py`

    Final output files:
    - `data/us_treasury_yields.csv`

    Source and download links:
    - https://home.treasury.gov/resource-center/data-chart-center/interest-rates
    - https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml

    Notes:
    Values are kept in percent units because the notebooks and dataio rate loader expect percent-style par-yield inputs.

    Rebuild from the repository root:

    ```bash
    python data/us_treasury_yields/download.py
    ```

    Raw/manual files, when required, belong in this folder's `raw/` directory and should not be committed.
