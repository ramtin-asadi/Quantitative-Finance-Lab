# Macro Factors

    Source: FRED, Statistics Canada, and Bank of Canada APIs.

    Script: `download.py`

    Final output files:
    - `data/us_macro_factors.csv`
- `data/canada_macro_factors.csv`
- `data/macro_factor_summary.csv`
- `data/macro_download_issues.csv`

    Source and download links:
    - https://fred.stlouisfed.org/
- https://www150.statcan.gc.ca/n1/en/type/data
- https://www.bankofcanada.ca/valet/docs

    Notes:
    NFCI is intentionally handled by chicago_fed_nfci instead of this macro bundle.

    Rebuild from the repository root:

    ```bash
    python data/macro_factors/download.py
    ```

    Raw/manual files, when required, belong in this folder's `raw/` directory and should not be committed.
