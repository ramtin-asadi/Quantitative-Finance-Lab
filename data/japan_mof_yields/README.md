# Japan Mof Yields

    Source: Japan Ministry of Finance historical Japanese Government Bond interest-rate CSV.

    Script: `download.py`

    Final output files:
    - `data/japan_mof_yields.csv`

    Source and download links:
    - https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/index.htm
- https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/historical/jgbcme_all.csv

    Notes:
    The script downloads the official MOF CSV and fails clearly if the URL, encoding, or CSV layout changes.

    Rebuild from the repository root:

    ```bash
    python data/japan_mof_yields/download.py
    ```

    Raw/manual files, when required, belong in this folder's `raw/` directory and should not be committed.
