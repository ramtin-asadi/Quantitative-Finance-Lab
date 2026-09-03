# Federal Reserve Credit

Downloads the Federal Reserve Board's permanent excess-bond-premium CSV and
writes `data/fed_credit.parquet`.

```powershell
.\.venv\Scripts\python.exe data\fed_credit\download.py
```

The output preserves the four official columns: month, Gilchrist-Zakrajsek
spread, excess bond premium, and model-estimated 12-month recession
probability. The source is the permanent URL documented by the Board:

https://www.federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv

The Board states that the entire history can revise. `update.py` therefore uses
HTTP validators plus a SHA-256 content comparison and atomically rewrites the
Parquet only when the source changed:

```powershell
.\.venv\Scripts\python.exe data\fed_credit\update.py
```
