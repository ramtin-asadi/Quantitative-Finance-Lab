# Treasury Credit Curves

Downloads the official U.S. Treasury curve workbooks and writes one tidy long
file, `data/treasury_credit_curves.parquet`.

Included data:

- HQM high-quality corporate spot curves, monthly average and month-end,
  1984–present;
- HQM corporate par yields, monthly average and month-end;
- TNC nominal Treasury spot curves, monthly average and month-end,
  1976–present;
- TNC nominal Treasury par yields, monthly average and month-end.

The output columns are `curve_family`, `rate_type`, `observation_type`, `date`,
`maturity_years`, `yield_percent`, and `source_file`. TIPS/TRC, breakeven/TBI,
quarterly averages, 10-year averages, forwards, and on-the-run tables are not
included because they are not needed for the corporate-credit curve comparison.

Official pages:

- https://home.treasury.gov/data/treasury-coupon-issues-and-corporate-bond-yield-curve/corporate-bond-yield-curve
- https://home.treasury.gov/data/treasury-coupon-issues-and-corporate-bond-yield-curves/treasury-coupon-issues

Build the full history:

```powershell
.\.venv\Scripts\python.exe data\treasury_credit_curves\download.py
```

Incremental maintenance:

```powershell
.\.venv\Scripts\python.exe data\treasury_credit_curves\update.py
```

The updater discovers only the latest rolling spot workbooks and the latest par
workbooks. It uses ETag/Last-Modified validators plus SHA-256 comparison when a
server returns a full response. Historical five-year spot blocks remain in the
ignored cache and are not requested again.
