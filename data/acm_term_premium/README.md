# ACM Term Premium

Builds `data/acm_term_premium.csv` from the Federal Reserve Bank of New York Adrian-Crump-Moench Treasury term-premia dataset.

## Source

- NY Fed Treasury Term Premia page: https://www.newyorkfed.org/research/data_indicators/term-premia-tabs
- Official direct file: https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls

## What To Put In `raw/`

Nothing. `download.py` downloads the official NY Fed Excel file directly and does not read manual raw files.

## Rebuild

```bash
python data/acm_term_premium/download.py
```

The output preserves all ACM yield, term-premium, and risk-neutral-yield columns that are present in the official file.
