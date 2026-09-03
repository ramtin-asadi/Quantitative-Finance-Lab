# Fama French Us

Source: Kenneth R. French Data Library.

Script: `download.py`

Final output files:

- `data/fama_french_us_5_factors.csv`
- `data/fama_french_us_momentum.csv`
- `data/fama_french_us_12_industries.csv`

Source and download links:

- https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

Notes: Raw percent returns are converted to decimal returns in the final files.

Rebuild from the repository root:

```bash
python data/fama_french_us/download.py
```

Raw/manual files, when required, belong in this folder's `raw/` directory and should not be committed.
