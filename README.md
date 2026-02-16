# quantitative-finance-lab


[![Pages](https://img.shields.io/badge/Website-QuantFinLab-0A66C2?logo=githubpages&logoColor=white)](https://ramtin-asadi.github.io/Quantitative-Finance-Lab/)

This repository is organized as a **series of end-to-end projects** (research notes + mathematic background + reproducible code + visual results) and a growing Python library, **`quantfinlab`**, that collects reusable building blocks developed across the projects. The projects are published as a Quarto site for easy reading and sharing.

## What’s inside

- **Project notebooks (Quarto)**: narrative, math, experiments, and plots  
  Located in `notebooks/` and rendered into the published site.
- **Reusable Python library**: production-style modules shared across projects  
  Located in `quantfinlab/` (e.g., fixed income utilities, portfolio tools, plotting helpers).
- **Reproducibility tooling**: formatting, linting, tests, CI  
  Configuration in `pyproject.toml`, `.github/workflows/`, and `.pre-commit-config.yaml`.

## Projects (current)

1. **Yield Curve Construction, Bond Pricing, and Risk**
   - Curve construction with four different models (discount factors / zero rates), simulating a bond portfolio, bond pricing, and risk metrics
   - Duration/convexity, PV sensitivities, and implementing the whole project on Japan data using quantfinlab

2. **Portfolio Optimization (Mean–Variance Models)**
   -  multiple covariance estimators and mean momentum estimator.
   - implementation of Mean-Variance, Min-Variance and Max-Sharpe models
   - Backtest-style evaluation and comparison across model choices
   - implementation on Hong Kong stock market with quantfinlab

> New projects will be added continuously with the same structure: clean narrative + reusable code extracted into `quantfinlab`.

Getting started
1) Clone the repository
```
git clone https://github.com/ramtin-asadi/Quantitative-Finance-Lab.git
cd Quantitative-Finance-Lab
```


2) Create a virtual environment and install dependencies

Using pip:

```python
-m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -U pip
pip install -e .
```
---

## Running the projects
### Option A — Read the rendered site

The projects are published as a Quarto website (recommended for browsing results).

### Option B — Run locally

Open notebooks in Jupyter/VS Code and run from top to bottom.


***Note:*** datasets are not committed to the repository by default due to legal issues. If a notebook expects local data files, place them under data/ (ignored by git) and update paths as needed. every notebook has the link for downloading data from the official source.