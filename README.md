# Quantitative Finance Lab

[![Website](https://img.shields.io/badge/Website-QuantFinLab-0A66C2?logo=githubpages&logoColor=white)](https://ramtin-asadi.github.io/Quantitative-Finance-Lab/)

**Quantitative Finance Lab** is a project-based portfolio of financial engineering and quantitative finance work. Each project is developed as an end-to-end research notebook: motivation, mathematical background, implementation, experiments, diagnostics, interpretation, and reusable Python code.

The purpose of this repository is to show how financial models can be studied, implemented on real market data, analyzed with realistic assumptions, and then refactored into reusable code through the package, **`quantfinlab`**.

This repository is designed to show a combination of:

- financial mathematics and model intuition;
- real-market data cleaning and experiment design;
- reusable Python implementation of quantitative workflows;
- realistic evaluation with costs, constraints, diagnostics, and failure modes;
- clear communication of results through notebooks, plots, and written interpretation.

The notebooks are the main research artifacts. The library extracts the parts that are reusable enough to run again on secondary datasets, alternative markets, or later projects.

## Current projects

### 01. Yield Curve Construction, Bond Pricing, and Risk

Builds yield curves from market rate data, converts rates into discount factors and forward curves, prices bond portfolios, and analyzes fixed-income risk through duration, convexity, PV01, and key-rate duration. The project also studies a duration-targeted ladder strategy and applies the reusable fixed-income workflow to a secondary interest-rate market.

### 02. Portfolio Construction: Mean Variance models

Compares equal-weight, minimum-variance, mean-variance, max-Sharpe, and frontier-style allocation methods under walk-forward rebalancing. The project includes liquidity-based universe selection, different expected-return estimators and covariance estimators, regularization, transaction costs, turnover control, grid search, and performance diagnostics. The reusable portfolio workflow is then applied to a secondary equity market.

### 03. Risk Report Engine and CAPM

Builds an institutional style risk analysis workflow for assets and strategy portfolios. The report covers performance, drawdowns, distribution diagnostics, VaR and expected shortfall, VaR backtesting, stress testing, CAPM beta analysis, rolling beta behavior, and risk contribution. The project connects the portfolio outputs from Project 02 to a reusable risk-reporting pipeline.

### 04. Black–Scholes, Implied Volatility, Greeks, and Hedging

Develops a market-facing options workflow around Black–Scholes pricing, put-call parity, implied volatility, bid/ask-aware quote cleaning, fast IV solvers, analytic and autodiff Greeks, Greek uncertainty, and hedging P&L. The project compares numerical solvers, evaluates quote quality, and applies the reusable options workflow to a secondary options dataset.

## Repository structure

```text
notebooks/       Research notebooks rendered into the project website
quantfinlab/     Reusable Python components extracted from the notebooks
config/          Shared configuration files
.github/         CI and project automation
docs/            Rendered GitHub Pages site
```

## How to read the project

The rendered website is the best way to read the notebooks:

https://ramtin-asadi.github.io/Quantitative-Finance-Lab/

For each project, start with the notebook narrative, then read the final library implementation section. The notebook explains the research logic; the final section shows how the reusable components can reproduce the workflow on another dataset or market.

## Installation

Clone the repository:

```bash
git clone https://github.com/ramtin-asadi/Quantitative-Finance-Lab.git
cd Quantitative-Finance-Lab
```

Create and activate a virtual environment:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install the package in editable mode:

```bash
pip install -U pip
pip install -e .
```

Some optional functionality may require additional packages depending on the project, such as optimization, JAX and Numba dependencies. but the library mostly has fallbacks for when these dependencies are not available.

## Data

Large or licensed datasets are not committed to the repository. When a notebook requires local data, the notebook explains the expected file location, source, and required format.
The data folder and better guidance for downloading data and scripts for processing them will be added in the future for more reproducibility.

## Status

This repository is under active development. The current focus is on completing the implemented notebooks, improving interpretation and result discussion, and gradually moving repeated workflows into `quantfinlab`.

## Disclaimer

This repository is for research, education, and portfolio demonstration only. Nothing here is investment advice. Results depend on data quality, assumptions, costs, constraints, market regime, and implementation details.