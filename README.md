# Quantitative Finance Lab

[![Website](https://img.shields.io/badge/Website-QuantFinLab-0A66C2?logo=githubpages&logoColor=white)](https://ramtin-asadi.github.io/Quantitative-Finance-Lab/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Quantitative Finance Lab** is a research and engineering sequence of end-to-end projects. It can be used a **self-study curriculum** that forces every model to be derived, coded, and tested against real market data, as a **research notebook series** that treats each topic the way a working paper would (motivation, mathematics, implementation, diagnostics, discussion), and as a **reusable Python library**, `quantfinlab`, that turns the parts of each project into tested, documented, importable code instead of just the notebook cells.

In every project, we do the same thing: build the model, run it on real data with realistic frictions (costs, liquidity, look-ahead controls), write the reusable version into the library, and then prove the library version works by re-running the same workflow on a **second, independent dataset** (usually a different market, asset class, or country). A model that only works on the data it was built on hasn't been tested; it's been fit. All projects use only the extracted library code for the second run in notebook's final cell, and repeated parts from earlier notebooks also use library instead of re-implementing something.

- Every notebook explains the mathematics before the code, with derivations.
- Every reusable component lives in `quantfinlab`, has type hints, has tests, and is importable outside the notebook that created it (like `from quantfinlab.options import bsm`)
- Every dataset used has a documented, scripted path back to its source (see [Data](#data-and-reproducibility) below).

## Repository structure

```text
notebooks/       20 notebooks, rendered into the project website
quantfinlab/      Reusable Python library extracted from the notebooks (see quantfinlab/README.md)
data/             Data reproducibility layer: one script per data source, no redistributed data
tests/            pytest suite for the library (171 tests, run in CI)
cpp/              C++ pricing kernels (LSM, PDE, tree, Fourier/COS) bound to Python via pybind11
config/           Local, non-sensitive configuration
docs/             Rendered GitHub Pages site (built by Quarto)
```

## How to read this project

The rendered website is the best way to read the work. code, output, and plots all together are easily visible in website:

**https://ramtin-asadi.github.io/Quantitative-Finance-Lab/**

## Projects

**01. Yield Curve, Bond Pricing & Risk.** Builds discount, zero-rate, and forward curves from par-yield data and prices bonds off the resulting curve. Quantifies fixed-income risk through duration, convexity, PV01, and key-rate duration, then tests a duration-targeted bond ladder.

**02. Portfolio Optimization (Mean–Variance Models).** Compares equal-weight, minimum-variance, mean-variance, max-Sharpe (Tangency), and frontier allocation under realistic walk-forward rebalancing. Covers liquidity-based universe selection, multiple return/covariance estimators, regularization, and transaction costs.

**03. Risk Report Engine & CAPM.** Builds an institutional-style risk report (drawdowns, VaR/ES with backtesting, stress scenarios, CAPM beta, risk contribution) and applies it to the portfolios produced in Project 02 and stocks.

**04. Black–Scholes, IV, Greeks & Hedging.** Implements Black–Scholes pricing, put-call parity checks, fast implied-volatility solvers, and analytic/autodiff Greeks, then evaluates delta and delta-vega hedging P&L under transaction costs and quote noise.

**05. GARCH Forecasting & Variance Risk Premium.** Forecasts realized volatility with GARCH-family models and compares the forecast to option-implied volatility to study the variance risk premium and its trading implications.

**06. Black–Litterman with Learned Confidence.** Uses uses equilibrium returns, views, confidence mapping, posterior expected returns, and constrained portfolio optimization. Extends classical Black–Litterman by learning the confidence assigned to each view from data rather than fixing it by hand, and evaluates the resulting allocations against a benchmark.

**07. Dynamic Hedge Ratios & Residual Trading.** Estimates time-varying hedge ratios between related instruments using static, rolling, and dynamic methods, and builds a residual-based relative-value strategy around the spread that's left over.

**08. Volatility Surface & Local Volatility.** Fits arbitrage-aware implied-volatility surfaces and derives the corresponding local-volatility surface via Dupire's formula, then checks pricing consistency against quoted option prices.

**09. Short-Rate & Term-Structure Models.** Calibrates short-rate models to the yield curve and term-premium data, builds PCA curve shocks, uses them for scenario generation, duration overlays, and swap-style overlays.

**10. Tail Risk, Risk Parity & Robust Portfolios.** Builds CVaR-aware and robust (box/ellipsoid/Wasserstein) portfolio optimizers that explicitly account for estimation error and tail risk, compared against risk-parity, HRP and classical mean-variance baselines.

**11 — Stochastic Volatility & Model-Risk Relative Value.** Calibrates Heston, SABR, SVI/SSVI, Merton and Bates models to the option surface and uses the spread between model families as a model-risk-aware relative-value signal for option trading.

**12. Macro Financial Conditions Index.** Builds a financial-conditions index from macro and market variables using transformations, standardization, dimensionality reduction or supervised components, compares to the Chicago Fed NFCI, and uses it for stress classification and allocation rules.

**13. American Options & Numerical Pricing.** Prices American options with binomial trees, finite-difference PDE solvers (PSOR), and Longstaff–Schwartz Monte Carlo, with the performance-critical kernels written in C++ (bound to Python via pybind11) and Numba.

**14. Fourier-Based Option Pricing.** Implements Carr–Madan FFT and COS-method pricing under Lévy/affine models (including Merton jump-diffusion, Heston and Variance Gamma) and benchmarks them for speed and accuracy against direct integration, again backed by compiled C++ and Numba kernels.

**15. Factor Investing.** uses Fama-French factors, industry portfolios, factor proxies and validation-weighted scoring for evaluating portfolio construction choices and factor-timing.

**16. Regime-Switching Portfolio Allocation.** Detects macro and market regimes with different Econometrics and Machine learning models (Markov-switching, clustering, or classification) and adapts portfolio allocation rules conditional on the detected regime.

**17. Network Portfolio Construction.** Builds dependence networks (Dense, MST and PMFG) from the equity panel correlation and Copula tail dependence, and uses network structure for portfolio construction using centrality measures.

**18. Rough Volatility.** Implements rough Bergomi and rough Heston-style models, estimates the Hurst roughness parameter from realized variance, and compares rough-volatility dynamics against classical (Markovian) stochastic-volatility models.

**19 — ML Forecasting & Kelly Allocation.** Builds return-forecasting models (gradient-boosted trees, sequence models, probabilistic models, Neural Networks) with proper evaluation (rank metrics, pinball loss, coverage), and converts forecasts into position sizes through a fractional-Kelly allocation and a Forecast-Gated Max-Sharpe model.

**20. RL Portfolio Allocation.** builds reinforcement-learning environment,trains PPO, recurrent PPO, and SAC policies to allocate across the asset set directly, using a differential Sharpe ratio reward, and evaluates the learned policies against the rule-based strategies from earlier projects.

## Link to each project and data used:

| # | Project | Primary data | Secondary (library-only repeat) | Link |
|---|---|---|---|---|
| 01 | Yield Curve, Bond Pricing & Risk | US Treasury par yields (FRED) | Japan JGB par yields (MOF) | [Open »](https://ramtin-asadi.github.io/Quantitative-Finance-Lab/notebooks/01_yield_curve_bond_pricing_and_risk.html) |
| 02 | Portfolio Optimization (Mean–Variance) | NASDAQ US equities (Stooq) | Hong Kong equities (HKEX) | [Open »](https://ramtin-asadi.github.io/Quantitative-Finance-Lab/notebooks/02_portfolio_optimization_MV_models.html) |
| 03 | Risk Report Engine & CAPM | Portfolios from Project 02 (US equities), NVDA and AAPL | HKEX portfolios from project 2 | [Open »](https://ramtin-asadi.github.io/Quantitative-Finance-Lab/notebooks/03_risk_report_engine_and_CAPM.html) |
| 04 | Black–Scholes, IV, Greeks & Hedging | Equity index options (SPX option chain) | BTC options (Deribit) | [Open »](https://ramtin-asadi.github.io/Quantitative-Finance-Lab/notebooks/04_BSM_IV_greeks_hedging.html) |
| 05 | GARCH Forecasting & Variance Risk Premium | Equity index returns/options (SPX) | BTC returns/options (Deribit) | [Open »](https://ramtin-asadi.github.io/Quantitative-Finance-Lab/notebooks/05_GARCH_forecasting_VRP.html) |
| 06 | Black–Litterman with Learned Confidence | Global cross-asset ETFs | US sector ETFs | [Open »](https://ramtin-asadi.github.io/Quantitative-Finance-Lab/notebooks/06_portfolio_black_littreman.html) |
| 07 | Dynamic Hedge Ratios & Residual Trading | Cross-asset / sector ETF pairs | International hedging ETFs (EW of different countries stock markets) | [Open »](https://ramtin-asadi.github.io/Quantitative-Finance-Lab/notebooks/07_dynamic_hedge_ratios.html) |
| 08 | Volatility Surface & Local Volatility | Equity index options (SPX) | BTC options (Deribit) | [Open »](https://ramtin-asadi.github.io/Quantitative-Finance-Lab/notebooks/08_vol_surface_local_vol.html) |
| 09 | Short-Rate & Term-Structure Models | US Treasury yields + ACM term premia | Japan JGB yields (MOF) | [Open »](https://ramtin-asadi.github.io/Quantitative-Finance-Lab/notebooks/09_term_structure_models.html) |
| 10 | Tail Risk, Risk Parity & Robust Portfolios | Cross-asset ETFs | US sector ETFs | [Open »](https://ramtin-asadi.github.io/Quantitative-Finance-Lab/notebooks/10_portfolio_tail_parity_robust.html) |
| 11 | Stochastic Volatility & Model-Risk RV | Equity index options (SPX) | BTC options (Deribit) | [Open »](https://ramtin-asadi.github.io/Quantitative-Finance-Lab/notebooks/11_stochastic_volatility.html) |
| 12 | Macro Financial Conditions Index | US macro factors and sector ETFs | Canada macro factors and sector ETFs | [Open »](https://ramtin-asadi.github.io/Quantitative-Finance-Lab/notebooks/12_financial_conditions_index.html) |
| 13 | American Options & Numerical Pricing | American SPY options | QQQ options | [Open »](https://ramtin-asadi.github.io/Quantitative-Finance-Lab/notebooks/13_american_option_numerics.html) |
| 14 | Fourier-Based Option Pricing | SPX index options | BTC options (Deribit) | [Open »](https://ramtin-asadi.github.io/Quantitative-Finance-Lab/notebooks/14_fourier_option_pricing.html) |
| 15 | Factor Investing | US factor-proxy ETFs + Fama-French US factors | International country ETFs + Fama-French developed ex-US factors | [Open »](https://ramtin-asadi.github.io/Quantitative-Finance-Lab/notebooks/15_factor_investing.html) |
| 16 | Regime-Switching Portfolio Allocation | US cross-asset and macro factors | Sector ETFs | [Open »](https://ramtin-asadi.github.io/Quantitative-Finance-Lab/notebooks/16_regime_switching_portfolio.html) |
| 17 | Network Portfolio Construction | US equities (Nasdaq) | Hong Kong equities (HKEX) | [Open »](https://ramtin-asadi.github.io/Quantitative-Finance-Lab/notebooks/17_network_portfolio.html) |
| 18 | Rough Volatility (rBergomi / rough Heston) | Equity index options (SPX) | BTC options (Deribit) | [Open »](https://ramtin-asadi.github.io/Quantitative-Finance-Lab/notebooks/18_rough_volatility.html) |
| 19 | ML Forecasting & Kelly Allocation | cross-asset ETFs and macro factors | US Sector ETFs | [Open »](https://ramtin-asadi.github.io/Quantitative-Finance-Lab/notebooks/19_forecasting_kelly_allocation.html) |
| 20 | RL Portfolio Allocation | cross-asset ETFs and macro factors | US Sector ETFs | [Open »](https://ramtin-asadi.github.io/Quantitative-Finance-Lab/notebooks/20_rl_portfolio_allocation.html) |


## Data and reproducibility

**The notebooks are not directly re-runnable on a fresh clone.** I order to avoid legal and copy-right problems, Large or licensed market data is never committed to this repository. But there is a `data/` folder: a reproducibility layer of scripts, and README files for guidance of downloading data, one per data source, that either download data automatically if API exists (FRED, NY Fed, Bank of Japan MOF, yfinance, Kenneth French data library, Stooq) or tell you exactly what licensed file to place where (OptionsDX option chains, Stooq bulk equity files) and scripts for processing the raw files to the files that can be exactly used in notebooks.

To reproduce a notebook:

1. Read `data/README.md` for the full source list and the one-line description of what each script does.
2. For automatic sources, just run the relevant script, like `python data/us_treasury_yields/download.py`.
3. For manual/licensed sources (OptionsDX option chains, Stooq bulk downloads), download the files yourself from the linked source and drop them in the matching `data/<source>/raw/` folder, then run the corresponding `build.py`.
4. Run the notebooks.

See `data/README.md` for the complete script list, raw-folder checklist, and licensing/citation notes per source.

## The library: `quantfinlab`

`quantfinlab` is the part of this repository meant to be more reusable than individual notebooks. It is a typed, tested Python package covering fixed income, options pricing (including a compiled C++ pricing core), portfolio construction, risk reporting, volatility modeling, hedging, macro indicators, and ML/RL applications for finance. See **[`quantfinlab/README.md`](quantfinlab/README.md)** for the full module map, installation instructions, and worked code examples (mean-variance optimization, implied volatility, yield-curve loading, and more).

Quick example: mean-variance optimization with turnover control:

```python
from quantfinlab.portfolio import covariance, optimizers

cov_ann = covariance.estimate_covariance(returns_window, method="LedoitWolf", return_df=True)
weights = optimizers.mean_variance(
    mu_excess_ann=expected_excess_returns,
    cov_ann=cov_ann, w_max=0.25,
    turnover_penalty_bps=10.0)
```

Quick example: implied volatility from a quote:

```python
from quantfinlab.options import iv

sigma = iv.implied_vol(
    option_type="call", price=4.35, forward=101.2, strike=100.0, tau=30 / 365)
```

Full installation instructions, the complete module list, and more examples are in [`quantfinlab/README.md`](quantfinlab/README.md).

## Installation

```bash
git clone https://github.com/ramtin-asadi/Quantitative-Finance-Lab.git
cd Quantitative-Finance-Lab

python -m venv .venv
source .venv/Scripts/activate  

pip install -U pip
pip install -e .
```

The C++ extension (used by Projects 13 and 14) builds automatically on install via `scikit-build-core`, `pybind11`, and CMake. A working C++ compiler and CMake must be available, but no manual build step is required. Optional dependencies (JAX/Numba for acceleration, PyTorch for ML/RL, `networkx` for the network-portfolio project) are listed as extras in `pyproject.toml`. The library has working fallbacks when they're absent, so `pip install -e ".[all]"` is only needed to run every notebook end to end. for example if running projects 19 and 20 is not needs, user doesn't need to install torch. See `pyproject.toml` for the full extras list, or `quantfinlab/README.md` for the per-module dependency breakdown.

Run the test suite with:

```bash
pip install -e ".[dev]"
pytest
```

## Status

Under active development. Twenty projects are complete with library extraction, and rendered output, ongoing work is on deepening interpretation/discussion sections, completing markdowns for all projects, closing remaining test-coverage gaps, and a first tagged PyPI release of `quantfinlab`.

## Disclaimer

This repository is for research, education, and demonstration of quantitative methods only. Nothing here is investment advice. All results depend on data quality, modeling assumptions, transaction costs, constraints, the market regime studied, and implementation details. They should not be read as claims about live trading performance.