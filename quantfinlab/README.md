# quantfinlab Library

`quantfinlab` is the reusable Python library extracted from the [Quantitative Finance Lab](https://github.com/ramtin-asadi/Quantitative-Finance-Lab) project series. It covers the methods developed across 24 projects: fixed income, options pricing, portfolio construction, risk reporting, volatility modeling, hedging, fundamental equity research, corporate credit, real-time macro, dependence networks, ML/RL and financial analysis with a local LLM.

The test suite checks real model properties rather than notebook snapshots: weights summing to one, CVaR behavior, curve/discount consistency, American option engine fallbacks, implied-volatility diagnostics, and PSD matrix reconstruction. The current package checks pass with a clean `ruff` lint pass.

## Installation

From PyPI:

```bash
pip install quantfinlab
```

For optional dependencies, install the extras you need. The source checkout defines:

```bash
pip install "quantfinlab[numerics]"   # JAX / Numba acceleration (autodiff Greeks, fast IV)
pip install "quantfinlab[volatility]" # arch, statsmodels (GARCH, HAR)
pip install "quantfinlab[hedging]"    # statsmodels (dynamic hedge ratios)
pip install "quantfinlab[ml]"         # PyTorch (sequence models, RL policies)
pip install "quantfinlab[network]"    # networkx (dependence networks)
pip install "quantfinlab[data]"       # PyArrow / DuckDB source readers
pip install "quantfinlab[credit]"     # LightGBM / statsmodels default models and source readers
pip install "quantfinlab[macro]"      # statsmodels state-space models and source readers
pip install "quantfinlab[analyst]"    # document retrieval and local financial analysis
pip install "quantfinlab[plotting]"   # matplotlib, seaborn
pip install "quantfinlab[all]"        # everything above
```

For development from the repository root:

```bash
pip install -e .
```

Source installs build the optional C++ extension (`quantfinlab._kernels`) automatically via `scikit-build-core`, CMake, and pybind11 when a C++ compiler is available. The extension accelerates the heaviest American-option, Fourier/COS, Monte Carlo, and calibration kernels, but the package also works from a pure-Python wheel when native builds are not available.

To disable native kernels during a source install:

```bash
pip install . --config-settings=cmake.define.quantfinlab_build_cpp=off
```

Functions that accept `engine="auto"` prefer C++ when available, then Numba when installed, then NumPy/SciPy fallback paths where implemented. If you explicitly request `engine="cpp"` without the extension installed, `quantfinlab` raises `MissingKernelsError` with installation and fallback guidance.

Most of the library works with just the core dependencies (NumPy, pandas, SciPy, cvxpy, and scikit-learn). Optional dependencies are checked at the point of use. Functions either fall back to a pure NumPy/SciPy implementation, or raise a clear, specific error telling you which extra to install, rather than failing on package import. Tests that need optional dependencies use `pytest.importorskip` and skip cleanly rather than fail when the dependency is absent.

## Module map

| Module | Covers |
|---|---|
| `quantfinlab.dataio` | Source-normalized data loading: yield curves, equity/ETF panels, option chains, SEC facts and filings, macro vintages and surveys; each source family returns a stable schema regardless of the underlying data vendor. See [Data loading](#data-loading) below. |
| `quantfinlab.fixed_income` | Curve bootstrapping, discounting, forward rates, bond pricing and cashflows, duration/convexity/key-rate duration, short-rate/term-structure models, swaps, scenario generation, duration-targeted laddering, overnight-rate compounding. |
| `quantfinlab.options` | Black–Scholes/Black-76 pricing, put-call parity, quote cleaning, implied volatility (Newton-bisection and "Let's Be Rational"-style solvers, with an optional Numba backend), analytic and autodiff Greeks (optional JAX), American option pricing (tree/PDE/LSM, C++ and numba backed), Fourier/COS pricing, Heston/SABR/SVI/SSVI/rough-volatility/Merton/variance-gamma models, local volatility, and model-risk diagnostics. |
| `quantfinlab.portfolio` | Expected-return models, covariance estimation (sample/Ledoit-Wolf/OAS/EWMA), mean-variance/min-variance/max-Sharpe/ridge optimizers, constraints, transaction costs, walk-forward backtesting harness, Black-Litterman (with learned-confidence views and factor/regime conditioning), HRP/NCO clustering allocation, risk parity, CVaR and robust (box/ellipsoid/Wasserstein) optimization, factor construction, regime models, dependence-network construction and network-based signals, universe selection and position sizing. |
| `quantfinlab.risk` | VaR/expected shortfall (historical, Cornish-Fisher, filtered historical simulation), VaR backtesting, drawdown analysis, performance metrics, CAPM beta, correlation diagnostics, stress testing, risk contribution/attribution. |
| `quantfinlab.volatility` | Realized-volatility estimators, GARCH/HAR forecasting, rough-volatility estimation, variance risk premium analysis. |
| `quantfinlab.hedging` | Dynamic hedge-ratio estimation, hedge policies, residual-spread construction, hedging performance metrics. |
| `quantfinlab.macro` | Financial-conditions indicators and allocation; vintage/as-of transforms, completed-month GDP bridges, inflation components, MIDAS, grouped mixed-frequency DFM and Kalman news, monthly Minnesota BVAR, policy forecasts and release evaluation. |
| `quantfinlab.fundamentals` | Point-in-time statement reconstruction, accounting ratios, corporate/financial-company diagnostics, peer scoring, stock selection and score validation. |
| `quantfinlab.credit` | Filing-based default labels, logit/spline/LightGBM default models, Merton inversion, intensity curves, CDS valuation, credit portfolios and tranche losses. |
| `quantfinlab.ml` | Feature engineering, forecasting evaluation (rank metrics, pinball loss, coverage), probabilistic/uncertainty models, release-aware validation and forecast combinations, sequence models (e.g. TCN-based forecasters), regime classifiers, RL environments, reward shaping (differential Sharpe ratio), and RL policies (PPO, recurrent PPO, SAC). |
| `quantfinlab.backtest` | Shared backtesting engines for portfolios, fixed income, hedging, and options strategies, with cost models and overlay support. |
| `quantfinlab.reports` | Risk and fundamental equity reports, combining outputs from `risk`, `portfolio`, and `plotting` into a single executive summary. |
| `quantfinlab.numerics` | Finite-difference schemes, Fourier transforms, interpolation, Monte Carlo path generation, Gaussian and Student-t copulas. Shared numerical primitives used across pricing, portfolios and risk. |
| `quantfinlab.calibration` | Model calibration. American option numerics, FFT/COS calibration, jump-diffusion model fitting, LSM regression. |
| `quantfinlab.analyst` | Official documents, FTS5 retrieval, financial contexts, cached local GGUF inference and readable company, macro, market and daily reports. |
| `quantfinlab.plotting` | Consistent plotting utilities per domain (curves, options, portfolio, risk, volatility, macro, regimes, ML, hedging, fixed income, fundamentals, credit) and explanatory diagrams. |
| `quantfinlab.common` | Shared contracts/dataclasses (`Curve`, `Bond`, `PortfolioState`, `BacktestResult`, ...), error types, date utilities, cache identities, and input validation used across every other module. |

The optional C++ pricing kernels (`cpp/`, exposed as `quantfinlab._kernels`) implement the LSM regression solver, the PSOR finite-difference PDE solver, the binomial tree, Monte Carlo paths, and the Fourier/COS pricer. They are written in C++ and bound via pybind11 for speed; pure-Python installs keep the public APIs importable and use automatic fallbacks where those methods exist.

## Data loading

`quantfinlab.dataio` is what every notebook uses to turn a raw downloaded file into a clean, analysis-ready dataset. It is also the boundary the project series treats most carefully. see the [project README](https://github.com/ramtin-asadi/Quantitative-Finance-Lab#data-and-reproducibility) and [data README](https://github.com/ramtin-asadi/Quantitative-Finance-Lab/tree/main/data) for how raw data is obtained in the first place, `dataio` is what runs after having the data files for turning them into analysis ready dataframes.

```python
from quantfinlab.dataio import load_par_yield_curve, load_yfinance_panel

us_curve = load_par_yield_curve("data/us_treasury_yields.csv", source="us_treasury")
jp_curve = load_par_yield_curve("data/japan_mof_yields.csv", source="japan_mof")

panel = load_yfinance_panel(
    "data/core_cross_asset_etfs.csv", fields=("close", "volume"),
    source="yfinance_export", start="2010-01-01")

close, volume = panel["close"], panel["volume"]
```

Loaders normalize each source family to a consistent shape. par-yield curves come back as a `DatetimeIndex`-sorted DataFrame with standard tenor columns (`1M`...`30Y`) in decimal form. equities and prices come back as `{field: DataFrame}` with tickers as columns, numeric-coerced, deduplicated, sorted. This is what makes the "secondary market" repeat at the end of most notebooks possible with no extra glue code. swap the `path`/`source`, get the same schema back. SEC readers retain filing versions and availability dates; real-time macro readers retain observation dates separately from publication or snapshot dates. A vintage label is not automatically an exact release timestamp.

## Examples

A few representative examples. See the notebooks for the full derivations behind each of these and the full repeat of each project completely using the library.

### Mean-variance portfolio optimization with turnover control

```python
from quantfinlab.portfolio import covariance, optimizers

cov_ann = covariance.estimate_covariance(returns_window, method="LedoitWolf", return_df=True)

weights = optimizers.mean_variance(
    mu_excess_ann=expected_excess_returns,
    cov_ann=cov_ann, mv_lambda=6.0,
    w_max=0.25, turnover_penalty_bps=10.0,
    long_only=True)
```

`optimizers` also exposes `equal_weight`, `minimum_variance`, `ridge_mean_variance`, `max_sharpe_slsqp`, and `max_sharpe_frontier_grid` with the same calling convention, plus a `walkforward` module that runs any of these through a full rolling rebalance/backtest loop given a returns panel and rebalance schedule.

### Implied volatility from a market quote

```python
from quantfinlab.options import iv

sigma = iv.implied_vol(
    option_type="call", price=4.35, forward=101.2,
    strike=100.0, tau=30 / 365,
    engine="auto", solver="lbr_lite")
```

For a full option-chain DataFrame at once, `iv.compute_iv_table(quotes, ...)` runs the same solver vectorized across all rows and returns solver-status diagnostics alongside the implied vols (used in the notebooks to check solver success rate and pricing residuals before trusting a fitted surface).

### Risk parity and CVaR-aware allocation

```python
from quantfinlab.portfolio import risk_parity, cvar

erc_weights = risk_parity.equal_risk_contribution_weights(cov_ann, tickers=cov_ann.index, w_max=0.40)
contrib_table = risk_parity.risk_contribution_table(erc_weights, cov_ann)

cvar_weights = cvar.min_cvar_weights(returns_window, alpha=0.95, w_max=0.40)
loss = cvar.portfolio_cvar_loss(returns_window, cvar_weights, alpha=0.95)
```

### Yield curve construction and bond pricing

```python
from quantfinlab.dataio import load_par_yield_curve
from quantfinlab.fixed_income import bootstrap, bond_pricing

curve = load_par_yield_curve("data/us_treasury_yields.csv", source="us_treasury")
discount_curve = bootstrap.bootstrap_discount_factors(curve.loc["2024-06-01"])

price = bond_pricing.bond_price(coupon=0.045, maturity_years=10.0, df_func=discount_curve, freq=2)
```

### A full risk report

```python
from quantfinlab.reports import risk_report

report = risk_report.risk_report(
    returns=strategy_returns, benchmark_returns=benchmark_returns,
    weights=weights_history, cov_ann=cov_ann)
```

This produces the same VaR/ES, drawdown, CAPM, and risk-contribution summary used throughout Project 03 and applied to every backtest in later projects — one call instead of re-deriving the report each time.


### Default probabilities and CDS pricing

```python
from quantfinlab.credit.curves import hazards_from_pd
from quantfinlab.credit.pricing import cds_spread
from quantfinlab.fixed_income.discounting import discount_from_zero

T = [0.25, 0.5, 1.0, 2.0]
lambda_q = hazards_from_pd([0.003, 0.006, 0.015, 0.035], T)
discount = discount_from_zero(T, [0.035, 0.034, 0.033, 0.032])
spread = cds_spread(lambda_q, T, discount, 2.0, R=0.4)
```

Rates, probabilities and recoveries are decimals; maturities are years. Pricing requires a risk-neutral hazard assumption, distinct from an empirical default forecast. The CDS functions use explicit cash-flow approximations, not the ISDA standard model; `cds_cs01` bumps spreads and recalibrates the hazard curve, while `risky_pv01` measures annuity sensitivity.

### A historical macro information set

```python
from quantfinlab.dataio.realtime import read_alfred
from quantfinlab.macro.realtime import vintage_asof, release_growth

releases = read_alfred("data/alfred_realtime.parquet", series=["INDPRO"])
known = vintage_asof(releases, "2020-04-15", wide=True)
first_growth = release_growth(releases, periods=1, scale=1200, max_delay=100)
```

Growth uses numerator and denominator from the same release vintage. Archive backfill is not automatically first-release truth; `max_delay` is a bound measured from the reference period's start. State-space news, factor-augmented forecasts and the monthly BVAR with a separate quarterly bridge remain distinct APIs; their assumptions and return values are documented in the functions.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

Tests are organized to mirror the package layout (`tests/portfolio`, `tests/options`, `tests/cpp`, ...) and use a small set of deterministic synthetic-data generators (`tests/synthetic/generators.py`) rather than real market data, so the suite is fast, reproducible, and has no external data dependency.
