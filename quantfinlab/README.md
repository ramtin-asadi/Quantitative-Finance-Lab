# quantfinlab Library

`quantfinlab` is the reusable Python library extracted from the [Quantitative Finance Lab](../README.md) project series. It is not a general-purpose finance library. it is a focused, package covering exactly the methods developed across the twenty projects: fixed income, options pricing, portfolio construction, risk reporting, volatility modeling, hedging, macro indicators, dependence networks, and ML/RL components for finance.

Every public function is covered by at least one test in `tests/` that checks a real property of the model (weights summing to one, CVaR being bounded by the benchmark, a PSD square root reconstructing its matrix). 171 tests pass in CI, alongside a clean `ruff` lint pass.

## Installation

From the repository root:

```bash
pip install -e .
```

This builds the C++ extension (`quantfinlab._kernels`, used by `options.american` and `calibration`) automatically via `scikit-build-core`, CMake, and pybind11. a C++ compiler and CMake need to be available, but no manual build step is required.

Most of the library works with just the core dependencies (NumPy, pandas, SciPy, cvxpy, scikit-learn, matplotlib). A handful of modules need optional extras, installed as needed:

```bash
pip install -e ".[numerics]"   # JAX / Numba acceleration (autodiff Greeks, fast IV)
pip install -e ".[volatility]" # arch, statsmodels (GARCH, HAR)
pip install -e ".[hedging]"    # statsmodels (dynamic hedge ratios)
pip install -e ".[ml]"         # PyTorch (sequence models, RL policies)
pip install -e ".[network]"    # networkx (dependence networks)
pip install -e ".[plotting]"   # seaborn
pip install -e ".[all]"        # everything above
```

**Every module degrades without its optional dependency**. Functions either fall back to a pure NumPy/SciPy implementation, or raise a clear, specific error telling you which extra to install, rather than failing on import. This is enforced by the test suite: tests that need an optional dependency use `pytest.importorskip` and skip cleanly rather than fail when the dependency is absent.

## Module map

| Module | Covers |
|---|---|
| `quantfinlab.dataio` | Source-normalized data loading: yield curves, equity/ETF panels, option chains, macro series, every loader returns a stable schema regardless of the underlying data vendor. See [Data loading](#data-loading) below. |
| `quantfinlab.fixed_income` | Curve bootstrapping, discounting, forward rates, bond pricing and cashflows, duration/convexity/key-rate duration, short-rate/term-structure models, swaps, scenario generation, duration-targeted laddering. |
| `quantfinlab.options` | Black–Scholes/Black-76 pricing, put-call parity, quote cleaning, implied volatility (Newton-bisection and "Let's Be Rational"-style solvers, with an optional Numba backend), analytic and autodiff Greeks (optional JAX), American option pricing (tree/PDE/LSM, C++ and numba backed), Fourier/COS pricing, Heston/SABR/SVI/SSVI/rough-volatility/Merton/variance-gamma models, local volatility, and model-risk diagnostics. |
| `quantfinlab.portfolio` | Expected-return models, covariance estimation (sample/Ledoit-Wolf/OAS/EWMA), mean-variance/min-variance/max-Sharpe/ridge optimizers, constraints, transaction costs, walk-forward backtesting harness, Black-Litterman (with learned-confidence views and factor/regime conditioning), HRP/NCO clustering allocation, risk parity, CVaR and robust (box/ellipsoid/Wasserstein) optimization, factor construction, regime models, dependence-network construction and network-based signals, universe selection and position sizing. |
| `quantfinlab.risk` | VaR/expected shortfall (historical, Cornish-Fisher, filtered historical simulation), VaR backtesting, drawdown analysis, performance metrics, CAPM beta, correlation diagnostics, stress testing, risk contribution/attribution. |
| `quantfinlab.volatility` | Realized-volatility estimators, GARCH/HAR forecasting, rough-volatility estimation, variance risk premium analysis. |
| `quantfinlab.hedging` | Dynamic hedge-ratio estimation, hedge policies, residual-spread construction, hedging performance metrics. |
| `quantfinlab.macro` | Macro indicator construction (financial-conditions index, NFCI-style PCA), macro-conditioned allocation models. |
| `quantfinlab.ml` | Feature engineering, forecasting evaluation (rank metrics, pinball loss, coverage), probabilistic/uncertainty models, sequence models (e.g. TCN-based forecasters), regime classifiers, RL environments, reward shaping (differential Sharpe ratio), and RL policies (PPO, recurrent PPO, SAC). |
| `quantfinlab.backtest` | Shared backtesting engines for portfolios, fixed income, hedging, and options strategies, with cost models and overlay support. |
| `quantfinlab.reports` | risk report generation, combining outputs from `risk`, `portfolio`, and `plotting` into a single executive summary. |
| `quantfinlab.numerics` | Finite-difference schemes, Fourier transforms, interpolation, Monte Carlo path generation. Shared numerical primitives used across `options` and `calibration`. |
| `quantfinlab.calibration` | Model calibration. American option numerics, FFT/COS calibration, jump-diffusion model fitting, LSM regression. |
| `quantfinlab.plotting` | Consistent plotting utilities per domain (curves, options, portfolio, risk, volatility, macro, regimes, ML, hedging, fixed income) and explanatory diagrams. |
| `quantfinlab.common` | Shared contracts/dataclasses (`Curve`, `Bond`, `PortfolioState`, `BacktestResult`, ...), error types, date utilities, and input validation used across every other module. |

The C++ pricing kernels (`cpp/`, exposed as `quantfinlab._kernels`) implement the LSM regression solver, the PSOR finite-difference PDE solver, the binomial tree, and the Fourier/COS pricer (the numerically heaviest loops in `options.american` and `calibration`), written in C++ and bound via pybind11 for speed, with a pure-Python fallback path used automatically when the extension isn't built.

## Data loading

`quantfinlab.dataio` is what every notebook uses to turn a raw downloaded file into a clean, analysis-ready dataset. It is also the boundary the project series treats most carefully. see the [`README.md`](../README.md#data-and-reproducibility) and [`data/README.md`](../data/README.md) for how raw data is obtained in the first place, `dataio` is what runs after having the data files for turning them into analysis ready dataframes.

```python
from quantfinlab.dataio import load_par_yield_curve, load_yfinance_panel

us_curve = load_par_yield_curve("data/us_treasury_yields.csv", source="us_treasury")
jp_curve = load_par_yield_curve("data/japan_mof_yields.csv", source="japan_mof")

panel = load_yfinance_panel(
    "data/core_cross_asset_etfs.csv", fields=("close", "volume"),
    source="yfinance_export", start="2010-01-01")

close, volume = panel["close"], panel["volume"]
```

Every loader in `dataio` normalizes to the same shape. par-yield curves come back as a `DatetimeIndex`-sorted DataFrame with standard tenor columns (`1M`...`30Y`) in decimal form. equities and prices come back as `{field: DataFrame}` with tickers as columns, numeric-coerced, deduplicated, sorted. This is what makes the "secondary market" repeat at the end of most notebooks possible with no extra glue code. swap the `path`/`source`, get the same schema back.

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

## Testing

```bash
pip install -e ".[dev]"
pytest
```

Tests are organized to mirror the package layout (`tests/portfolio`, `tests/options`, `tests/cpp`, ...) and use a small set of deterministic synthetic-data generators (`tests/synthetic/generators.py`) rather than real market data, so the suite is fast, reproducible, and has no external data dependency.
