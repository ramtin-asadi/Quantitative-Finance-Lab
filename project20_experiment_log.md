# Project 20 — RL Portfolio Allocation: Experiment Log

## Design diagnosis (why the old setup could not meet the criteria)

The previous reward was an **active information ratio against Forecast-Gated MaxSharpe**
(`reward_mode="active_te"`). Optimising tracking error against a benchmark teaches the
policy to *replicate* that benchmark, which structurally guarantees high correlation
(fails criterion 2) and removes any path to a genuinely different edge (fails criterion 1).
Two further blockers:

- **85% minimum equity exposure** — the action transform forced `min_exposure=0.85`, so the
  agent physically could not de-risk in 2022 (fails criterion 3).
- **Prior strategy weights in the observation** (Mean-CVaR, BL, FG-MaxSharpe, ML Regime in
  both the asset tensor and the global frame) — feeding benchmark allocations to the agent
  pulls its policy toward them, reinforcing the correlation problem.

It was also slow: every environment step ran a *full* benchmark sub-rollout plus three
pandas `reindex`/DataFrame constructions per step (benchmark path, reference daily returns,
active-path stats), inside a Python loop over hundreds of weeks × hundreds of epochs.

## Changes applied (Run 1)

### Reward — Differential Sharpe Ratio (DSR)
- New `DifferentialSharpe` / `DifferentialSortino` (online EMA Sharpe, Moody & Saffell) in
  `quantfinlab/ml/rewards.py`; updated **per daily return** within each week.
- New `reward_mode="dsr"` branch in `reward_components`: primary signal `dsr_scale·DSR`
  minus penalties for drawdown (below `-0.12`), vol (above `0.14`), turnover cost,
  concentration (HHI > 0.15) and **cash hoarding** (cash > 0.30). No benchmark term.
- Notebook local `reward_components` / `reward_settings` rewritten to the same DSR form.

### Action space — defensive capability
- `min_exposure` floor lowered `0.85 → 0.50` everywhere (notebook `action_to_weights`,
  library `action_to_weights` default, all policy constructors). Cash is now bounded by a
  reward penalty, not a hard floor, so the agent can go defensive in stress regimes.

### Observation — decorrelation + regime awareness
- **Prior strategy weights removed** from the RL observation (asset tensor and global frame).
  `build_state_tables(..., include_prior_weights=False)` is the new default; prior frames are
  still kept for benchmark comparison tables, just not fed to the agent.
- **VIX features added** to the global state: 20d level z-score, VIX / 63d-MA ratio, 252d
  percentile (`load_vix` + `vix_feature_frame` in `quantfinlab/dataio/panel.py`,
  `data/vix_data.csv` fetched 2000→present). Macro/FCI data remains only in the
  TCN/logistic models, never in the RL observation.

### Training
- **Randomised episode start points** in PPO, Recurrent PPO and SAC (replaces the biased
  sequential sliding window).
- **SAC gamma 0.0 → 0.97**, replay buffer 20k → 100k, windowed rollouts.
- GAE `gamma=0.97, lam=0.90`; LSTM forced-reset cadence 26 → 52 steps.
- Best-checkpoint selection switched from total reward to **validation Sharpe**.

### Ensemble
- `blend_policy_weights` (validation-Sharpe-weighted blend of PPO/RPPO/SAC) added as a
  fourth "Ensemble RL" strategy in both the main and sector comparison tables.

### Runtime
- DSR removes the per-step benchmark sub-rollout and the three per-step pandas operations.
- Daily windows precomputed once to contiguous NumPy arrays (`daily_windows_np`,
  `StateTables.daily_windows_np`); the hot loop is now pure NumPy + one batch-1 forward.

## Consistency
Every change is mirrored in both the **notebook local implementation** (cells 1–N) and the
**`quantfinlab` library** used by the final sector cell, so the two stay in lock-step.

## Run 1 results (executed end-to-end, 0 cell errors, fresh TCN + RL checkpoints)

### Main universe (12 ETFs), test 2021+ — Sharpe
| Strategy | Sharpe | CAGR | Max DD | Turnover | Eff. N |
|----------|--------|------|--------|----------|--------|
| **Forecast-Gated MaxSharpe (best benchmark)** | **0.665** | 0.124 | -0.162 | 0.31 | 3.3 |
| ML Regime-Aware | 0.529 | 0.089 | -0.198 | 0.22 | 7.7 |
| RandomForest Kelly | 0.484 | 0.092 | -0.237 | 0.39 | 5.2 |
| SAC (best RL) | 0.410 | 0.080 | -0.219 | 0.02 | 11.9 |
| Equal Weight | 0.391 | 0.077 | -0.219 | 0.02 | 12.0 |
| Recurrent PPO | 0.303 | 0.063 | **-0.135** | 0.11 | 8.1 |
| Ensemble RL | 0.269 | 0.059 | -0.172 | 0.07 | 11.0 |
| PPO | 0.006 | 0.036 | -0.182 | 0.12 | 8.2 |

### Sector universe (9 sectors), test 2021+ — Sharpe
FG-MaxSharpe 0.828 / MaxSharpe 0.812 / HGB-Kelly 0.700 / EW 0.698 / BL 0.696 /
**SAC 0.690 (best RL)** / Ensemble 0.649 / RPPO 0.616 / PPO 0.591 / ML-Regime 0.581 / CVaR 0.534.

### 2022 stress (cumulative, main universe) — RL vs EW
EW -18.7% · FG-MS -9.1% · **Recurrent PPO -11.6% (best RL, max_dd -12.9%)** · Ensemble -14.5% ·
PPO -14.8% · SAC -18.5%. **2023 rebound:** EW +13.9% · **SAC +14.7% (best)** · Ensemble +10.1% · RPPO +9.0%.

### Criteria check
- [ ] **Criterion 1 — RL Sharpe ≥ best benchmark + 0.15: FAILED.** Best RL trails the forecast
  benchmark (main: 0.41 vs 0.67; sector: 0.69 vs 0.83). Active IR vs FG-MS is negative for all RL.
- [~] **Criterion 2 — corr < 0.88: not jointly satisfied.** The competitive RL models (SAC/Ensemble)
  are near-equal-weight (Eff. N ≈ 9–12, turnover ≈ 0.02) so they correlate highly with EW; the one
  low-exposure model (PPO) has near-zero Sharpe. (Matrix rendered as a figure in the notebook.)
- [x] **Criterion 3 — 2022 stress / regime credibility: MET.** Dropping the 0.85 floor worked:
  Recurrent PPO posted the best 2022 drawdown of *any* strategy (-12.9%) and beat EW by ~7pp; SAC
  captured the most 2023 upside (+14.7%). No RL model underperformed EW by >5pp in any stress window.

## Diagnosis
Validation `mean_reward` is -111 to -219 **per step** — the run is dominated by the penalty terms,
not the DSR primary (`dsr_scale·DSR ≈ +10/step` vs drawdown/vol penalties reaching -100…-200/step
because they scale as `10000·excess²`). So the agents minimise risk penalties and collapse toward a
diversified, low-turnover, near-equal-weight book. That is genuinely *defensive and regime-aware*
(criterion 3) but cannot out-alpha a well-tuned forecast benchmark (criterion 1).

## Run 2 results (executed by user)
Main universe much better: SAC 0.558 / Ensemble 0.550 / PPO 0.548 / RPPO 0.508 — all now beat EW
(0.391), RF-Kelly (0.484), ML-Regime (0.529); exposures healthy (0.77–0.95, no cash-hoarding);
active IR vs FG-MS improved from ~-0.85 to -0.07 (SAC). Still below FG-MaxSharpe (0.665); EW
correlation still high; mean_reward still mildly negative (-0.48 to -0.68). Sector still weakest.

### Diagnosis after Run 2
- Penalties still outweigh the DSR primary *in the mean* (primary mean ~ daily-Sharpe ~0.05 at
  dsr_scale=1; cash penalty `2·cash` alone was -0.10 to -0.47/step) → agent still leans to a
  diversified, near-EW book → high EW correlation + negative reward.
- Sector training window was only 5y (2013–2017) although all 9 SPDRs + SHY are available from 2005.

## Run 3 changes (this iteration)
1. **Reward rebalanced so DSR/alpha dominates**: `dsr_scale 1→4`; penalties relaxed
   (`lambda_vol 0.5→0.25`, `lambda_conc 0.5→0.1`, `hhi_target 0.22→0.30`, `target_vol→0.18`,
   `drawdown_floor→-0.18`). Normal-regime reward is now positive.
2. **Cash penalty made cap-based**: `lambda_cash·max(0, cash-0.15)` instead of `lambda_cash·cash`
   — stops punishing healthy ~5% cash (fixes the persistent negative reward) while still
   discouraging >15% cash hoarding.
3. **Explicit decorrelation penalty**: `lambda_decorr·100·max(0, corr63(policy,EW)-0.90)²` added to
   the reward (rolling 63-day daily-return correlation to Equal Weight), in both the notebook hot
   loop and the library `_collect_rollout`. Directly targets the EW-correlation criterion.
4. **Sector data window doubled**: SPDR forecasts + RL training extended **2013→2008** (HGB
   `train_window 8y→6y`, `prediction_dates`/`train_period` start 2008) — ~10y of weekly data
   spanning the GFC, the biggest lever for the weak sector models. Main stays ~2010 (HYG-capped).
5. Carries over: forecast-alpha learnable tilt in all actors, randomized starts, SAC γ=0.97.

## Plan for Run 2 (superseded)
1. **Rebalance the reward**: shrink penalty scales ~10×, or raise `dsr_scale`, so DSR drives the policy
   instead of the penalties (e.g. `lambda_drawdown 2→0.2`, `lambda_vol 1→0.1`, `lambda_cash 1→0.2`).
2. **Seed the action mean at the TCN forecast tilt** so the policy learns *deviations from a momentum/
   alpha baseline* rather than from zero (prompt Step 6 item 6).
3. Optional: curriculum (DSR-only for first 40% of epochs) and a regime-conditioned actor.
