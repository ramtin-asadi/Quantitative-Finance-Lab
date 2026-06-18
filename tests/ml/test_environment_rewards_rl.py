from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.ml import features
from quantfinlab.ml.environment import (
    action_to_weights,
    align_weight_priors,
    blend_policy_weights,
    build_state_tables,
    make_decision_dates,
    policy_weight_frame,
    portfolio_state_vector,
    portfolio_step_path_return,
    portfolio_step_return,
    portfolio_turnover,
    rollout_weights,
)
from quantfinlab.ml.rewards import (
    DifferentialSharpe,
    DifferentialSortino,
    active_reward,
    concentration_penalty,
    drawdown_penalty,
    portfolio_reward,
    reward_component_table,
    reward_components,
    turnover_penalty,
    vol_band_penalty,
)
from quantfinlab.ml.rl import (
    TrainingResult,
    evaluate_policy,
    policy_checkpoints,
    validation_policy_table,
)
from tests.synthetic.generators import price_panel, volume_panel

pytestmark = pytest.mark.ml


ASSETS = ("SPY", "QQQ", "IEF", "GLD", "SHY")


def _state_fixture():
    close = price_panel(n=140, assets=ASSETS)
    returns = close.pct_change(fill_method=None).fillna(0.0)
    volume = volume_panel(n=140, assets=ASSETS)
    risky = ASSETS[:-1]
    asset_x = features.build_asset_feature_block(close, volume, returns, assets=risky)
    context = features.build_cross_asset_feature_block(close, returns, assets=risky, cash_ticker="SHY")
    decision_dates = make_decision_dates(close.index, freq="W-FRI", min_history_days=70)
    equal = pd.DataFrame(0.25, index=close.index[::10], columns=list(risky))
    equal["SHY"] = 0.0
    priors = {"Equal": equal}
    state = build_state_tables(
        asset_features=asset_x,
        context_features=context,
        prior_weights=priors,
        include_prior_weights=True,
        returns=returns,
        decision_dates=decision_dates,
        assets=risky,
        cash_ticker="SHY",
        active_benchmark="Equal",
    )
    return state, returns, priors


class ConstantPolicy:
    def __init__(self, n_assets: int):
        import torch

        self.raw = torch.nn.Parameter(torch.zeros(int(n_assets) + 1))
        self.min_exposure = 0.60
        self.max_exposure = 0.90
        self.max_weight = 0.50

    def parameters(self):
        return iter([self.raw])

    def to(self, device):
        self.raw.data = self.raw.data.to(device)
        return self

    def act(self, asset_x, global_x, portfolio_x=None, *, deterministic: bool = False):
        import torch

        raw = self.raw.expand(asset_x.shape[0], -1)
        weights = action_to_weights(
            raw,
            min_exposure=self.min_exposure,
            max_exposure=self.max_exposure,
            max_weight=self.max_weight,
        )
        zero = torch.zeros(asset_x.shape[0], dtype=asset_x.dtype, device=asset_x.device)
        return raw, zero, zero, zero, weights

    def deterministic_weights(self, asset_x, global_x, portfolio_x=None):
        raw = self.raw.expand(asset_x.shape[0], -1)
        return action_to_weights(
            raw,
            min_exposure=self.min_exposure,
            max_exposure=self.max_exposure,
            max_weight=self.max_weight,
        )


def test_state_tables_align_priors_and_encode_portfolio_inputs() -> None:
    state, _, priors = _state_fixture()
    aligned = align_weight_priors(
        priors,
        decision_dates=state.dates,
        assets=state.assets,
        cash_ticker=state.cash_ticker,
    )
    copied = state.copy_with()
    period_idx = state.period_indices((state.dates[1], state.dates[3]))
    portfolio_x = portfolio_state_vector([0.20, 0.30, 0.20, 0.10, 0.20], previous_turnover=0.10)

    assert state.n_dates >= 10
    assert state.n_assets == 4
    assert state.asset_state.shape == (state.n_dates, state.n_assets, state.n_asset_features)
    assert state.global_state.shape[0] == state.n_dates
    assert state.prior_weights.shape == (state.n_dates, state.n_assets, 1)
    assert state.active_benchmark == "Equal"
    assert aligned["Equal"].sum(axis=1).eq(1.0).all()
    assert copied.asset_state is not state.asset_state
    assert period_idx.tolist() == [1, 2, 3]
    assert len(portfolio_x) == state.n_assets + 5
    assert portfolio_x[state.n_assets] == pytest.approx(0.80)


def test_action_weight_conversion_step_returns_and_blended_policy_frames() -> None:
    action = np.array([0.10, 0.20, 0.30, 0.40, 0.00])
    weights = action_to_weights(action, min_exposure=0.50, max_exposure=1.00, max_weight=0.50)
    path_return, path_turnover, path_cost, end_weights, daily_net, nav_path = portfolio_step_path_return(
        [0.40, 0.60],
        pd.DataFrame([[0.01, 0.02], [0.00, 0.01]]),
        w_prev=[0.50, 0.50],
        cost_bps=10.0,
    )
    one_step = portfolio_step_return([0.50, 0.50], [0.01, 0.02], w_prev=[0.40, 0.60], cost_bps=5.0)
    frames = {
        "a": pd.DataFrame({"SPY": [0.50], "SHY": [0.50]}, index=[pd.Timestamp("2024-01-05")]),
        "b": pd.DataFrame({"SPY": [0.25], "QQQ": [0.25], "SHY": [0.50]}, index=[pd.Timestamp("2024-01-05")]),
    }
    blended = blend_policy_weights(frames, blend_weights={"a": 2.0, "b": 1.0})

    assert weights.sum() == pytest.approx(1.0)
    assert weights[:-1].max() <= 0.50 + 1e-12
    assert portfolio_turnover([0.50, 0.50], [0.20, 0.80]) == pytest.approx(0.30)
    assert one_step[0] == pytest.approx(0.01495)
    assert path_return > 0.0
    assert path_turnover == pytest.approx(0.10)
    assert path_cost == pytest.approx(0.00010)
    assert end_weights.sum() == pytest.approx(1.0)
    assert len(daily_net) == len(nav_path) == 2
    assert blended.sum(axis=1).iloc[0] == pytest.approx(1.0)


def test_reward_components_penalties_and_component_table() -> None:
    sharpe = DifferentialSharpe(eta=0.20)
    sortino = DifferentialSortino(eta=0.20)
    sharpe_values = [sharpe.update(x) for x in [0.01, -0.005, 0.012]]
    sortino_values = [sortino.update(x) for x in [0.01, -0.005, 0.012]]
    components = reward_components(
        portfolio_return=0.012,
        benchmark_return=0.004,
        turnover=0.18,
        cost=0.0001,
        weights=[0.40, 0.30, 0.20, 0.10],
        realized_vol=0.20,
        drawdown=-0.12,
        reward_mode="active_te",
        tracking_vol=0.08,
    )
    dsr_components = reward_components(portfolio_return=0.01, reward_mode="dsr", dsr=0.25, cash_weight=0.20)
    table = reward_component_table(component_rows={"policy": [components, dsr_components]})

    assert sharpe_values[0] == pytest.approx(0.0)
    assert sortino_values[0] == pytest.approx(0.0)
    assert all(np.isfinite(v) for v in sharpe_values + sortino_values)
    assert vol_band_penalty(0.20, vol_high=0.16) > 0.0
    assert drawdown_penalty(-0.15, drawdown_floor=-0.10) > 0.0
    assert turnover_penalty(0.20, turnover_budget=0.12) > 0.0
    assert concentration_penalty([0.70, 0.20, 0.10]) > 0.0
    assert active_reward(0.02, 0.01) == pytest.approx(75.0)
    assert portfolio_reward(portfolio_return=0.012, benchmark_return=0.004, turnover=0.05) == pytest.approx(
        reward_components(portfolio_return=0.012, benchmark_return=0.004, turnover=0.05)["reward"]
    )
    assert components["active_return"] == pytest.approx(0.008)
    assert dsr_components["primary_reward"] == pytest.approx(25.0)
    assert table.loc["policy", "reward"] == pytest.approx(np.mean([components["reward"], dsr_components["reward"]]))


def test_rollout_evaluate_policy_validation_table_and_checkpoints(tmp_path) -> None:
    pytest.importorskip("torch")
    state, returns, _ = _state_fixture()
    policy = ConstantPolicy(state.n_assets)
    period = (state.dates[0], state.dates[4])
    weights = rollout_weights(policy, state, period=period)
    selected = policy_weight_frame(policy=policy, state=state, period=period, assets=state.assets, cash_ticker="SHY")
    result = evaluate_policy(
        policy=policy,
        state=state,
        period=period,
        reward_settings={"cost_bps": 5.0, "active_benchmark": "Equal"},
    )
    manual = TrainingResult(
        "manual",
        pd.DataFrame({"epoch": [1]}),
        {"return": 0.10, "sharpe": 1.0},
        weights,
        result.returns,
        result.components,
        tmp_path / "manual.pt",
    )
    (tmp_path / "a.pt").write_bytes(b"checkpoint")
    (tmp_path / "notes.txt").write_text("not a checkpoint", encoding="utf-8")
    validation = validation_policy_table({"constant": result, "manual": manual})
    checkpoints = policy_checkpoints(tmp_path)

    assert np.allclose(weights.sum(axis=1), 1.0)
    assert selected.columns.tolist() == [*state.assets, "SHY"]
    assert result.validation["avg_exposure"] == pytest.approx(0.75)
    assert len(result.components) == len(result.weights)
    assert not result.returns.empty
    assert {"return", "sharpe", "checkpoint"}.issubset(validation.columns)
    assert checkpoints["checkpoint"].tolist() == ["a.pt"]
    assert returns.columns.tolist() == list(ASSETS)
