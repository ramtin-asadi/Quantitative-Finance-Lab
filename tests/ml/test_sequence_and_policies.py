from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.ml import sequence_models

pytestmark = pytest.mark.ml


class DummyHMM:
    n_components = 2
    covariance_type = "diag"

    def predict_proba(self, x):
        arr = np.asarray(x, dtype=float)
        p0 = 1.0 / (1.0 + np.exp(-arr[:, 0]))
        return np.column_stack([p0, 1.0 - p0])

    def predict(self, x):
        return self.predict_proba(x).argmax(axis=1)

    def score(self, x):
        return -12.5


def test_hmm_probability_quality_and_pca_helpers() -> None:
    index = pd.bdate_range("2024-01-01", periods=20)
    x = pd.DataFrame({"growth": np.linspace(-1.0, 1.0, len(index)), "vol": np.cos(np.arange(len(index)))}, index=index)
    model = DummyHMM()
    proba = sequence_models.hmm_proba_frame(model, x)
    pca_x, scaler, pca = sequence_models.pca_hmm_inputs(x, n_components=2)
    transformed, _, _ = sequence_models.pca_hmm_inputs(x.tail(3), n_components=2, scaler=scaler, pca=pca)
    aligned = sequence_models.align_state_probabilities(pd.DataFrame({"s0": [0.20, 0.80], "s1": [0.80, 0.20]}), [1, 0])
    quality = sequence_models.hmm_quality_row("hmm", model, x, outcomes=x["growth"])

    assert proba.index.equals(index)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert list(pca_x.columns) == ["PC1", "PC2"]
    assert transformed.shape == (3, 2)
    assert aligned.columns.tolist() == ["state_0", "state_1"]
    assert np.allclose(aligned.sum(axis=1), 1.0)
    assert quality["states"] == 2
    assert np.isfinite(quality["aic"])
    assert isinstance(sequence_models.torch_available(), bool)


def test_sequence_array_builder_aligns_per_asset_rolling_windows() -> None:
    dates = pd.bdate_range("2024-01-01", periods=20)
    data = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "asset_id": [0] * len(dates) + [1] * len(dates),
            "f1": np.r_[np.arange(len(dates)), np.arange(len(dates)) + 1],
            "f2": 1.0,
            "target": np.r_[np.arange(len(dates)), np.arange(len(dates)) + 2],
        }
    )
    x, asset_id, y, row_index = sequence_models.build_sequence_arrays(
        data,
        features=["f1", "f2"],
        target="target",
        lookback=5,
    )
    x_no_target, asset_no_target, y_none, row_index_no_target = sequence_models.build_sequence_arrays(
        data,
        features=["f1", "f2"],
        target=None,
        lookback=5,
    )

    assert x.shape == (32, 5, 2)
    assert asset_id.shape == (32,)
    assert y.shape == (32,)
    assert row_index[:3].tolist() == [4, 5, 6]
    assert x[0, :, 0].tolist() == pytest.approx([0, 1, 2, 3, 4])
    assert x_no_target.shape == x.shape
    assert asset_no_target.tolist() == asset_id.tolist()
    assert y_none is None
    assert row_index_no_target.equals(row_index)


def test_torch_forecast_losses_and_model_output_shapes() -> None:
    torch = pytest.importorskip("torch")

    target = torch.tensor([0.0, 0.3], dtype=torch.float32)
    quantile_pred = torch.stack([target - 0.10, target, target + 0.10], dim=1)
    gaussian_pred = torch.column_stack([target, torch.zeros_like(target)])
    mlp = sequence_models.MlpForecast(
        n_features=2,
        n_assets=2,
        hidden_sizes=(4,),
        output_size=3,
        ordered_quantiles=True,
        dropout=0.0,
    )
    lstm = sequence_models.LstmForecast(
        n_features=2,
        n_assets=2,
        hidden_size=4,
        output_size=1,
        dropout=0.0,
    )
    tcn = sequence_models.TcnForecast(
        n_features=2,
        n_assets=2,
        channels=(4,),
        output_size=1,
        dropout=0.0,
    )
    tabular_x = torch.ones(3, 2)
    sequence_x = torch.ones(3, 5, 2)
    asset_id = torch.tensor([0, 1, 0])

    quantile_out = mlp(tabular_x, asset_id)

    assert sequence_models.pinball_loss_torch(quantile_pred, target, (0.10, 0.50, 0.90)).item() >= 0.0
    assert sequence_models.gaussian_nll_loss_torch(gaussian_pred, target).item() == pytest.approx(0.0)
    assert quantile_out.shape == (3, 3)
    assert bool((quantile_out[:, 0] <= quantile_out[:, 1]).all())
    assert bool((quantile_out[:, 1] <= quantile_out[:, 2]).all())
    assert lstm(sequence_x, asset_id).shape == (3, 1)
    assert tcn(sequence_x, asset_id).shape == (3, 1)


def test_policy_networks_emit_normalized_weight_actions() -> None:
    torch = pytest.importorskip("torch")
    from quantfinlab.ml.policies import PpoPolicy, RecurrentPpoPolicy, SacPolicy, StateEncoder

    asset_x = torch.ones(2, 3, 4)
    global_x = torch.ones(2, 5)
    portfolio_x = torch.ones(2, 8)
    encoder = StateEncoder(
        n_asset_features=4,
        n_global_features=5,
        n_assets=3,
        n_portfolio_features=8,
        hidden_size=8,
        attention_heads=3,
        dropout=0.0,
    )
    ppo = PpoPolicy(
        n_asset_features=4,
        n_global_features=5,
        n_assets=3,
        n_portfolio_features=8,
        hidden_size=8,
        attention_heads=2,
        dropout=0.0,
        max_weight=0.60,
    )
    recurrent = RecurrentPpoPolicy(
        n_asset_features=4,
        n_global_features=5,
        n_assets=3,
        n_portfolio_features=8,
        hidden_size=8,
        recurrent_size=6,
        attention_heads=2,
        dropout=0.0,
        max_weight=0.60,
    )
    sac = SacPolicy(
        n_asset_features=4,
        n_global_features=5,
        n_assets=3,
        n_portfolio_features=8,
        hidden_size=8,
        attention_heads=2,
        dropout=0.0,
        max_weight=0.60,
    )

    raw, log_prob, entropy, value, ppo_weights = ppo.act(asset_x, global_x, portfolio_x, deterministic=True)
    rec_raw, rec_log_prob, rec_entropy, rec_value, rec_weights, hidden = recurrent.act(
        asset_x,
        global_x,
        portfolio_x,
        deterministic=True,
    )
    sac_raw, sac_log_prob, sac_entropy, sac_weights = sac.sample_action(asset_x, global_x, portfolio_x, deterministic=True)
    q1, q2 = sac.q_values(asset_x, global_x, portfolio_x, sac_raw)

    assert encoder(asset_x, global_x, portfolio_x).shape == (2, 24)
    assert raw.shape == (2, 4)
    assert log_prob.shape == entropy.shape == value.shape == (2,)
    assert torch.allclose(ppo_weights.sum(dim=1), torch.ones(2))
    assert rec_raw.shape == (2, 4)
    assert rec_log_prob.shape == rec_entropy.shape == rec_value.shape == (2,)
    assert torch.allclose(rec_weights.sum(dim=1), torch.ones(2))
    assert len(hidden) == 2
    assert sac_raw.shape == (2, 4)
    assert sac_log_prob.shape == sac_entropy.shape == (2,)
    assert torch.allclose(sac_weights.sum(dim=1), torch.ones(2))
    assert q1.shape == q2.shape == (2,)
