from __future__ import annotations

try:
    import torch
    from torch import nn
    from torch.distributions import Normal
except Exception:  # pragma: no cover
    torch = None
    nn = None
    Normal = None

from quantfinlab.ml.environment import action_to_weights


def _require_torch():
    if torch is None or nn is None:  # pragma: no cover
        raise ImportError("PyTorch is required for Project 20 RL policies.")


if torch is not None and nn is not None:

    def _valid_heads(hidden_size: int, requested: int) -> int:
        heads = max(1, min(int(requested), int(hidden_size)))
        while int(hidden_size) % heads != 0 and heads > 1:
            heads -= 1
        return heads


    class StateEncoder(nn.Module):
        """Asset-preserving encoder used by the actor and critic networks."""

        def __init__(
            self,
            *,
            n_asset_features: int,
            n_global_features: int,
            n_assets: int,
            n_portfolio_features: int | None = None,
            hidden_size: int = 128,
            attention_heads: int = 4,
            dropout: float = 0.05,
        ):
            super().__init__()
            self.n_assets = int(n_assets)
            self.n_asset_features = int(n_asset_features)
            self.n_global_features = int(n_global_features)
            self.n_portfolio_features = int(n_portfolio_features or (int(n_assets) + 5))
            h = int(hidden_size)
            heads = _valid_heads(h, int(attention_heads))
            self.asset_encoder = nn.Sequential(
                nn.LayerNorm(self.n_asset_features),
                nn.Linear(self.n_asset_features, h),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(h, h),
                nn.SiLU(),
            )
            self.asset_attention = nn.MultiheadAttention(h, heads, dropout=float(dropout), batch_first=True)
            self.asset_norm = nn.LayerNorm(h)
            self.pool_score = nn.Linear(h, 1)
            self.global_encoder = nn.Sequential(
                nn.LayerNorm(self.n_global_features),
                nn.Linear(self.n_global_features, h),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(h, h),
                nn.SiLU(),
            )
            self.portfolio_encoder = nn.Sequential(
                nn.LayerNorm(self.n_portfolio_features),
                nn.Linear(self.n_portfolio_features, h),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(h, h),
                nn.SiLU(),
            )
            self.hidden_size = h
            self.output_dim = 3 * h
            self.asset_score_dim = 4 * h

        def encode(self, asset_x, global_x, portfolio_x=None):
            asset_x = asset_x.float()
            global_x = global_x.float()
            if portfolio_x is None:
                portfolio_x = torch.zeros(
                    asset_x.shape[0],
                    self.n_portfolio_features,
                    dtype=asset_x.dtype,
                    device=asset_x.device,
                )
            portfolio_x = portfolio_x.float()
            h_asset = self.asset_encoder(asset_x)
            attn, _ = self.asset_attention(h_asset, h_asset, h_asset, need_weights=False)
            h_asset = self.asset_norm(h_asset + attn)
            pool_w = torch.softmax(self.pool_score(h_asset).squeeze(-1), dim=-1)
            context = torch.sum(h_asset * pool_w.unsqueeze(-1), dim=1)
            global_h = self.global_encoder(global_x)
            portfolio_h = self.portfolio_encoder(portfolio_x)
            fused = torch.cat([context, global_h, portfolio_h], dim=-1)
            return h_asset, context, global_h, portfolio_h, fused

        def forward(self, asset_x, global_x, portfolio_x=None):
            return self.encode(asset_x, global_x, portfolio_x)[-1]


    class PpoPolicy(nn.Module):
        def __init__(
            self,
            *,
            n_asset_features: int,
            n_global_features: int,
            n_assets: int,
            n_portfolio_features: int | None = None,
            hidden_size: int = 128,
            attention_heads: int = 4,
            min_exposure: float = 0.55,
            max_exposure: float = 1.00,
            max_weight: float = 0.35,
            alpha_feature_index: int | None = None,
            dropout: float = 0.05,
        ):
            super().__init__()
            self.n_assets = int(n_assets)
            self.min_exposure = float(min_exposure)
            self.max_exposure = float(max_exposure)
            self.max_weight = float(max_weight)
            self.alpha_index = None if alpha_feature_index is None else int(alpha_feature_index)
            self.alpha_gain = nn.Parameter(torch.tensor(4.0)) if alpha_feature_index is not None else None
            self.encoder = StateEncoder(
                n_asset_features=n_asset_features,
                n_global_features=n_global_features,
                n_assets=n_assets,
                n_portfolio_features=n_portfolio_features,
                hidden_size=hidden_size,
                attention_heads=attention_heads,
                dropout=dropout,
            )
            h = int(hidden_size)
            self.score_head = nn.Sequential(
                nn.LayerNorm(self.encoder.asset_score_dim),
                nn.Linear(self.encoder.asset_score_dim, h),
                nn.SiLU(),
                nn.Linear(h, 1),
            )
            self.exposure_head = nn.Sequential(
                nn.LayerNorm(self.encoder.output_dim),
                nn.Linear(self.encoder.output_dim, h),
                nn.SiLU(),
                nn.Linear(h, 1),
            )
            self.log_std = nn.Parameter(torch.full((self.n_assets + 1,), -0.65))
            self.value = nn.Sequential(
                nn.LayerNorm(self.encoder.output_dim),
                nn.Linear(self.encoder.output_dim, h),
                nn.SiLU(),
                nn.Linear(h, 1),
            )

        def _alpha_bias(self, risky_logits, asset_x):
            if self.alpha_index is None:
                return risky_logits
            return risky_logits + self.alpha_gain * asset_x.float()[..., self.alpha_index]

        def action_mean(self, asset_x, global_x, portfolio_x=None):
            h_asset, context, global_h, portfolio_h, fused = self.encoder.encode(asset_x, global_x, portfolio_x)
            n = h_asset.shape[1]
            x = torch.cat(
                [
                    h_asset,
                    context.unsqueeze(1).expand(-1, n, -1),
                    global_h.unsqueeze(1).expand(-1, n, -1),
                    portfolio_h.unsqueeze(1).expand(-1, n, -1),
                ],
                dim=-1,
            )
            risky_logits = self._alpha_bias(self.score_head(x).squeeze(-1), asset_x)
            exposure_logit = self.exposure_head(fused).squeeze(-1)
            return torch.cat([risky_logits, exposure_logit.unsqueeze(-1)], dim=-1), fused

        def distribution(self, asset_x, global_x, portfolio_x=None):
            mean, fused = self.action_mean(asset_x, global_x, portfolio_x)
            std = torch.exp(self.log_std).clamp(0.05, 2.0).expand_as(mean)
            value = self.value(fused).squeeze(-1)
            return Normal(mean, std), value

        def forward(self, asset_x, global_x, portfolio_x=None):
            dist, value = self.distribution(asset_x, global_x, portfolio_x)
            return dist.mean, value

        def act(self, asset_x, global_x, portfolio_x=None, *, deterministic: bool = False):
            dist, value = self.distribution(asset_x, global_x, portfolio_x)
            raw = dist.mean if deterministic else dist.sample()
            log_prob = dist.log_prob(raw).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            weights = action_to_weights(
                raw,
                min_exposure=self.min_exposure,
                max_exposure=self.max_exposure,
                max_weight=self.max_weight,
            )
            return raw, log_prob, entropy, value, weights

        def evaluate_actions(self, asset_x, global_x, portfolio_x, actions):
            dist, value = self.distribution(asset_x, global_x, portfolio_x)
            log_prob = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            weights = action_to_weights(
                actions,
                min_exposure=self.min_exposure,
                max_exposure=self.max_exposure,
                max_weight=self.max_weight,
            )
            return log_prob, entropy, value, weights

        def deterministic_weights(self, asset_x, global_x, portfolio_x=None):
            _, _, _, _, weights = self.act(asset_x, global_x, portfolio_x, deterministic=True)
            return weights


    class RecurrentPpoPolicy(nn.Module):
        def __init__(
            self,
            *,
            n_asset_features: int,
            n_global_features: int,
            n_assets: int,
            n_portfolio_features: int | None = None,
            hidden_size: int = 128,
            recurrent_size: int = 96,
            attention_heads: int = 4,
            sequence_length: int = 52,
            min_exposure: float = 0.55,
            max_exposure: float = 1.00,
            max_weight: float = 0.35,
            dropout: float = 0.05,
            alpha_feature_index: int | None = None,
        ):
            super().__init__()
            self.n_assets = int(n_assets)
            self.sequence_length = int(sequence_length)
            self.min_exposure = float(min_exposure)
            self.max_exposure = float(max_exposure)
            self.max_weight = float(max_weight)
            self.alpha_index = None if alpha_feature_index is None else int(alpha_feature_index)
            self.alpha_gain = nn.Parameter(torch.tensor(4.0)) if alpha_feature_index is not None else None
            self.encoder = StateEncoder(
                n_asset_features=n_asset_features,
                n_global_features=n_global_features,
                n_assets=n_assets,
                n_portfolio_features=n_portfolio_features,
                hidden_size=hidden_size,
                attention_heads=attention_heads,
                dropout=dropout,
            )
            h = int(hidden_size)
            r = int(recurrent_size)
            self.lstm = nn.LSTM(self.encoder.output_dim, r, batch_first=True)
            self.score_head = nn.Sequential(
                nn.LayerNorm(self.encoder.asset_score_dim + r),
                nn.Linear(self.encoder.asset_score_dim + r, h),
                nn.SiLU(),
                nn.Linear(h, 1),
            )
            self.exposure_head = nn.Sequential(
                nn.LayerNorm(self.encoder.output_dim + r),
                nn.Linear(self.encoder.output_dim + r, h),
                nn.SiLU(),
                nn.Linear(h, 1),
            )
            self.log_std = nn.Parameter(torch.full((self.n_assets + 1,), -0.70))
            self.value = nn.Sequential(
                nn.LayerNorm(self.encoder.output_dim + r),
                nn.Linear(self.encoder.output_dim + r, h),
                nn.SiLU(),
                nn.Linear(h, 1),
            )

        def _apply_alpha(self, mean, asset_x):
            if self.alpha_index is None:
                return mean
            n = self.n_assets
            risky = mean[..., :n] + self.alpha_gain * asset_x.float()[..., self.alpha_index]
            return torch.cat([risky, mean[..., n:]], dim=-1)

        def _heads(self, h_asset, context, global_h, portfolio_h, fused, memory):
            n = h_asset.shape[-2]
            if memory.ndim == 2:
                score_memory = memory.unsqueeze(1).expand(-1, n, -1)
                head_base = torch.cat([fused, memory], dim=-1)
                risky_x = torch.cat(
                    [
                        h_asset,
                        context.unsqueeze(1).expand(-1, n, -1),
                        global_h.unsqueeze(1).expand(-1, n, -1),
                        portfolio_h.unsqueeze(1).expand(-1, n, -1),
                        score_memory,
                    ],
                    dim=-1,
                )
                risky = self.score_head(risky_x).squeeze(-1)
                exposure = self.exposure_head(head_base).squeeze(-1)
                value = self.value(head_base).squeeze(-1)
                return torch.cat([risky, exposure.unsqueeze(-1)], dim=-1), value

            b, t = memory.shape[0], memory.shape[1]
            score_memory = memory.unsqueeze(2).expand(-1, -1, n, -1)
            risky_x = torch.cat(
                [
                    h_asset,
                    context.unsqueeze(2).expand(-1, -1, n, -1),
                    global_h.unsqueeze(2).expand(-1, -1, n, -1),
                    portfolio_h.unsqueeze(2).expand(-1, -1, n, -1),
                    score_memory,
                ],
                dim=-1,
            )
            risky = self.score_head(risky_x.reshape(b * t, n, -1)).reshape(b, t, n)
            head_base = torch.cat([fused, memory], dim=-1)
            exposure = self.exposure_head(head_base).squeeze(-1)
            value = self.value(head_base).squeeze(-1)
            return torch.cat([risky, exposure.unsqueeze(-1)], dim=-1), value

        def distribution(self, asset_x, global_x, portfolio_x=None, hidden=None):
            h_asset, context, global_h, portfolio_h, fused = self.encoder.encode(asset_x, global_x, portfolio_x)
            memory, hidden_next = self.lstm(fused.unsqueeze(1), hidden)
            memory_last = memory[:, -1, :]
            mean, value = self._heads(h_asset, context, global_h, portfolio_h, fused, memory_last)
            mean = self._apply_alpha(mean, asset_x)
            std = torch.exp(self.log_std).clamp(0.05, 2.0).expand_as(mean)
            return Normal(mean, std), value, hidden_next

        def sequence_distribution(self, asset_x, global_x, portfolio_x=None):
            b, t = asset_x.shape[0], asset_x.shape[1]
            flat_asset = asset_x.reshape(b * t, asset_x.shape[2], asset_x.shape[3])
            flat_global = global_x.reshape(b * t, global_x.shape[2])
            flat_port = None if portfolio_x is None else portfolio_x.reshape(b * t, portfolio_x.shape[2])
            h_asset, context, global_h, portfolio_h, fused = self.encoder.encode(flat_asset, flat_global, flat_port)
            h_asset = h_asset.reshape(b, t, h_asset.shape[1], h_asset.shape[2])
            context = context.reshape(b, t, -1)
            global_h = global_h.reshape(b, t, -1)
            portfolio_h = portfolio_h.reshape(b, t, -1)
            fused = fused.reshape(b, t, -1)
            memory, _ = self.lstm(fused)
            mean, value = self._heads(h_asset, context, global_h, portfolio_h, fused, memory)
            mean = self._apply_alpha(mean, asset_x)
            std = torch.exp(self.log_std).clamp(0.05, 2.0).view(1, 1, -1).expand_as(mean)
            return Normal(mean, std), value

        def act(self, asset_x, global_x, portfolio_x=None, *, hidden=None, deterministic: bool = False):
            dist, value, hidden_next = self.distribution(asset_x, global_x, portfolio_x, hidden=hidden)
            raw = dist.mean if deterministic else dist.sample()
            log_prob = dist.log_prob(raw).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            weights = action_to_weights(
                raw,
                min_exposure=self.min_exposure,
                max_exposure=self.max_exposure,
                max_weight=self.max_weight,
            )
            return raw, log_prob, entropy, value, weights, hidden_next

        def evaluate_actions(self, asset_x, global_x, portfolio_x, actions):
            dist, value, _ = self.distribution(asset_x, global_x, portfolio_x)
            log_prob = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            weights = action_to_weights(
                actions,
                min_exposure=self.min_exposure,
                max_exposure=self.max_exposure,
                max_weight=self.max_weight,
            )
            return log_prob, entropy, value, weights

        def evaluate_sequence_actions(self, asset_x, global_x, portfolio_x, actions):
            dist, value = self.sequence_distribution(asset_x, global_x, portfolio_x)
            log_prob = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            weights = action_to_weights(
                actions,
                min_exposure=self.min_exposure,
                max_exposure=self.max_exposure,
                max_weight=self.max_weight,
            )
            return log_prob, entropy, value, weights

        def deterministic_weights(self, asset_x, global_x, portfolio_x=None):
            _, _, _, _, weights, _ = self.act(asset_x, global_x, portfolio_x, deterministic=True)
            return weights


    class SacCritic(nn.Module):
        def __init__(
            self,
            *,
            n_asset_features: int,
            n_global_features: int,
            n_assets: int,
            n_portfolio_features: int | None = None,
            hidden_size: int = 128,
            attention_heads: int = 4,
            dropout: float = 0.05,
        ):
            super().__init__()
            self.encoder = StateEncoder(
                n_asset_features=n_asset_features,
                n_global_features=n_global_features,
                n_assets=n_assets,
                n_portfolio_features=n_portfolio_features,
                hidden_size=hidden_size,
                attention_heads=attention_heads,
                dropout=dropout,
            )
            h = int(hidden_size)
            q_input = self.encoder.output_dim + int(n_assets) + 1
            self.q1 = nn.Sequential(
                nn.LayerNorm(q_input),
                nn.Linear(q_input, h),
                nn.SiLU(),
                nn.Linear(h, h),
                nn.SiLU(),
                nn.Linear(h, 1),
            )
            self.q2 = nn.Sequential(
                nn.LayerNorm(q_input),
                nn.Linear(q_input, h),
                nn.SiLU(),
                nn.Linear(h, h),
                nn.SiLU(),
                nn.Linear(h, 1),
            )

        def forward(self, asset_x, global_x, portfolio_x, action):
            z = self.encoder(asset_x, global_x, portfolio_x)
            x = torch.cat([z, action.float()], dim=-1)
            return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


    class SacPolicy(nn.Module):
        def __init__(
            self,
            *,
            n_asset_features: int,
            n_global_features: int,
            n_assets: int,
            n_portfolio_features: int | None = None,
            hidden_size: int = 128,
            attention_heads: int = 4,
            min_exposure: float = 0.55,
            max_exposure: float = 1.00,
            max_weight: float = 0.35,
            dropout: float = 0.05,
            alpha_feature_index: int | None = None,
        ):
            super().__init__()
            self.n_assets = int(n_assets)
            self.min_exposure = float(min_exposure)
            self.max_exposure = float(max_exposure)
            self.max_weight = float(max_weight)
            self.alpha_index = None if alpha_feature_index is None else int(alpha_feature_index)
            self.alpha_gain = nn.Parameter(torch.tensor(4.0)) if alpha_feature_index is not None else None
            self.actor_encoder = StateEncoder(
                n_asset_features=n_asset_features,
                n_global_features=n_global_features,
                n_assets=n_assets,
                n_portfolio_features=n_portfolio_features,
                hidden_size=hidden_size,
                attention_heads=attention_heads,
                dropout=dropout,
            )
            h = int(hidden_size)
            self.actor_score_head = nn.Sequential(
                nn.LayerNorm(self.actor_encoder.asset_score_dim),
                nn.Linear(self.actor_encoder.asset_score_dim, h),
                nn.SiLU(),
                nn.Linear(h, 1),
            )
            self.actor_exposure_head = nn.Sequential(
                nn.LayerNorm(self.actor_encoder.output_dim),
                nn.Linear(self.actor_encoder.output_dim, h),
                nn.SiLU(),
                nn.Linear(h, 1),
            )
            self.actor_log_std_head = nn.Sequential(
                nn.LayerNorm(self.actor_encoder.output_dim),
                nn.Linear(self.actor_encoder.output_dim, h),
                nn.SiLU(),
                nn.Linear(h, self.n_assets + 1),
            )
            self.critic = SacCritic(
                n_asset_features=n_asset_features,
                n_global_features=n_global_features,
                n_assets=n_assets,
                n_portfolio_features=n_portfolio_features,
                hidden_size=hidden_size,
                attention_heads=attention_heads,
                dropout=dropout,
            )

        def actor_mean(self, asset_x, global_x, portfolio_x=None):
            h_asset, context, global_h, portfolio_h, fused = self.actor_encoder.encode(asset_x, global_x, portfolio_x)
            n = h_asset.shape[1]
            x = torch.cat(
                [
                    h_asset,
                    context.unsqueeze(1).expand(-1, n, -1),
                    global_h.unsqueeze(1).expand(-1, n, -1),
                    portfolio_h.unsqueeze(1).expand(-1, n, -1),
                ],
                dim=-1,
            )
            risky = self.actor_score_head(x).squeeze(-1)
            if self.alpha_index is not None:
                risky = risky + self.alpha_gain * asset_x.float()[..., self.alpha_index]
            exposure = self.actor_exposure_head(fused).squeeze(-1)
            return torch.cat([risky, exposure.unsqueeze(-1)], dim=-1), fused

        def actor_distribution(self, asset_x, global_x, portfolio_x=None):
            mean, fused = self.actor_mean(asset_x, global_x, portfolio_x)
            log_std = self.actor_log_std_head(fused).clamp(-5.0, 1.0)
            return Normal(mean, torch.exp(log_std))

        def sample_action(self, asset_x, global_x, portfolio_x=None, *, deterministic: bool = False):
            dist = self.actor_distribution(asset_x, global_x, portfolio_x)
            raw = dist.mean if deterministic else dist.rsample()
            log_prob = dist.log_prob(raw).sum(dim=-1)
            weights = action_to_weights(
                raw,
                min_exposure=self.min_exposure,
                max_exposure=self.max_exposure,
                max_weight=self.max_weight,
            )
            entropy = dist.entropy().sum(dim=-1)
            return raw, log_prob, entropy, weights

        def q_values(self, asset_x, global_x, portfolio_x, action):
            return self.critic(asset_x, global_x, portfolio_x, action)

        def deterministic_weights(self, asset_x, global_x, portfolio_x=None):
            _, _, _, weights = self.sample_action(asset_x, global_x, portfolio_x, deterministic=True)
            return weights


else:  # pragma: no cover

    class PpoPolicy:  # type: ignore[no-redef]
        def __init__(self, *_, **__):
            _require_torch()

    class RecurrentPpoPolicy:  # type: ignore[no-redef]
        def __init__(self, *_, **__):
            _require_torch()

    class SacPolicy:  # type: ignore[no-redef]
        def __init__(self, *_, **__):
            _require_torch()

    class SacCritic:  # type: ignore[no-redef]
        def __init__(self, *_, **__):
            _require_torch()


__all__ = [
    "PpoPolicy",
    "RecurrentPpoPolicy",
    "SacCritic",
    "SacPolicy",
    "StateEncoder",
]
