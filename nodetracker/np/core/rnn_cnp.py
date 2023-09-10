from typing import Tuple

import torch
from torch import nn

from nodetracker.library.building_blocks.mlp import MLP
from nodetracker.np.utils import to_scaled_relative_ts


class RNNCNPBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        target_dim: int,

        hidden_dim: int = 8,
        n_input2hidden_layers: int = 2,
        n_target2hidden_layers: int = 2,
        n_enc_layers: int = 2,
        n_agg_layers: int = 1,
        agg_dropout_ratio: float = 0.3
    ):
        super().__init__()
        self._input2hidden = MLP(
            input_dim=input_dim,
            output_dim=hidden_dim,
            n_layers=n_input2hidden_layers
        )

        self._target2hidden = MLP(
            input_dim=target_dim,
            output_dim=hidden_dim,
            n_layers=n_target2hidden_layers
        )

        self._enc = MLP(
            input_dim=2 * hidden_dim,
            output_dim=hidden_dim,
            n_layers=n_enc_layers
        )

        self._agg = nn.GRU(hidden_dim, hidden_dim, num_layers=n_agg_layers, batch_first=False)
        self._dropout = nn.Dropout(agg_dropout_ratio)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, y1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encoding observed nodes
        xh1 = self._input2hidden(x1)
        yh1 = self._target2hidden(y1)

        xyh1 = torch.cat([xh1, yh1], dim=-1)
        r1 = self._enc(xyh1)

        # Aggregation
        r_agg, _ = self._agg(r1)
        r_agg = self._dropout(r_agg)[-1]

        # Encoding target nodes
        xh2 = self._input2hidden(x2)

        return r_agg, xh2


class RNNCNP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        target_dim: int,

        hidden_dim: int = 8,
        n_input2hidden_layers: int = 2,
        n_target2hidden_layers: int = 2,
        n_enc_layers: int = 2,

        n_head_layers: int = 2,
        n_agg_layers: int = 2,
    ):
        super().__init__()
        self._backbone = RNNCNPBackbone(
            input_dim=input_dim,
            target_dim=target_dim,
            hidden_dim=hidden_dim,
            n_input2hidden_layers=n_input2hidden_layers,
            n_target2hidden_layers=n_target2hidden_layers,
            n_enc_layers=n_enc_layers,
            n_agg_layers=n_agg_layers
        )

        self._head = nn.Sequential(
            MLP(
                input_dim=2 * hidden_dim,
                output_dim=hidden_dim,
                n_layers=n_head_layers
            ),
            nn.Linear(hidden_dim, 2 * target_dim, bias=True)
        )

    def forward(self, x1: torch.Tensor, y1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        n_nodes, _ , _ = x2.shape

        r_agg, xh2 = self._backbone(x1, x2, y1)
        r_agg = r_agg.unsqueeze(0).repeat(n_nodes, 1, 1)

        xhr2 = torch.cat([xh2, r_agg], dim=-1)
        return self._head(xhr2)


class RNNCNPFilter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        target_dim: int,

        hidden_dim: int = 8,
        n_input2hidden_layers: int = 2,
        n_target2hidden_layers: int = 2,
        n_enc_layers: int = 2,

        n_head_layers: int = 2,
        n_agg_layers: int = 2,
        t_scale: float = 5.0,
        bounded_variance: bool = False,
        bounded_value: float = 0.1
    ):
        super().__init__()
        self._t_scale = t_scale
        self._bounded_variance = bounded_variance
        self._bounded_value = bounded_value

        self._backbone = RNNCNPBackbone(
            input_dim=input_dim,
            target_dim=target_dim,
            hidden_dim=hidden_dim,
            n_input2hidden_layers=n_input2hidden_layers,
            n_target2hidden_layers=n_target2hidden_layers,
            n_enc_layers=n_enc_layers,
            n_agg_layers=n_agg_layers
        )

        self._prior_head = nn.Sequential(
            MLP(
                input_dim=2 * hidden_dim,
                output_dim=hidden_dim,
                n_layers=n_head_layers
            ),
            nn.Linear(hidden_dim, 2 * target_dim, bias=True)
        )

        self._evidence2hidden = MLP(
            input_dim=target_dim,
            output_dim=hidden_dim,
            n_layers=n_target2hidden_layers
        )

        self._posterior_head = nn.Sequential(
            MLP(
                input_dim=3 * hidden_dim,
                output_dim=hidden_dim,
                n_layers=n_head_layers
            ),
            nn.Linear(hidden_dim, 2 * target_dim, bias=True)
        )

    def estimate_prior(self, x1: torch.Tensor, y1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n_nodes, _, _ = x2.shape
        r_agg, xh2 = self._backbone(x1, x2, y1)
        r_agg = r_agg.unsqueeze(0).repeat(n_nodes, 1, 1)

        xhr2 = torch.cat([xh2, r_agg], dim=-1)
        prior = self._prior_head(xhr2)

        return prior, xhr2

    def estimate_posterior(self, xhr2: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        yh2 = self._evidence2hidden(y2)
        xyhr2 = torch.cat([xhr2, yh2], dim=-1)
        return self._posterior_head(xyhr2)

    def _forward(self, x1: torch.Tensor, y1: torch.Tensor, x2: torch.Tensor, y2: torch.Tensor):
        prior, xhr2 = self.estimate_prior(x1, y1, x2)
        posterior = self.estimate_posterior(xhr2, y2)
        return prior, posterior

    def forward(self, x_obs: torch.Tensor, ts_obs: torch.Tensor, x_unobs: torch.Tensor, ts_unobs: torch.Tensor, metadata: dict) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            ts_obs, ts_unobs = to_scaled_relative_ts(ts_obs, ts_unobs, self._t_scale)

        _ = metadata  # Ignored
        return self._forward(ts_obs, x_obs, ts_unobs, x_unobs)

    def unpack_output(self, x_unobs_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_unobs_hat = x_unobs_hat.view(*x_unobs_hat.shape[:-1], -1, 2)
        x_unobs_hat_mean = x_unobs_hat[..., 0]
        x_unobs_hat_log_var = x_unobs_hat[..., 1]
        if not self._bounded_variance:
            x_unobs_hat_var = torch.exp(x_unobs_hat_log_var)
        else:
            x_unobs_hat_var = self._bounded_value + (1 - self._bounded_value) * torch.nn.functional.softplus(x_unobs_hat_log_var)

        return x_unobs_hat_mean, x_unobs_hat_var
