from typing import Tuple, Optional

import torch
from torch import nn

from nodetracker.library.building_blocks.mlp import MLP


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
        n_classes: Optional[int] = None,

        hidden_dim: int = 8,
        n_input2hidden_layers: int = 2,
        n_target2hidden_layers: int = 2,
        n_enc_layers: int = 2,

        n_head_layers: int = 2,
        n_agg_layers: int = 2,
    ):
        super().__init__()
        self._is_classifier = (n_classes is not None)

        self._backbone = RNNCNPBackbone(
            input_dim=input_dim,
            target_dim=target_dim,
            hidden_dim=hidden_dim,
            n_input2hidden_layers=n_input2hidden_layers,
            n_target2hidden_layers=n_target2hidden_layers,
            n_enc_layers=n_enc_layers,
            n_agg_layers=n_agg_layers
        )

        output_dim = n_classes if self._is_classifier else 2 * target_dim
        self._head = nn.Sequential(
            MLP(
                input_dim=2 * hidden_dim,
                output_dim=hidden_dim,
                n_layers=n_head_layers
            ),
            nn.Linear(hidden_dim, output_dim, bias=True)
        )

    def forward(self, x1: torch.Tensor, y1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        n_nodes, _ , _ = x2.shape

        r_agg, xh2 = self._backbone(x1, x2, y1)
        r_agg = r_agg.unsqueeze(0).repeat(n_nodes, 1, 1)

        return self._head(torch.cat([xh2, r_agg], dim=-1))
