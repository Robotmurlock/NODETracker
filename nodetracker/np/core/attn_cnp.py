from typing import Tuple, Optional

import torch
from torch import nn

from nodetracker.library.building_blocks.attention import TemporalFirstMultiHeadSelfAttention, TemporalFirstMultiHeadCrossAttention
from nodetracker.library.building_blocks.mlp import MLP


class AttentionCNPBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        n_heads: int,

        hidden_dim: int = 8,
        n_input2hidden_layers: int = 2,
        n_target2hidden_layers: int = 2,
        n_enc_layers: int = 2
    ):
        super().__init__()
        assert hidden_dim % n_heads == 0

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

        self._enc = nn.Sequential(
            MLP(
                input_dim=2 * hidden_dim,
                output_dim=hidden_dim,
                n_layers=n_enc_layers
            ),
            TemporalFirstMultiHeadSelfAttention(
                n_heads=n_heads,
                dim=hidden_dim
            )
        )

        self._mhca = TemporalFirstMultiHeadCrossAttention(n_heads=n_heads)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, y1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encoding observed nodes
        xh1 = self._input2hidden(x1)  # Keys
        yh1 = self._target2hidden(y1)

        xyh1 = torch.cat([xh1, yh1], dim=-1)
        r1 = self._enc(xyh1)  # Values

        # Encoding target nodes
        xh2 = self._input2hidden(x2)  # Queries
        r_all = self._mhca(q=xh2, k=xh1, v=r1)

        return r_all, xh2


class AttnCNP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        n_heads: int,

        n_classes: Optional[int] = None,

        hidden_dim: int = 8,
        n_input2hidden_layers: int = 2,
        n_target2hidden_layers: int = 2,
        n_enc_layers: int = 2,

        n_head_layers: int = 2
    ):
        super().__init__()
        self._is_classifier = (n_classes is not None)

        self._backbone = AttentionCNPBackbone(
            input_dim=input_dim,
            target_dim=target_dim,
            hidden_dim=hidden_dim,
            n_input2hidden_layers=n_input2hidden_layers,
            n_target2hidden_layers=n_target2hidden_layers,
            n_enc_layers=n_enc_layers,
            n_heads=n_heads
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

        r_all, xh2 = self._backbone(x1, x2, y1)

        return self._head(torch.cat([xh2, r_all], dim=-1))
