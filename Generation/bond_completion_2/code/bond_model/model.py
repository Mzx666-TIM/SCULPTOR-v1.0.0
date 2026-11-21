from __future__ import annotations

import torch
from torch import nn


class MeanMessageLayer(nn.Module):
    """Small dependency-light message passing layer."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            aggregated = torch.zeros_like(x)
        else:
            source, target = edge_index
            aggregated = torch.zeros_like(x)
            aggregated.index_add_(0, target, x[source])
            degree = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            degree.index_add_(0, target, torch.ones_like(target, dtype=x.dtype))
            aggregated = aggregated / degree.clamp_min(1).unsqueeze(-1)

        out = self.self_linear(x) + self.neighbor_linear(aggregated)
        return self.norm(torch.relu(out))


class BondCompletionModel(nn.Module):
    """Scores candidate edges for a base half graph."""

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input = nn.Linear(node_feature_dim, hidden_dim)
        self.layers = nn.ModuleList(
            MeanMessageLayer(hidden_dim) for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data) -> torch.Tensor:
        x = torch.relu(self.input(data.x))
        for layer in self.layers:
            x = self.dropout(layer(x, data.edge_index))

        left = x[data.candidate_edge_index[0]]
        right = x[data.candidate_edge_index[1]]
        edge_features = torch.cat(
            [left, right, torch.abs(left - right), left * right], dim=-1
        )
        return self.edge_mlp(edge_features).squeeze(-1)
