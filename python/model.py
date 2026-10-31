import torch
from torch import nn

from z3_map_gen import BOARD_SIZE, Catalog


class OutcomeModel(nn.Module):
    def __init__(
        self,
        catalog: Catalog,
        *,
        room_embedding_dim: int,
        position_embedding_dim: int,
        hidden_layers: int,
        hidden_width: int,
    ):
        super().__init__()
        if min(room_embedding_dim, position_embedding_dim, hidden_layers, hidden_width) <= 0:
            raise ValueError("Embedding dimensions, hidden layer count, and width must be positive")
        room_count = len(catalog.room_ids)
        cell_count = BOARD_SIZE * BOARD_SIZE
        self.room_embedding = nn.Embedding(catalog.room_cell_count, room_embedding_dim)
        self.position_embedding = nn.Embedding(cell_count, position_embedding_dim)
        input_width = cell_count * room_embedding_dim + room_count * position_embedding_dim
        layers = []
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(input_width, hidden_width), nn.ReLU()])
            input_width = hidden_width
        layers.append(nn.Linear(input_width, room_count + cell_count))
        self.mlp = nn.Sequential(*layers)

    def forward(self, room_cells: torch.Tensor, room_positions: torch.Tensor) -> torch.Tensor:
        cells = self.room_embedding(room_cells.clamp_min(0))
        cells = cells.masked_fill((room_cells < 0).unsqueeze(-1), 0)
        positions = self.position_embedding(room_positions.clamp_min(0))
        positions = positions.masked_fill((room_positions < 0).unsqueeze(-1), 0)
        inputs = torch.cat((cells.flatten(-2), positions.flatten(-2)), dim=-1)
        return self.mlp(inputs)
