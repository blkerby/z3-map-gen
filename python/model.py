import torch
from torch import nn
from torch.nn import functional as F

from edges import EdgePairs
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
        widths, heights = torch.tensor(catalog.room_sizes).T
        cells = torch.arange(cell_count)
        valid_anchors = (cells % BOARD_SIZE + widths[:, None] <= BOARD_SIZE) & (
            cells // BOARD_SIZE + heights[:, None] <= BOARD_SIZE
        )
        valid_classes = F.pad(valid_anchors, (0, 1), value=True)  # Last class means never placed.
        anchor_count = int(valid_classes.sum())
        anchor_indices = torch.zeros_like(valid_classes, dtype=torch.long)
        anchor_indices[valid_classes] = torch.arange(1, anchor_count + 1)
        self.register_buffer("anchor_indices", anchor_indices, persistent=False)
        self.edge_pairs = EdgePairs(catalog)
        self.anchor_end = cell_count + anchor_count
        self.room_embedding = nn.Embedding(catalog.room_cell_count + 1, room_embedding_dim)
        self.position_embedding = nn.Embedding(cell_count + 1, position_embedding_dim)
        input_width = room_count * position_embedding_dim + cell_count * room_embedding_dim
        layers = []
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(input_width, hidden_width), nn.ReLU()])
            input_width = hidden_width
        layers.append(nn.Linear(input_width, self.anchor_end + self.edge_pairs.ids.numel()))
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        with torch.no_grad():
            # Give placement and non-placement equal prior probability.
            never_placed = cell_count + anchor_indices[:, -1] - 1
            layers[-1].bias[never_placed] = valid_anchors.sum(dim=-1).float().log()
        self.mlp = nn.Sequential(*layers)

    def forward(
        self, room_positions: torch.Tensor, room_cells: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = self.position_embedding(room_positions + 1)
        cells = self.room_embedding((room_cells + 1).clamp_min(0))
        inputs = torch.cat((positions.flatten(-2), cells.flatten(-2)), dim=-1)
        logits = self.mlp(inputs)
        cell_logits = logits[:, :BOARD_SIZE**2]
        anchor_logits = logits[:, BOARD_SIZE**2:self.anchor_end]
        edge_logits = logits[:, self.anchor_end:]
        # shortcut: this masks board bounds, add state-dependent legality if needed.
        anchor_logits = F.pad(anchor_logits, (1, 0), value=-torch.inf)[:, self.anchor_indices]
        return anchor_logits, cell_logits, edge_logits
