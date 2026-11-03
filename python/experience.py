import bisect
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from z3_map_gen import BOARD_SIZE


def _collision_balance(counts, possible_partners):
    totals = counts.sum(dim=1)
    possible_partners = torch.tensor(possible_partners, dtype=torch.float64)
    expected = (totals * (totals - 1) / possible_partners).sum().item()
    return (counts * (counts - 1)).sum().item() / expected if expected else float("nan")


def placement_balance(actions, room_sizes):
    placements = actions[actions[:, :, 0] >= 0]
    indices = placements[:, 0] * BOARD_SIZE**2 + placements[:, 2] * BOARD_SIZE + placements[:, 1]
    counts = torch.bincount(indices, minlength=len(room_sizes) * BOARD_SIZE**2).reshape(
        len(room_sizes), BOARD_SIZE**2,
    )
    position_counts = [
        (BOARD_SIZE - width + 1) * (BOARD_SIZE - height + 1) for width, height in room_sizes
    ]
    return _collision_balance(counts, position_counts)


def edge_pair_balance(actions, room_sizes):
    edges = torch.full((len(actions), 4, BOARD_SIZE, BOARD_SIZE), -1, dtype=torch.long)
    total_width = sum(width for width, height in room_sizes)
    total_height = sum(height for width, height in room_sizes)
    possible_partners = []
    segment_count = 0
    for room, (width, height) in enumerate(room_sizes):
        episode, step = (actions[:, :, 0] == room).nonzero(as_tuple=True)
        x, y = actions[episode, step, 1:].unbind(dim=1)
        # Match catalog segment order: north, south, east, west within each room.
        for dx in range(width):
            edges[episode, 0, y, x + dx] = segment_count + dx
            edges[episode, 1, y + height - 1, x + dx] = segment_count + width + dx
        for dy in range(height):
            edges[episode, 2, y + dy, x + width - 1] = segment_count + 2 * width + dy
            edges[episode, 3, y + dy, x] = segment_count + 2 * width + height + dy
        segment_count += 2 * (width + height)
        possible_partners.extend([total_width - width] * (2 * width))
        possible_partners.extend([total_height - height] * (2 * height))
    first = torch.cat((edges[:, 1, :-1, :].flatten(), edges[:, 2, :, :-1].flatten()))
    second = torch.cat((edges[:, 0, 1:, :].flatten(), edges[:, 3, :, 1:].flatten()))
    paired = (first >= 0) & (second >= 0)
    counts = torch.bincount(
        first[paired] * segment_count + second[paired], minlength=segment_count**2,
    ).reshape(segment_count, segment_count)
    return _collision_balance(counts + counts.T, possible_partners)


def collect_unique_maps(actions, outcomes):
    valid_actions = actions[outcomes.all(dim=1)]
    # Room-ordered anchors identify layouts independently of placement order.
    positions = torch.empty(valid_actions.shape[:2], dtype=torch.uint8)
    positions.scatter_(
        1, valid_actions[:, :, 0],
        (valid_actions[:, :, 2] * BOARD_SIZE + valid_actions[:, :, 1]).to(torch.uint8),
    )
    unique_positions, counts = positions.unique(dim=0, return_counts=True)
    return {bytes(row) for row in unique_positions.tolist()}, counts


@dataclass
class ExperienceRound:
    actions: list[list[tuple[int, int, int]]]
    outcomes: torch.Tensor
    cumulative_lengths: list[int]

    @classmethod
    def load(cls, path: Path, catalog_hash: str):
        with safe_open(path, framework="pt", device="cpu") as file:
            if file.metadata()["catalog_hash"] != catalog_hash:
                raise ValueError(f"Experience catalog does not match: {path}")
            stored_actions = file.get_tensor("actions").tolist()
            lengths = file.get_tensor("lengths").tolist()
            outcomes = file.get_tensor("outcomes")
        actions = []
        cumulative_lengths = []
        total = 0
        for episode, length in zip(stored_actions, lengths):
            actions.append([tuple(action) for action in episode[:length]])
            total += length
            cumulative_lengths.append(total)
        return cls(actions=actions, outcomes=outcomes, cumulative_lengths=cumulative_lengths)

    def sample_state(self):
        index = random.randrange(self.cumulative_lengths[-1])
        episode = bisect.bisect_right(self.cumulative_lengths, index)
        start = self.cumulative_lengths[episode - 1] if episode else 0
        length = index - start + 1
        return self.actions[episode][:length], self.outcomes[episode]
