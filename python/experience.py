import bisect
import random

import torch
from z3_map_gen import BOARD_SIZE


def collision_totals(counts, possible_partners):
    totals = counts.sum(dim=1)
    expected = (totals * (totals - 1) / possible_partners).sum()
    return (counts * (counts - 1)).sum(), expected


def _collision_balance(counts, possible_partners):
    observed, expected = collision_totals(counts, possible_partners)
    return (observed / expected).item()


def placement_balance(actions, room_sizes):
    placements = actions[actions[:, :, 0] >= 0]
    indices = placements[:, 0] * BOARD_SIZE**2 + placements[:, 2] * BOARD_SIZE + placements[:, 1]
    counts = torch.bincount(indices, minlength=len(room_sizes) * BOARD_SIZE**2).reshape(
        len(room_sizes), BOARD_SIZE**2,
    )
    position_counts = [
        (BOARD_SIZE - width + 1) * (BOARD_SIZE - height + 1) for width, height in room_sizes
    ]
    return _collision_balance(counts, torch.tensor(position_counts, dtype=torch.float64))


def edge_partner_counts(room_sizes):
    total_width = sum(width for width, height in room_sizes)
    total_height = sum(height for width, height in room_sizes)
    possible_partners = []
    for width, height in room_sizes:
        possible_partners.extend([total_width - width] * (2 * width))
        possible_partners.extend([total_height - height] * (2 * height))
    return torch.tensor(possible_partners, dtype=torch.float64)


def edge_pair_counts(actions, room_sizes):
    edges = torch.full((len(actions), 4, BOARD_SIZE, BOARD_SIZE), -1, dtype=torch.long)
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
    first = torch.cat((edges[:, 1, :-1, :].flatten(), edges[:, 2, :, :-1].flatten()))
    second = torch.cat((edges[:, 0, 1:, :].flatten(), edges[:, 3, :, 1:].flatten()))
    paired = (first >= 0) & (second >= 0)
    counts = torch.bincount(
        first[paired] * segment_count + second[paired], minlength=segment_count**2,
    ).reshape(segment_count, segment_count)
    return counts + counts.T


def edge_pair_balance(actions, room_sizes):
    return _collision_balance(edge_pair_counts(actions, room_sizes), edge_partner_counts(room_sizes))


def map_anchors(valid_actions):
    # Room-ordered anchors identify layouts independently of placement order.
    positions = torch.empty(valid_actions.shape[:2], dtype=torch.uint8)
    positions.scatter_(
        1, valid_actions[:, :, 0],
        (valid_actions[:, :, 2] * BOARD_SIZE + valid_actions[:, :, 1]).to(torch.uint8),
    )
    return positions


class ExperienceRound:
    def __init__(self, actions, lengths, outcomes):
        self.tensors = {"actions": actions.to(torch.int8), "lengths": lengths, "outcomes": outcomes}
        self.cumulative_lengths = lengths.cumsum(dim=0).tolist()

    def sample_state(self):
        index = random.randrange(self.cumulative_lengths[-1])
        episode = bisect.bisect_right(self.cumulative_lengths, index)
        start = self.cumulative_lengths[episode - 1] if episode else 0
        length = index - start + 1
        episode_length = self.tensors["lengths"][episode]
        actions = self.tensors["actions"][episode, :episode_length].tolist()
        outcomes = self.tensors["outcomes"][episode]
        cell_count = BOARD_SIZE**2
        anchors = [cell_count] * (outcomes.numel() - cell_count)
        for room, x, y in actions:
            anchors[room] = y * BOARD_SIZE + x
        return [tuple(action) for action in actions[:length]], outcomes, anchors
