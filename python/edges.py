import torch
from torch import nn

from experience import collision_totals
from z3_map_gen import BOARD_SIZE


class EdgePairs(nn.Module):
    def __init__(self, catalog):
        super().__init__()
        north, south, east, west = [], [], [], []
        segment = 0
        room_sizes = catalog.room_sizes
        for room, (width, height) in enumerate(room_sizes):
            # Catalog order is north, south, east, west within each room.
            for side, offsets in (
                (north, [(x, 0) for x in range(width)]),
                (south, [(x, height - 1) for x in range(width)]),
                (east, [(width - 1, y) for y in range(height)]),
                (west, [(0, y) for y in range(height)]),
            ):
                for x, y in offsets:
                    side.append((segment, room, x, y))
                    segment += 1

        matches = catalog.segment_matches
        pairs = [
            (a, b, room_a, room_b, x_a + step_x - x_b, y_a + step_y - y_b)
            for first, second, step_x, step_y in ((south, north, 0, 1), (east, west, 1, 0))
            for a, room_a, x_a, y_a in first
            for b, room_b, x_b, y_b in second
            if room_a != room_b and matches[a][b]
        ]
        segments_a, segments_b, rooms_a, rooms_b, dx, dy = torch.tensor(
            pairs, dtype=torch.long,
        ).reshape(-1, 6).T
        segment_count = segment + 4 * BOARD_SIZE
        self.edge_bins = segment_count**2
        self.pair_bins = segments_a * segment_count + segments_b
        closed = catalog.segment_closed
        wall_pairs = []
        wall_anchors = []
        for side, segments in enumerate((north, south, east, west)):
            for segment_id, room, offset_x, offset_y in segments:
                if not closed[segment_id]:
                    continue
                width, height = room_sizes[room]
                for slot in range(BOARD_SIZE):
                    if side < 2:
                        x, y = slot - offset_x, 0 if side == 0 else BOARD_SIZE - height
                    else:
                        x, y = BOARD_SIZE - width if side == 2 else 0, slot - offset_y
                    if 0 <= x <= BOARD_SIZE - width and 0 <= y <= BOARD_SIZE - height:
                        wall_pairs.append((segment_id, segment + side * BOARD_SIZE + slot))
                        wall_anchors.append(room * BOARD_SIZE**2 + y * BOARD_SIZE + x)
        wall_a, wall_b = torch.tensor(wall_pairs, dtype=torch.long).reshape(-1, 2).T
        self.wall_bins = wall_a * segment_count + wall_b
        self.wall_anchors = torch.tensor(wall_anchors, dtype=torch.long)
        self.room_count = len(room_sizes)
        degrees = torch.cat((wall_a, wall_b)).bincount(minlength=segment_count)
        interior_partners = []
        for width, height in room_sizes:
            interior_partners.extend([len(north) - width] * (2 * width))
            interior_partners.extend([len(east) - height] * (2 * height))
        degrees[:segment] += torch.tensor(interior_partners, dtype=degrees.dtype)
        self.possible_partners = degrees.double().masked_fill(degrees == 0, float("inf"))
        # Each stored contact represents two endpoints in the uniform collision rate.
        self.uniform_collisions = 0.5 * self.possible_partners.reciprocal().sum().item()

        widths, heights = torch.tensor(room_sizes).T
        anchors = torch.arange(BOARD_SIZE**2)[:, None]
        x, y = anchors % BOARD_SIZE, anchors // BOARD_SIZE
        self.register_buffer("rooms", torch.stack((rooms_a, rooms_b)), persistent=False)
        self.register_buffer("ids", torch.arange(len(pairs)), persistent=False)
        for name, rooms, required_x, required_y in (
            ("second_anchors", rooms_b, x + dx, y + dy),
            ("first_anchors", rooms_a, x - dx, y - dy),
        ):
            in_bounds = (
                (required_x >= 0) & (required_x + widths[rooms] <= BOARD_SIZE)
                & (required_y >= 0) & (required_y + heights[rooms] <= BOARD_SIZE)
            )
            required = (required_y * BOARD_SIZE + required_x).masked_fill(~in_bounds, -1)
            # Input anchor -1 indexes the last row for an unplaced room.
            required = torch.cat((required, required.new_full((1, len(pairs)), -1)))
            self.register_buffer(name, required, persistent=False)

    def collision_normalizer(self, layout_count):
        return layout_count * (layout_count - 1) * self.uniform_collisions

    def balance(self, counts, layout_count):
        normalizer = self.collision_normalizer(layout_count)
        counts = counts.double()
        return counts.dot(counts - 1).item() / normalizer if normalizer else float("nan")

    def observed_balance(self, counts):
        matrix = counts.reshape(self.possible_partners.numel(), -1).double()
        observed, expected = collision_totals(matrix + matrix.T, self.possible_partners)
        return (observed / expected).item() if expected > 0 else float("nan")

    def contacts(self, room_positions):
        first, second = room_positions[:, self.rooms].unbind(dim=1)
        required_first = self.first_anchors[second, self.ids]
        required_second = self.second_anchors[first, self.ids]
        first_placed, second_placed = first >= 0, second >= 0
        formed = second_placed & (required_second == second)
        # shortcut: check bounds only, add overlap and other contacts if needed.
        unresolved = (
            ~(first_placed & second_placed)
            & (~first_placed | (required_second >= 0))
            & (~second_placed | (required_first >= 0))
        )
        return formed, unresolved


class EdgeController:
    def __init__(self, pairs, device):
        self.pairs = pairs
        self.prices = torch.zeros(pairs.ids.numel(), device=device)
        self.wall_prices = torch.zeros(pairs.wall_bins.numel())

    def wall_anchor_prices(self):
        prices = torch.bincount(
            self.pairs.wall_anchors, weights=self.wall_prices, minlength=self.pairs.room_count * BOARD_SIZE**2,
        )
        return prices.reshape(self.pairs.room_count, BOARD_SIZE**2).to(self.prices)

    @torch.no_grad()
    def update(self, counts, episode_count, *, gain, leak):
        self.prices.mul_(1 - leak)
        self.wall_prices.mul_(1 - leak)
        if episode_count and gain:
            self.prices.add_(counts[self.pairs.pair_bins].to(self.prices), alpha=gain / episode_count)
            self.wall_prices.add_(counts[self.pairs.wall_bins], alpha=gain / episode_count)
        prices = torch.cat((self.prices, self.wall_prices.to(self.prices)))
        return {
            "edge_pair_balance": self.pairs.observed_balance(counts),
            "edge_price_rms": prices.square().mean().sqrt().item(),
        }
