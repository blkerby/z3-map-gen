import torch
from torch import nn

from experience import collision_totals, edge_partner_counts
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
        widths, heights = torch.tensor(room_sizes).T
        anchors = torch.arange(BOARD_SIZE**2)[:, None]
        x, y = anchors % BOARD_SIZE, anchors // BOARD_SIZE
        self.register_buffer("segments", torch.stack((segments_a, segments_b)), persistent=False)
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
    def __init__(self, pairs, room_sizes, device):
        self.segments = pairs.segments.cpu()
        self.possible_partners = edge_partner_counts(room_sizes)
        self.prices = torch.zeros(pairs.ids.numel(), device=device)

    @torch.no_grad()
    def update(self, counts, episode_count, *, gain, leak):
        observed, expected = collision_totals(counts, self.possible_partners)
        balance = (observed / expected).item()
        self.prices.mul_(1 - leak)
        if expected > 0 and gain:
            first, second = self.segments
            # Each added contact affects both endpoints of the metric denominator.
            endpoint_rates = counts.sum(dim=1) / self.possible_partners
            marginal = counts[first, second] - 0.5 * balance * (
                endpoint_rates[first] + endpoint_rates[second]
            )
            self.prices.add_(marginal.to(self.prices.device), alpha=gain / episode_count)
        return {"edge_pair_balance": balance, "edge_price_rms": self.prices.square().mean().sqrt().item()}
