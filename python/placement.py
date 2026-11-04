import torch

from z3_map_gen import BOARD_SIZE


class PlacementController:
    """Leaky integral prices, updated once per round using all generated maps.

    Frequencies are conditional on each room being placed. Prices are in the
    same units as candidate log-success scores, before temperature scaling.
    Each update is (1 - leak) * price + gain * (observed / target - 1).
    A room with no observations receives decay only.
    """

    def __init__(self, room_sizes, device, *, gain, leak):
        self.gain = gain
        self.leak = leak
        self.target = torch.zeros((len(room_sizes), BOARD_SIZE, BOARD_SIZE), device=device)
        for room, (width, height) in enumerate(room_sizes):
            if (width, height) == (2, 2):
                self.target[room, :7, :7] = 1 / 49
            else:
                self.target[room] = 17 / 1568
                self.target[room, [0, -1], :] = 33 / 1568
                self.target[room, :, [0, -1]] = 33 / 1568
                self.target[room, 0, [0, -1]] = 41 / 1568
                self.target[room, -1, [0, -1]] = 41 / 1568
        self.target = self.target.flatten(1)
        self.prices = torch.zeros_like(self.target)

    @torch.no_grad()
    def update(self, actions):
        placements = actions[actions[:, :, 0] >= 0].to(self.prices.device)
        indices = placements[:, 0] * BOARD_SIZE**2 + placements[:, 2] * BOARD_SIZE + placements[:, 1]
        counts = torch.bincount(indices, minlength=self.prices.numel()).reshape_as(self.prices)
        totals = counts.sum(dim=1, keepdim=True)
        observed = counts / totals.clamp_min(1)
        supported = self.target > 0
        error = observed / self.target.masked_fill(~supported, 1) - 1
        error.masked_fill_(~supported | (totals == 0), 0)
        self.prices.mul_(1 - self.leak).add_(error, alpha=self.gain)
        imbalance = (error.square() * self.target).sum(dim=1)
        return {
            "placement_target_imbalance": imbalance.sum().item() / (totals > 0).sum().clamp_min(1).item(),
            "placement_price_rms": (self.prices.square() * self.target).sum(dim=1).mean().sqrt().item(),
        }
