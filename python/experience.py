import bisect
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from z3_map_gen import BOARD_SIZE


def collect_unique_maps(actions, outcomes):
    valid_actions = actions[outcomes.all(dim=1)]
    # Room-ordered anchors identify layouts independently of placement order.
    positions = torch.empty(valid_actions.shape[:2], dtype=torch.uint8)
    positions.scatter_(
        1, valid_actions[:, :, 0],
        (valid_actions[:, :, 2] * BOARD_SIZE + valid_actions[:, :, 1]).to(torch.uint8),
    )
    return {bytes(row) for row in positions.unique(dim=0).tolist()}


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
