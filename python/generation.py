import torch
from torch.nn import functional as F

from model import OutcomeModel
from z3_map_gen import CandidateFeatures, EnvironmentGroup


CELL_FAILED = -1
NUM_ROOMS = 40

@torch.no_grad()
def score_candidates(
    model: OutcomeModel,
    candidates: CandidateFeatures,
    log_success_floor: float,
) -> torch.Tensor:
    """Score actual candidates using a finite negative log-success floor."""
    device = model.room_embedding.weight.device
    if candidates.environment_offsets[-1] == 0:
        return torch.empty(0, device=device)
    features = candidates.features
    room_cells = torch.tensor(features.room_cells, dtype=torch.long).to(device)
    n = room_cells.shape[0]
    room_positions = torch.tensor(features.room_positions, dtype=torch.long).to(device)
    failed_cells = room_cells == CELL_FAILED
    failed_position = torch.zeros([n, NUM_ROOMS], dtype=torch.bool, device=device)
    failed = torch.cat([failed_position, failed_cells], dim=-1)
    successful = torch.cat((room_positions >= 0, room_cells >= 0), dim=-1)
    logits = model(room_cells, room_positions)
    log_success = F.logsigmoid(logits).clamp_min(log_success_floor)
    log_success = log_success.masked_fill(failed, log_success_floor)
    log_success = log_success.masked_fill(successful, 0)
    return log_success.sum(dim=-1)


@torch.no_grad()
def sample_actions(
    environments: EnvironmentGroup,
    model: OutcomeModel,
    *,
    temperature: float,
    log_success_floor: float,
) -> list[tuple[int, int, int] | None]:
    """Sample once per environment; temperature must be positive and finite."""
    candidates = environments.extract_candidate_features()
    actions = candidates.actions
    offsets = candidates.environment_offsets
    environment_count = len(offsets) - 1
    if not actions:
        return [None] * environment_count
    scores = score_candidates(model, candidates, log_success_floor)
    counts = [end - start for start, end in zip(offsets[:-1], offsets[1:])]
    device = scores.device
    counts_tensor = torch.tensor(counts, device=device)
    rows = torch.arange(environment_count, device=device).repeat_interleave(
        counts_tensor, output_size=len(actions)
    )
    starts = torch.tensor(offsets[:-1], device=device).repeat_interleave(
        counts_tensor, output_size=len(actions)
    )
    columns = torch.arange(len(actions), device=device) - starts
    padded_scores = scores.new_full((environment_count, max(counts)), -torch.inf)
    padded_scores[rows, columns] = scores / temperature
    # Exhausted rows have one dummy slot, so their softmax remains defined.
    padded_scores[counts_tensor == 0, 0] = 0
    probabilities = torch.softmax(padded_scores, dim=-1)
    selected = torch.multinomial(probabilities, 1).squeeze(-1).tolist()
    return [
        actions[offsets[i] + column] if counts[i] else None
        for i, column in enumerate(selected)
    ]
