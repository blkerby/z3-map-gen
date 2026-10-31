import torch
from torch.nn import functional as F

from model import OutcomeModel
from z3_map_gen import CandidateFeatures, Environment


@torch.no_grad()
def score_candidates(
    model: OutcomeModel,
    candidates: CandidateFeatures,
    log_success_floor: float,
) -> torch.Tensor:
    """Score actual candidates using a finite negative log-success floor."""
    device = model.room_embedding.weight.device
    features = candidates.features
    if not features:
        return torch.empty(0, device=device)
    room_cells = torch.tensor([f.room_cells for f in features], dtype=torch.long, device=device)
    room_positions = torch.tensor([f.room_positions for f in features], dtype=torch.long, device=device)
    failed = torch.tensor(candidates.failed_outcomes, dtype=torch.bool, device=device)
    successful = torch.cat((room_positions >= 0, room_cells >= 0), dim=-1)
    logits = model(room_cells, room_positions)
    log_success = F.logsigmoid(logits).clamp_min(log_success_floor)
    log_success = log_success.masked_fill(failed, log_success_floor)
    log_success = log_success.masked_fill(successful, 0)
    return log_success.sum(dim=-1)


@torch.no_grad()
def sample_action(
    environment: Environment,
    model: OutcomeModel,
    *,
    temperature: float,
    log_success_floor: float,
) -> tuple[int, int, int] | None:
    """Sample without committing; temperature must be positive and finite."""
    candidates = environment.extract_candidate_features()
    actions = candidates.actions
    if not actions:
        return None
    scores = score_candidates(model, candidates, log_success_floor)
    probabilities = torch.softmax(scores / temperature, dim=0)
    index = torch.multinomial(probabilities, 1).item()
    return actions[index]
