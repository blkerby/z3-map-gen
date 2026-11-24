import json
import os
import random
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def round_files(directory):
    paths = {}
    for path in directory.glob("round_*.safetensors"):
        match = re.fullmatch(r"round_(0|[1-9][0-9]*)\.safetensors", path.name)
        if match and not path.is_symlink() and path.is_file():
            paths[int(match[1])] = path
    return paths


def prune_checkpoints(directory, completed_round, period):
    paths = round_files(directory)
    periodic = [number for number in paths if 0 < number <= completed_round and number % period == 0]
    keep = set()
    if periodic:
        number = max(periodic)
        keep.add(number)
        remaining = (number - min(periodic)) // period
        gap = period
        while remaining:
            # Keep one or two gaps at this size and merge the older gaps in pairs.
            count = 1 if remaining % 2 else 2
            for _ in range(count):
                number -= gap
                keep.add(number)
            remaining = (remaining - count) // 2
            gap *= 2

    for number, path in paths.items():
        if completed_round - number > period and number not in keep:
            path.unlink()


def save_tensors(path: Path, tensors: dict, metadata: dict):
    stored = {}
    for name, tensor in tensors.items():
        stored[name] = tensor.detach().cpu().contiguous()
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(stored, temporary, metadata=metadata)
    os.replace(temporary, path)


def save_checkpoint(
    path, model, optimizer, completed_round, config, catalog_hash, room_ids, replay,
    placement_prices, edge_prices, wall_prices,
):
    tensors = {}
    for name, tensor in model.state_dict().items():
        tensors[f"model.{name}"] = tensor
    optimizer_state = optimizer.state_dict()
    for index, state in optimizer_state["state"].items():
        for name, tensor in state.items():
            tensors[f"optimizer.{index}.{name}"] = tensor
    tensors["rng.cpu"] = torch.get_rng_state()
    tensors["placement_prices"] = placement_prices
    tensors["edge_prices"] = edge_prices
    tensors["wall_prices"] = wall_prices
    for number, experience in replay.items():
        for name, tensor in experience.tensors.items():
            tensors[f"replay.{number}.{name}"] = tensor
    device = model.room_embedding.weight.device
    if device.type == "cuda":
        tensors["rng.device"] = torch.cuda.get_rng_state(device)
    metadata = {
        "completed_round": str(completed_round),
        "config": json.dumps(config),
        "catalog_hash": catalog_hash,
        "room_ids": json.dumps(room_ids),
        "replay_rounds": json.dumps(list(replay)),
        "optimizer_param_groups": json.dumps(optimizer_state["param_groups"]),
        "python_rng": json.dumps(random.getstate()),
    }
    save_tensors(path, tensors, metadata)


def load_checkpoint(path, model, optimizer, config, catalog_hash, placement_prices, edge_prices, wall_prices):
    with safe_open(path, framework="pt", device="cpu") as file:
        metadata = file.metadata()
        completed_round = int(metadata["completed_round"])
        if completed_round > config["rounds"]:
            raise ValueError("Configured rounds precede the checkpoint")
        if metadata["catalog_hash"] != catalog_hash:
            raise ValueError(f"Checkpoint catalog does not match: {path}")
        if "replay_rounds" not in metadata:
            raise ValueError(f"Checkpoint does not contain replay: {path}")
        if "edge_prices" not in file.keys() or "wall_prices" not in file.keys():
            raise ValueError(f"Checkpoint lacks edge prices for wall-slot balance: {path}")
        replay = {
            number: {
                name: file.get_tensor(f"replay.{number}.{name}")
                for name in ("actions", "lengths", "outcomes")
            }
            for number in json.loads(metadata["replay_rounds"])[-config["train"]["experience_rounds"]:]
        }
        model_state = {}
        optimizer_state = {}
        for key in file.keys():
            if key.startswith("model."):
                model_state[key.removeprefix("model.")] = file.get_tensor(key)
            elif key.startswith("optimizer."):
                _, index, name = key.split(".")
                optimizer_state.setdefault(int(index), {})[name] = file.get_tensor(key)
        model.load_state_dict(model_state)
        placement_prices.copy_(file.get_tensor("placement_prices"))
        edge_prices.copy_(file.get_tensor("edge_prices"))
        wall_prices.copy_(file.get_tensor("wall_prices"))
        optimizer.load_state_dict({
            "state": optimizer_state,
            "param_groups": json.loads(metadata["optimizer_param_groups"]),
        })
        torch.set_rng_state(file.get_tensor("rng.cpu"))
        device = model.room_embedding.weight.device
        if device.type == "cuda":
            torch.cuda.set_rng_state(file.get_tensor("rng.device"), device)
        version, state, gaussian = json.loads(metadata["python_rng"])
        random.setstate((version, tuple(state), gaussian))
    return completed_round, replay
