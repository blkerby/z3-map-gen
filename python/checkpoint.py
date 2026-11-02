import json
import os
import random
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def save_tensors(path: Path, tensors: dict, metadata: dict):
    stored = {}
    for name, tensor in tensors.items():
        stored[name] = tensor.detach().cpu().contiguous()
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(stored, temporary, metadata=metadata)
    os.replace(temporary, path)


def save_checkpoint(path, model, optimizer, completed_round, config, catalog_hash, room_ids, unique_maps):
    tensors = {}
    for name, tensor in model.state_dict().items():
        tensors[f"model.{name}"] = tensor
    optimizer_state = optimizer.state_dict()
    for index, state in optimizer_state["state"].items():
        for name, tensor in state.items():
            tensors[f"optimizer.{index}.{name}"] = tensor
    tensors["rng.cpu"] = torch.get_rng_state()
    if unique_maps is not None:
        tensors["unique_maps"] = torch.tensor(
            [list(layout) for layout in sorted(unique_maps)], dtype=torch.uint8,
        ).reshape(-1, len(room_ids))
    device = model.room_embedding.weight.device
    if device.type == "cuda":
        tensors["rng.device"] = torch.cuda.get_rng_state(device)
    metadata = {
        "completed_round": str(completed_round),
        "config": json.dumps(config),
        "catalog_hash": catalog_hash,
        "room_ids": json.dumps(room_ids),
        "optimizer_param_groups": json.dumps(optimizer_state["param_groups"]),
        "python_rng": json.dumps(random.getstate()),
    }
    save_tensors(path, tensors, metadata)


def load_checkpoint(path, model, optimizer, config):
    with safe_open(path, framework="pt", device="cpu") as file:
        metadata = file.metadata()
        completed_round = int(metadata["completed_round"])
        if completed_round > config["rounds"]:
            raise ValueError("Configured rounds precede the checkpoint")
        unique_maps = None
        if "unique_maps" in file.keys():
            unique_maps = {bytes(row) for row in file.get_tensor("unique_maps").tolist()}
        model_state = {}
        optimizer_state = {}
        for key in file.keys():
            if key.startswith("model."):
                model_state[key.removeprefix("model.")] = file.get_tensor(key)
            elif key.startswith("optimizer."):
                _, index, name = key.split(".")
                optimizer_state.setdefault(int(index), {})[name] = file.get_tensor(key)
        model.load_state_dict(model_state)
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
    return completed_round, unique_maps
