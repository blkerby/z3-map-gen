import argparse
import hashlib
import json
import logging
import math
import os
import random
import signal
import sys
from bisect import bisect_right
from datetime import datetime
from pathlib import Path

import torch
from safetensors import safe_open
from torch.nn import functional as F

from checkpoint import load_checkpoint, save_checkpoint, save_tensors
from experience import ExperienceRound, collect_unique_maps, edge_pair_balance, placement_balance
from generation import sample_actions
from model import OutcomeModel
from z3_map_gen import EPISODE_LENGTH, Catalog, EnvironmentGroup


def load_config(path):
    config = json.loads(path.read_text())
    for key in ("rounds", "checkpoint_period"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for group, keys in {
        "generation": ("environment_count", "batches_per_round"),
        "train": ("batch_size", "batches_per_round", "experience_rounds"),
    }.items():
        for key in keys:
            if type(config[group][key]) is not int or config[group][key] <= 0:
                raise ValueError(f"{group}.{key} must be a positive integer")
    scheduled_settings = (
        ("optimizer", "learning_rate"),
        ("optimizer", "beta1"),
        ("optimizer", "beta2"),
        ("generation", "temperature"),
    )
    if any(isinstance(config[group][key], dict) for group, key in scheduled_settings):
        knots = config["knot_episodes"]
        if not isinstance(knots, list) or not knots:
            raise ValueError("knot_episodes must be a nonempty list")
        previous = -1
        for knot in knots:
            if type(knot) is not int or knot <= previous:
                raise ValueError("knot_episodes must be strictly increasing nonnegative integers")
            previous = knot
    for group, key in scheduled_settings:
        setting = config[group][key]
        kind = None
        values = [setting]
        if isinstance(setting, dict):
            if len(setting) != 1 or next(iter(setting)) not in ("linear", "log", "step"):
                raise ValueError(f"{group}.{key} must have exactly one of linear, log, or step")
            kind, values = next(iter(setting.items()))
            if not isinstance(values, list) or len(values) != len(knots):
                raise ValueError(f"{group}.{key} values must match knot_episodes")
        for value in values:
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{group}.{key} values must be nonnegative and finite")
            if key == "temperature" and value == 0:
                raise ValueError("generation.temperature values must be positive")
            if key in ("beta1", "beta2") and value >= 1:
                raise ValueError(f"optimizer.{key} values must be less than one")
            if kind == "log" and value == 0:
                raise ValueError(f"{group}.{key} values must be positive for a log schedule")
    floor = config["generation"]["log_success_floor"]
    if not math.isfinite(floor) or floor >= 0:
        raise ValueError("generation.log_success_floor must be finite and negative")
    config["room_data"] = str(Path(config["room_data"]).resolve())
    return config


def instantiate_schedule(value, knots, position):
    if not isinstance(value, dict):
        return value
    kind, values = next(iter(value.items()))
    index = max(0, bisect_right(knots, position) - 1)
    if kind == "step" or position <= knots[0] or index == len(knots) - 1:
        return values[index]
    fraction = (position - knots[index]) / (knots[index + 1] - knots[index])
    start, end = values[index:index + 2]
    if kind == "log":
        return math.exp((1 - fraction) * math.log(start) + fraction * math.log(end))
    return (1 - fraction) * start + fraction * end


def instantiate_schedules(settings, knots, position):
    return {key: instantiate_schedule(value, knots, position) for key, value in settings.items()}


class TrainingSession:
    def __init__(self, config, resume, log_unique_maps=False, log_cumulative_unique_maps=False):
        self.config = config
        self.log_unique_maps = log_unique_maps
        self.unique_maps = set() if log_cumulative_unique_maps else None
        if resume is None:
            self.run_dir = Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        else:
            self.run_dir = resume.resolve().parent.parent
        self.catalog = Catalog(config["room_data"])
        self.catalog_hash = hashlib.sha256(Path(config["room_data"]).read_bytes()).hexdigest()
        self.device = torch.device(config["device"])
        random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        self.model = OutcomeModel(self.catalog, **config["model"]).to(self.device)
        self.model.compile(dynamic=True)
        settings = instantiate_schedules(config["optimizer"], config.get("knot_episodes"), 0)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=settings["learning_rate"],
            betas=(settings["beta1"], settings["beta2"]),
            eps=1e-8,
            weight_decay=0,
        )
        self.completed_round = 0
        self.experience = {}
        self.stop_requested = False
        if resume is None:
            self.run_dir.mkdir(parents=True, exist_ok=False)
        else:
            self.completed_round, unique_maps = load_checkpoint(
                resume, self.model, self.optimizer, config,
            )
            if unique_maps is not None:
                self.unique_maps = unique_maps
            elif log_cumulative_unique_maps:
                for number in range(1, self.completed_round + 1):
                    path = self.run_dir / "experience" / f"round_{number}.safetensors"
                    with safe_open(path, framework="pt", device="cpu") as file:
                        if file.metadata()["catalog_hash"] != self.catalog_hash:
                            raise ValueError(f"Experience catalog does not match: {path}")
                        unique_maps, _ = collect_unique_maps(
                            file.get_tensor("actions"), file.get_tensor("outcomes"),
                        )
                        self.unique_maps.update(unique_maps)
        for name in ("experience", "checkpoints", "metrics"):
            (self.run_dir / name).mkdir(exist_ok=True)
        (self.run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    def request_stop(self, signum, _frame):
        if self.stop_requested:
            os._exit(128 + signum)
        self.stop_requested = True
        logging.info(
            "Finishing the current round, then saving and stopping; "
            "interrupt again to exit immediately."
        )

    def generate_round(self):
        self.model.eval()
        settings = self.config["generation"]
        num_episodes = self.completed_round * settings["environment_count"] * settings["batches_per_round"]
        temperature = instantiate_schedule(
            settings["temperature"], self.config.get("knot_episodes"), num_episodes,
        )
        action_batches = []
        outcome_batches = []
        for _ in range(settings["batches_per_round"]):
            environments = EnvironmentGroup(self.catalog, settings["environment_count"])
            steps = []
            for _ in range(EPISODE_LENGTH):
                actions = sample_actions(
                    environments, self.model,
                    temperature=temperature,
                    log_success_floor=settings["log_success_floor"],
                )
                environments.apply_actions(actions)
                steps.append([action if action is not None else (-1, -1, -1) for action in actions])
            action_batches.append(torch.tensor(steps, dtype=torch.long).permute(1, 0, 2))
            features = environments.extract_features()
            outcomes = torch.cat(
                (features.room_positions >= 0, features.room_cells >= 0),
                dim=1,
            )
            outcome_batches.append(outcomes)
        actions = torch.cat(action_batches)
        return {
            "actions": actions,
            "lengths": (actions[:, :, 0] >= 0).sum(dim=1),
            "outcomes": torch.cat(outcome_batches),
        }

    def train_round(self, round_number):
        self.model.train()
        generation = self.config["generation"]
        num_episodes = round_number * generation["environment_count"] * generation["batches_per_round"]
        optimizer_settings = instantiate_schedules(
            self.config["optimizer"], self.config.get("knot_episodes"), num_episodes,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = optimizer_settings["learning_rate"]
            group["betas"] = (optimizer_settings["beta1"], optimizer_settings["beta2"])
        settings = self.config["train"]
        first_round = max(1, round_number - settings["experience_rounds"] + 1)
        for old_round in list(self.experience):
            if old_round < first_round:
                del self.experience[old_round]
        for number in range(first_round, round_number + 1):
            if number not in self.experience:
                path = self.run_dir / "experience" / f"round_{number}.safetensors"
                self.experience[number] = ExperienceRound.load(path, self.catalog_hash)
        total_loss = 0.0
        for _ in range(settings["batches_per_round"]):
            episodes = []
            labels = []
            for _ in range(settings["batch_size"]):
                number = random.randrange(first_round, round_number + 1)
                actions, outcomes = self.experience[number].sample_state()
                episodes.append(actions)
                labels.append(outcomes)
            features = EnvironmentGroup.replay(self.catalog, episodes).extract_features()
            room_positions = features.room_positions.to(self.device)
            room_cells = features.room_cells.to(self.device)
            targets = torch.stack(labels).to(device=self.device, dtype=torch.float32)
            unresolved = torch.cat((room_positions < 0, room_cells < 0), dim=1)
            logits = self.model(room_positions, room_cells)
            losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            loss = losses.masked_fill(~unresolved, 0).sum()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return {
            **optimizer_settings,
            "loss_per_prefix": total_loss / (settings["batches_per_round"] * settings["batch_size"]),
        }

    def save_checkpoint(self):
        path = self.run_dir / "checkpoints" / f"round_{self.completed_round}.safetensors"
        save_checkpoint(
            path, self.model, self.optimizer, self.completed_round, self.config,
            self.catalog_hash, self.catalog.room_ids,
            self.unique_maps,
        )
        logging.info("Saved checkpoint: %s", path)

    def run(self):
        last_saved_round = None
        for number in range(self.completed_round + 1, self.config["rounds"] + 1):
            if self.stop_requested:
                break
            data = self.generate_round()
            save_tensors(
                self.run_dir / "experience" / f"round_{number}.safetensors", data,
                {
                    "round": str(number),
                    "catalog_hash": self.catalog_hash,
                    "room_ids": json.dumps(self.catalog.room_ids),
                    "config": json.dumps(self.config),
                },
            )
            metrics = self.train_round(number)
            room_count = len(self.catalog.room_ids)
            outcomes = data["outcomes"].float()
            valid_actions = data["actions"][data["outcomes"].all(dim=1)]
            metrics.update({
                "round": number,
                "complete_map_frequency": outcomes[:, :room_count].all(dim=1).float().mean().item(),
                "mean_rooms_unplaced": (1.0 - outcomes[:, :room_count]).sum(dim=1).mean().item(),
                "mean_unoccupied_cells": (1.0 - outcomes[:, room_count:]).sum(dim=1).mean().item(),
                "room_placement_frequency": outcomes[:, :room_count].mean(dim=0).tolist(),
                "cell_fill_frequency": outcomes[:, room_count:].mean(dim=0).tolist(),
                "placement_balance": placement_balance(data["actions"], self.catalog.room_sizes),
                "placement_balance_valid": placement_balance(valid_actions, self.catalog.room_sizes),
                "edge_pair_balance": edge_pair_balance(data["actions"], self.catalog.room_sizes),
                "edge_pair_balance_valid": edge_pair_balance(valid_actions, self.catalog.room_sizes),
            })
            unique_log = ""
            if self.log_unique_maps or self.unique_maps is not None:
                unique_maps, counts = collect_unique_maps(data["actions"], data["outcomes"])
                if self.log_unique_maps:
                    metrics["unique_maps"] = len(unique_maps)
                    unique_log += f" unique={len(unique_maps)}"
                    valid_count = counts.sum().item()
                    collision_probability = None
                    if valid_count >= 2:
                        collision_probability = (counts * (counts - 1)).sum().item() / (
                            valid_count * (valid_count - 1)
                        )
                    metrics["map_collision_probability"] = collision_probability
                    if collision_probability is None:
                        unique_log += " collision=n/a"
                    else:
                        unique_log += f" collision={collision_probability:.5f}"
                if self.unique_maps is not None:
                    self.unique_maps.update(unique_maps)
                    metrics["cumulative_unique_maps"] = len(self.unique_maps)
                    unique_log += f" cumulative_unique={len(self.unique_maps)}"
            (self.run_dir / "metrics" / f"round_{number}.json").write_text(json.dumps(metrics) + "\n")
            self.completed_round = number
            logging.info(
                "Round %s: complete=%.3f rooms=%.2f cells=%.2f loss=%.4f lr=%.6g "
                "bal=%.5f,%.5f edge=%.5f,%.5f%s",
                number,
                metrics["complete_map_frequency"], metrics["mean_rooms_unplaced"],
                metrics["mean_unoccupied_cells"], metrics["loss_per_prefix"], metrics["learning_rate"],
                metrics["placement_balance"], metrics["placement_balance_valid"],
                metrics["edge_pair_balance"], metrics["edge_pair_balance_valid"], unique_log,
            )
            if number % self.config["checkpoint_period"] == 0:
                self.save_checkpoint()
                last_saved_round = number
            if self.stop_requested:
                break
        if last_saved_round != self.completed_round:
            self.save_checkpoint()


def main():
    parser = argparse.ArgumentParser(description="Train the overworld outcome predictor")
    parser.add_argument("config", type=Path)
    parser.add_argument("--resume", type=Path, help="Resume a safetensors checkpoint in the same run")
    parser.add_argument(
        "--log-unique-maps", action="store_true",
        help="Log unique valid maps and estimated collision probability in each round",
    )
    parser.add_argument(
        "--log-cumulative-unique-maps", action="store_true",
        help="Log unique valid maps since the run began; resume saved tracking or reconstruct from experience",
    )
    args = parser.parse_args()
    session = TrainingSession(
        load_config(args.config), args.resume, args.log_unique_maps, args.log_cumulative_unique_maps,
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(session.run_dir / "train.log", encoding="utf-8"),
        ],
    )
    logging.info("Run directory: %s", session.run_dir)
    logging.info("Config:\n%s", json.dumps(session.config, indent=2))
    signal.signal(signal.SIGINT, session.request_stop)
    signal.signal(signal.SIGTERM, session.request_stop)
    session.run()


if __name__ == "__main__":
    main()
