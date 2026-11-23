import argparse
import hashlib
import json
import logging
import math
import os
import random
import shutil
import signal
import sys
from bisect import bisect_right
from datetime import datetime
from itertools import accumulate
from pathlib import Path

import torch
from torch.nn import functional as F

from checkpoint import load_checkpoint, prune_checkpoints, round_files, save_checkpoint, save_tensors
from experience import ExperienceRound, edge_pair_balance, map_anchors, placement_balance
from generation import sample_actions
from model import OutcomeModel
from placement import PlacementController
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
        ("placement_controller", "gain"),
        ("placement_controller", "leak"),
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
            if key == "leak" and not 0 < value <= 1:
                raise ValueError("placement_controller.leak values must be in (0, 1]")
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
    def __init__(self, config, resume, log_unique_maps=False):
        self.config = config
        self.log_unique_maps = log_unique_maps
        self.catalog = Catalog(config["room_data"])
        self.catalog_hash = hashlib.sha256(Path(config["room_data"]).read_bytes()).hexdigest()
        self.device = torch.device(config["device"])
        self.placement_controller = PlacementController(self.catalog.room_sizes, self.device)
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
        self.pending_maps = []
        self.stop_requested = False
        run_name = datetime.now().strftime("%Y%m%d%H%M%S")
        if resume is None:
            self.run_dir = Path("runs") / run_name
            self.run_dir.mkdir(parents=True, exist_ok=False)
        else:
            resume = resume.resolve()
            self.completed_round, replay = load_checkpoint(
                resume, self.model, self.optimizer, config, self.catalog_hash,
                self.placement_controller.prices,
            )
            self.experience = {number: ExperienceRound(**data) for number, data in replay.items()}
            self.run_dir = resume.parent.parent
            checkpoints = round_files(self.run_dir / "checkpoints")
            if checkpoints.get(self.completed_round) != resume:
                raise ValueError(f"Expected a checkpoint in its run directory: {resume}")
            batches = round_files(self.run_dir / "successful_maps")
            if self.completed_round and self.completed_round not in batches:
                raise ValueError(f"Checkpoint is missing its successful-map batch: {resume}")
            if self.completed_round < max(checkpoints):
                self.run_dir = self.run_dir.with_name(f"{run_name}-from-{self.completed_round}")
                self.run_dir.mkdir()
                (self.run_dir / "checkpoints").mkdir()
                (self.run_dir / "successful_maps").mkdir()
                for number, path in batches.items():
                    if number <= self.completed_round:
                        shutil.copy2(path, self.run_dir / "successful_maps" / path.name)
                shutil.copy2(resume, self.run_dir / "checkpoints" / resume.name)
            else:
                for number, path in batches.items():
                    if number > self.completed_round:
                        path.unlink()
        for name in ("successful_maps", "checkpoints", "metrics"):
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
                    placement_prices=self.placement_controller.prices,
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
        replay_rounds = tuple(self.experience)
        total_loss = 0.0
        for _ in range(settings["batches_per_round"]):
            episodes = []
            labels = []
            anchors = []
            for _ in range(settings["batch_size"]):
                number = random.choice(replay_rounds)
                actions, outcomes, terminal_anchors = self.experience[number].sample_state()
                episodes.append(actions)
                labels.append(outcomes)
                anchors.append(terminal_anchors)
            features = EnvironmentGroup.replay(self.catalog, episodes).extract_features()
            room_positions = features.room_positions.to(self.device)
            room_cells = features.room_cells.to(self.device)
            anchor_logits, cell_logits = self.model(room_positions, room_cells)
            anchor_targets = torch.tensor(anchors, device=self.device)
            anchor_targets.masked_fill_(room_positions >= 0, -100)
            anchor_loss = F.cross_entropy(
                anchor_logits.flatten(0, 1), anchor_targets.flatten(), reduction="sum",
            )
            cell_targets = torch.stack(labels)[:, room_positions.shape[1]:].to(
                device=self.device, dtype=torch.float32,
            )
            cell_losses = F.binary_cross_entropy_with_logits(cell_logits, cell_targets, reduction="none")
            loss = anchor_loss + cell_losses.masked_fill(room_cells >= 0, 0).sum()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return {
            **optimizer_settings,
            "loss_per_prefix": total_loss / (settings["batches_per_round"] * settings["batch_size"]),
        }

    def save_checkpoint(self):
        filename = f"round_{self.completed_round}.safetensors"
        if self.pending_maps:
            save_tensors(
                self.run_dir / "successful_maps" / filename,
                {
                    "anchors": torch.cat(self.pending_maps),
                    "offsets": torch.tensor(list(accumulate(map(len, self.pending_maps), initial=0))),
                },
                {
                    "catalog_hash": self.catalog_hash,
                    "room_ids": json.dumps(self.catalog.room_ids),
                },
            )
        path = self.run_dir / "checkpoints" / filename
        save_checkpoint(
            path, self.model, self.optimizer, self.completed_round, self.config,
            self.catalog_hash, self.catalog.room_ids, self.experience, self.placement_controller.prices,
        )
        self.pending_maps.clear()
        prune_checkpoints(path.parent, self.completed_round, self.config["checkpoint_period"])
        logging.info("Saved checkpoint: %s", path)

    def run(self):
        last_saved_round = self.completed_round if self.completed_round else None
        torch.set_float32_matmul_precision('high')
        for number in range(self.completed_round + 1, self.config["rounds"] + 1):
            if self.stop_requested:
                break
            data = self.generate_round()
            self.experience[number] = ExperienceRound(**data)
            metrics = self.train_round(number)
            controller_settings = instantiate_schedules(
                self.config["placement_controller"], self.config.get("knot_episodes"),
                number * len(data["actions"]),
            )
            metrics.update(self.placement_controller.update(data["actions"], **controller_settings))
            room_count = len(self.catalog.room_ids)
            outcomes = data["outcomes"].float()
            valid_actions = data["actions"][data["outcomes"].all(dim=1)]
            anchors = map_anchors(valid_actions)
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
            if self.log_unique_maps:
                unique_maps, counts = anchors.unique(dim=0, return_counts=True)
                metrics["unique_maps"] = len(unique_maps)
                unique_log += f" uniq={len(unique_maps)}"
                valid_count = len(anchors)
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
            (self.run_dir / "metrics" / f"round_{number}.json").write_text(json.dumps(metrics) + "\n")
            self.pending_maps.append(anchors)
            self.completed_round = number
            logging.info(
                "Round %s: complete=%.3f rooms=%.2f cells=%.2f loss=%.4f lr=%.6g "
                "bal=%.2f (%.2f) edge=%.2f (%.2f) target=%.3f price=%.3f%s",
                number,
                metrics["complete_map_frequency"], metrics["mean_rooms_unplaced"],
                metrics["mean_unoccupied_cells"], metrics["loss_per_prefix"], metrics["learning_rate"],
                metrics["placement_balance"], metrics["placement_balance_valid"],
                metrics["edge_pair_balance"], metrics["edge_pair_balance_valid"],
                metrics["placement_target_imbalance"], metrics["placement_price_rms"], unique_log,
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
    parser.add_argument("--resume", type=Path, help="Resume a checkpoint and fork the run if a newer checkpoint exists")
    parser.add_argument(
        "--log-unique-maps", action="store_true",
        help="Log unique valid maps and estimated collision probability in each round",
    )
    args = parser.parse_args()
    session = TrainingSession(load_config(args.config), args.resume, args.log_unique_maps)
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
