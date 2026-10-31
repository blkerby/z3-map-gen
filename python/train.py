import argparse
import hashlib
import json
import logging
import math
import os
import random
import signal
from pathlib import Path

import torch
from torch.nn import functional as F

from checkpoint import load_checkpoint, save_checkpoint, save_tensors
from experience import ExperienceRound
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
    for group, key in (("generation", "temperature"), ("optimizer", "learning_rate")):
        value = config[group][key]
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{group}.{key} must be positive and finite")
    floor = config["generation"]["log_success_floor"]
    if not math.isfinite(floor) or floor >= 0:
        raise ValueError("generation.log_success_floor must be finite and negative")
    for key in ("room_data", "run_dir"):
        config[key] = str(Path(config[key]).resolve())
    return config


class TrainingSession:
    def __init__(self, config, resume):
        self.config = config
        self.run_dir = Path(config["run_dir"])
        self.catalog = Catalog(config["room_data"])
        self.catalog_hash = hashlib.sha256(Path(config["room_data"]).read_bytes()).hexdigest()
        self.device = torch.device(config["device"])
        random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        self.model = OutcomeModel(self.catalog, **config["model"]).to(self.device)
        settings = config["optimizer"]
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
            self.completed_round = load_checkpoint(
                resume, self.model, self.optimizer, config, self.catalog_hash,
            )
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
        action_batches = []
        outcome_batches = []
        for _ in range(settings["batches_per_round"]):
            environments = EnvironmentGroup(self.catalog, settings["environment_count"])
            steps = []
            for _ in range(EPISODE_LENGTH):
                actions = sample_actions(
                    environments, self.model,
                    temperature=settings["temperature"],
                    log_success_floor=settings["log_success_floor"],
                )
                environments.apply_actions(actions)
                steps.append([action if action is not None else (-1, -1, -1) for action in actions])
            action_batches.append(torch.tensor(steps, dtype=torch.long).permute(1, 0, 2))
            features = environments.extract_features()
            outcomes = torch.cat(
                (torch.tensor(features.room_positions) >= 0, torch.tensor(features.room_cells) >= 0),
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
        supervised_entries = 0
        for _ in range(settings["batches_per_round"]):
            episodes = []
            labels = []
            for _ in range(settings["batch_size"]):
                number = random.randrange(first_round, round_number + 1)
                actions, outcomes = self.experience[number].sample_state()
                episodes.append(actions)
                labels.append(outcomes)
            features = EnvironmentGroup.replay(self.catalog, episodes).extract_features()
            room_cells = torch.tensor(features.room_cells, dtype=torch.long, device=self.device)
            room_positions = torch.tensor(features.room_positions, dtype=torch.long, device=self.device)
            targets = torch.stack(labels).to(device=self.device, dtype=torch.float32)
            unresolved = torch.cat((room_positions < 0, room_cells < 0), dim=1)
            logits = self.model(room_cells, room_positions)
            losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            loss = losses.masked_fill(~unresolved, 0).sum()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            supervised_entries += unresolved.sum().item()
        return {
            "mean_batch_loss": total_loss / settings["batches_per_round"],
            "loss_per_entry": total_loss / supervised_entries if supervised_entries else 0.0,
        }

    def save_checkpoint(self):
        path = self.run_dir / "checkpoints" / f"round_{self.completed_round}.safetensors"
        save_checkpoint(
            path, self.model, self.optimizer, self.completed_round, self.config,
            self.catalog_hash, self.catalog.room_ids,
        )
        logging.info("Saved checkpoint: %s", path)

    def run(self):
        last_saved_round = None
        for number in range(self.completed_round + 1, self.config["rounds"] + 1):
            if self.stop_requested:
                break
            logging.info("Generating round %s", number)
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
            logging.info("Training round %s", number)
            metrics = self.train_round(number)
            room_count = len(self.catalog.room_ids)
            outcomes = data["outcomes"].float()
            metrics.update({
                "round": number,
                "complete_map_frequency": outcomes[:, :room_count].all(dim=1).float().mean().item(),
                "mean_rooms_placed": outcomes[:, :room_count].sum(dim=1).mean().item(),
                "mean_occupied_cells": outcomes[:, room_count:].sum(dim=1).mean().item(),
                "room_placement_frequency": outcomes[:, :room_count].mean(dim=0).tolist(),
                "cell_fill_frequency": outcomes[:, room_count:].mean(dim=0).tolist(),
            })
            (self.run_dir / "metrics" / f"round_{number}.json").write_text(json.dumps(metrics) + "\n")
            self.completed_round = number
            logging.info(
                "Round %s: complete=%.3f rooms=%.2f cells=%.2f loss=%.4f", number,
                metrics["complete_map_frequency"], metrics["mean_rooms_placed"],
                metrics["mean_occupied_cells"], metrics["mean_batch_loss"],
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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    session = TrainingSession(load_config(args.config), args.resume)
    signal.signal(signal.SIGINT, session.request_stop)
    signal.signal(signal.SIGTERM, session.request_stop)
    session.run()


if __name__ == "__main__":
    main()
