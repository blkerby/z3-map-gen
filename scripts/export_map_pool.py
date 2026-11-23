#!/usr/bin/env python3
"""Collect recent distinct layouts from a training run's successful-map batches."""

import argparse
from bisect import bisect_left
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import torch
from safetensors import safe_open

from compute_pool_stats import DEFAULT_ROOM_DATA, POOL_HEADER, format_layout_stats, load_pool_catalog, validate_pool_anchors

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from checkpoint import round_files
from placement import PlacementController
from z3_map_gen import BOARD_SIZE, EPISODE_LENGTH


def select_balanced_layouts(anchors, count, room_sizes):
    target = PlacementController(room_sizes, "cpu").target.flatten()
    denominator = target.masked_fill(target == 0, float("inf"))
    inverse_target = denominator.double().reciprocal()
    indices = torch.arange(EPISODE_LENGTH) * BOARD_SIZE**2 + anchors
    baseline_counts = torch.bincount(indices[:count].flatten(), minlength=target.numel())
    expected_counts = baseline_counts.double()
    probabilities = torch.zeros(len(anchors), dtype=torch.float64)
    probabilities[:count] = 1

    # Frank-Wolfe uses the lowest-scoring subset as the next search direction.
    # shortcut: 50 iterations cap export cost, revisit if larger pools still improve substantially at the cap.
    for _ in range(50):
        gradient = (2 * expected_counts - 1) * inverse_target
        scores = gradient[indices].sum(dim=1)
        chosen = scores.topk(count, largest=False, sorted=False).indices
        chosen_counts = torch.bincount(indices[chosen].flatten(), minlength=target.numel())
        change = chosen_counts - expected_counts
        improvement = -gradient.dot(change).item()
        if improvement <= 0:
            break
        curvature = 2 * change.square().dot(inverse_target).item()
        step = min(1.0, improvement / curvature)
        expected_counts.add_(change, alpha=step)
        probabilities.mul_(1 - step)
        probabilities[chosen] += step

    # A random shift of the integer grid hits each interval with probability equal to its length.
    order = torch.randperm(len(anchors))
    cumulative = probabilities[order].cumsum(dim=0)
    thresholds = (cumulative + torch.rand((), dtype=torch.float64)).floor()
    thresholds[-1] = count  # Pin the total against floating-point drift.
    selected = order[thresholds.diff(prepend=thresholds.new_zeros(1)) > 0]
    if len(selected) == count:
        rounded_counts = torch.bincount(indices[selected].flatten(), minlength=target.numel())
        rounded_score = (rounded_counts * (rounded_counts - 1) / denominator).sum()
        baseline_score = (baseline_counts * (baseline_counts - 1) / denominator).sum()
        if rounded_score < baseline_score:
            return selected.tolist()
    return range(count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Training run directory containing checkpoints and successful_maps")
    parser.add_argument("--count", type=int, default=100_000, help="Number of maps (default: 100000)")
    parser.add_argument("--balance", action="store_true", help="Sample a pool with lower placement target imbalance")
    parser.add_argument("--candidate-count", type=int, help="Maximum distinct layouts to consider with --balance")
    parser.add_argument("--end-round", type=int, help="Newest eligible round (inclusive)")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    parser.add_argument("--room-data", type=Path, default=DEFAULT_ROOM_DATA, help="Source room catalog JSON")
    args = parser.parse_args()
    if not 1 <= args.count <= 0xFFFFFFFF:
        parser.error("count must fit a positive u32")
    if args.balance and (args.candidate_count is None or args.candidate_count < args.count):
        parser.error("--balance requires --candidate-count at least as large as --count")
    if not args.balance and args.candidate_count is not None:
        parser.error("--candidate-count requires --balance")
    candidate_count = args.candidate_count if args.balance else args.count
    latest_round = max(round_files(args.run / "checkpoints"), default=0)
    batches = round_files(args.run / "successful_maps")
    if latest_round and latest_round not in batches:
        parser.error(f"checkpoint {latest_round} is missing its successful-map batch")
    ending_round = min(latest_round, args.end_round) if args.end_round is not None else latest_round
    if ending_round <= 0:
        parser.error("no eligible successful-map batches found")
    rounds = sorted(number for number in batches if 0 < number <= latest_round)
    start_index = bisect_left(rounds, ending_round)
    run_name = args.run.resolve().name
    output_path = args.run / "exports" / f"map-pool-{run_name}-{ending_round}.bin"
    if not args.overwrite and output_path.exists():
        parser.error(f"output already exists: {output_path} (use --overwrite to replace it)")

    torch.set_num_threads(1)
    seen = set()
    sample = []
    catalog_hash = None
    room_ids = None
    for index in range(start_index, -1, -1):
        number = rounds[index]
        previous_round = rounds[index - 1] if index else 0
        path = batches[number]
        with safe_open(path, framework="pt", device="cpu") as file:
            metadata = file.metadata()
            file_hash = bytes.fromhex(metadata["catalog_hash"])
            file_rooms = json.loads(metadata["room_ids"])
            if catalog_hash is None:
                catalog_hash = file_hash
                room_ids = file_rooms
                if len(catalog_hash) != 32 or len(room_ids) != 40:
                    raise ValueError(f"Expected a SHA-256 hash and 40 room IDs: {path}")
            elif file_hash != catalog_hash or file_rooms != room_ids:
                raise ValueError(f"Successful-map catalog does not match: {path}")
            anchors = file.get_tensor("anchors")
            offset_tensor = file.get_tensor("offsets")
        if offset_tensor.dtype != torch.int64 or offset_tensor.ndim != 1:
            raise ValueError(f"Invalid successful-map offsets: {path}")
        offsets = offset_tensor.tolist()
        if (
            anchors.dtype != torch.uint8 or anchors.ndim != 2 or anchors.shape[1] != 40
            or len(offsets) != number - previous_round + 1
            or offsets[0] != 0 or offsets[-1] != len(anchors)
            or any(start > end for start, end in zip(offsets, offsets[1:]))
        ):
            raise ValueError(f"Invalid successful-map batch: {path}")

        last_round = min(ending_round, number) - previous_round
        for offset in range(last_round, 0, -1):
            layouts = {bytes(row) for row in anchors[offsets[offset - 1]:offsets[offset]].tolist()}
            for layout in sorted(layouts):
                if layout in seen:
                    continue
                seen.add(layout)
                sample.append(layout)
                if len(sample) == candidate_count:
                    break
            if len(sample) == candidate_count:
                break
        files_read = start_index - index + 1
        if files_read % 1000 == 0 or index == 0 or len(sample) == candidate_count:
            print(f"{files_read}/{start_index + 1} files: {len(seen)} distinct successful maps", flush=True)
        if len(sample) == candidate_count:
            break

    if len(sample) < args.count:
        raise ValueError(f"Requested {args.count} maps, found only {len(sample)} distinct successful maps")
    catalog = load_pool_catalog(args.room_data, catalog_hash, room_ids)
    room_sizes = catalog.room_sizes
    anchors = torch.frombuffer(bytearray(b"".join(sample)), dtype=torch.uint8).reshape(-1, EPISODE_LENGTH)
    validate_pool_anchors(anchors, room_sizes)
    if 1 < args.count < len(sample):
        selected = select_balanced_layouts(anchors, args.count, room_sizes)
        sample = [sample[index] for index in selected]
        anchors = anchors[selected]
    else:
        sample = sample[:args.count]
        anchors = anchors[:args.count]
    pool = bytearray(POOL_HEADER.pack(b"Z3MAPPL1", len(sample), catalog_hash, *room_ids))
    pool.extend(b"".join(sample))
    stats = format_layout_stats(anchors, room_sizes)
    output_path.parent.mkdir(exist_ok=True)
    if args.overwrite:
        with TemporaryDirectory(dir=output_path.parent) as directory:
            staged = Path(directory) / output_path.name
            staged.write_bytes(pool)
            staged.replace(output_path)
    else:
        with output_path.open("xb") as output:
            output.write(pool)
    print(f"Wrote {len(sample)} maps ({output_path.stat().st_size} bytes) to {output_path}")
    print(stats)


if __name__ == "__main__":
    main()
