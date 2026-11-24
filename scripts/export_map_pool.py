#!/usr/bin/env python3
"""Collect recent distinct layouts from a training run's successful-map batches."""

import argparse
from bisect import bisect_left
import json
import logging
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import torch
from safetensors import safe_open

from compute_pool_stats import DEFAULT_ROOM_DATA, POOL_HEADER, format_layout_stats, load_pool_catalog, validate_pool_anchors

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from checkpoint import round_files
from edges import EdgePairs
from experience import edge_contact_indices
from placement import PlacementController
from z3_map_gen import BOARD_SIZE, EPISODE_LENGTH


logger = logging.getLogger(__name__)
DEFAULT_ITERATIONS = 100


def layout_edge_indices(anchors, room_sizes):
    contacts_per_map = sum(width + height for width, height in room_sizes) + 2 * BOARD_SIZE
    indices = torch.empty((len(anchors), contacts_per_map), dtype=torch.int32)
    for start in range(0, len(anchors), 4096):
        chunk = anchors[start:start + 4096]
        rooms = torch.arange(len(room_sizes)).expand_as(chunk)
        actions = torch.stack((rooms, chunk % BOARD_SIZE, chunk // BOARD_SIZE), dim=-1)
        indices[start:start + len(chunk)] = edge_contact_indices(actions, room_sizes)
    return indices


def select_balanced_layouts(anchors, count, room_sizes, edge_pairs, edge_weight=0.0, *, iterations=DEFAULT_ITERATIONS):
    target = PlacementController(room_sizes, "cpu").target.flatten()
    denominator = target.masked_fill(target == 0, float("inf"))
    inverse_target = denominator.double().reciprocal()
    indices = torch.arange(EPISODE_LENGTH) * BOARD_SIZE**2 + anchors
    baseline_counts = torch.bincount(indices[:count].flatten(), minlength=target.numel())
    expected_counts = baseline_counts.double()
    placement_normalizer = count * (count - 1) * len(room_sizes)
    scale = max(1.0, edge_weight)
    inverse_target /= placement_normalizer
    inverse_target /= scale
    edge_bins = edge_pairs.edge_bins
    edge_indices = layout_edge_indices(anchors, room_sizes)
    baseline_edges = torch.bincount(edge_indices[:count].flatten(), minlength=edge_bins).double()
    expected_edges = baseline_edges
    if edge_weight:
        edge_weight /= scale
        edge_coefficient = edge_weight / edge_pairs.collision_normalizer(count)
    probabilities = torch.zeros(len(anchors), dtype=torch.float64)
    probabilities[:count] = 1

    # Frank-Wolfe uses the lowest-scoring subset as the next search direction.
    for iteration in range(1, iterations + 1):
        gradient = (2 * expected_counts - 1) * inverse_target
        scores = gradient[indices].sum(dim=1)
        if edge_weight:
            edge_gradient = (2 * expected_edges - 1) * edge_coefficient
            scores += edge_gradient.index_select(0, edge_indices.flatten()).view_as(edge_indices).sum(dim=1)
        chosen = scores.topk(count, largest=False, sorted=False).indices
        chosen_counts = torch.bincount(indices[chosen].flatten(), minlength=target.numel())
        change = chosen_counts - expected_counts
        slope = gradient.dot(change).item()
        curvature = change.square().dot(inverse_target).item()
        chosen_edges = torch.bincount(edge_indices[chosen].flatten(), minlength=edge_bins)
        edge_change = chosen_edges - expected_edges
        if edge_weight:
            slope += edge_gradient.dot(edge_change).item()
            curvature += edge_coefficient * edge_change.square().sum().item()
        if slope < 0:
            step = min(1.0, -slope / (2 * curvature))
            expected_edges = expected_edges + step * edge_change
            expected_counts.add_(change, alpha=step)
            probabilities.mul_(1 - step)
            probabilities[chosen] += step
        estimated_edge = edge_pairs.balance(expected_edges, count)
        estimated_target = (expected_counts * (expected_counts - 1) / denominator).sum() / placement_normalizer
        logger.info(
            "Iteration %d (from expected counts): edge=%.3f target=%.4f",
            iteration, estimated_edge, estimated_target,
        )
        if slope >= 0:
            break

    # A random shift of the integer grid hits each interval with probability equal to its length.
    order = torch.randperm(len(anchors))
    cumulative = probabilities[order].cumsum(dim=0)
    thresholds = (cumulative + torch.rand((), dtype=torch.float64)).floor()
    thresholds[-1] = count  # Pin the total against floating-point drift.
    selected = order[thresholds.diff(prepend=thresholds.new_zeros(1)) > 0]
    if len(selected) == count:
        rounded_counts = torch.bincount(indices[selected].flatten(), minlength=target.numel())
        rounded_score = (rounded_counts.double() * (rounded_counts - 1)).dot(inverse_target)
        baseline_score = (baseline_counts.double() * (baseline_counts - 1)).dot(inverse_target)
        if edge_weight:
            rounded_edges = torch.bincount(edge_indices[selected].flatten(), minlength=edge_bins)
            rounded_score += edge_weight * edge_pairs.balance(rounded_edges, count)
            baseline_score += edge_weight * edge_pairs.balance(baseline_edges, count)
        if rounded_score < baseline_score:
            return selected.tolist()
    return range(count)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Training run directory containing checkpoints and successful_maps")
    parser.add_argument("--count", type=int, default=100_000, help="Number of maps (default: 100000)")
    parser.add_argument("--balance", action="store_true", help="Select a pool with lower weighted imbalance")
    parser.add_argument("--candidate-count", type=int, help="Maximum distinct layouts to consider with --balance")
    parser.add_argument(
        "--iterations", type=int, help=f"Maximum optimization iterations with --balance (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--edge-weight", type=float,
        help="Weight of edge imbalance relative to placement target imbalance (default: 0)",
    )
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
    if args.iterations is not None:
        if not args.balance:
            parser.error("--iterations requires --balance")
        if args.iterations < 1:
            parser.error("--iterations must be positive")
    if args.edge_weight is not None:
        if not args.balance:
            parser.error("--edge-weight requires --balance")
        if not math.isfinite(args.edge_weight) or args.edge_weight < 0:
            parser.error("--edge-weight must be finite and nonnegative")
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
    original_count = 0
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
            start, end = offsets[offset - 1], offsets[offset]
            original_count += end - start
            layouts = {bytes(row) for row in anchors[start:end].tolist()}
            layouts.difference_update(seen)
            seen.update(layouts)
            sample.extend(sorted(layouts)[:candidate_count - len(sample)])
            if len(sample) == candidate_count:
                break
        files_read = start_index - index + 1
        if files_read % 1000 == 0 or index == 0 or len(sample) == candidate_count:
            logger.info(
                "%d/%d files: %d successful maps before deduplication, %d distinct successful maps across scanned rounds",
                files_read, start_index + 1, original_count, len(seen),
            )
        if len(sample) == candidate_count:
            break

    if len(sample) < args.count:
        raise ValueError(f"Requested {args.count} maps, found only {len(sample)} distinct successful maps")
    catalog = load_pool_catalog(args.room_data, catalog_hash, room_ids)
    room_sizes = catalog.room_sizes
    edge_pairs = EdgePairs(catalog)
    anchors = torch.frombuffer(bytearray(b"".join(sample)), dtype=torch.uint8).reshape(-1, EPISODE_LENGTH)
    validate_pool_anchors(anchors, room_sizes)
    if 1 < args.count < len(sample):
        selected = select_balanced_layouts(
            anchors, args.count, room_sizes, edge_pairs, args.edge_weight or 0.0,
            iterations=args.iterations if args.iterations is not None else DEFAULT_ITERATIONS,
        )
        sample = [sample[index] for index in selected]
        anchors = anchors[selected]
    else:
        sample = sample[:args.count]
        anchors = anchors[:args.count]
    pool = bytearray(POOL_HEADER.pack(b"Z3MAPPL1", len(sample), catalog_hash, *room_ids))
    pool.extend(b"".join(sample))
    stats = format_layout_stats(anchors, room_sizes, edge_pairs)
    output_path.parent.mkdir(exist_ok=True)
    if args.overwrite:
        with TemporaryDirectory(dir=output_path.parent) as directory:
            staged = Path(directory) / output_path.name
            staged.write_bytes(pool)
            staged.replace(output_path)
    else:
        with output_path.open("xb") as output:
            output.write(pool)
    logger.info("Wrote %d maps (%d bytes) to %s", len(sample), output_path.stat().st_size, output_path)
    logger.info("%s", stats)


if __name__ == "__main__":
    main()
