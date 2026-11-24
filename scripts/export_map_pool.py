#!/usr/bin/env python3
"""Collect recent distinct layouts from a training run's successful-map batches."""

import argparse
from bisect import bisect_left
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import torch
from safetensors import safe_open

from compute_pool_stats import DEFAULT_ROOM_DATA, POOL_HEADER, format_layout_stats, load_pool_catalog, validate_pool_anchors

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from checkpoint import round_files
from experience import edge_partner_counts, facing_segments
from placement import PlacementController
from z3_map_gen import BOARD_SIZE, EPISODE_LENGTH


def layout_edge_indices(anchors, room_sizes):
    contacts_per_map = sum(width + height for width, height in room_sizes) - 2 * BOARD_SIZE
    indices = torch.empty((len(anchors), contacts_per_map), dtype=torch.int32)
    for start in range(0, len(anchors), 4096):
        chunk = anchors[start:start + 4096]
        rooms = torch.arange(len(room_sizes)).expand_as(chunk)
        actions = torch.stack((rooms, chunk % BOARD_SIZE, chunk // BOARD_SIZE), dim=-1)
        first, second, segment_count = facing_segments(actions, room_sizes)
        paired = (first >= 0) & (second >= 0)
        if not (paired.sum(dim=1) == contacts_per_map).all():
            raise ValueError("Edge balancing requires complete room layouts")
        indices[start:start + len(chunk)] = (
            first[paired] * segment_count + second[paired]
        ).reshape_as(indices[start:start + len(chunk)])
    return indices


def edge_balance_terms(counts, inverse_partners):
    matrix = counts.reshape(inverse_partners.numel(), -1)
    endpoints = matrix.sum(dim=0) + matrix.sum(dim=1)
    expected = (endpoints * (endpoints - 1)).dot(inverse_partners)
    observed = 2 * counts.dot(counts - 1)
    return observed / expected, expected, endpoints


def select_balanced_layouts(anchors, count, room_sizes, edge_weight=0.0):
    target = PlacementController(room_sizes, "cpu").target.flatten()
    denominator = target.masked_fill(target == 0, float("inf"))
    inverse_target = denominator.double().reciprocal()
    indices = torch.arange(EPISODE_LENGTH) * BOARD_SIZE**2 + anchors
    baseline_counts = torch.bincount(indices[:count].flatten(), minlength=target.numel())
    expected_counts = baseline_counts.double()
    placement_normalizer = count * (count - 1) * len(room_sizes)
    inverse_partners = edge_partner_counts(room_sizes).reciprocal()
    edge_bins = inverse_partners.numel()**2
    if edge_weight:
        scale = max(1.0, edge_weight)
        inverse_target /= placement_normalizer
        inverse_target /= scale
        edge_weight /= scale
        edge_indices = layout_edge_indices(anchors, room_sizes)
        baseline_edges = torch.bincount(edge_indices[:count].flatten(), minlength=edge_bins).double()
        expected_edges = baseline_edges
    probabilities = torch.zeros(len(anchors), dtype=torch.float64)
    probabilities[:count] = 1

    # Frank-Wolfe uses the lowest-scoring subset as the next search direction.
    # shortcut: 50 iterations cap export cost, revisit if larger pools still improve substantially at the cap.
    for _ in range(50):
        gradient = (2 * expected_counts - 1) * inverse_target
        scores = gradient[indices].sum(dim=1)
        if edge_weight:
            balance, expected, endpoints = edge_balance_terms(expected_edges, inverse_partners)
            endpoint_gradient = (2 * endpoints - 1) * inverse_partners
            denominator_gradient = (endpoint_gradient[:, None] + endpoint_gradient).flatten()
            edge_gradient = (2 * (2 * expected_edges - 1) - balance * denominator_gradient) / expected
            scores += edge_weight * edge_gradient[edge_indices].sum(dim=1)
        chosen = scores.topk(count, largest=False, sorted=False).indices
        chosen_counts = torch.bincount(indices[chosen].flatten(), minlength=target.numel())
        change = chosen_counts - expected_counts
        slope = gradient.dot(change).item()
        curvature = change.square().dot(inverse_target).item()
        if edge_weight:
            chosen_edges = torch.bincount(edge_indices[chosen].flatten(), minlength=edge_bins)
            edge_change = chosen_edges - expected_edges
            total_slope = slope + edge_weight * edge_gradient.dot(edge_change).item()
        else:
            total_slope = slope
        if total_slope >= 0:
            break
        if edge_weight:
            # The edge metric is a ratio of quadratics, so check the actual objective along the step.
            step = 1.0
            for _ in range(24):
                next_edges = expected_edges + step * edge_change
                next_balance, _, _ = edge_balance_terms(next_edges, inverse_partners)
                objective_change = step * (slope + step * curvature) + edge_weight * (next_balance - balance).item()
                if objective_change <= 1e-4 * step * total_slope:
                    break
                step *= 0.5
            else:
                break
            expected_edges = next_edges
        else:
            step = min(1.0, -slope / (2 * curvature))
        expected_counts.add_(change, alpha=step)
        probabilities.mul_(1 - step)
        probabilities[chosen] += step

    if not edge_weight:
        expected_edges = torch.zeros(edge_bins, dtype=torch.float64)
        for chunk, weights in zip(anchors.split(4096), probabilities.split(4096)):
            contacts = layout_edge_indices(chunk, room_sizes)
            expected_edges += torch.bincount(
                contacts.flatten(), weights=weights.repeat_interleave(contacts.size(1)), minlength=edge_bins,
            )
    estimated_edge = edge_balance_terms(expected_edges, inverse_partners)[0]
    estimated_target = (expected_counts * (expected_counts - 1) / denominator).sum() / placement_normalizer
    print(f"Before sampling (from expected counts): edge={estimated_edge:.3f} target={estimated_target:.4f}", flush=True)

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
        if edge_weight:
            rounded_edges = torch.bincount(edge_indices[selected].flatten(), minlength=edge_bins).double()
            rounded_score = rounded_score / placement_normalizer / scale + edge_weight * edge_balance_terms(
                rounded_edges, inverse_partners,
            )[0]
            baseline_score = baseline_score / placement_normalizer / scale + edge_weight * edge_balance_terms(
                baseline_edges, inverse_partners,
            )[0]
        if rounded_score < baseline_score:
            return selected.tolist()
    return range(count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Training run directory containing checkpoints and successful_maps")
    parser.add_argument("--count", type=int, default=100_000, help="Number of maps (default: 100000)")
    parser.add_argument("--balance", action="store_true", help="Select a pool with lower weighted imbalance")
    parser.add_argument("--candidate-count", type=int, help="Maximum distinct layouts to consider with --balance")
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
            print(
                f"{files_read}/{start_index + 1} files: {original_count} successful maps before deduplication, "
                f"{len(seen)} distinct successful maps across scanned rounds",
                flush=True,
            )
        if len(sample) == candidate_count:
            break

    if len(sample) < args.count:
        raise ValueError(f"Requested {args.count} maps, found only {len(sample)} distinct successful maps")
    catalog = load_pool_catalog(args.room_data, catalog_hash, room_ids)
    room_sizes = catalog.room_sizes
    anchors = torch.frombuffer(bytearray(b"".join(sample)), dtype=torch.uint8).reshape(-1, EPISODE_LENGTH)
    validate_pool_anchors(anchors, room_sizes)
    if 1 < args.count < len(sample):
        selected = select_balanced_layouts(anchors, args.count, room_sizes, args.edge_weight or 0.0)
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
