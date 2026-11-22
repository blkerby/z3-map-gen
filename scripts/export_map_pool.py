#!/usr/bin/env python3
"""Collect distinct successful layouts from newest experience into a map pool."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import torch
from safetensors import safe_open

from compute_pool_stats import DEFAULT_ROOM_DATA, POOL_HEADER, format_pool_stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from experience import collect_unique_maps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experience", type=Path, help="Directory containing round_*.safetensors")
    parser.add_argument("output", type=Path, help="Output map pool file or existing directory")
    parser.add_argument("--count", type=int, default=100_000, help="Number of maps (default: 100000)")
    parser.add_argument("--end-round", type=int, help="Newest eligible round (inclusive)")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    parser.add_argument("--room-data", type=Path, default=DEFAULT_ROOM_DATA, help="Source room catalog JSON")
    args = parser.parse_args()
    if not 1 <= args.count <= 0xFFFFFFFF:
        parser.error("count must fit a positive u32")
    rounds = sorted(
        ((int(path.stem.removeprefix("round_")), path) for path in args.experience.glob("round_*.safetensors")),
        reverse=True,
    )
    if args.end_round is not None:
        rounds = [(number, path) for number, path in rounds if number <= args.end_round]
    if not rounds:
        parser.error("no eligible experience files found")
    output_path = args.output
    if output_path.is_dir():
        run_name = args.experience.resolve().parent.name
        ending_round, _ = rounds[0]
        output_path = output_path / f"map-pool-{run_name}-{ending_round}.bin"
    if not args.overwrite and output_path.exists():
        parser.error(f"output already exists: {output_path} (use --overwrite to replace it)")

    torch.set_num_threads(1)
    seen = set()
    sample = []
    catalog_hash = None
    room_ids = None
    for index, (_, path) in enumerate(rounds, 1):
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
                raise ValueError(f"Experience catalog does not match: {path}")
            layouts, _ = collect_unique_maps(file.get_tensor("actions"), file.get_tensor("outcomes"))

        # Sort each batch so hash randomization cannot affect the pool.
        for layout in sorted(layouts):
            if layout in seen:
                continue
            seen.add(layout)
            sample.append(layout)
            if len(sample) == args.count:
                break
        if index % 1000 == 0 or index == len(rounds) or len(sample) == args.count:
            print(f"{index}/{len(rounds)} files: {len(seen)} distinct successful maps", flush=True)
        if len(sample) == args.count:
            break

    if len(sample) < args.count:
        raise ValueError(f"Requested {args.count} maps, found only {len(sample)} distinct successful maps")
    pool = bytearray(POOL_HEADER.pack(b"Z3MAPPL1", len(sample), catalog_hash, *room_ids))
    pool.extend(b"".join(sample))
    stats = format_pool_stats(pool, args.room_data)
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
