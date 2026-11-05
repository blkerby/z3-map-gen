#!/usr/bin/env python3
"""Collect distinct successful layouts from newest experience into a map pool."""

import argparse
import json
from pathlib import Path
import struct
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from experience import collect_unique_maps
from z3_map_gen import BOARD_SIZE, EPISODE_LENGTH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experience", type=Path, help="Directory containing round_*.safetensors")
    parser.add_argument("output", type=Path, help="New output file; existing files are not overwritten")
    parser.add_argument("--count", type=int, default=100_000, help="Number of maps (default: 100000)")
    args = parser.parse_args()
    if not 1 <= args.count <= 0xFFFFFFFF:
        parser.error("count must fit a positive u32")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if BOARD_SIZE != 8 or EPISODE_LENGTH != 40:
        parser.error("Z3MAPPL1 requires an 8x8 board with 40 rooms")

    paths = sorted(
        args.experience.glob("round_*.safetensors"),
        key=lambda path: int(path.stem.removeprefix("round_")),
        reverse=True,
    )
    if not paths:
        parser.error("no experience files found")

    torch.set_num_threads(1)
    seen = set()
    sample = []
    catalog_hash = None
    room_ids = None
    for index, path in enumerate(paths, 1):
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
        if index % 1000 == 0 or index == len(paths) or len(sample) == args.count:
            print(f"{index}/{len(paths)} files: {len(seen)} distinct successful maps", flush=True)
        if len(sample) == args.count:
            break

    if len(sample) < args.count:
        raise ValueError(f"Requested {args.count} maps, found only {len(sample)} distinct successful maps")
    header = struct.pack("<8sI32s40H", b"Z3MAPPL1", len(sample), catalog_hash, *room_ids)
    with args.output.open("xb") as output:
        output.write(header)
        for layout in sample:
            output.write(layout)
    print(f"Wrote {len(sample)} maps ({args.output.stat().st_size} bytes) to {args.output}")


if __name__ == "__main__":
    main()
