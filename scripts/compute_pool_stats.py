#!/usr/bin/env python3
"""Compute placement and edge balance for a Z3MAPPL1 map pool."""

import argparse
import hashlib
from pathlib import Path
import struct
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from experience import edge_pair_balance, placement_balance
from placement import PlacementController
from z3_map_gen import BOARD_SIZE, EPISODE_LENGTH, Catalog


POOL_HEADER = struct.Struct("<8sI32s40H")
DEFAULT_ROOM_DATA = Path(__file__).resolve().parents[1] / "data" / "rooms.json"


def format_pool_stats(pool, room_data):
    if BOARD_SIZE != 8 or EPISODE_LENGTH != 40:
        raise ValueError("Z3MAPPL1 requires an 8x8 board with 40 rooms")
    if len(pool) < POOL_HEADER.size:
        raise ValueError("Truncated map pool header")
    magic, count, catalog_hash, *room_ids = POOL_HEADER.unpack_from(pool)
    if magic != b"Z3MAPPL1" or not count or len(pool) != POOL_HEADER.size + count * EPISODE_LENGTH:
        raise ValueError("Invalid Z3MAPPL1 magic, map count, or file size")
    if hashlib.sha256(room_data.read_bytes()).digest() != catalog_hash:
        raise ValueError(f"Catalog hash does not match the pool. Pass the source catalog with --room-data: {room_data}")
    catalog = Catalog(room_data)
    if catalog.room_ids != room_ids:
        raise ValueError("Catalog room IDs do not match the pool")

    anchors = torch.frombuffer(pool, dtype=torch.uint8, offset=POOL_HEADER.size).reshape(count, EPISODE_LENGTH)
    x, y = anchors % BOARD_SIZE, anchors // BOARD_SIZE
    room_sizes = catalog.room_sizes
    widths, heights = torch.tensor(room_sizes).T
    if ((x + widths > BOARD_SIZE) | (y + heights > BOARD_SIZE)).any():
        raise ValueError("Map pool contains a room placement outside the board")
    rooms = torch.arange(EPISODE_LENGTH).expand_as(anchors)
    actions = torch.stack((rooms, x, y), dim=-1)

    bal = placement_balance(actions, room_sizes)
    edge = edge_pair_balance(actions, room_sizes)
    controller = PlacementController(room_sizes, "cpu", gain=0, leak=1)
    target = controller.update(actions)["placement_target_imbalance"]
    return f"bal={bal:.2f} edge={edge:.2f} target={target:.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pool", type=Path, help="Map pool .bin file")
    parser.add_argument("--room-data", type=Path, default=DEFAULT_ROOM_DATA, help="Source room catalog JSON")
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(format_pool_stats(bytearray(args.pool.read_bytes()), args.room_data))


if __name__ == "__main__":
    main()
