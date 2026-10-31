#!/usr/bin/env python3
"""Export overworld footprints and supported edge variants for map generation."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to a z3-json-data checkout")
    parser.add_argument("output", type=Path, help="Output JSON path")
    args = parser.parse_args()

    rooms = []
    total_area = 0
    for path in (args.source / "rooms" / "overworld").glob("*.json"):
        source = json.loads(path.read_text(encoding="utf-8"))
        width, height = source["size"]
        width //= 2
        height //= 2
        edges = {}
        for direction in ("north", "south", "east", "west"):
            edges[direction] = [
                sorted(segment["supported"])
                for segment in source["edgeVariants"][direction]
            ]
        rooms.append({"id": source["id"], "size": [width, height], "edges": edges})
        total_area += width * height

    if total_area != 64:
        parser.error(f"exported room area must total 64 cells, got {total_area}")
    rooms.sort(key=lambda room: room["id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"rooms": rooms}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
