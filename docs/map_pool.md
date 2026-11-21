# Map pool format

`Z3MAPPL1` stores layouts of the 40 source rooms on an 8×8 board. All integers are unsigned and little-endian. The file is uncompressed, allowing a reader to seek directly to an individual map without Python or a decompression library.

| Offset | Bytes | Meaning |
| --- | --- | --- |
| 0 | 8 | ASCII magic/version `Z3MAPPL1` |
| 8 | 4 | Map count (`u32`) |
| 12 | 32 | SHA-256 of the source room-catalog JSON bytes |
| 44 | 80 | 40 source room IDs (`u16` each), in record order |
| 124 | 40 × map count | Map records |

Each record contains one byte per room in header order, giving its top-left anchor as `y * 8 + x`. Decode with `x = position % 8` and `y = position / 8` using integer division. Coordinates range from 0 to 7, with x increasing east and y increasing south. Room sizes and supported edges come from the source catalog; the records do not store placement order, themes, or edge-variant choices. Source room IDs are authored identities, not vanilla map IDs or board positions.

Map `i` (zero-based) starts at byte `124 + 40 * i`. The total file size is `124 + 40 * map_count`; 100,000 maps occupy 4,000,124 bytes. A Rust reader needs only byte slices and `u16::from_le_bytes` / `u32::from_le_bytes`. Future changes to board dimensions, room count, or record encoding require a new magic/version.

# Exporting experience

Run from the project root with the `map-gen` conda environment active:

```sh
python scripts/export_map_pool.py runs/20261104-051118-246400/experience runs/20261104-051118-246400/map_pool.bin --count 100000 --end-round 500
```

The exporter reads `round_*.safetensors` in descending numeric round order and requires matching catalog hashes and ordered room IDs across the files it reads. With `--end-round`, only rounds at or below that number are eligible. If omitted, collection starts at the newest available round. Only episodes for which every stored outcome flag is true are eligible, matching training's definition of success. Layouts are deduplicated by room anchors regardless of placement order. Collection stops as soon as the requested number of distinct eligible layouts is reached. Within each round, layouts are sorted by their encoded bytes for reproducible selection. The pool favors recent experience rather than sampling uniformly across the run.

Input tensors are read one file at a time, while the collected layouts stay in memory for deduplication. Memory is bounded by the requested pool size plus one input file's tensors and layouts. Use a completed, stable input directory; empty or damaged inputs encountered during collection cause an error rather than being silently skipped. Fewer eligible layouts than requested also cause an error. The exporter writes only after collecting the requested number of maps. Pass `--overwrite` to replace an existing output file with the completed pool.

The exporter prints placement balance (`bal`), edge-pair balance (`edge`), and placement target imbalance (`target`) over the exported layouts. To print the same stats for an existing pool, run:

```sh
python scripts/compute_pool_stats.py path/to/map_pool.bin
```

Both commands default to `data/rooms.json` from the project. For a pool or experience from a different catalog, pass `--room-data path/to/rooms.json` with the original catalog JSON matching the stored hash.
