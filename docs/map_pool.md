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

# Successful-map batches

Training writes a safetensors file under `successful_maps/` for each checkpoint that advances the run. The file name `round_<number>.safetensors` gives the last round in the batch. Its `anchors` tensor contains one 40-byte row per successful episode, in round order, using the same room-anchor encoding as the map pool. The `offsets` tensor starts at zero and gives the cumulative row count after each round. The metadata contains the catalog SHA-256 and ordered room IDs. An episode is successful when every outcome flag is true.

Batches are published before their checkpoints. The latest published checkpoint bounds the committed batches, and checkpoint retention preserves the map batches. Resuming an older checkpoint creates a sibling run with the map batches through that checkpoint.

# Exporting a map pool

Run from the project root with the `map-gen` conda environment active:

```sh
python scripts/export_map_pool.py runs/20261104051118 --count 100000 --end-round 500
```

The exporter writes `<run>/exports/map-pool-<run-name>-<ending-round>.bin`. The ending round is the latest committed round, capped by `--end-round` when supplied.

The exporter reads successful-map rounds newest first and requires matching catalog hashes and ordered room IDs across the files it reads. Within each round, layouts are sorted by their encoded bytes for reproducible selection. By default, collection stops as soon as the requested number of distinct layouts is reached.

Input tensors are read one batch at a time. By default, memory is bounded by the requested pool size, one batch's tensors, and one round's decoded layouts. Use a completed, stable run directory. The exporter writes the pool after collecting the requested number of maps. Pass `--overwrite` to replace an existing output file with the completed pool.

Use `--balance --candidate-count 300000` to consider up to 300,000 distinct layouts for the export. Set `--candidate-count` at least as high as `--count`. If fewer candidates are available, the exporter can still proceed as long as it can fill the pool. Memory and optimization time scale with the number of candidates.

With `--balance`, use `--edge-weight 1` to give a one-point change in edge imbalance the same weight as a one-point change in placement target imbalance. The default weight of zero balances room placements. The exporter minimizes the weighted sum of the reported `target` and `edge` metrics.

Balanced export optimizes layout inclusion probabilities, then uses randomized rounding to draw the requested number of distinct layouts. It keeps the newest-first pool if the draw fails to lower the weighted objective. The output path and overwrite rules are the same for both modes.

The exporter prints placement balance (`bal`), edge-pair balance (`edge`), and placement target imbalance (`target`) over the exported layouts. To print the same stats for an existing pool, run:

```sh
python scripts/compute_pool_stats.py path/to/map_pool.bin
```

Both commands default to `data/rooms.json` from the project. For a pool or run from a different catalog, pass `--room-data path/to/rooms.json` with the original catalog JSON matching the stored hash.
