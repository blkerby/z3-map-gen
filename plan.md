# Overworld Map Generation Plan

## Goal and scope

Create a standalone project that generates overworld layouts on an 8×8 grid. Place every overworld room exactly once, without overlap, while respecting neighboring edge compatibility and a closed outer boundary. Use a learned outcome predictor to score speculative placements, then sample a placement using a temperature-controlled softmax. Do not use backtracking.

Continue partially failed maps until no further room can be placed. These episodes still provide useful per-room and per-cell outcome labels for training. Complete placement is the initial success criterion; traversal connectivity and concrete edge-type selection are deferred.

## Verified source data

The current `../z3-map-rando/z3-json-data` contains 40 overworld rooms: 32 with source size `[2, 2]` and 8 with source size `[4, 4]`. Dividing each dimension by two gives grid footprints of 1×1 and 2×2 respectively. Their total area is `32 × 1 + 8 × 4 = 64`, exactly filling the board.

The vanilla positions cover all 64 cells without overlap. Every touching segment pair has a supported variant in common, and every outward-facing segment supports `closed`. This provides a known feasible layout for checking the environment without requiring the model to generate a successful map first.

Room IDs currently range from 1 through 40. Use explicit IDs from the data, not hexadecimal filename prefixes, to identify rooms. Light World and Dark World share each room's placement and supported variant choices; do not generate separate layouts for the two worlds.

## Minimal room-data export

Write a Python script that accepts the path to a `z3-json-data` checkout and an output JSON path. Read the overworld room records and produce one file with a `rooms` array sorted by room ID. Each room contains only:

- `id`: The source room ID.
- `size`: `[width, height]` in grid cells, already divided by two from the source screen dimensions.
- `edges`: An object with `north`, `south`, `east`, and `west`. Each value is an array of segments; each segment is an array of its supported variant names.

Preserve segment order: west to east on north/south edges and north to south on east/west edges. Sort variant names within each segment for deterministic output. Preserve the exact variant spellings from the source, including `horizonal-water-land`.

Do not export names, vanilla positions, defaults, variant-to-node arrays, node logic, or entrance data. The board size is fixed at 8×8 in this project. Check that the exported room area totals 64. Rely on the source project's schema and validator for the full room-data contract rather than duplicating them.

Implemented by `scripts/export_rooms.py` using only the Python standard library. From the project root, regenerate the current catalog with `python scripts/export_rooms.py ../z3-map-rando/z3-json-data data/rooms.json`.

## Environment and placement rules

Use Rust for environment state, candidate enumeration, speculative state features, and replay, exposed to Python through PyO3/maturin. Use Python/PyTorch for the model, scoring, sampling, and training. Take the division of responsibilities from `~/map-gen`, without copying its frontier machinery, balancing objectives, proposal networks, or generation pipeline.

Do not imitate `~/map-gen`'s `engine.rs` layer or its heavy use of `unsafe`. Implement the environment with ordinary owned Rust state and safe borrowing. Start with straightforward state copies for speculative placements on this small board, and return owned feature buffers through the Python bindings. Avoid a separate engine abstraction or custom unsafe memory handling; consider further optimization only if measurements justify it.

Represent each room in its original orientation. A placement specifies a room and the upper-left grid coordinate of its footprint. Large rooms may start at any coordinate where they fit; do not restrict them to even coordinates.

A real candidate is valid when all of the following hold:

1. The room has not already been placed.
2. Its entire footprint fits within the board and does not overlap any placed room.
3. Every segment facing outside the board supports `closed`.
4. Every segment touching a placed neighboring room shares at least one supported variant name with that neighboring segment.

Check the entire perimeter of a large room, including contacts with multiple neighboring rooms and both segments of a shared large-room edge. Match segment indices using their positions along the shared boundary. Matching uses exact variant-name intersection; defaults and node IDs do not participate. `closed` is a valid shared variant even though it supplies no passage.

An empty neighboring cell imposes no edge constraint. A skipped cell remains empty and must not be treated as a closed boundary. Borders between cells belonging to the same large room are internal to that room and have no segment-matching constraint.

Segment choices are independent according to the source-data contract, so checking nonempty intersections is enough for this placement phase. Do not select a concrete variant yet.

Implemented in `src/lib.rs` as a safe-Rust library. Load the export with `Catalog::load_json`, share it through `Arc`, and create an `Environment`. At startup, assign an integer index to every room edge segment and precompute a table indicating whether each segment pair shares a supported variant, plus a flag indicating whether each segment supports `closed`. Runtime edges store segment indices; generation uses table lookups and retains no variant strings. The JSON export still uses the original variant names. `can_place` checks geometry at any anchor using a catalog room index; `list_candidates` applies the row-major generation rule and returns actions carrying explicit source room IDs. `apply_action` trusts that the caller supplies an action valid for the current state and updates the board without re-enumerating or validating candidates. `replay` likewise trusts the recorded actions and reconstructs a real-action prefix or an episode with dummy padding, up to 40 slots. `get_cells` and `get_positions` expose read-only state in row-major and catalog order respectively. An empty candidate list indicates exhaustion. A dummy action is an unconditional no-op; the generation caller is responsible for using it only after exhaustion and for supplying the 40-slot episode loop.

## Generation loop and batching

For each environment, scan empty cells in row-major order: smallest `y` first, then smallest `x`. At each cell, enumerate valid unused rooms anchored there. Skip cells with no candidates and stop scanning at the first cell with candidates. Offer all valid candidates at that cell; do not offer placements at later cells while an earlier cell has a valid candidate.

Evaluate each candidate as a separate speculative placement from the same parent state. Extract its resulting state features without committing it or affecting the other candidates. Score these states with the outcome model, sample one candidate, and commit only that placement. Repeat until no empty cell has any valid candidate.

Do not stop because an earlier cell is permanently unfillable or full-map success is otherwise impossible. Continue placing other rooms to obtain more complete terminal labels. Do not restart or backtrack within an episode.

Use 40 action slots per episode, one for each possible real room placement. Once an environment is exhausted, fill its remaining slots with a dummy action that leaves its state unchanged. A completed map has 40 real placements; an exhausted incomplete map has a dummy tail. Dummy actions are padding, not alternatives to valid placements. Candidate-count padding is likewise masked out of sampling.

Organize generation settings under a `generation` group, following the configuration structure in `~/map-gen`. Make `generation.environment_count` a required positive integer hyperparameter controlling the number of environments advanced concurrently during generation, and store the sampling temperature as `generation.temperature`. Keep the environment count independent of the training batch size; batching environments does not require a worker thread per environment. `EnvironmentGroup` owns a vector of independent Rust environments sharing one catalog. One Python-to-Rust call extracts candidates for the whole group, one model forward pass scores all actual candidates, and one Python-to-Rust call applies the selected actions to the whole group. Pad only the per-environment score rows for batched sampling, masking padding with negative infinity and giving exhausted rows a single dummy slot.

## Model input features

Encode the existing room placements in two complementary ways, using separate learned embedding tables shared across all cells or rooms respectively:

1. For each of the 40 rooms, embed its anchor position using a table with 64 learned rows and embedding dimension `position_embedding_dim`. An unplaced room contributes an exact zero vector. Positions use row-major indices `8 * y + x`, including index 0 for the valid position `[0, 0]`.
2. For each of the 64 grid cells, embed the occupying room and the cell's relative position within that room using a table with embedding dimension `room_embedding_dim`. Each small room has one learned row; each large room has four distinct learned rows for its top-left, top-right, bottom-left, and bottom-right cells. These are specific to each room, not shared quadrant embeddings. The current catalog therefore requires `32 + 8 * 4 = 64` learned rows. An empty cell contributes an exact zero vector.

Both embedding dimensions are required configurable model hyperparameters. The zero vectors represent absence and are not additional learned rows. In particular, do not reserve a real room-cell or position index as padding. Assign room-cell indices in room-ID order, then row-major order within each room's footprint; derive this mapping from the exported sizes without adding export fields. Rust supplies room-cell indices per grid cell and anchor-position indices per room, with a separate sentinel such as `-1` for absence; Python performs the learned lookups and masks absent entries to zero.

Order rooms by the fixed room-ID ordering and cells by row-major position. Flatten both sets of embedding vectors and concatenate room-position embeddings first, then room-cell embeddings, as the input to a small MLP. This matches the room-first, cell-second outcome ordering. Its input width is `40 * position_embedding_dim + 64 * room_embedding_dim`. Do not pool over cells or rooms, since their slots identify absolute positions and room identities. Train both embedding tables jointly with the predictor and save their dimensions and weights in checkpoints.

Make the MLP's number of hidden layers (`model.hidden_layers`) and common hidden-layer width (`model.hidden_width`) required positive integer model hyperparameters. The hidden-layer count excludes the final 104-logit output layer. Save both values in checkpoints alongside the embedding dimensions so the architecture can be reconstructed.

These inputs describe the speculative post-placement state during candidate evaluation and the corresponding replayed state during training. They replace the earlier proposal for raw placement flags and coordinates; no extra occupancy or geometry feature channels are required initially. The fixed room catalog supplies room sizes and edge rules to the environment. The model learns against this catalog; it does not initially need to generalize across different room sets.

Implemented by `Environment::extract_features` and `Environment::extract_candidate_features`, with batched PyO3 bindings in `src/python.rs` and maturin configuration in `pyproject.toml`. Build and install into the existing conda environment with `conda run -n map-gen maturin develop`. The Python module is `z3_map_gen`: `Catalog(path)` loads the catalog, and `EnvironmentGroup(catalog, environment_count)` creates the group. Catalog properties `room_ids` and `room_cell_count` expose the room ordering and embedding row count. `extract_features()` returns owned integer matrices named `room_cells` and `room_positions`, with one row per environment and `-1` for absence. `extract_candidate_features()` returns all real candidates in a flat batch: `actions`, `features`, and `failed_outcomes` share candidate ordering. `features.room_cells` and `features.room_positions` each have one row per candidate; failure masks have 40 room entries followed by 64 cell entries. `environment_offsets` has one more entry than the number of environments and delimits each environment's candidate range; equal consecutive offsets indicate exhaustion. Python placement actions are `(room_id, x, y)` tuples, and `None` represents dummy padding. `apply_actions(actions)` takes one trusted action per environment, checking only the group length. `EnvironmentGroup.replay(catalog, episodes)` reconstructs one environment per recorded action sequence in a single call.

The learned model is implemented in `python/model.py` as `OutcomeModel(catalog, **config["model"])`. Its forward method accepts `(room_positions, room_cells)` index tensors and returns raw outcome logits. It masks absent embeddings to exact zero while retaining trainable index-zero rows, flattens the two embedding arrays separately, and concatenates them in that order before the configured MLP.

## Outcome predictor and candidate score

The model outputs 104 logits: one per room for the probability that the room will be present in the terminal map, and one per grid cell for the probability that the cell will be filled in the terminal map after continuing from state `s` under the generation policy. Order the 40 room outputs by room ID and the 64 cell outputs in row-major order. These are outcome predictions, not a distribution over the next room to place.

For candidate `a`, let `s_a` be the state after applying it. Already-placed rooms and filled cells, including those supplied by the candidate, have known success probability 1 and contribute zero to the score. A known-failed outcome contributes the finite negative `generation.log_success_floor`, initially `-50`. Clamp predicted log-success values at the same floor:

```text
floor = generation.log_success_floor
log_success_i(s_a) = 0                                  if outcome i is already successful
                    floor                              if outcome i is known to have failed
                    max(floor, logsigmoid(z_i(s_a)))    otherwise
score(a) = sum over all 40 room and 64 cell outcomes of log_success_i(s_a)
P(select a) = softmax(score(a) / temperature)
```

An unused room is known to have failed if it has no valid placement anywhere on the current board. An empty cell is known to have failed if no valid placement of any unused room can cover it. For a large room, covering a cell does not require anchoring at that cell. These checks consider all legal anchors, not only the current generation candidates. Subsequent placements only remove rooms and space or add constraints, so these failures are permanent. Do not mark every unresolved outcome as failed merely because full-map completion is impossible.

Use a positive temperature coefficient and sample from this distribution. Keep temperature fixed within an initial training run; it controls exploration and changes the continuation policy whose outcomes the model predicts. Compute log probabilities directly using `logsigmoid` on logits, rather than taking the logarithm of rounded sigmoid probabilities.

The score sums equally weighted, floored log-success values for room and cell outcomes. These outcomes are dependent, so this is a ranking surrogate, not a calibrated joint success probability. Accept the possibility that the surrogate can misrank candidates even with accurate marginal predictions; the experimental hope is that training improves the continuation policy and raises success probabilities toward 1. After full success becomes impossible, continue ranking candidates with this same score. Each known failure contributes `-50` rather than negative infinity, allowing candidates with fewer failed outcomes to remain preferable.

Implemented in `python/generation.py`: `score_candidates(model, candidates, log_success_floor)` returns a flat score tensor for the whole candidate batch, using one model forward pass on the model's device. `sample_actions(environments, model, temperature=..., log_success_floor=...)` extracts the group's speculative states in one call, scores them without gradients, and samples one action per environment using masked temperature-softmax rows and a batched multinomial draw. It returns a list of placement tuples or `None` for exhausted environments without committing them; pass that list to `environments.apply_actions` to advance the group. Rust computes known failures with `find_failed_outcomes`, checking all legal anchors and all cells covered by each valid footprint. To use these Python modules interactively from the project root, set `PYTHONPATH=python`; move the model to the configured device with `.to(config["device"])` before scoring.

## Training targets and experience

At the end of each episode, construct a binary vector `y` with 104 entries in model-output order: 40 room labels indicating whether each room was placed anywhere in the terminal map, followed by 64 cell labels indicating whether each cell was filled. Use this same terminal vector as the outcome target for every real visited post-placement state in that episode.

Train with binary cross-entropy with logits, independently over the room and cell outcomes. Already-placed rooms and filled cells have known success and need no learned prediction; mask them out of the learned loss. Exclude dummy padding so shorter failed episodes do not gain extra weight from repeated frozen states. Sum the loss over all supervised state-outcome entries in the training batch; do not take a mean. An empty set of supervised entries contributes zero loss. Apply the `-50` floor only to candidate scoring, not to the logits used for binary cross-entropy.

Make `train.batch_size` a required positive integer hyperparameter specifying the number of replayed post-placement states per training minibatch. Retain the training and generation hyperparameters with the run artifacts. Because the loss is summed, changing the training batch size also changes its scale.

Only the chosen trajectory receives these labels. Do not attach its terminal vector to unchosen speculative candidate states: those candidates were not continued, so their counterfactual outcomes are unknown. Scoring every candidate does not require producing training labels for every candidate.

Store experience in safetensors files, one file per generation round, following the storage approach in `~/map-gen`. Each file contains the ordered placement actions, real-action lengths distinguishing dummy padding, and terminal outcome vectors for all episodes collected in that round. Reconstruct visited states and their features by replaying the real actions through the same Rust environment. Retain the room ordering, catalog identity, temperature, and model configuration with training artifacts so outputs can be interpreted consistently. Keep the initial experience format simple and specific to this project.

Alternate generation under the current model with supervised updates from collected experience. Accumulate experience rounds indefinitely. Define a required positive integer hyperparameter `train.experience_rounds` giving the number of most recent collected rounds eligible for training, including the latest completed generation round. If fewer rounds exist, use all available rounds. Sample rounds uniformly within this window, then sample experience uniformly within the selected round. Retain older rounds even when they fall outside the sampling window. Historical labels reflect the policies that generated them; accept this mixture within the configured window rather than treating fixed temperature as making the continuation policy stationary. Full successes are not required to begin learning: failed episodes still contain positive and negative room and cell labels. Do not replace this objective with a reward based on room count or occupied area.

## Training configuration

Use one JSON config with `model`, `optimizer`, `generation`, and `train` groups, following the grouping in `~/map-gen` without importing its specialized options. The following is a proposed complete initial config; the numeric values are illustrative starting points, not tuned settings. Every field shown is required, with no silent defaults for missing config fields.

```json
{
  "room_data": "data/rooms.json",
  "device": "cpu",
  "seed": 0,
  "rounds": 100,
  "checkpoint_period": 10,
  "model": {
    "room_embedding_dim": 16,
    "position_embedding_dim": 8,
    "hidden_layers": 2,
    "hidden_width": 256
  },
  "optimizer": {
    "learning_rate": 0.0003,
    "beta1": 0.9,
    "beta2": 0.999
  },
  "generation": {
    "environment_count": 64,
    "batches_per_round": 4,
    "temperature": 1.0,
    "log_success_floor": -50.0
  },
  "train": {
    "batch_size": 256,
    "batches_per_round": 100,
    "experience_rounds": 10
  }
}
```

Top-level settings describe the run:

- `room_data`: Path to the exported room catalog. Resolve relative paths against the working directory from which training is launched.
- `device`: Single common PyTorch device used for both generation inference and training, such as `cpu` or `cuda:0`. Generation and training alternate on this device; do not introduce separate generation/training device settings or multi-device orchestration. The Rust environment runs on the CPU. Do not silently fall back to another device.
- `seed`: Seed for model initialization, placement sampling, and training-experience sampling. This controls randomness but does not promise identical results across devices.
- `rounds`: Number of generation-and-training rounds to execute.
- `checkpoint_period`: Required positive integer interval between periodic checkpoints, measured in completed generation-and-training rounds. The example saves every ten rounds.

Model and optimizer settings define a plain MLP and its updates. Each of the `model.hidden_layers` hidden layers has `model.hidden_width` units followed by ReLU; the output layer produces 104 raw logits. Use Adam with the configured learning rate and betas, fixed epsilon `1e-8`, and no weight decay. Keep optimizer state across rounds. Do not initially add activation or optimizer selection, dropout, mixed precision, or gradient accumulation.

`optimizer.learning_rate`, `optimizer.beta1`, and `optimizer.beta2` each accept a constant or a schedule, following `~/map-gen`. For example, set top-level `"knot_episodes": [0, 100000, 1000000]` and `"learning_rate": {"log": [0.0003, 0.0001, 0.00003]}` inside `optimizer`. A schedule must contain exactly one of `linear`, `log`, or `step`, with one value per knot. Knots are strictly increasing nonnegative episode counts; `knot_episodes` is required only when using a schedule. Linear schedules interpolate values, log schedules interpolate their logarithms, and step schedules switch values at each knot. Values outside the knot range hold the nearest endpoint. All values must be finite and nonnegative; beta values must also be less than one. Log schedules require strictly positive values. For example, `"beta1": {"linear": [0.9, 0.8, 0.7]}` uses the same episode knots as the learning-rate example.

Resolve all three optimizer parameters after generation, using the cumulative number of generated episodes including the current round: `round_number * generation.environment_count * generation.batches_per_round`. Apply them to all training minibatches in that round without resetting Adam state. Record all three values in the round metrics and the learning rate in the compact log. Resume derives the same episode count from the checkpoint's completed round and unchanged generation settings; the full schedule remains in the saved config.

Generation and training have separate batch controls:

- `generation.environment_count`: Number of concurrent episodes in one generation batch. Each episode has 40 action slots, including any dummy tail.
- `generation.batches_per_round`: Number of complete generation batches collected before training. A round contains exactly `environment_count * batches_per_round` episodes; the example collects 256 episodes. Hold the model weights fixed throughout this generation phase.
- `generation.temperature`: Positive sampling temperature, fixed throughout the run.
- `generation.log_success_floor`: Finite negative floor for candidate-score contributions, including known failures. This does not clamp training logits or the loss.
- `train.batch_size`: Number of real post-placement states sampled per training minibatch, not episodes or individual room/cell labels.
- `train.batches_per_round`: Number of sampled minibatches processed after each generation phase. Each minibatch supplies one summed masked BCE loss and one optimizer step; this is a fixed update budget, not a number of passes over the growing experience archive.
- `train.experience_rounds`: Number of most recent rounds eligible for sampling, including the newly collected round. For each sampled state, choose an eligible round uniformly, then choose uniformly among its real visited post-placement states. Sample with replacement so the configured batch size remains usable during startup. Dummy states never enter the sampling population.

With the example settings, each round collects 256 episodes and then performs 100 updates using 256 sampled states each. The first round samples only itself; once ten rounds exist, each eligible round has probability `1/10` for each sample. Retain every round's safetensors experience file on disk, including rounds outside this window.

Save checkpoints in safetensors format at `runs/<timestamp>/checkpoints/round_<completed_round>.safetensors`, following `~/map-gen`'s checkpoint approach. Save after each completed round whose number is divisible by `checkpoint_period`, and at normal completion if that round has not already been saved. Each checkpoint contains the model and optimizer state, completed-round count, config, catalog identity, and random-generator states. Store tensor state as named tensors and non-tensor state as safetensors string metadata, using JSON where needed. Write to a temporary file in the same directory, then atomically rename it to the final path. Keep experience in its separate per-round safetensors files.

Follow `~/map-gen`'s interruption behavior: the first SIGINT or SIGTERM requests a graceful stop, allowing the current generation-and-training round to finish and its experience to be stored before saving a checkpoint and exiting. Log that shutdown waits for the current round. A second signal exits immediately without checkpointing. Avoid writing the same completed-round checkpoint twice if periodic saving already handled it. Do not save a checkpoint from a failed partial round after an unexpected exception.

Write compact `HH:MM:SS` logs to stdout and `train.log` within the run directory, appending when resuming. Routine progress consists of one summary line per completed round with complete-map frequency, mean rooms and cells placed, and loss per sampled prefix. Log the run directory once at startup and retain the graceful-stop message; omit separate generation, training, and checkpoint messages.

Implemented in `python/train.py`, `python/experience.py`, and `python/checkpoint.py`, with the initial config in `configs/initial.json`. Start a run from the project root with `conda run --no-capture-output -n map-gen python python/train.py configs/initial.json`. Resume with the same command plus `--resume runs/<timestamp>/checkpoints/round_10.safetensors`, substituting the checkpoint to restore. A new run creates `runs/<timestamp>` relative to the launch directory, using the local start time formatted as `YYYYMMDD-HHMMSS-microseconds`, and logs that directory. The run directory is not a config setting. Resume derives the existing run directory from the checkpoint path and uses the same config, allowing the total `rounds` to change; it checks the catalog hash before restoring model, Adam, and random-generator state. Existing experience and metrics for later rounds are overwritten as those rounds are regenerated. Training reads only rounds through its current progress, so later files do not enter the sampling window early.

Each experience file contains `actions` with shape `[episodes, 40, 3]` and integer `(room_id, x, y)` entries, `lengths` with one real-action count per episode, and boolean `outcomes` with shape `[episodes, 104]`. Dummy action rows contain `[-1, -1, -1]`; only real post-placement states are eligible for replay sampling. The archive retains every round, while the in-memory cache contains only the recent sampling window. Each round also writes `metrics/round_<number>.json` with complete-map frequency, mean room and cell counts, individual room/cell frequencies, loss per sampled prefix (`loss_per_prefix`), computed by dividing the total summed loss by `train.batches_per_round * train.batch_size`. Checkpoints include both embedding tables, the MLP, Adam state, config, catalog hash, room ordering, completed-round count, and Python/PyTorch RNG states, including the configured CUDA device when used.

Pass `--log-unique-maps` to log the number of distinct valid layouts within each round, or `--log-cumulative-unique-maps` to log the number seen since the start of the run; the flags can be combined. Enabled counts appear in the round log and metrics JSON as `unique_maps` and `cumulative_unique_maps`. Only fully successful episodes count, and layouts are identified by their room positions independently of placement order. Cumulative tracking stores the exact set of layouts in checkpoints and resumes automatically without repeating the flag. Enabling it on a checkpoint without tracking reconstructs the set from experience rounds 1 through the checkpoint round, requiring those files to be available. Cumulative memory and checkpoint size grow with the number of distinct layouts; checkpoint layout data uses 40 bytes per map. Per-round logging must be requested again when resuming.

The 8×8 board, 40 action slots, catalog-derived embedding row counts, 104 outputs, equal outcome weights, and summed loss reduction are part of the design rather than configurable options.

## Implementation sequence

1. Implement the exporter and define its minimal JSON format. Verify the current inventory and total area.
2. Implement the Rust environment, placement validity, row-major scanning with skipped cells, exhaustion, dummy actions, and deterministic replay.
3. Add speculative feature extraction and Python bindings for the per-cell room-cell indices and per-room anchor-position indices. Check that speculative features match a separately committed copy of the same state and leave the parent unchanged.
4. Implement both learned embedding tables with separately configurable dimensions, their flattened concatenation, the 104-output MLP with configurable hidden-layer count and width, known-outcome checks, floored summed log-success score, and temperature-softmax sampling over valid candidates.
5. Implement episode generation with a configurable concurrent environment count, terminal room and cell labels, indefinite experience storage with uniform sampling from the configured recent-round window, replay, summed masked binary cross-entropy training with configurable minibatch size, and checkpoint save/load.
6. Run a small end-to-end training experiment and inspect learning and generation statistics before adding architectural or throughput complexity.

Use `~/map-gen` as a reference for PyO3/maturin setup and the concepts of speculative features, binary cross-entropy, `logsigmoid` scoring, and dummy padding, not for its `engine.rs` implementation. Do not modify that project or the source-data project as part of this implementation.

## Verification and initial measurements

Use temporary checks during development and remove any added test or verification files before completing implementation, in accordance with the project instructions. Check the vanilla layout directly from the source data rather than adding vanilla positions to the export.

- Verify exporter geometry, ordering, exact variant names, and the 64-cell area total.
- Verify the vanilla layout satisfies overlap, boundary, and segment compatibility rules.
- Exercise large-room contacts, skipped cells followed by later placements, exhaustion, and dummy tails.
- Compare speculative features with actual placement features and confirm replay reconstructs the same states and final outcomes.
- Check distinct room-specific embeddings for all four cells of a large room, zero embeddings for empty cells and unplaced rooms, valid anchor-position embeddings for `[0, 0]`, and the concatenated input width for different embedding dimensions.
- Exercise different MLP hidden-layer counts and widths, training batch sizes, and concurrent generation environment counts; confirm checkpoint loading restores the configured model architecture.
- Check room and cell terminal labels, known-failure detection including large-room coverage from other anchors, finite score floors, summed loss masking including zero supervised entries, candidate sampling masks, and checkpoint round trips in a small training run.
- Check retention of older experience, the recent-round sampling window when fewer than the configured number of rounds exist, and uniform sampling of eligible rounds.
- Check safetensors checkpoint save/load, the configured round interval, final saving on normal completion, and graceful interruption during generation and training. Confirm a second signal exits immediately and a failed partial round does not produce a checkpoint.
- Track complete-map frequency, rooms placed, occupied cells, per-room placement frequency, per-cell fill frequency, and outcome prediction loss. These are measurements, not additional training objectives.

## Deferred work

Choose concrete edge variants in a later phase from the intersections of the touching segments' supported sets. Require `closed` at the outer boundary. Persist room IDs and generated placement coordinates so the later phase can recover the original room data without expanding this geometry export.

Connectivity validation must eventually use the original node endpoints, intra-room routing, Light World and Dark World rules, Mirror transitions, and caves or other entrances. Room adjacency alone is insufficient, and `closed` compatibility does not imply a usable passage. Define whether that later checker evaluates a concrete seam selection or searches for a suitable selection. Reject maps that fail the required connectivity, and measure the resulting acceptance rate separately from complete placement success.

Do not implement connectivity checking, backtracking, room rotation, additional training objectives, or the older project's specialized generation machinery in the first version.
