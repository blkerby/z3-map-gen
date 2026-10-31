# Overworld Map Generation Plan

## Goal and scope

Create a standalone project that generates overworld layouts on an 8×8 grid. Place every overworld room exactly once, without overlap, while respecting neighboring edge compatibility and a closed outer boundary. Use a learned outcome predictor to score speculative placements, then sample a placement using a temperature-controlled softmax. Do not use backtracking.

Continue partially failed maps until no further room can be placed. These episodes still provide useful per-room outcome labels for training. Complete placement is the initial success criterion; traversal connectivity and concrete edge-type selection are deferred.

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

## Environment and placement rules

Use Rust for environment state, candidate enumeration, speculative state features, and replay, exposed to Python through PyO3/maturin. Use Python/PyTorch for the model, scoring, sampling, and training. Take the division of responsibilities from `~/map-gen`, without copying its frontier machinery, balancing objectives, proposal networks, or generation pipeline.

Represent each room in its original orientation. A placement specifies a room and the upper-left grid coordinate of its footprint. Large rooms may start at any coordinate where they fit; do not restrict them to even coordinates.

A real candidate is valid when all of the following hold:

1. The room has not already been placed.
2. Its entire footprint fits within the board and does not overlap any placed room.
3. Every segment facing outside the board supports `closed`.
4. Every segment touching a placed neighboring room shares at least one supported variant name with that neighboring segment.

Check the entire perimeter of a large room, including contacts with multiple neighboring rooms and both segments of a shared large-room edge. Match segment indices using their positions along the shared boundary. Matching uses exact variant-name intersection; defaults and node IDs do not participate. `closed` is a valid shared variant even though it supplies no passage.

An empty neighboring cell imposes no edge constraint. A skipped cell remains empty and must not be treated as a closed boundary. Borders between cells belonging to the same large room are internal to that room and have no segment-matching constraint.

Segment choices are independent according to the source-data contract, so checking nonempty intersections is enough for this placement phase. Do not select a concrete variant yet.

## Generation loop and batching

For each environment, scan empty cells in row-major order: smallest `y` first, then smallest `x`. At each cell, enumerate valid unused rooms anchored there. Skip cells with no candidates and stop scanning at the first cell with candidates. Offer all valid candidates at that cell; do not offer placements at later cells while an earlier cell has a valid candidate.

Evaluate each candidate as a separate speculative placement from the same parent state. Extract its resulting state features without committing it or affecting the other candidates. Score these states with the outcome model, sample one candidate, and commit only that placement. Repeat until no empty cell has any valid candidate.

Do not stop because an earlier cell is permanently unfillable or full-map success is otherwise impossible. Continue placing other rooms to obtain more complete terminal labels. Do not restart or backtrack within an episode.

Use 40 action slots per episode, one for each possible real room placement. Once an environment is exhausted, fill its remaining slots with a dummy action that leaves its state unchanged. A completed map has 40 real placements; an exhausted incomplete map has a dummy tail. Dummy actions are padding, not alternatives to valid placements. Candidate-count padding is likewise masked out of sampling.

## Model input features

Encode the existing room placements in two complementary ways, using separate learned embedding tables shared across all cells or rooms respectively:

1. For each of the 64 grid cells, embed the occupying room and the cell's relative position within that room using a table with embedding dimension `room_embedding_dim`. Each small room has one learned row; each large room has four distinct learned rows for its top-left, top-right, bottom-left, and bottom-right cells. These are specific to each room, not shared quadrant embeddings. The current catalog therefore requires `32 + 8 * 4 = 64` learned rows. An empty cell contributes an exact zero vector.
2. For each of the 40 rooms, embed its anchor position using a table with 64 learned rows and embedding dimension `position_embedding_dim`. An unplaced room contributes an exact zero vector. Positions use row-major indices `8 * y + x`, including index 0 for the valid position `[0, 0]`.

Both embedding dimensions are required configurable model hyperparameters. The zero vectors represent absence and are not additional learned rows. In particular, do not reserve a real room-cell or position index as padding. Assign room-cell indices in room-ID order, then row-major order within each room's footprint; derive this mapping from the exported sizes without adding export fields. Rust supplies room-cell indices per grid cell and anchor-position indices per room, with a separate sentinel such as `-1` for absence; Python performs the learned lookups and masks absent entries to zero.

Order cells by row-major position and rooms by the fixed room-ID ordering. Flatten both sets of embedding vectors and concatenate them as the input to a small MLP. Its input width is `64 * room_embedding_dim + 40 * position_embedding_dim`. Do not pool over cells or rooms, since their slots identify absolute positions and room identities. Train both embedding tables jointly with the predictor and save their dimensions and weights in checkpoints.

These inputs describe the speculative post-placement state during candidate evaluation and the corresponding replayed state during training. They replace the earlier proposal for raw placement flags and coordinates; no extra occupancy or geometry feature channels are required initially. The fixed room catalog supplies room sizes and edge rules to the environment. The model learns against this catalog; it does not initially need to generalize across different room sets.

## Outcome predictor and candidate score

The model outputs one logit `z_r(s)` per room, representing the probability that room `r` will be present in the terminal map after continuing from state `s` under the generation policy. This is an outcome prediction, not a distribution over the next room to place.

For candidate `a`, let `s_a` be the state after applying it. Already-placed rooms, including the candidate room, have known placement probability 1 and contribute zero to the score. For the other rooms, use the model's predictions:

```text
p_r(s_a) = sigmoid(z_r(s_a))
score(a) = sum over rooms not yet placed in s_a of log(p_r(s_a))
P(select a) = softmax(score(a) / temperature)
```

Use a positive temperature coefficient and sample from this distribution. Keep temperature fixed within an initial training run; it controls exploration and changes the continuation policy whose outcomes the model predicts. Compute log probabilities directly using `logsigmoid` on logits, rather than taking the logarithm of rounded sigmoid probabilities.

The score is the log of the product of predicted room-placement probabilities, approximating full placement success under an independence assumption. It is a ranking surrogate, not a calibrated joint success probability. After an unfillable cell makes full success impossible, continue using the individual predictions to rank candidates. Do not hard-set failed outcomes to zero probability and collapse every candidate score to negative infinity.

## Training targets and experience

At the end of each episode, construct a binary vector `y` with one entry per room: 1 if the room was placed anywhere in the terminal map, otherwise 0. Use this same terminal vector as the outcome target for every real visited post-placement state in that episode.

Train with binary cross-entropy with logits, independently over the room outcomes. Already-placed rooms have known success and need no learned prediction; mask them out of the learned loss. Exclude dummy padding so shorter failed episodes do not gain extra weight from repeated frozen states. Aggregate over the supervised state-room entries.

Only the chosen trajectory receives these labels. Do not attach its terminal vector to unchosen speculative candidate states: those candidates were not continued, so their counterfactual outcomes are unknown. Scoring every candidate does not require producing training labels for every candidate.

Store the ordered real placement actions and the terminal outcome vector. Reconstruct visited states and their features by replaying the actions through the same Rust environment. Retain the room ordering, catalog identity, temperature, and model configuration with training artifacts so outputs can be interpreted consistently. Keep the initial experience format simple and specific to this project.

Alternate generation under the current model with supervised updates from collected experience. Keep experience reasonably current because the continuation policy changes as the model improves. Full successes are not required to begin learning: failed episodes still contain positive and negative per-room labels. Do not replace this objective with a reward based on room count or occupied area.

## Implementation sequence

1. Implement the exporter and define its minimal JSON format. Verify the current inventory and total area.
2. Implement the Rust environment, placement validity, row-major scanning with skipped cells, exhaustion, dummy actions, and deterministic replay.
3. Add speculative feature extraction and Python bindings for the per-cell room-cell indices and per-room anchor-position indices. Check that speculative features match a separately committed copy of the same state and leave the parent unchanged.
4. Implement both learned embedding tables with separately configurable dimensions, their flattened concatenation, the small outcome model, summed log-probability score, and temperature-softmax sampling over valid candidates.
5. Implement batched episode generation, terminal room labels, experience replay, masked binary cross-entropy training, and checkpoint save/load.
6. Run a small end-to-end training experiment and inspect learning and generation statistics before adding architectural or throughput complexity.

Use `~/map-gen` as a reference for PyO3/maturin setup, speculative features, binary cross-entropy, `logsigmoid` scoring, and dummy padding. Do not modify that project or the source-data project as part of this implementation.

## Verification and initial measurements

Use temporary checks during development and remove any added test or verification files before completing implementation, in accordance with the project instructions. Check the vanilla layout directly from the source data rather than adding vanilla positions to the export.

- Verify exporter geometry, ordering, exact variant names, and the 64-cell area total.
- Verify the vanilla layout satisfies overlap, boundary, and segment compatibility rules.
- Exercise large-room contacts, skipped cells followed by later placements, exhaustion, and dummy tails.
- Compare speculative features with actual placement features and confirm replay reconstructs the same states and final outcomes.
- Check distinct room-specific embeddings for all four cells of a large room, zero embeddings for empty cells and unplaced rooms, valid anchor-position embeddings for `[0, 0]`, and the concatenated input width for different embedding dimensions.
- Check terminal labels, loss masking, candidate sampling masks, and checkpoint round trips in a small training run.
- Track complete-map frequency, rooms placed, occupied cells, per-room placement frequency, and outcome prediction loss. These are measurements, not additional training objectives.

The earlier exploratory uniform-candidate sampler completed 0 of 1,000 attempts both with and without a closed-boundary requirement. Those trials stopped at the first unfillable cell and therefore do not measure the continued-placement policy specified here. Establish a fresh uniform-candidate baseline with the actual continuation rules before comparing model-guided generation.

## Deferred work

Choose concrete edge variants in a later phase from the intersections of the touching segments' supported sets. Require `closed` at the outer boundary. Persist room IDs and generated placement coordinates so the later phase can recover the original room data without expanding this geometry export.

Connectivity validation must eventually use the original node endpoints, intra-room routing, Light World and Dark World rules, Mirror transitions, and caves or other entrances. Room adjacency alone is insufficient, and `closed` compatibility does not imply a usable passage. Define whether that later checker evaluates a concrete seam selection or searches for a suitable selection. Reject maps that fail the required connectivity, and measure the resulting acceptance rate separately from complete placement success.

Do not implement connectivity checking, backtracking, room rotation, additional training objectives, or the older project's specialized generation machinery in the first version.
