# SGN-C owner decision request

**Status:** all seven decisions approved by the owner on 2026-10-09
**Decision gate:** complete; implementation may proceed
**Default reply format:** “Approve all recommendations” or list changes by
question number

The paper and pinned source leave several choices that materially affect the
model. The recommendations below use only paper/source evidence and
training-only diagnostics; no SGN evaluation result exists.

## Decision summary

| # | Topic | Recommendation | Status |
|---:|---|---|---|
| 1 | grouping initializer | `G=2`, 2,000 contract-stratified training rows, project-native deterministic K-means | approved |
| 2 | period and hierarchy | fixed `P=16`, four stages, depths `[2,2,2,1]` | approved |
| 3 | group semantics | source-style weighted sum and one shared group embedding | approved |
| 4 | shifted boundary | preserve source first-half restoration after inverse shift | approved |
| 5 | width/block/loss | `D=64`, seven kernels, ratio 2, dropout 0.1, `beta=0.1`, source temperature schedule | approved |
| 6 | optimizer/runtime | Adam `1e-3`, no weight decay/scheduler, physical batch 256, fp32, clip 4.0 | approved, subject to resource admission |
| 7 | licence boundary | independent implementation only; no upstream source reuse | approved |

## 1. Group count and BDC initializer

**Recommendation:** use `G=2`. Compute BDC independently per walk from exactly
2,000 deterministic, label-free, contract-stratified chronological-quantile
training rows using the accepted stored OHLCV values without a second scaler.
Run an independently authored deterministic K-means over the five BDC row
profiles; fail on empty groups or non-replay.

**Why:** both read-only walk probes recover `{O,H,L,C}` versus `{V}` at
`G=2`. `G=3/4` arbitrarily separates OHLC variables whose BDC values exceed
`0.9984`, and assignments vary more across walks. Two groups retain meaningful
intra- and inter-group paths without near-singleton over-partitioning.

**Alternative:** `G=3` is closer to the paper's relatively high group counts,
but has weaker empirical structure for these five variables.

**Owner decision:** approved.

## 2. Period and hierarchy

**Recommendation:** freeze `P=16` before training and use four stages with
depths `[2,2,2,1]`. Record a training-only FFT diagnostic for each walk and
require 16 to remain in both top-five non-full-length candidate sets.

**Why:** both walk probes yield candidates `32,21,16,13,11`. `P=16` divides
64 exactly, produces four windows, supports a complete `4->3->2->1` merge
hierarchy, avoids zero padding, and retains three shifted blocks. It is a
transparent project adaptation.

**Alternatives:** a literal reading of Equation 5 selects `P=11`, pads the
sequence to 66, and also supports four stages. `P=32` is simpler but leaves
only two windows and two stages. The official source supplies configured
periods and does not implement FFT selection.

**Owner decision:** approved.

**Where this belongs in SGN:** `P` first belongs to **Small Periodic Window
Partitioning**, the opening part of MGWM in paper Section 3.2. It decides how
the 64-hour sequence is cut into local windows before multi-scale convolution.
The number of resulting windows then controls **PWSM** in Section 3.3: shifted
blocks communicate across those window boundaries, and period merging creates
the hierarchy. Therefore Decision 2 is not part of variable grouping/VGE; it
connects MGWM's input partition to PWSM's depth and merge schedule.

## 3. Group fusion and embedding semantics

**Recommendation:** preserve the source's assignment-weighted **sum** across
variables and reuse one shared temporal embedding module across groups.

**Why:** these are observable released behaviors and keep the implementation
compact and group-permutation symmetric. They also avoid inventing separate
embedding capacity not present in the released model.

**Trade-off:** a group containing four OHLC channels has larger raw magnitude
than the one-variable volume group, and the paper's phrase “independent
embeddings” could reasonably imply separate group parameters. Dividing by
assignment mass would remove cardinality scaling but would be a more material
departure from the source.

The project input already standardizes volume from each walk's training
interval, while OHLC remains bounded probability data. This keeps channels in
comparable numerical ranges, although a four-channel OHLC sum is not
mathematically the same scale as one standardized volume channel.

**Owner decision:** approved.

## 4. Shift boundary behavior

**Recommendation:** after the inverse half-period roll, restore the first
`P//2` positions from the block input, matching the pinned source.

**Why:** this prevents artificial wrap-around communication between the oldest
and newest timestamps while retaining shifted cross-window interaction over
the interior.

**Alternative:** the paper describes a pure cyclic left/right shift with no
restoration. That is simpler but permits endpoint interaction.

**Owner decision:** approved.

## 5. Model width, convolution, assignment, and loss

**Recommendation:** freeze:

```text
D=64; kernels=7 (1..13); block_num=1; ratio=2; dropout=0.1
tau_init=1.0; tau_min=0.1; exponential rate=0.0003 per optimizer step
soft noisy train assignment; deterministic hard evaluation assignment
beta=0.1 with source reduction and cosine rescaling
native LayerNorm + mean pool + Linear(128,3) head
```

**Why:** width 64, seven kernels, ratio 2, beta 0.1, and the temperature
parameters are the best-supported common paper/source settings. They preserve
the defining SGN mechanisms without commissioning a tuning sweep.

**Owner decision:** approved.

## 6. Optimizer and physical batch

**Recommendation:** use source-specific Adam at `1e-3`, zero weight decay, no
scheduler, float32, clip global norm at `4.0`, physical batch 256, shuffled
seed-0 training, exactly 50 epochs, snapshots 5/15/50, and epoch 50 primary.
Batch reduction is permitted only by a pre-training resource admission shared
by both walks.

**Why:** `1e-3`, Adam, and clip 4 follow the SGN source; batch 256 is used by
its large PTB-XL dataset. Fixed 50 epochs and no early stopping are mandatory
project rules. This intentionally differs from H0/Raw LSTM's `1e-4`/batch-512
recipe and must be reported as complete-system optimization freedom.

**Alternative:** use the project-wide learned-control recipe (`1e-4`, batch
512). That is optimizer-matched but less source-aligned and may undertrain SGN
within 50 epochs.

“Physical batch 256” means 256 sequences are resident for one actual
forward/backward pass before one optimizer update; it is not a larger
effective batch simulated by gradient accumulation. The current WSL runtime
has PyTorch `2.12.1+cu126` with CUDA access to an RTX 4060 Laptop GPU with
8,188 MiB total VRAM. Batch 256 is a plausible starting point for the proposed
compact SGN, but approval does not assert that it fits: the implementation
must still pass a real-row batch-256 plus remainder forward/backward/update
resource smoke. Any reduction is frozen before training and shared by both
walks.

**Owner decision:** approved, subject to the mandatory resource admission.

## 7. Source/licence boundary

**Recommendation:** write every SGN-C file independently from the paper and
the approved specification. Do not copy, translate, vendor, or modify any
upstream source. Retain repository URL, commit, observed behavior, and hashes
as attribution. Revisit only if the authors publish a clear licence or grant
permission.

**Why:** the official repository has no software licence. Public visibility is
not permission to redistribute derivative code.

**Owner decision:** approved.

## What approval does and does not authorize

Approval freezes the implementation specification and permits model/lifecycle
coding plus CPU tests. It does **not** authorize full GPU training. The later
manifest/resource gate remains non-training by default, and real trajectories
still require explicit `--execute` authorization under the Phase 6.9 plan.
