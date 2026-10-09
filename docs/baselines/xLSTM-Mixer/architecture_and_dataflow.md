# xLSTM-Mixer architecture and Phase 6.9 data flow

## 1. Scope

This document separates three layers of evidence:

1. the architecture described in the NeurIPS 2025 paper;
2. the exact `FULL` path in official commit
   `730b0531aa9456e498765028f3c22ca3677de42e`; and
3. the proposed Phase 6.9 adaptation for `[64,5]` OHLCV to `[8,5]`.

No implementation or training is authorized by this document. Choices marked
“proposed” remain subject to the owner decisions in
`upstream_clarification_request.md`.

## 2. Released `FULL` forward path

Let:

```text
B = batch size
T = 64 historical hours
V = 5 variates in [open, high, low, close, volume] order
H = 8 future hours
D = sLSTM embedding width
M = number of learned initial/memory tokens
```

The official source accepts time-major tensors:

```text
x_enc: [B,T,V]
```

Timestamp markers are accepted by the interface but ignored by
`xLSTMMixer.forecast`.

### 2.1 Instance normalization

Official `Normalize` computes detached mean and standard deviation over the
time dimension for each sample and variate:

```text
mu    = mean(x_enc, dim=time)                    [B,1,V]
sigma = sqrt(var(x_enc, unbiased=False) + 1e-5) [B,1,V]
x     = (x_enc - mu) / sigma                    [B,T,V]
```

The source sets `affine=False`; there are no learned RevIN scale/offset
parameters despite the paper equation.

### 2.2 Shared NLinear time mixing

The final normalized input value is detached and used as an NLinear anchor:

```text
last        = x[:,-1:,:].detach()               [B,1,V]
centered    = x - last                          [B,T,V]
preliminary = Linear(T,H)(centered^T)^T + last [B,H,V]
tokens      = transpose(preliminary)            [B,V,H]
```

One `Linear(64,8)` is shared across all five variates.

### 2.3 Up-projection and optional learned tokens

```text
z = Linear(H,D)(tokens)                         [B,V,D]
```

If `M>0`, learned parameters of shape `[M,D]` are repeated over the batch and
prepended:

```text
z_with_memory = concat(memory,z, axis=tokens)  [B,M+V,D]
```

The paper says one initial token. The source constructor defaults to zero and
the experiment scripts use zero through four.

### 2.4 sLSTM recurrence and released reverse view

For the normal `FULL` mode, the xLSTM block stack receives tokens along axis
1 and width `D` along axis 2:

```text
forward_view = sLSTM(z_with_memory)             [B,M+V,D]
reverse_in   = flip(z_with_memory, dim=-1)      [B,M+V,D]
reverse_view = same_sLSTM(reverse_in)           [B,M+V,D]
mixed        = concat(forward_view,
                      reverse_view, dim=-1)     [B,M+V,2D]
```

The crucial fact is `dim=-1`: the source reverses latent coordinates inside
every token. It does **not** reverse the `[M+V]` token order. It also does not
flip the second output back before concatenation.

This differs from the Phase 6.6 provisional phrase “reversed variate-order
views” and from one interpretation of the paper's ensemble language. Exact
released behavior and paper-intended behavior must not be conflated.

### 2.5 Projection and denormalization

After the learned-token slice is removed:

```text
variates = mixed[:,M:,:]                        [B,V,2D]
y_norm   = Linear(2D,H)(variates)               [B,V,H]
y_norm   = transpose(y_norm)                    [B,H,V]
y        = y_norm * sigma + mu                  [B,H,V]
```

The same output projection is shared over variates.

## 3. Proposed project adaptation

The fixed scientific mapping is:

```text
accepted historical context             [B,64,5]
  (OHLC in [0,1], existing walk-scaled volume)
-> released xLSTM-Mixer core            [B,8,5]
-> extract [:,7,close_index]             [B]
-> existing absolute_price_h8 metrics
```

The proposed preliminary architecture, pending owner approval, is:

```text
D=128
M=1 learned initial token (owner-approved)
1 sLSTM block
8 heads
convolution kernel disabled (0)
dropout=0.1
packing=1
NLinear backbone
backcast/two-view path enabled
non-affine RevIN
released feature-axis reversal
```

This is not a result-selected configuration. It uses the paper's single token,
the smallest common source architecture family for ETT-like low-variate data,
and the released default/full pathway. No published script covers the exact
`T=64,H=8,V=5` regime.

## 4. Project input units and loss domain

The official loaders add a dataset-level `StandardScaler` before RevIN. The
Phase 6.9 owner decision instead keeps the project's accepted inputs unchanged:
OHLC probabilities are already in `[0,1]`, and volume already uses the
walk-training-only Phase 5 volume transform. No second xLSTM-specific
five-channel scaler is fitted.

The future-path builder applies that same existing volume transform to target
volume while leaving future OHLC probabilities in their accepted units. The
model's non-affine RevIN operates internally and is inverted before loss.
Training uses unweighted mean L1 over all `8 x 5` outputs in these accepted
units. The upstream volume-scaler identity, parameters, population, and hash
remain part of each walk's data manifest.

## 5. Target bundle and identity contract

For each existing `absolute_price_h8` decision row, a deterministic metadata
join checks that the same contract and accepted segment contain observed,
non-imputed, finite bars at every `t+1,...,t+8` for all five channels.

The auxiliary bundle stores:

```text
walk
split
contract identity
decision timestamp
eight target timestamps
segment identity
observed/imputed flags
context row identity
full target [8,5]
source hashes
```

A read-only identity check already proves that the full-path population is a
strict subset. The expected intersections are:

| Walk | Split | Existing price rows | Fully observed common rows |
|---:|---|---:|---:|
| 1 | train | 36,773 | 32,470 |
| 1 | evaluation | 29,834 | 27,786 |
| 2 | train | 56,652 | 53,112 |
| 2 | evaluation | 13,506 | 12,115 |

The preparation stage must freeze and independently assert those identities
while constructing complete OHLCV targets. `XM-MV8`, `H0-D0`, and Raw LSTM
then receive the same intersected train and evaluation rows. No target values
or model metrics may influence the intersection.

## 6. Optimization and checkpoint flow

The proposed project-native fixed-budget path is:

```text
walk-training rows only
-> shuffled seed-0 minibatches (project batch 512, if admitted)
-> Adam at 1e-4, L1 full-path loss, gradient clipping 1.0
-> snapshots at epochs 5, 15, 50
-> epoch 50 fixed before evaluation
-> complete [N_eval,8,5] prediction artifact
-> extracted close[t+8] report
```

There is no validation split, early stopping, Optuna search, best-loss restart,
or best-on-evaluation selection. This deliberately differs from the official
Lightning runner, which tests the best validation-MSE checkpoint.

## 7. Information and comparison boundary

At decision time the model sees exactly the same historical `[64,5]` context
as the established price task. It sees no future input. During supervised
training it receives 40 targets per row rather than the one `close[t+8]`
target used by H0-D0 and Raw LSTM. Therefore:

- identical context and headline endpoint support a useful task-level
  comparison;
- extra target supervision prevents a target-matched architectural claim;
- a win does not prove that sLSTM is inherently better than H0 or Raw LSTM;
- a loss does not disprove the paper's long-horizon public-benchmark claims;
  and
- the model provides no direct evidence about reusable representation quality.

## 8. Required tensor tests

Before full-data execution, focused tests must prove:

1. exact shapes at every stage above;
2. shared time/up/output projection parameters across variates;
3. the selected reversal axis with a hand-constructed tensor;
4. learned-token insertion and removal without row displacement;
5. exact reuse of the existing walk-training volume transform for contexts
   and future targets;
6. no target bar enters any input context;
7. output extraction uses horizon index 7 and close index 3;
8. batch-size invariance in evaluation mode;
9. checkpoint/config/source hash rejection on mismatch; and
10. same-backend CUDA checkpoint reload and fixed-probe replay in the Lumid
    Sandbox. Cross-backend CPU/CUDA numerical equivalence is not required.
