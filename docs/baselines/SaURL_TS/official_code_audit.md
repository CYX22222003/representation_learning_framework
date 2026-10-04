# SaURL-TS official-code audit

**Audit date:** 2026-10-04  \
**Repository:** <https://github.com/YusenL/SAURL-TS>  \
**Pinned commit:** [`96f39fe16646e866b6e6b9131e5216ac021c8950`](https://github.com/YusenL/SAURL-TS/tree/96f39fe16646e866b6e6b9131e5216ac021c8950)  \
**Remote refs found:** `refs/heads/main` only; no tags  \
**Commit history:** six commits, 2024-08-09 through 2024-10-16  \
**Software licence:** none found in the repository or README  \
**Audit type:** static source inspection; no paper result reproduction or Phase 6.7 training

## 1. Repository contents

The repository contains one ETTh1 CSV, a minimal README, a training entry point, `saurl_ts.py`, CoST-style encoders, BYOL wrappers, several unused model files, downstream forecasting/classification helpers, and a broad environment dump in `requirements.txt`.

The source provides useful evidence for an earlier SaURL implementation:

- separate time, frequency, and cross encoders;
- learned time- and frequency-domain augmentation networks;
- hard stochastic masks with straight-through gradients;
- magnitude/phase decomposition with `rfft` and `irfft`;
- BYOL online and EMA-target networks; and
- an alternating augmentation/representation update loop.

It is not a release-quality implementation of the final 2026 paper.

## 2. Paper-to-code crosswalk

| Paper component | Public source behavior | Audit result |
|---|---|---|
| 128-dimensional representation | CLI default is 320; ETTh1 config does not override it | mismatch |
| RwAM with avg/max pooling and shared Conv1d MLP | no RwAM implementation found | missing |
| Attention-weighted sum of three branches | extraction attempts a concatenation weighted by global STL-derived scalars | mismatch |
| Per-sample adaptive attention | train-wide trend/seasonality strengths initialize scalar loss weights | mismatch |
| Adaptive domain loss | adaptive weighted loss is computed, then immediately overwritten by an unweighted sum | dead behavior |
| Frequency-domain encoder | training builds frequency-derived reconstructed views; inference feeds the same raw `x` to all three encoders | mismatch |
| Cross-domain encoder on time/frequency inputs | training passes time- and frequency-derived views through one encoder; exact paper behavior remains unclear | ambiguous |
| Clean frozen extraction | extraction contains unresolved `out` and `self.net` references | broken |
| Paper optimizer settings | source ETTh1 config uses `lr=1e-5`, `meta_lr=0.012`; paper states `1e-4` and `1e-2` | mismatch |
| Paper Python/PyTorch environment | README says Python 3.9 and pins PyTorch 1.13.1; paper says Python 3.8.18/PyTorch 1.12.1 | mismatch |

## 3. Critical implementation findings

### 3.1 Missing software licence

There is no `LICENSE` file and no licence declaration in the README. The paper's CC BY 4.0 licence applies to the article, not automatically to the software. The source may be inspected for feasibility, but it should not be copied, modified, or vendored into this repository without an explicit software licence or author permission.

### 3.2 Repository predates the final paper

The latest public commit is dated 2024-10-16, while the paper was revised in December 2025 and accepted in January 2026. Only `main` exists and no release tag identifies the code used for the published results. This strongly suggests version drift.

### 3.3 RwAM is absent

The paper's central RwAM module is not implemented in the SaURL model. The repository contains an unused generic channel-attention helper, but it is not the paper's avg/max-pooling Conv1d block. A separate `AdaptiveBYOLLoss` holds three trainable scalar loss weights; it weights losses rather than representations. See [`models/losses.py` lines 13–25](https://github.com/YusenL/SAURL-TS/blob/96f39fe16646e866b6e6b9131e5216ac021c8950/models/losses.py#L13-L25).

During training, the adaptive loss result is assigned and then overwritten by `loss_fft + loss_time + loss_cross`; see [`saurl_ts.py` lines 724–727](https://github.com/YusenL/SAURL-TS/blob/96f39fe16646e866b6e6b9131e5216ac021c8950/saurl_ts.py#L724-L727).

### 3.4 Frozen extraction is not operational

The generic `encode`, `save`, and `load` methods reference `self.net`, which is never constructed; the class only defines `net_time`, `net_freq`, and `net_cross`. See [`saurl_ts.py` lines 766–807 and 908–923](https://github.com/YusenL/SAURL-TS/blob/96f39fe16646e866b6e6b9131e5216ac021c8950/saurl_ts.py#L766-L923).

The multi-branch `_eval_with_pooling` computes `out_time`, `out_freq`, and `out_cross` but uses an undefined `out` in several branches. Only one slicing branch creates a concatenated output, and it uses global STL strength scalars rather than RwAM. See [`saurl_ts.py` lines 925–976](https://github.com/YusenL/SAURL-TS/blob/96f39fe16646e866b6e6b9131e5216ac021c8950/saurl_ts.py#L925-L976).

### 3.5 Five-channel inputs hit hard-coded forecasting assumptions

Before constructing the loader, `fit` calculates trend/seasonality strength from `train_data[:,:,7:]`, assuming the first seven channels are calendar covariates. A `[N,64,5]` OHLCV tensor therefore yields an empty feature slice. See [`saurl_ts.py` lines 485–520](https://github.com/YusenL/SAURL-TS/blob/96f39fe16646e866b6e6b9131e5216ac021c8950/saurl_ts.py#L485-L520).

### 3.6 Validation is embedded in training

The official entry point constructs train/validation/test slices and passes all of them into `fit`; `fit` periodically evaluates the downstream task. See [`train.py` lines 64–128](https://github.com/YusenL/SAURL-TS/blob/96f39fe16646e866b6e6b9131e5216ac021c8950/train.py#L64-L128). That procedure conflicts with the Phase 6.7 no-validation, no-evaluation-driven-selection contract and cannot be reused.

### 3.7 Configuration and documentation disagree

The paper specifies representation width 128, Adam learning rates `1e-4`/`1e-2`, batch size 32, and dropout 0.1. The CLI defaults to width 320, and the ETTh1 configuration changes the learning rates to `1e-5`/`0.012` with 20 epochs. See [`train.py` lines 23–52](https://github.com/YusenL/SAURL-TS/blob/96f39fe16646e866b6e6b9131e5216ac021c8950/train.py#L23-L52) and [`config.py` lines 2–13](https://github.com/YusenL/SAURL-TS/blob/96f39fe16646e866b6e6b9131e5216ac021c8950/config.py#L2-L13).

The requirements file is an environment dump with many unrelated and obsolete packages rather than a minimal reproducible dependency set.

## 4. Static feasibility verdict

| Gate | Status | Reason |
|---|---|---|
| Paper identity | pass | title, authors, journal, DOI, and official repository link match |
| Code identity | partial | repository is official but no paper-matching tag/release exists |
| Software licence | fail | no licence or permission found |
| 64-by-5 API | fail | hard-coded `[:,:,7:]` assumption and no usable adapter |
| Paper-faithful architecture | fail | RwAM absent; output width/fusion differ |
| Frozen extraction | fail | unresolved attributes/variables and undefined extraction semantics |
| Project split compatibility | fail as-is | validation/test evaluation is embedded during encoder training |
| Hardware estimate | paper-only | paper reports A40 measurements; local model cannot be trusted until architecture is resolved |

**Direct-source decision:** the audited public snapshot is not admissible for
copying, modification, vendoring, or execution as the Phase 6.7 baseline. This
verdict applies to direct source reuse. A later 2026-10-04 project-owner
decision admits a separately documented, independently authored paper-guided
reimplementation, with this snapshot retained only as attributed behavioural
evidence.

## 5. What would unblock direct reuse of the upstream implementation

All of the following are needed before code implementation begins:

1. an explicit software licence or written permission covering use and adaptation;
2. the exact source commit/release used for the 2026 article;
3. a working RwAM and frozen-representation extraction path;
4. confirmation of the native output width and pooling rule;
5. the exact frequency and cross-encoder inputs at training and inference;
6. the SaDA/SaSSL loss weights and update schedule; and
7. a configuration that maps cleanly to `[N,64,5]` without downstream validation.

These conditions remain applicable to direct upstream reuse. The approved
independent candidate instead uses the reporting label **SaURL-TS-Frozen
(paper-guided reimplementation)** and must not be described as the official
implementation.
