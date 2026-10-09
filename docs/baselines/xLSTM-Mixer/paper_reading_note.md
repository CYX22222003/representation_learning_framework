# xLSTM-Mixer paper reading note

**Title:** xLSTM-Mixer: Multivariate Time Series Forecasting by Mixing via
Scalar Memories

**Authors:** Maurice Kraus; Felix Divo; Devendra Singh Dhami; Kristian Kersting

**Venue:** 39th Conference on Neural Information Processing Systems
(NeurIPS 2025)

**Source read:** local 34-page arXiv v4/NeurIPS PDF, dated 20 November 2025

**Local SHA-256:**
`dc4d9c7a731489d4932aac993d6d5c04eea741c11b4f989b395a77f909d7151d`

## 1. One-paragraph summary

xLSTM-Mixer is an end-to-end multivariate forecaster that first produces a
channel-independent NLinear forecast, up-projects each variate's complete
forecast horizon into a token, and refines the variate-token sequence with a
stack of scalar-memory xLSTM blocks. A learned initial token conditions the
recurrent stack. Two transformed views share the same sLSTM weights and are
concatenated before a shared linear map returns each variate to the requested
forecast horizon. RevIN surrounds the model. The main contribution is the
combination of explicit time mixing, recurrent cross-variate mixing, and
two-view reconciliation. The paper reports strong long-horizon point
forecasting, probabilistic GIFT-Eval, and classification results, but its
principal evidence concerns public regular-grid benchmarks rather than
financial prediction markets.

## 2. Problem

For context `X in R^(V x T)`, predict the jointly sampled future path
`Y in R^(V x H)`. The paper assumes a regular grid with all variates observed
together and conventional chronological train, validation, and test portions
(Section 2, p. 3). It targets two perceived weaknesses in recent forecasting:

- attention-based models may scale poorly with long time/variate sequences;
- pure channel-independent models regularize well but cannot directly learn
  cross-variate relationships; and
- ordinary recurrence over time does not exploit the inverted-token view in
  which one token represents an entire variate history or forecast.

The proposed compromise first learns temporal structure independently and
then recurrently mixes a much shorter sequence of variate tokens.

## 3. Main idea

The method has three stages (Figure 2 and Section 3, pp. 2--5):

1. **Time mixing:** normalize each sample/variate with RevIN, subtract the last
   normalized value, and apply one shared `Linear(T,H)` NLinear map to every
   variate.
2. **Joint mixing:** apply one shared `Linear(H,D)` up-projection to every
   preliminary variate forecast and run an sLSTM stack over the resulting
   variate tokens, with a learned initial token prepended.
3. **View mixing:** process an original and reversed latent view with the same
   sLSTM stack, concatenate their outputs, project `2D -> H` independently
   with shared weights, and invert RevIN.

This is direct supervised full-path forecasting. It does not pretrain an
encoder and then freeze a reusable representation.

## 4. Contributions

- **Architecture:** combines NLinear time mixing with sLSTM recurrence over
  variates and shared-weight two-view refinement.
- **Efficiency:** moves recurrence to the number-of-variates axis after the
  temporal horizon is compressed into each token.
- **Empirical breadth:** evaluates deterministic long-horizon forecasting,
  probabilistic forecasting, classification, resource scaling, ordering
  sensitivity, and component ablations.
- **Recurrent-model evidence:** argues that sLSTM memory mixing is more useful
  than mLSTM, ordinary LSTM, or GRU for the evaluated long-horizon setting.

## 5. Method

### 5.1 RevIN and shared NLinear forecast

Section 3.1 (p. 4) normalizes each variate over the input time axis, retaining
the instance statistics for inversion. In the paper's notation:

```text
x_norm    = RevIN(x)
x_initial = FC(x_norm[1:T] - x_norm[T]) + x_norm[T]
```

`FC: R^T -> R^H` has bias and is shared across all variates. Weight sharing
keeps the parameter count independent of `V` for this stage and acts as a
channel-independence regularizer.

The paper's RevIN equation includes learnable affine parameters `gamma` and
`beta`. The released source disables them. This is a reproducibility decision,
not an inconsequential implementation detail.

### 5.2 sLSTM refinement over variates

Section 3.2 (pp. 4--5) maps each length-`H` preliminary forecast to width `D`
with a shared up-projection. This produces `V` tokens, one per variate. A stack
of `M` sLSTM blocks then strides over the variate order rather than the
original `T` timestamps.

sLSTM extends LSTM with exponential input/forget gates, a normalizer state,
and a stabilizer state. The paper prefers it to mLSTM because sLSTM preserves
hidden-to-hidden recurrence in the gates, enabling state-dependent memory
updates (Section 2.1, p. 3; Appendix B, p. 24). Multi-head block-diagonal
recurrent matrices restrict mixing within feature groups while the input
weights still act across the full token.

A learned token `eta in R^D` is prepended to condition the initial recurrent
memory. The method text specifies one token; the official run scripts vary the
number between zero and four.

### 5.3 Multi-view mixing

Section 3.3 (p. 5) describes an original latent sequence and a reversed latent
sequence, both processed by the same sLSTM stack. The two outputs are
concatenated and a shared `Linear(2D,H)` produces each final variate forecast.

The exact reversal semantics are unclear in the paper. It says “the order of
the latent dimensions including the representation of eta is inverted,” but
also interprets the mechanism as an ensemble over variate orderings. Appendix
H attributes each output only to preceding input variates, which is more
consistent with retaining one variate direction. The released `FULL` code
resolves its own behavior unambiguously by flipping the last feature axis
`D`, not the variate-token axis. The Phase 6.9 implementation must state which
meaning it adopts.

### 5.4 Objective and optimization

The deterministic forecasting runs use MAE/L1 loss, while results report MAE
and MSE (Section 4, p. 5; Appendix C, p. 25). Appendix C specifies:

- float32 training for up to 60 epochs;
- Adam with `beta=(0.9,0.999)` and no weight decay;
- cosine annealing;
- gradient clipping at norm 1.0;
- Optuna hyperparameter selection; and
- seeds 2021, 2022, and 2023.

The search spans batch sizes 16--512, learning rates `1e-2`--`1e-4`, lookbacks
96--2048, widths 32--1024, one to four blocks, four to 32 heads, convolution
kernels disabled/2/4, and dropout 0.1/0.25 (Table 4, p. 25). There is no
published recipe for `T=64`, `H=8`, or five OHLCV channels.

## 6. Experiments

### 6.1 Long-horizon forecasting

The main benchmark uses Weather, Electricity, Traffic, ETTh1/2, and ETTm1/2,
with horizons 96, 192, 336, and 720. The datasets contain 7--862 variates and
are sampled every 10 minutes, 15 minutes, or hour (Appendix D, p. 25).

Table 1 reports xLSTM-Mixer as best in 11 of 28 MSE cells and 16 of 28 MAE
cells, with a claimed new state of the art on six of seven datasets (pp. 5--6).
The paper tunes lookback and other hyperparameters per method rather than
fixing lookback 96. A Friedman/Conover-Holm analysis at horizon 96 reports
xLSTM-Mixer significantly better than every listed method except xLSTMTime,
with average ranks 1.5 and 4.0 respectively (p. 7 and Figure 5, p. 27).

These are broad public-benchmark results, not evidence on Polymarket or on an
eight-hour endpoint.

### 6.2 GIFT-Eval

The paper adds a quantile head and reports rank 2 by aggregate CRPS on the
then-current GIFT-Eval leaderboard, describing xLSTM-Mixer as the strongest
purely supervised method in that table (Section 4.2, p. 7; Appendix F,
pp. 28--31). The audited source repository does not contain this estimator or
quantile path, so these results cannot be reproduced from that revision alone.

### 6.3 Classification outlook

The paper replaces the final regression projection with a fully connected
classification head and trains end-to-end with cross-entropy. Table 14 reports
74.1% mean accuracy across ten UEA datasets, close to ModernTCN's 74.2%
(Appendix J, p. 34). The released `xLSTMMixer` class does not implement its
classification method, and no corresponding scripts are present.

### 6.4 Model analysis and ablations

- Figure 1 reports favorable memory scaling and one-to-two orders of magnitude
  less memory than TimeMixer in the selected setup.
- Figure 4 shows useful but dataset/horizon-dependent effects from larger
  latent width and longer lookback.
- Appendix H finds some sensitivity to variate permutation, especially an
  Electricity horizon-720 case, while most displayed changes are smaller.
- Table 13 finds two views stronger overall than one, five, eight, or ten on
  the displayed ETTh2 study.
- Table 3 reports the full model strongest overall. In the authors' aggregate
  summary, replacing sLSTM by LSTM raises MAE/MSE by 6.2%/7.0%; recurrence over
  time rather than variates by 4.3%/4.7%; removing time mixing by 2.7%/3.1%;
  and removing the initial token or view mixing by less than 1% each
  (pp. 8--9).

## 7. Evidence versus claims

| Claim | Assessment | Reason |
|---|---|---|
| xLSTM-Mixer is a strong recent long-horizon forecaster | Strongly supports on the tested public benchmarks | Broad datasets, three seeds, many baselines, full tables, and component ablations support the bounded claim. |
| sLSTM-over-variates is useful | Supports within the tested architecture | LSTM/GRU/mLSTM and time-axis ablations degrade average results, but the comparison changes recurrent primitives and is not a universal causal result. |
| Two-view mixing improves forecasting | Partially supports | The one-view ablation and view-count study are positive, but the paper/source meaning of “reversed” is inconsistent. |
| The model transfers to finance | Does not establish | Finance is motivational; no financial dataset, market walk, probability price, or trading evaluation is reported. |
| The model is generally state of the art | Partially supports | It is strong on the named benchmarks but not best in every metric/dataset, and performance depends on per-setting tuning. |
| The source exactly reproduces the final paper | Does not establish | No paper-v4 release/tag, final-paper additions are absent, and several paper/source details differ. |

## 8. Strengths

- It is a genuine recent top-conference task-specific forecaster.
- Its full-path objective maps naturally to eight future OHLCV bars.
- NLinear provides a strong, interpretable preliminary forecast before
  nonlinear cross-variate refinement.
- Parameter sharing keeps the time/up/down projections independent of the
  number of variates.
- The paper includes three seeds, full results, significance testing,
  ordering analysis, resource analysis, and extensive ablations.
- The small five-variate token sequence should be computationally manageable
  if the CUDA sLSTM dependency builds successfully.

## 9. Limitations

- The paper assumes a regular jointly sampled grid. Missing future target bars
  must be removed by a predeclared metadata-only intersection.
- The public benchmarks have at least seven and up to 862 variates; OHLCV has
  only five, giving much less cross-variate sequence depth.
- Forecast horizons in the main study start at 96; the project uses eight.
- Per-dataset/horizon Optuna tuning does not fit the project's no-validation,
  no-evaluation-selection contract.
- Full-path supervision gives `XM-MV8` more targets than H0-D0 or Raw LSTM.
- Source RevIN, initial-token count, and view reversal differ from or
  complicate the paper description.
- The method's output is not constrained to valid OHLC relationships or
  `[0,1]` probability bounds.
- Variate order is semantically consequential and must be frozen.
- The custom CUDA sLSTM requires compute capability at least 8.0 and a
  compatible compiler/toolchain.
- The pinned xLSTM dependency is AGPL-3.0.

## 10. Relation to prior project baselines

Unlike H0 and the Phase 6.7 methods, xLSTM-Mixer is trained directly with
future labels and cannot isolate representation quality. Unlike Raw LSTM, it
first compresses each complete temporal history into a preliminary future
path and then recurs over variates. Unlike persistence, it learns the entire
five-channel trajectory. It should therefore be reported as a complete-system
forecasting benchmark with extra supervision, not as a target-matched
architecture control.

## 11. Key takeaways

1. Preserve full `5 x 8` forecasting and extract only `close[t+8]` for the
   established price metric.
2. Under the approved project adaptation, keep OHLC in `[0,1]` and reuse the
   existing walk-training-only volume transform; do not fit a second
   xLSTM-specific scaler.
3. Do not describe the released second view as reversed variate order unless
   that behavior is explicitly implemented and labelled as a paper-guided
   correction.
4. Non-affine RevIN is approved. One initial token remains the recommended and
   only still-unconfirmed owner choice.
5. A minimal project-native adapter is preferable to importing the repository's
   Time-Series-Library and Lightning training stack.
6. The comparison demonstrates task competitiveness, not reusable
   representation quality.

## 12. Questions after reading

- Should “source-faithful” preserve the released feature-axis reversal or
  correct it to the paper's variate-order interpretation?
- Should RevIN follow the paper's affine equation or the source's non-affine
  instance normalization?
- Should the paper's single initial token override the scripts' tuned
  zero-to-four token counts?
- Which predeclared architecture is defensible for an unseen `T=64,H=8,V=5`
  regime?
- Should the project use the AGPL `xlstm==1.0.3` package, or separately
  implement the required sLSTM core?
- What CUDA/Python/PyTorch combination can build and replay the pinned custom
  kernel in the persistent sandbox?

These questions are converted into decisions in
`upstream_clarification_request.md`. Decisions 1--4 and 6--14 are resolved;
implementation remains gated on the token-count confirmation, and execution
also requires the Phase 6.9 data and Lumid Sandbox runtime gates.

## 13. Relevance to Phase 6.9

xLSTM-Mixer is well matched to Phase 6.9's price leg because it is recent,
top-conference, supervised, and purpose-built for multivariate forecasting.
It complements the representation comparison by asking whether H0's simple
frozen probe remains competitive with a specialized complete system. Its
former listing in Phase 6.6B is superseded. Phase 6.9 now owns the two
walk-specific runs and all canonical artifacts; the outdated Phase 6.6 text
must not launch or report a duplicate baseline.
