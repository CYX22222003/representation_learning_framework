# Framework Price Prediction Comparison

This file compares the first framework MVP price-prediction run against existing
4h price baselines. All numbers are characterization results on the locked test
split; no checkpoint is selected from test performance.

## Framework Result

Run:

```text
experiments/framework/price_prediction/4h_stat_transform_vae_contrastive_concat/
```

Configuration:

- task: price prediction
- timeframe: 4h
- sequence length: 64
- branches: `statistical`, `transformed`, `vae`, `contrastive`
- aggregation: concat
- head: `PriceRegressor`
- branch standardization: train-fitted z-score with fixed clipping at +/-10

| Epochs | MAE | RMSE | MSE | Corr |
|---:|---:|---:|---:|---:|
| 15 | 0.0575 | 0.0955 | 0.0091 | 0.9780 |
| 50 | 0.0695 | 0.1046 | 0.0109 | 0.9763 |
| 100 | 0.0720 | 0.1059 | 0.0112 | 0.9761 |

## Baseline Context

Raw-OHLCV MLP price sweep:

| Epochs | MAE | RMSE | MSE | Corr |
|---:|---:|---:|---:|---:|
| 15 | 0.0834 | 0.1067 | 0.0114 | 0.9952 |
| 50 | 0.0600 | 0.0805 | 0.0065 | 0.9942 |
| 100 | 0.0454 | 0.0683 | 0.0047 | 0.9927 |

LSTM price benchmark:

| Epochs | MAE | RMSE |
|---:|---:|---:|
| 15 | 0.0089 | 0.0166 |
| 20 | 0.0072 | 0.0157 |
| 25 | 0.0103 | 0.0171 |
| 50 | 0.0116 | 0.0185 |
| 100 | 0.0093 | 0.0167 |

## Interpretation

The MVP framework loop is now working end to end and produces valid downstream
test metrics. At the matched 15-epoch budget, the framework improves over the
Raw-OHLCV MLP on MAE and RMSE. At 50 and 100 epochs, the Raw-OHLCV MLP remains
stronger, while the framework begins to show worse test error despite lower
train loss.

The LSTM benchmark remains substantially stronger on this price-prediction
setup. This result should therefore be framed as a successful first framework
loop, not as evidence that the framework is superior to all baselines.

## LSTM Comparison Validity

The stored LSTM v5 artifacts are genuine held-out test evaluations. The LSTM
code builds `X_train/y_train` and `X_test/y_test` with an 80/20 chronological
split per market, trains only on the training loader, and evaluates with the
test loader. Recomputing the saved `predictions.npz` files gives `27,500`
predictions and targets for each v5 run, matching the processed 4h test sequence
count.

However, this LSTM result should be treated as an external benchmark
characterization rather than a strict final one-to-one comparison. The current
LSTM runner rebuilds close-only windows directly from raw feather files instead
of loading `data/processed/market_4h_seq64_top50.npz`. It also creates its own
next-close labels, while the current framework price task uses
`build_price_prediction_targets(..., horizon=1)` on the already-split processed
sequence arrays. Because that shared task builder drops the final sequence in
each split, the framework evaluates `27,499` test targets while the LSTM artifact
contains `27,500`.

For the MVP, this is still useful: it tells us the framework is being compared
against a real held-out LSTM result under the same broad 4h/top-50/seq64 setup.
For a final performance claim, the LSTM should be refit into the current project
contract so it consumes the same processed split, target builder, sample count,
epoch budgets, seed policy, and metrics as the framework and Raw-OHLCV MLP.

## Required Baseline Adjustments

To make the LSTM comparison final-report ready:

1. Add a processed-data LSTM runner that loads
   `data/processed/market_4h_seq64_top50.npz` instead of reading raw feather
   files directly.
2. Build labels through `src/tasks/price_prediction.py` using
   `build_price_prediction_targets()` so LSTM, Raw-OHLCV MLP, and the framework
   share exactly the same train/test targets.
3. Save the same artifacts as the framework run: `config.json`,
   `dataset_manifest.json`, `history.npz`, `predictions.npz`, `metrics.json`,
   `summary.md`, and a sweep-level `sweep_metrics.json`.
4. Run the same fixed-budget sweep, e.g. `15,50,100`, with the same seed and no
   validation split, early stopping, or best-on-test model selection.
5. Report the close-only LSTM as the external benchmark, then add richer variants
   only with clear names so they are not confused with the original benchmark.

Suggested LSTM variants:

| Variant | Input | Purpose |
|---|---|---|
| `lstm_close_processed` | processed close column only, exact shared target builder | strict replacement for the current LSTM artifact |
| `lstm_ohlcv_processed` | full processed OHLCV window `[64, 5]` | checks whether the LSTM benefits from the same raw information available to framework feature extraction |
| `lstm_ohlcv_plus_features` | LSTM final hidden state concatenated with frozen `statistical`, `transformed`, `vae`, and `contrastive` features before the final MLP | hybrid model testing whether framework features add value to a supervised sequence model |

The third variant is not a pure external LSTM baseline. It should be reported as
a hybrid feature-augmented model because it uses the framework's extracted
representations. Its feature branches must use the same train-fitted
standardization and clipping as the framework task head to avoid changing the
comparison through preprocessing.

## Next-Stage SSL Backbone Direction

The current neural feature branches use different base encoders:

| Branch | Current base model | Stored downstream feature |
|---|---|---|
| `vae` | flattened-sequence MLP encoder/decoder | latent mean `mu`, 64-d |
| `contrastive` | Conv1d/CNN sequence backbone + MLP projector | backbone hidden state `h`, 128-d |
| `byol` | Conv1d/CNN online/target backbone + MLP projector/predictor | online backbone hidden state `h`, 128-d |

The strong LSTM price benchmark suggests that sequence-aware recurrent encoders
are worth testing in the next stage, but this should not replace the current MVP
branches yet. The immediate priority remains: finish the MVP loop for the other
downstream tasks and add the missing BYOL feature branch. After that, add
LSTM-based SSL encoders as named variants so their contribution can be compared
cleanly against the current MLP/CNN SSL branches.

Implementation direction:

1. Keep the current MLP VAE and CNN contrastive/BYOL branches as the first MVP
   reference point.
2. Add `contrastive_lstm` as an LSTM backbone with the existing MLP projector
   and NT-Xent objective. The LSTM final hidden state should be the stored
   downstream feature.
3. Add `vae_lstm` as an LSTM encoder VAE. Start with an LSTM encoder and a
   simple decoder before considering a full sequence decoder.
4. Consider `byol_lstm` only after BYOL-CNN features are extracted and evaluated,
   so the current BYOL implementation has a fair baseline result.

The projector head should remain an MLP unless there is a specific experiment
testing projector architecture. In contrastive and BYOL-style methods, the
projector maps the encoder representation into the SSL loss space; it is not the
main temporal modeling component. If LSTM capacity is added, it should primarily
be added to the encoder/backbone that produces the frozen downstream embedding.

Next evaluation steps:

- inspect whether feature scaling/clipping should be adjusted before final runs
- refit the LSTM into the processed-npz/shared-target project contract before
  making a strict final LSTM comparison
- finish the MVP loop for the other tasks, add BYOL features, then test
  LSTM-based SSL encoder variants as a next-stage extension
- run statistical-only and other ablations after the MVP
- consider gated aggregation after concat behavior is understood
- run multi-seed confirmation before making stronger claims
