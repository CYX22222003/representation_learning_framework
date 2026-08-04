# GINN Volatility Prediction Limitation

This note records an observed limitation of applying the GINN baseline
(AR -> GARCH -> LSTM with fused loss) to Polymarket OHLCV data.

## Summary

The GINN baseline relies on the AR-GARCH preprocessing stage to provide a
meaningful auxiliary volatility target. On sparse or near-static Polymarket
contracts, the GARCH fit can converge numerically while producing an
economically invalid volatility scale. When this happens, the fused loss pulls
the neural model toward the corrupted GARCH target, producing poor volatility
predictions even when the LSTM itself trains successfully.

This is a limitation of transferring a GARCH-supervised benchmark to bounded,
sparse prediction-market probability series. It should be documented as a
domain-mismatch limitation, not as evidence that GARCH is generally unsuitable
for liquid financial time series.

## Observed Failure Case

The 4h GINN cache `data/processed/ginn_4h_seq64_top50.npz` contains one
dominant pathological contract:

```text
contract_id: 47
file: SingleMarketWillHillaryClintonWinNO20220111_USDC-4h.feather
```

The contract has a near-degenerate time series:

- Close prices are mostly repeated, with only 34 unique close values across
  5653 rows.
- Volume is zero in 5591 of 5653 rows.
- The close residual train variance is very small: `4.730274642294856e-05`.

Despite this low realised variation, the GARCH optimizer reports convergence
with the following parameters:

```text
omega = 411892.6649674495
alpha = 0.09990244634912097
beta  = 1.000000582817781e-08
```

For GARCH(1,1),

```text
sigma2_t = omega + alpha * eps2_{t-1} + beta * sigma2_{t-1}
```

With a very large `omega` and near-zero `beta`, the conditional variance is
approximately constant at `411892`, so the derived volatility target is:

```text
sqrt(411892) ~= 641.79
```

This exactly matches the observed corrupted GARCH target scale.

## Impact on the Dataset

On the full 4h test split:

```text
y_gt mean       = 0.0805900395
y_garch mean    = 26.1663475037
y_garch max     = 641.7886352539
GARCH corr      = -0.1154132618
GARCH MSE       = 16745.021484375
```

Excluding only contract `47`:

```text
y_gt mean       = 0.0837670416
y_garch mean    = 0.0778909996
GARCH corr      = 0.8715347838
GARCH MSE       = 0.0044977656
```

This shows that the global GARCH supervision failure is highly concentrated
but severe enough to dominate the fused training objective.

## Impact on GINN Training

GINN uses a fused loss:

```text
loss = (1 - lambda_garch) * MSE(pred, y_gt)
     + lambda_garch * MSE(pred, y_garch)
```

With `lambda_garch = 0.3`, a corrupted GARCH target of about `641.79` creates a
large gradient incentive for the LSTM to predict volatility values many orders
of magnitude larger than realised volatility. The 15-epoch characterization
runs reflect this:

| output_transform | MSE | Pearson corr | RMSE | negative prediction fraction |
|---|---:|---:|---:|---:|
| `linear` | 1887.2247 | 0.0488 | 43.4422 | 0.6284 |
| `softplus` | 1313.9036 | -0.0703 | 36.2478 | 0.0000 |

The `softplus` output transform removes invalid negative volatility
predictions, but it does not solve the target-scale problem. The softplus run
still predicts values much larger than the realised volatility target because
the corrupted GARCH supervision remains in the loss.

## Interpretation

The failure mode comes from the interaction between three properties:

1. Prediction-market prices are bounded probabilities, often with long flat
   intervals.
2. Some markets are sparsely traded, producing repeated OHLCV values and many
   zero-volume rows.
3. The current GARCH fitting accepts optimizer convergence without a
   scale-sanity check against realised volatility or residual variance.

In such cases, the optimizer can produce parameters that are mathematically
accepted by the objective but not meaningful for volatility supervision.

## Paper Framing

A suitable report statement:

> The GINN benchmark assumes that the AR-GARCH preprocessing stage produces a
> meaningful auxiliary volatility signal. In Polymarket contracts, some markets
> contain long flat probability intervals, sparse trading, and repeated OHLCV
> values. For these near-degenerate series, GARCH fitting can become unstable
> despite optimizer convergence, producing volatility estimates many orders of
> magnitude larger than realised volatility. This makes GINN sensitive to the
> sparse and bounded structure of prediction-market time series, limiting its
> reliability as a volatility benchmark without additional data-quality filters
> or robust GARCH fitting safeguards.

## Possible Mitigations

These mitigations should be treated as adaptations, not part of the original
paper-faithful baseline:

- Add a scale sanity check after GARCH fitting and fall back to rolling realised
  volatility or EWMA variance when the GARCH target is implausible.
- Bound `omega` relative to the training residual variance.
- Fit GARCH on standardized residuals and rescale the conditional variance
  afterward.
- Filter near-degenerate contracts using training-visible data only, such as
  minimum non-zero volume count, minimum number of close-price changes, or
  minimum realised-volatility threshold.
- Keep `softplus` as a non-negative output transform, but only after the GARCH
  supervision scale is corrected or documented as an adaptation.

The current project decision is to document this as a limitation of applying
GINN to Polymarket volatility prediction, rather than silently filtering the
problematic market or altering the GARCH target generation for the main
benchmark.
