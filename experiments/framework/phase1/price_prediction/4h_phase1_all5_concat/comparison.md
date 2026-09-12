# Phase-1 Price Prediction Baseline Comparison

The Raw-OHLCV MLP comparison is strict: both runs use the same processed split, price target builder, seed, budgets, and 27,499 test rows. The LSTM is external context only because its close-only artifact uses 27,500 rows and has a documented one-row target-alignment difference.

| epoch | Phase-1 MAE | MLP MAE | Phase-1 MAE improvement | Phase-1 RMSE | MLP RMSE | Phase-1 RMSE improvement | LSTM MAE (context) | LSTM RMSE (context) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 | 0.051227 | 0.083424 | +38.6% | 0.090831 | 0.106712 | +14.9% | 0.008900 | 0.016600 |
| 50 | 0.065077 | 0.060006 | -8.5% | 0.099259 | 0.080469 | -23.4% | 0.011600 | 0.018500 |
| 100 | 0.067843 | 0.045375 | -49.5% | 0.100879 | 0.068346 | -47.6% | 0.009300 | 0.016700 |

Positive improvement means the Phase-1 framework has lower error than the MLP; negative means higher error. Do not select an epoch from these locked-test results.

## Experiment summary

This run is the Phase-1 five-branch feature probe for one-step-ahead close-price prediction. The frozen representation contains `statistical` (70 dimensions), `transformed` (55), `vae` (64), `contrastive` (128), and `byol` (128), for 445 concatenated features. A train-fitted standardizer and the default `PriceRegressor` head were used; the representation branches were not updated during supervised training.

The comparison uses the chronological per-contract 80/20 split from `data/processed/market_4h_seq64_top50.npz`, with 109,840 training rows and 27,499 locked test rows. Both the Phase-1 framework and Raw-OHLCV MLP use the same processed data, price target, horizon, seed (`0`), epoch budgets, and regression metrics. Consequently, the MLP rows are a strict internal comparison. The LSTM values are retained as context only because the stored close-only LSTM artifact has a one-row target-alignment difference and 27,500 test rows.

## Detailed results

At 15 epochs, Phase-1 has the best observed framework result: MAE `0.051227`, RMSE `0.090831`, and MSE `0.008250`. Relative to the matched Raw-OHLCV MLP, this is a 38.6% lower MAE, 14.9% lower RMSE, and 27.6% lower MSE. This is the only tested budget at which the framework beats the MLP on all three error measures.

At 50 epochs, the MLP is stronger: its MAE is lower by `0.005071` (8.5%), RMSE by `0.018790` (23.4%), and MSE by `0.003377` (52.2%). At 100 epochs, the gap widens: the MLP is lower by `0.022468` MAE (49.5%), `0.032532` RMSE (47.6%), and `0.005505` MSE (54.1%). The MLP therefore improves monotonically over this sweep, while the framework reaches its lowest test error at 15 epochs and degrades at longer budgets.

Pearson correlation should be read separately from pointwise error. The MLP has higher correlation at every budget (`0.9952`, `0.9942`, `0.9927`) than the framework (`0.9794`, `0.9786`, `0.9784`), even where the framework has lower MAE and RMSE at 15 epochs. This indicates that correlation alone is not sufficient to judge forecast accuracy here; the framework's predictions track the direction/shape reasonably well but have larger pointwise deviations.

The framework's final training loss falls from `0.000997` at 15 epochs to `0.000388` at 100 epochs, while its locked-test MAE/RMSE rise. This train--test divergence is consistent with overfitting, representation/head optimization mismatch, or a feature scale/conditioning issue. It is not evidence that the 100-epoch model is better, and no budget should be selected from these test results after the fact.

## Interpretation

The result supports a limited claim: the frozen five-branch representation can provide useful price-prediction features, and at the short 15-epoch budget it outperforms the matched Raw-OHLCV MLP. It does not support a general claim that Phase-1 is superior to the MLP, because the advantage disappears by 50 epochs and reverses substantially by 100 epochs. It also does not establish superiority over the LSTM; the LSTM is a context benchmark with a different row alignment and close-only input contract.

The Phase-1 versus MVP difference should be interpreted as a new representation configuration, not as a controlled BYOL ablation. Phase-1 adds the 128-dimensional BYOL branch to the earlier four-branch MVP store, while also using the same simple supervised head and fixed concat aggregation. Since there is no paired Phase-1 run with BYOL removed in this report, the individual contribution of BYOL cannot be identified from these numbers. A direct claim about BYOL requires a matched five-branch-minus-BYOL ablation using the same split, seed, budgets, standardization, and head.

## Limitations and next actions

- This is a single-seed (`0`) characterization sweep; variability across seeds is unknown.
- The Raw-OHLCV MLP is a strict data/target comparison, but its decoder implementation and optimization configuration are not identical to the framework head, so the result is an internal benchmark rather than a pure representation-only causal test.
- The LSTM comparison should remain contextual until it is refit to the processed-NPZ/shared-target contract.
- Before making a final Phase-1 performance claim, run the matched no-BYOL ablation, inspect per-contract and residual/error distributions, and repeat the shortlisted predeclared budget over multiple seeds. Keep all budgets in the report and choose any final operating point by a training-only rule or by fixing the budget in advance.
