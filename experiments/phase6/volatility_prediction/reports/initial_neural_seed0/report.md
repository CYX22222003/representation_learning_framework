# Phase 6 H=8 Initial Neural Volatility Report

Epoch 50 is the predeclared principal snapshot. This is an interim target-validity report, not model selection: ten temporal/control configurations remain pending mandatory entries. GARCH--LSTM is deferred to a later round and is not part of the active matrix.

## By walk

| Walk | Model | Kind | MAE | RMSE | Pearson | Spearman |
|---:|---|---|---:|---:|---:|---:|
| 1 | Canonical framework H0 | learned | 0.0006261896 | 0.025583491 | 0.028055832 | 0.48291385 |
| 1 | Exact zero | non_learned_reference | 0.00066157896 | 0.025595147 | nan | nan |
| 1 | Training median | non_learned_reference | 0.00066994841 | 0.025594413 | nan | nan |
| 1 | Historical persistence | non_learned_reference | 0.0010663084 | 0.035566885 | 0.033861437 | 0.55861871 |
| 1 | Raw-OHLCV MLP | learned | 0.00062743726 | 0.025545443 | 0.12454981 | 0.48682557 |
| 1 | Raw LSTM | learned | 0.00066690178 | 0.025585498 | 0.025511805 | 0.46529823 |
| 2 | Canonical framework H0 | learned | 0.00050204011 | 0.0098587248 | 0.0016464696 | 0.57134454 |
| 2 | Exact zero | non_learned_reference | 0.00051292952 | 0.0098601164 | nan | nan |
| 2 | Training median | non_learned_reference | 0.0005113091 | 0.0098596014 | nan | nan |
| 2 | Historical persistence | non_learned_reference | 0.00056671967 | 0.010025514 | 0.0051886723 | 0.61675031 |
| 2 | Raw-OHLCV MLP | learned | 0.0004963727 | 0.0098551261 | 0.021014483 | 0.57844874 |
| 2 | Raw LSTM | learned | 0.00054072237 | 0.009814025 | 0.11168934 | 0.55514002 |

## Pooled descriptive view

Walks are concatenated only for descriptive scale. Walk-specific rows above remain the scientific comparison.

| Model | Kind | MAE | RMSE | Pearson | Spearman |
|---|---|---:|---:|---:|---:|
| Canonical framework H0 | learned | 0.00058851341 | 0.022031398 | 0.023831843 | 0.52455648 |
| Exact zero | non_learned_reference | 0.00061646767 | 0.022041015 | nan | nan |
| Training median | non_learned_reference | 0.00062180545 | 0.022040352 | 0.0031019243 | -0.0039526579 |
| Historical persistence | non_learned_reference | 0.000914696 | 0.030192876 | 0.032869213 | 0.57897944 |
| Raw-OHLCV MLP | learned | 0.00058766253 | 0.022000136 | 0.10358777 | 0.50981877 |
| Raw LSTM | learned | 0.00062860956 | 0.022026965 | 0.036833931 | 0.50851588 |

All values are in raw future realised-variance units. Correlation alone is not evidence of usefulness; learned rows must be interpreted against historical persistence and the zero-inflated, heavy-tailed target.
