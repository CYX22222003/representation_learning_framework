# Framework trend classification sweep

All epoch budgets are reported as a characterization sweep; no checkpoint is selected from test performance.

| epoch | accuracy | macro_f1 | weighted_f1 | train_loss |
|---:|---:|---:|---:|---:|
| 15 | 0.4730291963 | 0.3874601574 | 0.4529548612 | 0.4115946293 |
| 50 | 0.4952554703 | 0.4164474747 | 0.4812287516 | 0.3890349567 |
| 100 | 0.5071167946 | 0.4231608198 | 0.4900122401 | 0.3701592982 |

See `comparison.md` for majority-HOLD context, TA-MLP benchmark context, and
the remaining row-alignment caveat before making final comparison claims.
