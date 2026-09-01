"""Greedy, transparent non-redundant factor selection."""
import numpy as np


def select_non_redundant(candidates, scores, *, max_factors=10, correlation_threshold=0.9):
    ranked = sorted(candidates, key=lambda n: abs(scores[n].rank_ic), reverse=True)
    selected = []
    def corr(a, b):
        if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    for name in ranked:
        if len(selected) >= max_factors: break
        x = np.asarray(candidates[name], float)
        if all(abs(corr(x, np.asarray(candidates[other], float))) < correlation_threshold
               for other in selected):
            selected.append(name)
    return selected
