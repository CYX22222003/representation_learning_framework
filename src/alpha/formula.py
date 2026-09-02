"""Small, interpretable, protected formula grammar for alpha prototyping."""
from dataclasses import dataclass
import numpy as np


def _rank(x):
    order = np.argsort(x, kind="mergesort")
    out = np.empty(len(x), dtype=float); out[order] = np.arange(len(x), dtype=float)
    return out / max(1, len(x) - 1)


@dataclass(frozen=True)
class Expression:
    op: str
    children: tuple = ()
    name: str | None = None

    @property
    def depth(self):
        return 1 if not self.children else 1 + max(c.depth for c in self.children)

    def evaluate(self, primitives, rolling_window=5):
        if self.op == "primitive": return np.asarray(primitives[self.name], dtype=float)
        x = self.children[0].evaluate(primitives, rolling_window)
        if self.op == "neg": return -x
        if self.op == "abs": return np.abs(x)
        if self.op == "rank": return _rank(x)
        if self.op == "delay": return np.concatenate(([x[0]] * rolling_window, x[:-rolling_window])) if len(x) > rolling_window else x
        if self.op == "rolling_mean":
            if rolling_window < 1: raise ValueError("rolling_window must be positive")
            cs = np.concatenate(([0.0], np.cumsum(x)))
            out = np.empty_like(x)
            for i in range(len(x)):
                start = max(0, i + 1 - rolling_window)
                out[i] = (cs[i + 1] - cs[start]) / (i + 1 - start)
            return out
        y = self.children[1].evaluate(primitives, rolling_window)
        if self.op == "add": return x + y
        if self.op == "sub": return x - y
        if self.op == "mul": return x * y
        if self.op == "div": return x / np.where(np.abs(y) < 1e-8, np.sign(y) * 1e-8 + (y == 0) * 1e-8, y)
        raise ValueError(f"unknown operator {self.op}")

    def to_string(self):
        if self.op == "primitive": return self.name or "?"
        if len(self.children) == 1: return f"{self.op}({self.children[0].to_string()})"
        return f"({self.children[0].to_string()} {self.op} {self.children[1].to_string()})"


def candidate_formulas(primitive_names, max_depth=2):
    leaves = [Expression("primitive", name=n) for n in primitive_names]
    formulas = list(leaves)
    if max_depth >= 2:
        formulas += [Expression(op, (x,)) for x in leaves for op in ("neg", "abs", "rank")]
        formulas += [Expression(op, (a, b)) for a in leaves for b in leaves for op in ("add", "sub", "mul", "div")]
    return formulas
