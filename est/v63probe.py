"""s20 probe: minimal compliant estimator (zero predictions).

Purpose: account/infra diagnostic after 4 consecutive "Evaluation error" failures
(3 free-cut variants + raw s10a re-control) while other participants graded normally
in the same window. This estimator performs no flopscope operations: residual time
~0, wall time ~0, FLOPs 0 -> multiplier floor 0.1. Score will be poor by design;
the only question is whether it GRADES.
"""
import numpy as np

from whestbench import BaseEstimator


class Estimator(BaseEstimator):
    def setup(self, ctx) -> None:
        self.width = int(getattr(ctx, "width", 1024))
        self.depth = int(getattr(ctx, "depth", 16))

    def predict(self, mlp, flop_budget):
        return np.zeros((self.depth, self.width), dtype=np.float32)
