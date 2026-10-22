"""Minimal stand-in for the whestbench contract types (client-side harness only)."""
from dataclasses import dataclass
from typing import Optional


class BaseEstimator:
    def setup(self, context):
        pass

    def predict(self, mlp, budget):
        raise NotImplementedError

    def teardown(self):
        pass


@dataclass
class SetupContext:
    width: int
    depth: int
    flop_budget: int
    api_version: str = "1"
    scratch_dir: Optional[str] = None
    submission_dir: Optional[str] = None
    seed: int = 0
