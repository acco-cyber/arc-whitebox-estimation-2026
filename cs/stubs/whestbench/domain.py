from dataclasses import dataclass
from typing import Any


@dataclass
class MLP:
    width: int
    depth: int
    weights: Any
    seed: int = 0
    name: str = ""
