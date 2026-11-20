"""Label-free groups g_t (M3.3; CLAUDE.md §8.1 step 2, §8.5 G6).

A design belongs to one group of each family, so groups intersect:
  proposer=P1..P4              who proposed it
  aggr=1..3                    the proposer's aggressiveness level (P4 has none, so no aggr group)
  ood=1..3                     tercile of its OOD score (models/ood_score.py); the cut points are
                               the 1/3 and 2/3 percentiles of the validation split's scores, fitted
                               once by experiments/run_trajectories.py terciles and stored with the
                               SHA-256 of the ood_score.json they belong to
  edge=0|1                     the predicted hotspot (argmax of μ over all layers) lies within
                               EDGE_MM = 5 mm (§8.1) of a board edge, measured from the cell centre
Everything here is computed from the design, the proposer and the prediction, before the audit
coin exists (B8). A design's groups travel as one string, "|"-joined (the §10.4 group column and
GateInput.group); membership() splits it for G6 (gate/multivalid.py).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from gate.base import GateInput

__all__ = ["ALL_GROUPS", "EDGE_MM", "encode", "group_names", "hotspot_near_edge", "load_terciles",
           "membership", "ood_tercile", "ood_tercile_cuts", "save_terciles"]

EDGE_MM = 5.0  # CLAUDE.md §8.1: "hotspot within 5 mm of an edge"
BOARD_MM = 100.0  # CLAUDE.md §6.3, L = 0.100 m
PROPOSERS = ("P1", "P2", "P3", "P4")
LEVELS = (1, 2, 3)
TERCILES = (1, 2, 3)
SEP = "|"
ALL_GROUPS = ([f"proposer={p}" for p in PROPOSERS] + [f"aggr={a}" for a in LEVELS]
              + [f"ood={k}" for k in TERCILES] + ["edge=0", "edge=1"])
SCHEMA = "twinheat-ood-terciles/1"


def ood_tercile_cuts(val_scores: Sequence[float] | np.ndarray) -> tuple[float, float]:
    """The 1/3 and 2/3 percentiles of the validation OOD scores."""
    v = np.asarray(val_scores, dtype=np.float64).reshape(-1)
    if v.size < 3 or not np.isfinite(v).all():
        raise ValueError(f"need >= 3 finite validation scores, got {v.size}")
    lo, hi = (float(c) for c in np.percentile(v, [100 / 3, 200 / 3]))
    return lo, hi


def ood_tercile(score: float, cuts: tuple[float, float]) -> int:
    score = float(score)
    if not math.isfinite(score):
        raise ValueError(f"OOD score must be finite, got {score}")
    lo, hi = cuts
    return 1 if score <= lo else 2 if score <= hi else 3


def hotspot_near_edge(cell: tuple[int, int, int], n: int, board_mm: float = BOARD_MM) -> bool:
    """True when the centre of cell (layer, y, x) is within EDGE_MM of the nearest edge."""
    _, y, x = cell
    if not (0 <= y < n and 0 <= x < n):
        raise ValueError(f"cell {cell} is outside an {n} x {n} grid")
    h = board_mm / n
    cx, cy = (x + 0.5) * h, (y + 0.5) * h
    return bool(min(cx, cy, board_mm - cx, board_mm - cy) <= EDGE_MM)


def group_names(proposer: str, aggressiveness: int | None, ood_tercile: int,
                near_edge: bool) -> list[str]:
    if proposer not in PROPOSERS:
        raise ValueError(f"proposer must be one of {PROPOSERS}, got {proposer!r}")
    if aggressiveness is not None and aggressiveness not in LEVELS:
        raise ValueError(f"aggressiveness must be one of {LEVELS} or None, got {aggressiveness!r}")
    if ood_tercile not in TERCILES:
        raise ValueError(f"ood_tercile must be one of {TERCILES}, got {ood_tercile!r}")
    names = [f"proposer={proposer}"]
    if aggressiveness is not None:
        names.append(f"aggr={aggressiveness}")
    return names + [f"ood={ood_tercile}", f"edge={int(bool(near_edge))}"]


def encode(names: Sequence[str]) -> str:
    bad = [g for g in names if g not in ALL_GROUPS]
    if bad or not names:
        raise ValueError(f"unknown or empty groups {bad or names!r}")
    return SEP.join(names)


def membership(inp: GateInput) -> list[str]:
    """G6's membership function: the groups encoded in GateInput.group."""
    if not inp.group:
        raise ValueError(f"design {inp.design_id!r} has no group string")
    names = inp.group.split(SEP)
    encode(names)  # validates
    return names


def make_grouper(cuts: tuple[float, float]):
    """grouper(prediction, proposer) -> group string, for gate/proposers.py's loops."""
    def grouper(pred, proposer) -> str:
        return encode(group_names(proposer.name, proposer.level,
                                  ood_tercile(pred.ood_score, cuts), pred.edge_hotspot))
    return grouper


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_terciles(path: Path, cuts: tuple[float, float], *, n_val: int, ood_score_path: Path,
                  dataset_hash: str, commit: str) -> None:
    lo, hi = cuts
    if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
        raise ValueError(f"tercile cuts must be finite and ordered, got {cuts}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({
        "schema": SCHEMA, "cuts": [lo, hi], "percentiles": [100 / 3, 200 / 3], "split": "val",
        "n_val": int(n_val), "ood_score_sha256": _sha256(ood_score_path),
        "dataset_hash": dataset_hash, "commit": commit}, indent=2) + "\n", encoding="utf-8")


def load_terciles(path: Path, *, ood_score_path: Path) -> tuple[float, float]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {d.get('schema')!r}, expected {SCHEMA}")
    if d["ood_score_sha256"] != _sha256(ood_score_path):
        raise ValueError(f"{path}: fitted for another ood_score file than {ood_score_path}")
    lo, hi = (float(c) for c in d["cuts"])
    return lo, hi
