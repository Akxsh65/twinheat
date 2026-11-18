"""Label-free groups (M3.3; CLAUDE.md §8.1 step 2, §8.5): proposer, aggressiveness, OOD-score
tercile (cut points from validation scores) and hotspot within 5 mm of an edge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gate import groups as G
from gate.base import GateInput
from gate.coin import AuditCoin
from gate.multivalid import MultiValidGate


def test_tercile_cuts_split_validation_scores_into_thirds() -> None:
    v = np.random.default_rng(0).exponential(size=3000)
    cuts = G.ood_tercile_cuts(v)
    assert cuts == (pytest.approx(np.percentile(v, 100 / 3)),
                    pytest.approx(np.percentile(v, 200 / 3)))
    t = np.array([G.ood_tercile(s, cuts) for s in v])
    for k in (1, 2, 3):
        assert abs(np.mean(t == k) - 1 / 3) < 0.01
    with pytest.raises(ValueError):
        G.ood_tercile_cuts(np.array([1.0]))
    with pytest.raises(ValueError):
        G.ood_tercile(float("nan"), cuts)


def test_hotspot_near_edge_uses_the_cell_centre_and_5_mm() -> None:
    n = 256  # 0.390625 mm cells
    assert G.EDGE_MM == 5.0
    # cell 12: centre (12.5) * 0.390625 = 4.88 mm from the west edge -> near; cell 13: 5.27 mm
    assert G.hotspot_near_edge((0, 128, 12), n) is True
    assert G.hotspot_near_edge((0, 128, 13), n) is False
    assert G.hotspot_near_edge((2, n - 13, 128), n) is True  # north edge, any layer
    assert G.hotspot_near_edge((3, 128, 128), n) is False


def test_group_names_and_membership_round_trip() -> None:
    names = G.group_names(proposer="P1", aggressiveness=3, ood_tercile=2, near_edge=True)
    assert names == ["proposer=P1", "aggr=3", "ood=2", "edge=1"]
    assert G.group_names("P4", None, 1, False) == ["proposer=P4", "ood=1", "edge=0"]
    s = G.encode(names)
    gi = GateInput(design_id="d0", x_hash=hashlib.sha256(b"d0").hexdigest(), peak_hat=40.0,
                   s=0.5, r_hot=0.1, sd_hot=0.1, proposer="P1", aggressiveness=3, group=s)
    assert G.membership(gi) == names
    assert set(names) <= set(G.ALL_GROUPS)
    assert len(G.ALL_GROUPS) == len(set(G.ALL_GROUPS)) == 4 + 3 + 3 + 2
    with pytest.raises(ValueError):
        G.group_names("P9", 1, 1, False)
    with pytest.raises(ValueError):
        G.group_names("P1", 2, 4, False)


def test_g6_accepts_the_group_family() -> None:
    gate = MultiValidGate(alpha=0.1, tau=1.0, p=0.1, groups=G.ALL_GROUPS, m=10, r=2, eps=0.5,
                          noise_width=0.01, seed=0, membership=G.membership,
                          coin=AuditCoin(seed=0))
    gi = GateInput(design_id="d0", x_hash=hashlib.sha256(b"d0").hexdigest(), peak_hat=40.0,
                   s=0.5, r_hot=0.1, sd_hot=0.1, proposer="P2", aggressiveness=1,
                   group=G.encode(G.group_names("P2", 1, 3, False)))
    assert gate.decide(gi).decision in ("accept", "flag")


def test_terciles_file_round_trips_and_refuses_a_different_ood_score(tmp_path: Path) -> None:
    ood = tmp_path / "ood_score.json"
    ood.write_text(json.dumps({"schema": "twinheat-ood-score/1", "threshold": 1.0}), "utf-8")
    path = tmp_path / "terciles.json"
    G.save_terciles(path, (1.0, 2.0), n_val=900, ood_score_path=ood, dataset_hash="ab" * 32,
                    commit="abc1234")
    assert G.load_terciles(path, ood_score_path=ood) == (1.0, 2.0)
    ood.write_text(json.dumps({"schema": "twinheat-ood-score/1", "threshold": 2.0}), "utf-8")
    with pytest.raises(ValueError, match="ood_score"):
        G.load_terciles(path, ood_score_path=ood)
