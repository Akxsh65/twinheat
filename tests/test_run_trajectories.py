"""experiments/run_trajectories.py (M3.3): parquet trajectories with the §10.4 columns minus
oracle_peak, reproducible from the seed, on a tiny ensemble at a small grid."""

from __future__ import annotations

import copy
import json
import re

import numpy as np
import pyarrow.parquet as pq
import pytest
from test_proposers import tiny_ensemble  # noqa: E402  (tests/ is on sys.path under pytest)

from contracts.validate import validate
from data.sampler import design_board
from experiments import run_trajectories as RT
from gate import proposers as P
from models.diff_inputs import DesignInputs
from models.ood_score import OODScore

AUDIT_COLS = ["run_id", "gate", "seed", "t", "proposer", "aggressiveness", "group", "x_hash", "s",
              "r_hot", "sd_hot", "q_t", "B", "decision", "p_t", "audited", "T_lo", "T_hi",
              "S_hi", "err", "alpha_t", "iters_used"]  # §10.4 minus oracle_peak


@pytest.fixture(scope="module")
def setup():
    cfg = copy.deepcopy(RT.load_config(RT.DEFAULT_CONFIG))
    cfg.update(N=32, m_hot=16, eval_batch=8)
    cfg["P1"]["proposals"] = 6
    cfg["P2"]["steps"] = 6
    cfg["P3"].update(starts=3, steps=2, grad_chunk=2)
    ens = tiny_ensemble()
    ood = OODScore([np.zeros(8)] * 2, [np.eye(8)] * 2, threshold=1.0)
    oracle = P.GradientOracle(ens, DesignInputs(RT.SPEC, n=32), chunk=2)
    return cfg, ens, ood, oracle


def _run(setup, name, out, seed=0, level=2):
    cfg, ens, ood, oracle = setup
    return RT.run_one(cfg, name, level, seed, out, ensemble=ens, ood=ood, cuts=(0.5, 2.0),
                      commit="abc1234", oracle=oracle)


@pytest.mark.parametrize("name,rows", [("P1", 6), ("P3", 6)])
def test_trajectory_file_has_the_audit_columns_and_null_gate_fields(setup, tmp_path, name, rows):
    entry = _run(setup, name, tmp_path)
    t = pq.read_table(tmp_path / entry["file"])
    assert entry["rows"] == t.num_rows == rows
    cols = t.column_names
    assert set(AUDIT_COLS) <= set(cols) and "oracle_peak" not in cols
    d = t.to_pylist()
    for k, row in enumerate(d):
        assert row["t"] == k and row["proposer"] == name and row["aggressiveness"] == 2
        assert re.fullmatch(r"^.+-[0-9a-f]{7,40}-[0-9]+$", row["run_id"])
        for c in RT._NULL_GATE_COLS:
            assert row[c] is None, c
        x = np.array(row["design_vector"])
        assert P.x_hash(x) == row["x_hash"]
        design_board(x)  # N3.2 rebuilds the board from the stored vector
        assert row["group"].startswith(f"proposer={name}|aggr=2|ood=")
        assert row["ood_tercile"] in (1, 2, 3) and isinstance(row["edge_hotspot"], bool)
        assert np.isfinite(row["peak_hat"]) and row["r_hot"] >= 0 and row["sd_hot"] >= 0
    meta = json.loads(t.schema.metadata[b"twinheat"])
    assert meta["proposer"] == name and meta["commit"] == "abc1234"


def test_filled_rows_validate_as_audit_log_rows(setup, tmp_path) -> None:
    # once a gate fills its columns (run_gates), every row is a §10.4 row
    entry = _run(setup, "P2", tmp_path)
    for row in pq.read_table(tmp_path / entry["file"]).to_pylist():
        full = {c: row[c] for c in ["schema_version", *AUDIT_COLS]}
        full.update(gate="G0", s=row["r_hot"] + row["sd_hot"] + 1e-6, decision="accept",
                    p_t=0.0, audited=False, iters_used=0, oracle_peak=None)
        validate(full, "audit_log_row")


def test_same_seed_same_file_and_another_seed_differs(setup, tmp_path) -> None:
    a = _run(setup, "P1", tmp_path / "a")
    b = _run(setup, "P1", tmp_path / "b")
    c = _run(setup, "P1", tmp_path / "c", seed=1)
    ta = pq.read_table(tmp_path / "a" / a["file"]).to_pylist()
    tb = pq.read_table(tmp_path / "b" / b["file"]).to_pylist()
    tc = pq.read_table(tmp_path / "c" / c["file"]).to_pylist()
    assert ta == tb
    assert [r["x_hash"] for r in ta] != [r["x_hash"] for r in tc]


def test_levels_and_proposers_get_independent_streams() -> None:
    seeds = {RT.proposer_seed(0, n, lv) for n in RT.NAMES for lv in RT.LEVELS}
    assert len(seeds) == 9


def test_manifest_lists_files_with_hashes(setup, tmp_path) -> None:
    entries = [_run(setup, "P1", tmp_path), _run(setup, "P2", tmp_path)]
    path = RT.write_manifest(tmp_path, entries, RT.DEFAULT_CONFIG, "abc1234")
    m = json.loads(path.read_text("utf-8"))
    assert [r["file"] for r in m["runs"]] == sorted(e["file"] for e in entries)
    assert m["total_rows"] == sum(e["rows"] for e in entries)
    for r in m["runs"]:
        assert r["sha256"] == RT._sha256(tmp_path / r["file"])


def test_config_needs_three_levels_per_proposer(tmp_path) -> None:
    cfg = RT.load_config(RT.DEFAULT_CONFIG)
    cfg["P3"]["eta"] = [0.01, 0.03]
    bad = tmp_path / "bad.yaml"
    import yaml
    bad.write_text(yaml.safe_dump(cfg), "utf-8")
    with pytest.raises(RT.TrajectoryError, match="3 levels"):
        RT.load_config(bad)


def test_shipped_config_matches_the_prompt_and_decisions() -> None:
    cfg = RT.load_config(RT.DEFAULT_CONFIG)
    assert cfg["P1"]["lambda_per_K"] == [1.0, 3.0, 10.0] and cfg["P1"]["step"] == 0.05
    assert cfg["P2"]["steps"] == 1000
    assert (cfg["P3"]["starts"], cfg["P3"]["steps"]) == (64, 50)
    assert cfg["seeds"] == [0, 1, 2, 3, 4] and cfg["m_hot"] == 64 and cfg["N"] == 256


def test_calibration_set_is_p0_draws_in_the_run_gates_format(setup, tmp_path) -> None:
    from experiments import run_gates as RG

    cfg, ens, ood, _ = setup
    out = tmp_path / "calibration.parquet"
    e = RT.run_calibration(cfg, 5, 2026, out, ensemble=ens, ood=ood, commit="abc1234")
    assert e["rows"] == 5
    d = pq.read_table(out).to_pydict()
    for x, h in zip(d["design_vector"], d["x_hash"], strict=True):
        assert P.x_hash(np.array(x)) == h
        design_board(np.array(x))  # feasible: drawn from p0
    labels = RG.Labels({h: 40.0 for h in d["x_hash"]}, {})
    cal = RG.Calibration.from_parquet(out, labels)
    assert cal.design_vector.shape == (5, P.DIM)
    again = tmp_path / "again.parquet"
    RT.run_calibration(cfg, 5, 2026, again, ensemble=ens, ood=ood, commit="abc1234")
    assert pq.read_table(again).to_pydict()["x_hash"] == d["x_hash"]
