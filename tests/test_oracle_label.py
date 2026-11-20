"""Oracle labels for the gate experiments (N3.2; CLAUDE.md §11.5, §10.4, B3).

CPU tests run the whole pipeline at small N (the solver path is the same at every N): PyAMG-GMRES
workers and a PCG worker on NumPy share one queue. The 50-board CPU-vs-GPU agreement at N = 256 is
a gpu test (rvcse A100).

Failure modes covered: a design labelled twice or never (dedupe across files, resume after a kill),
a label that does not match the design (x_hash check), inputs that changed under the labeller
(manifest hashes), two solvers disagreeing (cross-check, < 1e-4 K), a solve that stopped early
(every relres ≤ 1e-8, no certify import), and labels run_gates cannot read.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments import oracle_label as OL
from gate.proposers import DIM, sample_p0, x_hash

ROOT = Path(__file__).resolve().parents[1]
N_SMALL = 32


def _designs(n: int, seed: int) -> list[np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(seed))
    return [sample_p0(rng) for _ in range(n)]


def _table(xs: list[np.ndarray]) -> pa.Table:
    return pa.table({
        "x_hash": [x_hash(x) for x in xs],
        "design_vector": pa.array([list(map(float, x)) for x in xs], pa.list_(pa.float64(), DIM)),
    })


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make_inputs(root: Path, trajs: dict[str, list[np.ndarray]], calib: list[np.ndarray]) -> Path:
    """root/trajectories/<name>.parquet + manifest.json, root/calibration.parquet."""
    td = root / "trajectories"
    td.mkdir(parents=True)
    runs = []
    for name, xs in trajs.items():
        p = td / f"{name}.parquet"
        pq.write_table(_table(xs), p)
        runs.append({"file": p.name, "rows": len(xs), "sha256": _sha(p)})
    (td / "manifest.json").write_text(json.dumps({"schema": "twinheat-trajectories/1",
                                                  "runs": runs}), encoding="utf-8")
    pq.write_table(_table(calib), root / "calibration.parquet")
    return root


def _labels(root: Path) -> dict:
    return pq.read_table(root / "labels.parquet").to_pydict()


# ------------------------------------------------------------------------------ solvers agree

def test_amg_and_pcg_agree_on_50_boards_cpu():
    xs = _designs(50, 1)
    pcg_rows = OL.solve_pcg(xs, N=N_SMALL, device="cpu")
    worst = 0.0
    for x, rp in zip(xs, pcg_rows, strict=True):
        ra = OL.solve_amg(x, N=N_SMALL, strength_theta=OL.DEFAULT_STRENGTH_THETA)
        for r in (ra, rp):
            assert r["relres"] <= OL.ORACLE_TOL
            assert r["energy_balance"] <= OL.ENERGY_TOL
        worst = max(worst, abs(ra["oracle_peak"] - rp["oracle_peak"]))
    assert worst < OL.CROSS_TOL_K, worst


@pytest.mark.gpu
@pytest.mark.slow
def test_cpu_and_gpu_labels_agree_on_50_boards_at_256():
    xs = _designs(50, 2)
    n = OL.SPEC.geometry.N.value
    gpu = []
    for k in range(0, len(xs), OL.GPU_BATCH):
        gpu += OL.solve_pcg(xs[k:k + OL.GPU_BATCH], N=n, device="gpu")
    worst = 0.0
    for x, rg in zip(xs, gpu, strict=True):
        rc = OL.solve_amg(x, N=n, strength_theta=OL.DEFAULT_STRENGTH_THETA)
        assert rg["relres"] <= OL.ORACLE_TOL and rc["relres"] <= OL.ORACLE_TOL
        worst = max(worst, abs(rc["oracle_peak"] - rg["oracle_peak"]))
    print(f"max |dpeak| CPU vs GPU over 50 boards at N = {n}: {worst:.3e} K")
    assert worst < OL.CROSS_TOL_K, worst


def test_row_fields():
    x = _designs(1, 3)[0]
    r = OL.solve_amg(x, N=N_SMALL, strength_theta=None)
    assert r["solver"] == "pyamg_gmres" and r["device"] == "cpu" and r["batch"] == 1
    assert 0 <= r["argmax_layer"] < 4 and 0 <= r["argmax_y"] < N_SMALL
    assert 0 <= r["argmax_x"] < N_SMALL and len(r["theta_sha256"]) == 64
    assert r["iters"] >= 1 and r["wall_s"] > 0 and r["oracle_peak"] > 0


def test_no_certified_early_stopping():
    tree = ast.parse((ROOT / "experiments" / "oracle_label.py").read_text(encoding="utf-8"))
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not any(m == "certify" or m.startswith("certify.") for m in mods), mods


# --------------------------------------------------------------------------- the whole pipeline

def test_dedupe_across_files_and_labels_readable_by_run_gates(tmp_path):
    a, b, c = _designs(5, 4), _designs(4, 5), _designs(3, 6)
    root = make_inputs(tmp_path, {"traj_P1_a1_s0": a + b[:2], "traj_P2_a1_s0": b + a[:1]},
                       calib=c + b[3:])
    s = OL.label(root, N=N_SMALL, cpu_workers=2, pcg_workers=1, pcg_device="cpu", gpu_batch=3,
                 flush_every=2)
    assert s["status"] == "complete" and s["labelled_this_run"] == 12
    OL.finalise(root)
    d = _labels(root)
    assert len(d["x_hash"]) == len(set(d["x_hash"])) == 12
    src = dict(zip(d["x_hash"], d["source"], strict=True))
    assert all(src[x_hash(x)] == "traj_P1_a1_s0" for x in a + b[:2])
    assert all(src[x_hash(x)] == "traj_P2_a1_s0" for x in b[2:])
    assert all(src[x_hash(x)] == "calibration" for x in c)
    assert {"pyamg_gmres", "pcg_linez"} >= set(d["solver"])
    assert max(d["relres"]) <= OL.ORACLE_TOL
    from experiments.run_gates import Labels
    lab = Labels.from_parquet(root / "labels.parquet")
    assert set(lab.oracle) == set(d["x_hash"])
    # a second run has nothing to do
    s2 = OL.label(root, N=N_SMALL, cpu_workers=1, flush_every=2)
    assert s2["labelled_this_run"] == 0


def test_x_hash_mismatch_refused(tmp_path):
    xs = _designs(2, 7)
    root = make_inputs(tmp_path, {"traj_P1_a1_s0": xs}, calib=xs[:1])
    t = _table(xs).to_pydict()
    t["x_hash"][1] = x_hash(xs[0])
    p = root / "trajectories" / "traj_P1_a1_s0.parquet"
    pq.write_table(pa.table({"x_hash": t["x_hash"], "design_vector": pa.array(
        t["design_vector"], pa.list_(pa.float64(), DIM))}), p)
    man = json.loads((root / "trajectories" / "manifest.json").read_text(encoding="utf-8"))
    man["runs"][0]["sha256"] = _sha(p)
    (root / "trajectories" / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(OL.OracleError, match="x_hash"):
        OL.collect_designs(root)


def test_manifest_mismatch_refused(tmp_path):
    xs = _designs(2, 8)
    root = make_inputs(tmp_path, {"traj_P1_a1_s0": xs}, calib=xs[:1])
    pq.write_table(_table(xs[:1]), root / "trajectories" / "traj_P1_a1_s0.parquet")
    with pytest.raises(OL.OracleError, match="sha256"):
        OL.collect_designs(root)
    pq.write_table(_table(xs), root / "trajectories" / "traj_P9_a1_s0.parquet")
    with pytest.raises(OL.OracleError, match="manifest"):
        OL.collect_designs(root)


def test_duplicate_parts_and_gaps_refused(tmp_path):
    xs = _designs(4, 9)
    root = make_inputs(tmp_path, {"traj_P1_a1_s0": xs[:3]}, calib=xs[3:])
    OL.label(root, N=N_SMALL, cpu_workers=1, flush_every=1)
    parts = sorted((root / "oracle" / "parts").glob("*.parquet"))
    assert len(parts) == 4
    (root / "oracle" / "parts" / "stray.parquet.tmp").write_bytes(b"half-written")
    OL.finalise(root)  # a .tmp left by a kill is ignored
    dup = root / "oracle" / "parts" / "traj_P1_a1_s0.99999.parquet"
    dup.write_bytes(parts[0].read_bytes())
    with pytest.raises(OL.OracleError, match="twice"):
        OL.finalise(root)
    dup.unlink()
    parts[1].unlink()
    with pytest.raises(OL.OracleError, match="1 of 4 unique designs have no label"):
        OL.finalise(root)


def test_resume_after_kill_no_duplicates_or_gaps(tmp_path):
    xs = _designs(60, 10)
    root = make_inputs(tmp_path, {"traj_P1_a1_s0": xs[:40], "traj_P2_a1_s0": xs[30:55]},
                       calib=xs[55:])
    cmd = [sys.executable, "-m", "experiments.oracle_label", "label", "--inputs", str(root),
           "--N", "64", "--cpu-workers", "2", "--flush-every", "3"]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    parts = root / "oracle" / "parts"
    t0 = time.monotonic()
    while not (parts.exists() and len(list(parts.glob("*.parquet"))) >= 2):
        assert proc.poll() is None, proc.stderr.read().decode()
        assert time.monotonic() - t0 < 180, "no part written in 180 s"
        time.sleep(0.05)
    proc.kill()
    proc.wait()
    done = sum(pq.read_metadata(p).num_rows for p in parts.glob("*.parquet"))
    assert 0 < done < 60, f"the kill did not interrupt the run ({done} of 60 labelled)"
    with pytest.raises(OL.OracleError, match="no label"):
        OL.finalise(root)
    s = OL.label(root, N=64, cpu_workers=2, flush_every=3)
    assert s["labelled_this_run"] == 60 - done
    OL.finalise(root)
    d = _labels(root)
    assert sorted(d["x_hash"]) == sorted({x_hash(x) for x in xs})
    peak = dict(zip(d["x_hash"], d["oracle_peak"], strict=True))
    for x in xs[::12]:  # the resumed labels equal fresh solves (to rounding: the workers run
        # one BLAS thread, this process may run several, so bits can differ)
        r = OL.solve_amg(x, N=64, strength_theta=OL.DEFAULT_STRENGTH_THETA)
        assert abs(peak[x_hash(x)] - r["oracle_peak"]) < 1e-9


def test_lock_refuses_a_second_labeller(tmp_path):
    with OL.RunLock(tmp_path / ".lock"):
        with pytest.raises(OL.OracleError, match="another labeller"):
            with OL.RunLock(tmp_path / ".lock"):
                pass
    with OL.RunLock(tmp_path / ".lock"):  # released on exit
        pass


# ------------------------------------------------------------------------ cross-check, report

def _labelled(tmp_path, n=6) -> Path:
    xs = _designs(n, 11)
    root = make_inputs(tmp_path, {"traj_P1_a1_s0": xs[:-2]}, calib=xs[-2:])
    OL.label(root, N=N_SMALL, cpu_workers=1, pcg_workers=1, pcg_device="cpu", gpu_batch=2,
             flush_every=2)
    OL.finalise(root)
    return root


def test_cross_check_passes_and_uses_the_other_solver(tmp_path):
    root = _labelled(tmp_path)
    r = OL.cross_check(root, fraction=1.0, seed=0, cpu_workers=1, pcg_workers=1,
                       pcg_device="cpu", gpu_batch=4)
    assert r["passed"] and r["n_checked"] == 6 and r["max_dpeak_K"] < OL.CROSS_TOL_K
    c = pq.read_table(root / "oracle" / "cross_check.parquet").to_pydict()
    assert all(a != b for a, b in zip(c["label_solver"], c["check_solver"], strict=True))


def test_cross_check_sample_size_and_failure(tmp_path):
    assert OL.sample_size(100_000, OL.CROSS_FRACTION) == 500
    assert OL.sample_size(10, OL.CROSS_FRACTION) == 1
    root = _labelled(tmp_path)
    t = pq.read_table(root / "labels.parquet").to_pydict()
    t["oracle_peak"][0] += 1e-3  # a label off by 1 mK must be caught
    meta = pq.read_schema(root / "labels.parquet").metadata
    pq.write_table(pa.Table.from_pydict(t, schema=pq.read_schema(root / "labels.parquet"))
                   .replace_schema_metadata(meta), root / "labels.parquet")
    with pytest.raises(OL.OracleError, match="cross-check failed"):
        OL.cross_check(root, fraction=1.0, seed=0, cpu_workers=1, pcg_workers=1,
                       pcg_device="cpu", gpu_batch=4)
    assert not json.loads((root / "oracle" / "cross_check.json").read_text())["passed"]


def test_report(tmp_path):
    root = _labelled(tmp_path)
    OL.cross_check(root, fraction=1.0, seed=0, cpu_workers=1, pcg_workers=1,
                   pcg_device="cpu", gpu_batch=4)
    out = OL.write_report(root, tmp_path / "oracle_report.md")
    text = out.read_text(encoding="utf-8")
    for s in ("Unique designs | 6", "Labelled | 6", "Cross-check | PASS", "boards/s",
              "CPU core-hours", "GPU-hours", "commit"):
        assert s in text, s
    assert "Status: DONE" in text
    assert not math.isnan(json.loads((root / "oracle" / "cross_check.json").read_text())[
        "max_dpeak_K"])
