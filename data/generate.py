"""Dataset generator: every split, reproducible, resumable, verifiable (N2.1; CLAUDE.md §7.4).

    python -m data.generate --split {train,val,cal,test,ood_<name>} --seed S --out DIR
                            [--count C] [--batch 32] [--N 256] [--shard-size 500] [--device auto]

Board i of a split has seed SHA-256(split, seed, i) -> 64-bit int (board_seed). Its mode is
"design" (x ~ U[0, 1]^36 drawn from that seed, data/sampler.py) for even i of --split train and
"id" for odd i; val, cal and test are "id"; ood_<name> uses that OOD mode. Pipeline per board:
sample -> rasterise -> coefficients -> batched_pcg (line-z block Jacobi, relative residual
1e-10, CLAUDE.md §6.4; float64) -> energy balance <= 1e-3 (else GenerateError naming the seed)
-> envelope (§6.5: peak > 100 K scales every P_i and θ; no re-solve).

Output: DIR/shard_XXXX/records.jsonl (one JSON object per board, sorted keys: i, seed, record
after the envelope, scaled, iters, energy_balance, record_sha256) and fields.npy (float16
[n, 4, N, N], C order), written to shard_XXXX.tmp and renamed into place; DIR/manifest.json is
rewritten atomically after each shard (parameters, commit, spec-file hashes, tolerance, SHA-256
per file, running rescale fraction, start/end times, boards/s).

Resume: rerun the same command. Completed shards in the manifest are hash-checked and skipped;
a leftover .tmp is deleted; a shard renamed into place but missing from the manifest (crash in
between) is regenerated and adopted only if byte-identical, else the run is refused. A DIR
holding another split's manifest, or any differing parameter, is refused. Boards in a shard are
solved in batches that never cross a shard boundary, so a resumed shard repeats the exact same
arithmetic as an uninterrupted run (byte-identical on the same device and commit).

Data hygiene (CLAUDE.md §5): a dirty git tree is refused unless --allow-dirty (released splits
are generated from a committed tree). ood_bottom_chips is refused: data/raster.py puts bottom-chip
heat in q_bot, which solvers/fvm.py does not read yet. If more than 20 % of boards were rescaled,
§6.5 says to lower the power range instead: the run finishes, the manifest flags it and the
CLI exits with status 2.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from data.envelope import THETA_MAX, apply_envelope
from data.raster import rasterise
from data.sampler import OOD_MODES, sample_board
from solvers.fvm import assemble_board, board_energy_balance, coefficients
from solvers.pcg import batched_pcg, line_z_block_jacobi
from twinheat import spec as specmod

__all__ = ["GenerateError", "ManifestError", "board_mode", "board_seed", "generate",
           "record_hash", "verify"]

ROOT = Path(__file__).resolve().parents[1]
SPEC = specmod.load(ROOT / "config" / "spec_v2.yaml")
_S1 = specmod.load(ROOT / "config" / "spec_v1.yaml")
TOL = _S1.discretisation.rel_residual_tol.value  # 1e-10, CLAUDE.md §6.4
ENERGY_TOL = _S1.discretisation.energy_balance_tol.value  # 1e-3, CLAUDE.md §6.4
RESCALE_LIMIT = 0.20  # CLAUDE.md §6.5: above 20 % rescaled, lower the power range instead
SPLITS = ("train", "val", "cal", "test")
UNSUPPORTED = {"ood_bottom_chips": "data/raster.py puts bottom-chip heat in q_bot, which "
                                   "solvers/fvm.py does not read yet (docs/decisions.md, OOD v2)"}
SPEC_FILES = ("config/spec_v1.yaml", "config/spec_v2.yaml", "config/design_task.yaml")
MANIFEST = "manifest.json"
SCHEMA = "twinheat-dataset-manifest/1"
Hook = Callable[[str, int], None]


class GenerateError(RuntimeError):
    """The run is refused or a board failed a check; the message names what and why."""


class ManifestError(GenerateError):
    """A file does not match the manifest."""


# --------------------------------------------------------------------------- seeds and modes

def board_seed(split: str, seed: int, i: int) -> int:
    """64-bit board seed: the first 8 bytes (big-endian) of SHA-256(split NUL seed NUL i)."""
    return int.from_bytes(hashlib.sha256(f"{split}\0{seed}\0{i}".encode()).digest()[:8], "big")


def board_mode(split: str, i: int) -> str:
    if split == "train":
        return "design" if i % 2 == 0 else "id"
    if split in SPLITS:
        return "id"
    if split.startswith("ood_"):
        if split in UNSUPPORTED:
            raise GenerateError(f"split {split} is not generated: {UNSUPPORTED[split]}")
        if split not in OOD_MODES:
            raise GenerateError(f"unknown OOD split {split!r}; spec_v2 ood.names: {OOD_MODES}")
        return split
    raise GenerateError(f"unknown split {split!r}; expected {SPLITS} or ood_<name>")


def default_count(split: str) -> int:
    if split in SPLITS:
        return int(getattr(SPEC.splits, split).value)
    return int(SPEC.ood.samples_per_set.value)


def record_hash(record: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"))
                          .encode()).hexdigest()


# ------------------------------------------------------------------------------- solving

def _device(device: str) -> str:
    if device not in ("auto", "cpu", "gpu"):
        raise GenerateError(f"--device must be auto, cpu or gpu, got {device!r}")
    if device == "cpu":
        return "cpu"
    try:
        import cupy  # noqa: F401
        from numba import config, cuda
        ok = cuda.is_available() and not config.ENABLE_CUDASIM
    except ImportError:
        ok = False
    if device == "gpu" and not ok:
        raise GenerateError("--device gpu: CuPy or a CUDA GPU (numba.cuda) is not available")
    return "gpu" if ok else "cpu"


def _solve(cases: list[dict[str, Any]], device: str) -> tuple[np.ndarray, np.ndarray]:
    """θ [B, 4, N, N] float64 and PCG iterations [B] for a batch, line-z preconditioned."""
    b = np.stack([c["coeffs"]["rhs"] for c in cases])
    diag = np.stack([c["coeffs"]["diag"] for c in cases])
    gup = np.stack([c["coeffs"]["gup"] for c in cases])
    if device == "gpu":
        import cupy as cp

        from solvers import cuda_kernels as ck
        from solvers.pcg import stencil_operator
        op = stencil_operator(ck.upload_coefficients([c["coeffs"] for c in cases], np.float64,
                                                     to_device=cp.asarray))
        bd = cp.asarray(b)
        x, iters = batched_pcg(op, bd, cp.zeros_like(bd),
                               line_z_block_jacobi(cp.asarray(diag), cp.asarray(gup)), TOL)
        return cp.asnumpy(x), iters
    mats = [c["A"] for c in cases]

    def op(x: np.ndarray) -> np.ndarray:
        return np.stack([(A @ xi.ravel()).reshape(xi.shape) for A, xi in zip(mats, x, strict=True)])
    return batched_pcg(op, b, np.zeros_like(b), line_z_block_jacobi(diag, gup), TOL)


def _shard(split: str, seed: int, k: int, first: int, n: int, N: int, batch: int,
           device: str, log: Callable[..., None], progress: dict[str, Any]
           ) -> tuple[list[str], np.ndarray, int]:
    """Lines of records.jsonl, fields [n, 4, N, N] float16, and the count of rescaled boards."""
    lines: list[str] = []
    fields = np.empty((n, 4, N, N), dtype=np.float16)
    rescaled = 0
    for b0 in range(0, n, batch):
        cases = []
        for i in range(first + b0, first + min(b0 + batch, n)):
            s = board_seed(split, seed, i)
            rec = sample_board(s, board_mode(split, i))
            r = rasterise(rec, SPEC, N)
            edges = rec["dirichlet_edges"]
            A, _ = assemble_board(SPEC, r, edges)
            cases.append({"i": i, "seed": s, "record": rec, "raster": r,
                          "coeffs": coefficients(SPEC, r, edges), "A": A})
        thetas, iters = _solve(cases, device)
        for j, c in enumerate(cases):
            theta = thetas[j]
            err = board_energy_balance(SPEC, c["raster"], c["record"]["dirichlet_edges"], theta)
            if not err <= ENERGY_TOL:
                raise GenerateError(f"split {split} board i = {c['i']} seed {c['seed']} "
                                    f"({c['record']['id']}): energy balance error {err:.3e} > "
                                    f"{ENERGY_TOL} (CLAUDE.md §6.4)")
            rec2, theta2, scaled = apply_envelope(c["record"], theta, THETA_MAX)
            rescaled += int(scaled)
            fields[c["i"] - first] = theta2.astype(np.float16)
            lines.append(json.dumps({
                "i": c["i"], "seed": c["seed"], "record": rec2, "scaled": bool(scaled),
                "iters": int(iters[j]), "energy_balance": float(err),
                "record_sha256": record_hash(rec2)}, sort_keys=True))
        progress["done"] += len(cases)
        rate = progress["done"] / max(time.perf_counter() - progress["t0"], 1e-9)
        left = (progress["total"] - progress["done"]) / rate if rate > 0 else math.inf
        log(f"{split} shard {k}: {progress['done']}/{progress['total']} boards this run, "
            f"{rate:.3f} boards/s, ETA {dt.timedelta(seconds=round(left))}", flush=True)
    return lines, fields, rescaled


# ------------------------------------------------------------------------- files, manifest

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_shard_files(d: Path, lines: list[str], fields: np.ndarray) -> dict[str, str]:
    d.mkdir(parents=True)
    (d / "records.jsonl").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    with (d / "fields.npy").open("wb") as f:
        np.save(f, np.ascontiguousarray(fields), allow_pickle=False)
    return {name: _sha256(d / name) for name in ("records.jsonl", "fields.npy")}


def _write_manifest(out: Path, man: dict[str, Any]) -> None:
    tmp = out / (MANIFEST + ".tmp")
    tmp.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out / MANIFEST)


def _check_shard(out: Path, sh: dict[str, Any]) -> None:
    d = out / sh["name"]
    for name, want in sh["files"].items():
        p = d / name
        if not p.exists():
            raise ManifestError(f"{sh['name']}/{name} is missing")
        got = _sha256(p)
        if got != want:
            raise ManifestError(f"{sh['name']}/{name}: SHA-256 {got} != manifest {want}")


def verify(out: Path) -> dict[str, Any]:
    """Check every shard file against manifest.json (hashes, row counts, field shape)."""
    out = Path(out)
    man = json.loads((out / MANIFEST).read_text(encoding="utf-8"))
    if man.get("schema") != SCHEMA:
        raise ManifestError(f"{out / MANIFEST}: schema {man.get('schema')!r} != {SCHEMA}")
    for sh in man["shards"]:
        _check_shard(out, sh)
        d = out / sh["name"]
        rows = (d / "records.jsonl").read_text(encoding="utf-8").splitlines()
        shape = np.load(d / "fields.npy", mmap_mode="r").shape
        if len(rows) != sh["count"] or shape != (sh["count"], 4, man["N"], man["N"]):
            raise ManifestError(f"{sh['name']}: {len(rows)} records, fields {shape}, manifest "
                                f"says {sh['count']} boards at N = {man['N']}")
    done = sum(sh["count"] for sh in man["shards"])
    if man["complete"] != (done == man["count"]):
        raise ManifestError(f"manifest complete = {man['complete']} but {done}/{man['count']} "
                            "boards are in shards")
    return man


def git_commit() -> str:
    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    return git("rev-parse", "--short", "HEAD") + ("-dirty" if git("status", "--porcelain") else "")


def _spec_hashes() -> dict[str, str]:
    return {f: _sha256(ROOT / f) for f in SPEC_FILES}


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


# ------------------------------------------------------------------------------- generate

def generate(split: str, count: int, seed: int, out: Path, *, batch: int = 32, N: int = 256,
             shard_size: int = 500, device: str = "auto", allow_dirty: bool = False,
             hook: Hook | None = None, log: Callable[..., None] = print) -> dict[str, Any]:
    """Generate (or resume) a split into `out`; returns the final manifest."""
    board_mode(split, 0)  # refuses unknown and unsupported splits
    for name, v in (("count", count), ("batch", batch), ("N", N), ("shard_size", shard_size)):
        if not (isinstance(v, int) and v >= 1):
            raise GenerateError(f"{name} must be an int >= 1, got {v!r}")
    if not (isinstance(seed, int) and seed >= 0):
        raise GenerateError(f"seed must be an int >= 0, got {seed!r}")
    commit = git_commit()
    if commit.endswith("-dirty") and not allow_dirty:
        raise GenerateError(f"git tree is dirty ({commit}): commit first, or pass --allow-dirty "
                            "for a run that is not a released split (CLAUDE.md §5)")
    dev = _device(device)
    hook = hook or (lambda e, s: None)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    params = {"split": split, "seed": seed, "count": count, "N": N, "batch": batch,
              "shard_size": shard_size, "tolerance": TOL, "energy_tol": ENERGY_TOL,
              "theta_max": THETA_MAX, "device": dev, "commit": commit,
              "spec_sha256": _spec_hashes()}
    if (out / MANIFEST).exists():
        man = json.loads((out / MANIFEST).read_text(encoding="utf-8"))
        if man.get("split") != split:
            raise GenerateError(f"{out} holds the manifest of split {man.get('split')!r}, not "
                                f"{split!r}: use a separate --out per split")
        diff = {k: (man.get(k), v) for k, v in params.items() if man.get(k) != v}
        if diff:
            raise GenerateError(f"{out}: cannot resume, parameters differ (manifest, now): "
                                + ", ".join(f"{k} {a!r} vs {b!r}" for k, (a, b) in diff.items()))
        for sh in man["shards"]:
            _check_shard(out, sh)
    else:
        if any(out.glob("shard_*")):
            raise GenerateError(f"{out} has shard folders but no {MANIFEST}: refusing to mix runs")
        man = {"schema": SCHEMA, **params, "shards": [], "complete": False,
               "rescaled": 0, "rescale_fraction": 0.0, "rescale_limit": RESCALE_LIMIT,
               "rescale_limit_exceeded": False, "start_time": _now(), "end_time": None}
        _write_manifest(out, man)

    n_shards = math.ceil(count / shard_size)
    done = {sh["name"] for sh in man["shards"]}
    todo = [k for k in range(n_shards) if f"shard_{k:04d}" not in done]
    progress = {"done": 0, "t0": time.perf_counter(),
                "total": sum(min(shard_size, count - k * shard_size) for k in todo)}
    for k in todo:
        name = f"shard_{k:04d}"
        first, n = k * shard_size, min(shard_size, count - k * shard_size)
        final, tmp = out / name, out / f"{name}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)  # an interrupted shard of this same run
        t0, start = time.perf_counter(), _now()
        lines, fields, rescaled = _shard(split, seed, k, first, n, N, batch, dev, log, progress)
        hashes = _write_shard_files(tmp, lines, fields)
        hook("tmp_written", k)
        if final.exists():  # renamed before a crash, never recorded: adopt only if identical
            on_disk = {f: _sha256(final / f) for f in hashes}
            if on_disk != hashes:
                shutil.rmtree(tmp)
                raise GenerateError(f"{final} exists, is not in the manifest and differs from a "
                                    "regenerated shard: refusing to overwrite it")
            shutil.rmtree(tmp)
        else:
            os.rename(tmp, final)
        hook("renamed", k)
        sec = time.perf_counter() - t0
        man["shards"].append({"name": name, "first": first, "count": n, "files": hashes,
                              "rescaled": rescaled, "start_time": start, "end_time": _now(),
                              "seconds": round(sec, 3), "boards_per_s": round(n / sec, 4)})
        man["shards"].sort(key=lambda s: s["name"])
        boards = sum(s["count"] for s in man["shards"])
        man["rescaled"] = sum(s["rescaled"] for s in man["shards"])
        man["rescale_fraction"] = man["rescaled"] / boards
        man["rescale_limit_exceeded"] = man["rescale_fraction"] > RESCALE_LIMIT
        man["complete"] = boards == count
        man["end_time"] = _now()
        _write_manifest(out, man)
        hook("manifest_written", k)
    if man["rescale_limit_exceeded"]:
        log(f"WARNING: {man['rescale_fraction']:.1%} of boards were rescaled by the envelope, "
            f"above {RESCALE_LIMIT:.0%}: CLAUDE.md §6.5 says lower the power range instead",
            flush=True)
    return man


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", required=True)
    ap.add_argument("--count", type=int, default=None, help="default: the spec_v2 split size")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--N", type=int, default=SPEC.geometry.N.value)
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "gpu"))
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)
    count = default_count(args.split) if args.count is None else args.count
    t0 = time.perf_counter()
    man = generate(args.split, count, args.seed, args.out, batch=args.batch, N=args.N,
                   shard_size=args.shard_size, device=args.device, allow_dirty=args.allow_dirty)
    verify(args.out)
    sec = time.perf_counter() - t0
    print(f"{args.split}: {count} boards in {len(man['shards'])} shards, device {man['device']}, "
          f"rescaled {man['rescale_fraction']:.1%}, verified; this run {sec:.1f} s")
    sys.exit(2 if man["rescale_limit_exceeded"] else 0)


if __name__ == "__main__":
    main()
