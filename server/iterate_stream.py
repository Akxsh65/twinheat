"""Anytime verification: stream certified PCG iterates to the viewer (F7; CLAUDE.md §8.1 step 5,
§8.4, §10.3, §13 T1.6-T1.7).

    verification_messages(record, mu, spec, n, msg_id=..., tau_v=..., every=..., kappa_iters=...)

yields, for one committed design, `iterate` messages every `every` PCG iterations (k = 0 is the
prediction μ itself) and a final `verified` message. PCG runs in FP64 from μ (line-z block
Jacobi, D10); κ̂ comes from certify.kappa with z by AMG-CG (decision "kappa-hat z by AMG-CG").

The band is rigorous for the field the viewer actually draws. At iterate k the Lemma 2(b) bound
c_k = certify.margin.residual_bound(κ̂, ‖b − A x_k‖∞, ...) covers |θ − x_k| in every cell; the
message carries x_k rounded to float16 (contracts §10), so cert_k = up(c_k + max|x_k − fl16(x_k)|),
which covers |θ − fl16(x_k)| in every cell. The run stops when cert_k <= tau_v ("certified") or at
maxit (the D14 cap). A board without a certificate (κ̂ None) is solved to 1e-8 with cert_k null
and stop_reason saying so: the viewer must then draw no guaranteed band.

z warm start (z_cache, z_warm_iters): Lemma 2(a) holds for ANY z > 0 with A z > 0, so z for the
new design is computed by z_warm_iters AMG-CG iterations started from the previous design's z
instead of kappa_iters iterations from 0. certify.kappa.kappa_from_z then checks z > 0 and
min(A z) > 0 with the derived FP64 margin exactly as for a cold z; if the check fails, z is
recomputed cold. Validity never depends on the warm start, only the time does: after a 1 mm chip
move (which also moves its via array, so the old z alone fails), 5 warm iterations gave a valid,
equally tight κ̂ in about a fifth of the cold time on a 256² laptop test.

    python -m server.iterate_stream --out server/demo/verify.jsonl --designs 3 --n 256 \
        --model ensemble --device cuda:0 --tau-v 0.1 --every 10 --kappa-iters 60 \
        [--records-out viewer/public/demo/records.json]

writes a pred -> iterate... -> verified stream per design (p₀ designs, seed 0) for
server/ws_mock_server.py, so the demo plays without a GPU or a solver. --model stub uses
server/model_stub.py (no GPU; the field is not an FNO prediction and its ids say "stub").
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from contracts.f16 import encode
from contracts.validate import validate

ROOT = Path(__file__).resolve().parents[1]
FIXED_TOL = 1e-8  # §11.5: the uncertified fallback solves to the oracle tolerance
D14_MAXIT = 20000  # docs/decisions.md D14


def _f16_error(x: np.ndarray) -> float:
    """Exact max |x − fl16(x)| (K), rounded up: the wire rounding of a field."""
    d = np.abs(x - x.astype("<f2").astype(np.float64))
    return float(np.nextafter(d.max(), math.inf))


def verification_messages(record: dict[str, Any], mu: np.ndarray, spec: Any, n: int, *,
                          msg_id: str, tau_v: float, every: int, kappa_iters: int,
                          maxit: int = D14_MAXIT, raster: Any = None,
                          z_cache: dict[str, Any] | None = None,
                          z_warm_iters: int | None = None) -> Iterator[dict[str, Any]]:
    """iterate messages, then one verified message, for one committed design (module docstring).
    raster: the design's raster at n, if the caller has it; z_cache: a dict kept across calls
    (keys "z", "hits", "misses") for the z warm start when z_warm_iters is given."""
    from certify.kappa import kappa_from_z, multigrid, positive_z
    from certify.margin import residual_bound, row_abs_sum_bound
    from data.raster import rasterise
    from solvers.fvm import assemble_board, coefficients
    from solvers.pcg import line_z_block_jacobi

    if not (math.isfinite(tau_v) and tau_v > 0):
        raise ValueError(f"tau_v must be > 0 K, got {tau_v}")
    for name, v in (("every", every), ("kappa_iters", kappa_iters), ("maxit", maxit)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"{name} must be an int >= 1, got {v!r}")
    t0 = time.perf_counter()
    r = rasterise(record, spec, n) if raster is None else raster
    A, b = assemble_board(spec, r, record["dirichlet_edges"])
    A = A.tocsr()
    c = coefficients(spec, r, record["dirichlet_edges"])
    shape = (4, n, n)
    x = np.asarray(mu, dtype=np.float64).reshape(-1).copy()
    if x.shape != b.shape:
        raise ValueError(f"mu has {x.size} values, the board {b.size}")
    kappa = None
    warm = z_cache is not None and z_warm_iters is not None
    ml = multigrid(A)
    zc = z_cache.get("z") if warm else None
    if zc is not None and zc.shape == x.shape:
        zw = np.asarray(ml.solve(np.ones(x.size), x0=zc.copy(), maxiter=z_warm_iters, tol=1e-14,
                                 accel="cg"), dtype=np.float64)
        if np.all(np.isfinite(zw)):
            kappa = kappa_from_z(A, zw)  # full FP64 check of Lemma 2(a)'s hypothesis
        if kappa is not None:
            z_cache["z"] = zw
            z_cache["hits"] = z_cache.get("hits", 0) + 1
    if kappa is None:
        pz = positive_z(A, kappa_iters, ml=ml, accel="cg")
        kappa = None if pz is None else kappa_from_z(A, pz[0])
        if warm:
            z_cache["misses"] = z_cache.get("misses", 0) + 1
            if kappa is not None:
                z_cache["z"] = pz[0]
    S, b_inf, bn = row_abs_sum_bound(A), float(np.abs(b).max()), float(np.linalg.norm(b))
    M = line_z_block_jacobi(c["diag"], c["gup"])

    def Minv(v: np.ndarray) -> np.ndarray:
        return M(v.reshape(shape)).ravel()

    def band() -> float | None:
        if kappa is None:
            return None
        cb = residual_bound(kappa, float(np.abs(b - A @ x).max()), S, b_inf,
                            float(np.abs(x).max()))
        return float(np.nextafter(cb + _f16_error(x), math.inf))

    def iterate(k: int, cert: float | None) -> dict[str, Any]:
        return {"schema_version": "1.0", "type": "iterate", "id": msg_id, "k": k,
                "theta": encode(x.reshape(shape)), "cert_k": cert, "shape": list(shape),
                "t_server_ns": time.monotonic_ns()}

    def verified(k: int, cert: float | None, why: str) -> dict[str, Any]:
        return {"schema_version": "1.0", "type": "verified", "id": msg_id,
                "theta": encode(x.reshape(shape)), "cert": cert, "iters": k,
                "ms": 1000 * (time.perf_counter() - t0), "stop_reason": why,
                "shape": list(shape), "t_server_ns": time.monotonic_ns()}

    rr = b - A @ x
    z = Minv(rr)
    p = z.copy()
    rz = float(rr @ z)
    k = 0
    while True:
        if k % every == 0 or k >= maxit:
            cert = band()
            yield iterate(k, cert)
            if cert is not None and cert <= tau_v:
                yield verified(k, cert, "certified")
                return
            if cert is None and float(np.linalg.norm(b - A @ x)) <= FIXED_TOL * bn:
                yield verified(k, None, "no certificate (κ̂ not found): solved to 1e-8")
                return
            if k >= maxit:
                yield verified(k, cert, "maxit")
                return
        Ap = A @ p
        alpha = rz / float(p @ Ap)
        x += alpha * p
        rr -= alpha * Ap
        z = Minv(rr)
        rz_new = float(rr @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
        k += 1


# ------------------------------------------------------------------------------- record mode

def _pred(record: dict[str, Any], mu: np.ndarray, sd: np.ndarray, msg_id: str) -> dict[str, Any]:
    return {"schema_version": "1.0", "type": "pred", "id": msg_id, "mu": encode(mu),
            "sd": encode(sd), "B": None, "decision": "flag", "group": None,
            "peak": float(mu.max()), "units": {"mu": "K", "sd": "K", "B": "K", "peak": "K"},
            "shape": list(mu.shape), "t_server_ns": time.monotonic_ns()}


def _predictor(model: str, spec: Any, n: int, device: str) -> Callable:
    from data.raster import rasterise
    if model == "stub":
        from server import model_stub

        def stub(record):
            r = rasterise(record, spec, n)
            return model_stub.predict(spec, r, record["dirichlet_edges"], record)
        return stub
    from models import inputs as inp
    from models.ensemble import Ensemble
    ens = Ensemble.load(ROOT / "models" / "registry.json", ROOT / "models" / "norm_stats.json",
                        stage="ft256", device=device)

    def fno(record):
        raw = inp.build(rasterise(record, spec, n), spec)[None]
        mu, sd = ens.predict(raw)
        return mu[0], sd[0]
    return fno


def record_stream(out: Path, *, designs: int, n: int, model: str, device: str, tau_v: float,
                  every: int, kappa_iters: int, seed: int = 0,
                  records_out: Path | None = None) -> int:
    from data.sampler import design_board
    from gate.proposers import sample_p0
    from twinheat import spec as specmod

    spec = specmod.load(ROOT / "config" / "spec_v2.yaml")
    predict = _predictor(model, spec, n, device)
    rng = np.random.Generator(np.random.PCG64(seed))
    lines, recs = [], {}
    for i in range(designs):
        rec = design_board(sample_p0(rng))
        mid = f"demo-{model}-{i}-{rec['id']}"
        mu, sd = predict(rec)
        msgs = [_pred(rec, np.asarray(mu), np.asarray(sd), mid)]
        msgs += list(verification_messages(rec, mu, spec, n, msg_id=mid, tau_v=tau_v,
                                           every=every, kappa_iters=kappa_iters))
        for m in msgs:
            validate(m, m["type"])  # B6
            lines.append(json.dumps(m))
        recs[mid] = rec
        last = msgs[-1]
        print(f"{mid}: {len(msgs) - 2} iterates, {last['stop_reason']}, cert {last['cert']} K, "
              f"{last['ms']:.0f} ms")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if records_out is not None:
        records_out.parent.mkdir(parents=True, exist_ok=True)
        records_out.write_text(json.dumps(recs), encoding="utf-8")
    return len(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--designs", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--model", choices=("ensemble", "stub"), required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tau-v", type=float, required=True, help="K (CLAUDE.md §8.1: e.g. 0.1)")
    ap.add_argument("--every", type=int, required=True, help="PCG iterations between iterates")
    ap.add_argument("--kappa-iters", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--records-out", type=Path, default=None)
    a = ap.parse_args(argv)
    k = record_stream(a.out, designs=a.designs, n=a.n, model=a.model, device=a.device,
                      tau_v=a.tau_v, every=a.every, kappa_iters=a.kappa_iters, seed=a.seed,
                      records_out=a.records_out)
    print(f"wrote {k} messages to {a.out}")


if __name__ == "__main__":
    main()
