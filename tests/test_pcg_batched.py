"""Batched certified PCG (N4.2 GPU path, CPU arithmetic in CI; CLAUDE.md §8.1, §8.4, §9.1, B7).

NumPy arrays [B, 4, N, N] with a CSR operator per board stand in for the CuPy stencil operator
(solvers/pcg.py is array-module generic, so the same code runs on both). Failure modes covered: a
certificate below the true error (B7), a board stopping later than the single-board CPU rule, a
board's result depending on the other boards in its batch, certifying on a drifted recursive
residual, an audit decided on the wrong side of q, and a warm start that is already certified
taking iterations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse.linalg as spla

from certify import pcg_batched as PB
from certify.kappa import kappa_hat
from certify.peak_interval import row_abs_sum_bound
from certify.stopping import pcg_certified
from data.raster import rasterise
from data.sampler import sample_board
from solvers.fvm import assemble_board, coefficients
from solvers.pcg import line_z_block_jacobi
from twinheat import spec as specmod

ROOT = Path(__file__).resolve().parents[1]
SPEC = specmod.load(ROOT / "config" / "spec_v2.yaml")
N = 24


def boards(seeds):
    out = []
    for s in seeds:
        rec = sample_board(s, "id")
        r = rasterise(rec, SPEC, N)
        A, b = assemble_board(SPEC, r, rec["dirichlet_edges"])
        c = coefficients(SPEC, r, rec["dirichlet_edges"])
        theta = spla.spsolve(A.tocsc(), b)
        rng = np.random.Generator(np.random.PCG64(s))
        mu = theta * (1 + 0.03 * rng.standard_normal(theta.size))
        k = kappa_hat(A, 60, accel="cg")
        assert k is not None
        out.append({"A": A.tocsr(), "b": b, "c": c, "theta": theta, "mu": mu, "kappa": k,
                    "S": row_abs_sum_bound(c)})
    return out


def batch_ops(bs):
    mats = [d["A"] for d in bs]

    def Aop(x):
        return np.stack([(A @ xi.ravel()).reshape(xi.shape) for A, xi in zip(mats, x, strict=True)])
    diag = np.stack([d["c"]["diag"] for d in bs])
    gup = np.stack([d["c"]["gup"] for d in bs])
    shape = (len(bs), 4, N, N)
    b = np.stack([d["b"] for d in bs]).reshape(shape)
    x0 = np.stack([d["mu"] for d in bs]).reshape(shape)
    return Aop, line_z_block_jacobi(diag, gup), b, x0


def run(bs, tau=0.1, **kw):
    Aop, M, b, x0 = batch_ops(bs)
    kw.setdefault("check_every", 1)
    kw.setdefault("maxit", 5000)
    return PB.batched_pcg_certified(
        Aop, Aop, b, x0, M, kappa=np.array([d["kappa"] for d in bs]),
        S=np.array([d["S"] for d in bs]), tau_v=np.full(len(bs), tau), **kw)


def test_certified_bounds_hold_and_no_later_than_the_cpu_rule():
    bs = boards(range(6))
    res = run(bs)
    for j, d in enumerate(bs):
        err = float(np.abs(d["theta"] - res.x[j].ravel()).max())
        assert res.stop[j] == "certified" and res.cert[j] <= 0.1
        assert res.cert[j] >= err  # B7
        M = line_z_block_jacobi(d["c"]["diag"], d["c"]["gup"])
        _, it_ref, _ = pcg_certified(d["A"], d["b"], d["mu"],
                                     lambda r, M=M: M(r.reshape(4, N, N)).ravel(), d["kappa"], 0.1)
        assert res.iters[j] <= it_ref


def test_boards_leave_independently():
    bs = boards(range(4))
    together = run(bs)
    for j in range(len(bs)):
        alone = run([bs[j]])
        assert alone.iters[0] == together.iters[j]
        assert np.array_equal(alone.x[0], together.x[j])


def test_certified_warm_start_takes_zero_iterations():
    bs = boards([1])
    bs[0]["mu"] = bs[0]["theta"].copy()
    res = run(bs, tau=1.0)
    assert res.iters[0] == 0 and res.stop[0] == "certified"


def test_drifted_recursive_residual_never_certifies():
    bs = boards([2])
    Aop, M, b, x0 = batch_ops(bs)
    res = PB.batched_pcg_certified(  # iterations use 0.5 A: the recursive residual goes to 0
        lambda x: 0.5 * Aop(x), Aop, b, x0, M, kappa=np.array([bs[0]["kappa"]]),
        S=np.array([bs[0]["S"]]), tau_v=np.array([0.1]), check_every=5, maxit=300)
    assert res.stop[0] == "maxit" and res.iters[0] == 300
    err = float(np.abs(bs[0]["theta"] - res.x[0].ravel()).max())
    assert res.cert[0] >= err  # the reported bound is still a true bound


def test_audit_boards_stop_on_the_true_side_of_q():
    bs = boards(range(4))
    Aop, M, b, x0 = batch_ops(bs)
    peak_hat = np.array([float(d["mu"].max()) for d in bs])
    S_true = np.array([abs(float(d["theta"].max()) - ph)
                       for d, ph in zip(bs, peak_hat, strict=True)])
    s = np.full(4, 0.5)
    for q in (0.3 * S_true / s, 3.0 * S_true / s):
        res = PB.batched_pcg_certified(
            Aop, Aop, b, x0, M, kappa=np.array([d["kappa"] for d in bs]),
            S=np.array([d["S"] for d in bs]), tau_v=np.full(4, np.nan), peak_hat=peak_hat, s=s,
            q=q, eps_S=np.full(4, 1e-9), check_every=1, maxit=5000)
        for j, d in enumerate(bs):
            Sj = S_true[j] / s[j]
            assert res.T_lo[j] <= float(d["theta"].max()) <= res.T_hi[j]  # B7
            assert res.S_lo[j] <= Sj <= res.S_hi[j]  # Lemma 1'
            if res.stop[j] == "below_q":
                assert Sj <= q[j]
            elif res.stop[j] == "above_q":
                assert Sj > q[j]


def test_bad_arguments_refused():
    bs = boards([0])
    Aop, M, b, x0 = batch_ops(bs)
    k, S = np.array([bs[0]["kappa"]]), np.array([bs[0]["S"]])
    with pytest.raises(ValueError, match="kappa"):
        PB.batched_pcg_certified(Aop, Aop, b, x0, M, kappa=np.array([-1.0]), S=S,
                                 tau_v=np.array([0.1]), check_every=1, maxit=10)
    with pytest.raises(ValueError, match="audit"):
        PB.batched_pcg_certified(Aop, Aop, b, x0, M, kappa=k, S=S, tau_v=np.array([np.nan]),
                                 check_every=1, maxit=10)
    with pytest.raises(ValueError, match="check_every"):
        PB.batched_pcg_certified(Aop, Aop, b, x0, M, kappa=k, S=S, tau_v=np.array([0.1]),
                                 check_every=0, maxit=10)
