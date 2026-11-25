"""The live G5 gate and z reuse in the server (server/live_gate.py, server/iterate_stream.py,
server/app.py; CLAUDE.md §8.1, §8.3, §8.4, B7, B8).

Failure modes covered: a query that draws a coin or changes q (B8), a flagged design that updates
q (Lemma 1 A3), an audited design that does not, an accepted unaudited design that is verified
anyway, and a reused z that gives a band missing the true field (B7).
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla
from fastapi.testclient import TestClient
from test_run_gates import FROZEN

from contracts.f16 import decode
from data.raster import rasterise
from data.sampler import design_board
from gate.proposers import sample_p0
from server import iterate_stream as IS
from server.app import create_app
from server.live_gate import LiveGate
from solvers.fvm import assemble_board, coefficients
from twinheat import spec as specmod

ROOT = Path(__file__).resolve().parents[1]
SPEC = specmod.load(ROOT / "config" / "spec_v2.yaml")
N = 32


def _record(seed: int = 3) -> dict:
    return design_board(sample_p0(np.random.Generator(np.random.PCG64(seed))))


def _theta(rec: dict) -> np.ndarray:
    A, b = assemble_board(SPEC, rasterise(rec, SPEC, N), rec["dirichlet_edges"])
    return spla.spsolve(A.tocsc(), b).reshape(4, N, N)


def _moved(rec: dict, dx: float = 1.0) -> dict:
    r = copy.deepcopy(rec)
    c, v = r["chips"][0], r["via_arrays"][0]
    c["x0"] = v["x0"] = round(c["x0"] + dx, 3)
    return r


def _gate(**kw) -> LiveGate:
    return LiveGate({**FROZEN, **kw}, np.full(1000, 0.5), seed=0)


def test_preview_is_pure_and_commit_draws_the_coin() -> None:
    rec = _record()
    th = _theta(rec)
    g = _gate()
    c = coefficients(SPEC, rasterise(rec, SPEC, N), rec["dirichlet_edges"])
    gi = g.gate_input("q", rec, th * 1.03, np.full_like(th, 0.05), c)
    before = (g.gate.q_t, g.gate.alpha_t, dict(g.coin._drawn))
    g.preview(gi)
    g.preview(gi)
    assert (g.gate.q_t, g.gate.alpha_t, dict(g.coin._drawn)) == before  # B8
    gd = g.gate_input(g.unique_id("d"), rec, th * 1.03, np.full_like(th, 0.05), c)
    dec, audited = g.commit(gd)
    assert gd.design_id in g.coin._drawn and dec.decision in ("accept", "flag")


def _serve(gate: LiveGate, rec: dict):
    th = _theta(rec)

    def predict(spec, raster, edges, record):
        return (th * 1.03).astype(np.float32), np.full_like(th, 0.05, dtype=np.float32)
    return create_app(n=N, predict=predict, verify={"tau_v": 0.1, "every": 5, "kappa_iters": 60},
                      gate=gate)


def _drain(ws, mid: str) -> list[dict]:
    got = [ws.receive_json()]
    while got[-1]["type"] not in ("verified", "error") and not (
            got[-1]["type"] == "pred" and got[-1]["decision"] in ("accept", "preview")):
        got.append(ws.receive_json())
    assert all(m["id"] == mid for m in got)
    return got


def test_audited_design_is_verified_and_updates_q() -> None:
    rec = _record()
    g = _gate(tau_K=1e6, p_audit=1.0)  # everything accepted, everything audited
    with TestClient(_serve(g, rec)) as client, client.websocket_connect("/") as ws:
        ws.send_json({"schema_version": "1.0", "type": "query", "id": "q1", "record": rec})
        q = _drain(ws, "q1")
        assert len(q) == 1 and q[0]["decision"] == "preview" and q[0]["B"] is not None
        assert g.gate.n_audits == 0 and not g.coin._drawn  # B8: the query drew nothing
        ws.send_json({"schema_version": "1.0", "type": "design", "id": "d1", "record": rec})
        got = _drain(ws, "d1")
        assert got[0]["decision"] == "audit" and got[-1]["type"] == "verified"
        th = _theta(rec)
        drawn = decode(got[-1]["theta"], got[-1]["shape"]).astype(np.float64)
        assert np.all(np.abs(th - drawn) <= got[-1]["cert"])  # B7
    assert g.gate.n_audits == 1  # the audit, and only the audit, updated q


def test_flagged_unaudited_is_verified_but_never_updates_q() -> None:
    rec = _record()
    g = _gate(tau_K=1e-9, p_audit=1e-9)  # everything flagged, (almost surely) nothing audited
    with TestClient(_serve(g, rec)) as client, client.websocket_connect("/") as ws:
        ws.send_json({"schema_version": "1.0", "type": "design", "id": "d1", "record": rec})
        got = _drain(ws, "d1")
        assert got[0]["decision"] == "flag" and got[-1]["type"] == "verified"
    assert g.gate.n_audits == 0


def test_accepted_unaudited_design_gets_its_pred_only() -> None:
    rec = _record()
    g = _gate(tau_K=1e6, p_audit=1e-9)
    with TestClient(_serve(g, rec)) as client, client.websocket_connect("/") as ws:
        ws.send_json({"schema_version": "1.0", "type": "design", "id": "d1", "record": rec})
        ws.send_json({"schema_version": "1.0", "type": "query", "id": "q2", "record": rec})
        first, second = ws.receive_json(), ws.receive_json()
        assert first["id"] == "d1" and first["decision"] == "accept"
        assert second["id"] == "q2"  # no iterate or verified came between them


def test_z_is_warm_started_after_a_small_move_and_the_band_still_holds() -> None:
    rec = _record()
    cache: dict = {}
    for r in (rec, _moved(rec), _moved(rec, 2.0)):
        th = _theta(r)
        msgs = list(IS.verification_messages(r, th * 1.03, SPEC, N, msg_id="d", tau_v=0.1,
                                             every=5, kappa_iters=60, z_cache=cache,
                                             z_warm_iters=5))
        for m in msgs:
            drawn = decode(m["theta"], m["shape"]).astype(np.float64)
            cert = m["cert_k"] if m["type"] == "iterate" else m["cert"]
            assert np.all(np.abs(th - drawn) <= cert)  # B7 with a reused z
    assert cache["misses"] == 1 and cache["hits"] == 2
