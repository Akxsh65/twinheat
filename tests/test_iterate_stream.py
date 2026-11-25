"""Anytime verification stream (server/iterate_stream.py, server/app.py --verify; F7; CLAUDE.md
§8.4, §10.3, §13 T1.7, B6, B7, B8).

Failure modes covered: a band that does not contain the exact field the viewer draws (float16
included), messages that break the contracts, a stream that never ends or ends uncertified, a
board without a certificate shown with a band, and a query that starts a verification (B8).
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse.linalg as spla
from fastapi.testclient import TestClient

from contracts.f16 import decode
from contracts.validate import validate
from data.raster import rasterise
from data.sampler import design_board
from gate.proposers import sample_p0
from server import iterate_stream as IS
from server.app import create_app
from solvers.fvm import assemble_board
from twinheat import spec as specmod

ROOT = Path(__file__).resolve().parents[1]
SPEC = specmod.load(ROOT / "config" / "spec_v2.yaml")
N = 32


def _board(seed: int = 0):
    rec = design_board(sample_p0(np.random.Generator(np.random.PCG64(seed))))
    A, b = assemble_board(SPEC, rasterise(rec, SPEC, N), rec["dirichlet_edges"])
    theta = spla.spsolve(A.tocsc(), b).reshape(4, N, N)
    return rec, theta


def test_every_band_contains_the_exact_field_as_drawn() -> None:
    for seed in range(3):
        rec, theta = _board(seed)
        msgs = list(IS.verification_messages(rec, theta * 1.03, SPEC, N, msg_id="d", tau_v=0.1,
                                             every=5, kappa_iters=60))
        its, ver = msgs[:-1], msgs[-1]
        assert [m["type"] for m in its] == ["iterate"] * len(its) and ver["type"] == "verified"
        assert [m["k"] for m in its] == sorted({m["k"] for m in its}) and its[0]["k"] == 0
        for m in msgs:
            validate(m, m["type"])  # B6
            drawn = decode(m["theta"], m["shape"]).astype(np.float64)
            cert = m["cert_k"] if m["type"] == "iterate" else m["cert"]
            assert np.all(np.abs(theta - drawn) <= cert)  # B7, for the float16 field shown
        assert ver["stop_reason"] == "certified" and ver["cert"] <= 0.1
        assert ver["iters"] == its[-1]["k"]


def test_no_certificate_means_no_band(monkeypatch) -> None:
    import certify.kappa as K
    monkeypatch.setattr(K, "kappa_hat", lambda *a, **k: None)
    rec, theta = _board(1)
    msgs = list(IS.verification_messages(rec, theta * 1.03, SPEC, N, msg_id="d", tau_v=0.1,
                                         every=20, kappa_iters=60))
    assert all(m["cert_k"] is None for m in msgs[:-1]) and msgs[-1]["cert"] is None
    assert msgs[-1]["stop_reason"].startswith("no certificate")


def test_bad_settings_refused() -> None:
    rec, theta = _board(0)
    with pytest.raises(ValueError, match="tau_v"):
        next(IS.verification_messages(rec, theta, SPEC, N, msg_id="d", tau_v=0.0, every=5,
                                      kappa_iters=60))
    with pytest.raises(ValueError, match="every"):
        next(IS.verification_messages(rec, theta, SPEC, N, msg_id="d", tau_v=0.1, every=0,
                                      kappa_iters=60))


def test_server_streams_a_design_and_never_a_query() -> None:
    rec, theta = _board(2)

    def predict(spec, raster, edges, record):  # a 3 % wrong "prediction" of the exact field
        return (theta * 1.03).astype(np.float32), np.full_like(theta, 0.1, dtype=np.float32)

    app = create_app(n=N, predict=predict,
                     verify={"tau_v": 0.1, "every": 5, "kappa_iters": 60})
    with TestClient(app) as client, client.websocket_connect("/") as ws:
        ws.send_json({"schema_version": "1.0", "type": "query", "id": "q1",
                      "record": copy.deepcopy(rec)})
        first = ws.receive_json()
        assert first["type"] == "pred" and first["decision"] == "preview"
        ws.send_json({"schema_version": "1.0", "type": "design", "id": "d1",
                      "record": copy.deepcopy(rec)})
        got = [ws.receive_json()]
        while got[-1]["type"] != "verified":
            got.append(ws.receive_json())
        assert got[0]["type"] == "pred" and got[0]["decision"] == "flag"
        assert {m["type"] for m in got[1:-1]} == {"iterate"} and all(m["id"] == "d1" for m in got)
        assert got[-1]["stop_reason"] == "certified"
        # B8: the query before it produced exactly one message (its pred), no iterate
        assert first["id"] == "q1"
