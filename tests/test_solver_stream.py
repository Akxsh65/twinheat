"""Real solver fields for the viewer (G1.3 DONE WHEN; server/solver_stream.py)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from contracts.f16 import decode
from contracts.validate import validate
from data.raster import rasterise
from data.sampler import sample_board
from server import solver_stream as ss
from solvers.fvm import assemble_board, solve_direct

N = 16


@pytest.fixture(scope="module")
def lines():
    return list(ss.messages(seeds=(0, 1, 2), n=N, orientation=True))


def test_one_verified_message_per_board_plus_the_orientation_field(lines):
    msgs = [json.loads(s) for s in lines]
    assert [m["id"] for m in msgs] == ["solver-id-0", "solver-id-1", "solver-id-2",
                                       "synthetic-orientation-hot-SE"]
    for m in msgs:
        validate(m, "verified")
        assert m["shape"] == [4, N, N]


@pytest.mark.parametrize("k", [0, 1, 2])
def test_fields_are_the_4_layer_direct_solve(lines, k):
    msg = json.loads(lines[k])
    rec = sample_board(k, "id")
    A, b = assemble_board(ss.SPEC, rasterise(rec, ss.SPEC, N), rec["dirichlet_edges"])
    theta = solve_direct(A, b).reshape(4, N, N)
    got = decode(msg["theta"], msg["shape"])
    np.testing.assert_allclose(got, theta, rtol=2.0**-10, atol=0)  # float16 transport
    assert msg["cert"] is None and msg["iters"] == 0 and "direct" in msg["stop_reason"]


def test_orientation_field_is_hot_at_the_south_east_corner_of_l1(lines):
    msg = json.loads(lines[-1])
    f = decode(msg["theta"], msg["shape"])
    hot = np.argwhere(f == f.max())  # rows of [layer, y, x]
    assert (hot[:, 0] == 0).all()  # only on L1
    assert hot[:, 1].max() < N // 2 and hot[:, 2].min() >= N // 2  # south half, east half
    assert f[0, 0, N - 1] == f.max()  # the corner cell itself
    assert "SYNTHETIC" in msg["stop_reason"]
