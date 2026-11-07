"""The viewer's physical constants equal the spec (G2.1; CLAUDE.md §0 rule 1)."""

from __future__ import annotations

import json
from pathlib import Path

from server import viewer_constants as vc

ROOT = Path(__file__).resolve().parents[1]


def test_committed_viewer_constants_match_the_spec():
    committed = json.loads((ROOT / "viewer" / "src" / "spec_constants.json").read_text(
        encoding="utf-8"))
    assert committed == vc.constants()


def test_via_fixture_was_made_from_the_current_solver():
    """viewer/test/fixtures/via_board.json comes from tests/fixtures/make_via_fixture.py;
    regenerate it if the solver or the spec changes (this checks it still solves the same)."""
    import numpy as np

    from tests.fixtures import make_via_fixture as mk
    fx = json.loads((ROOT / "viewer" / "test" / "fixtures" / "via_board.json").read_text(
        encoding="utf-8"))
    theta = mk.solve(fx["record"], fx["N"])
    np.testing.assert_allclose(np.array(fx["theta"]).reshape(theta.shape), theta, rtol=1e-6,
                               atol=1e-6)
