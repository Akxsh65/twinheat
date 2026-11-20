"""Proposers P1–P4, the online loop and the differentiable design inputs (M3.3; CLAUDE.md §11.2,
§8.1, §5 B8). Proposer logic runs against a fake, deterministic evaluator; the evaluator and the
differentiable rasteriser run against a tiny real FNO ensemble at a small grid."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from data.raster import rasterise
from data.sampler import InfeasibleDesignError, design_board
from gate import proposers as P
from gate.coin import AuditCoin
from models import inputs as inp
from models import zoo
from models.diff_inputs import DesignInputs
from models.ensemble import Ensemble
from models.normalise import Normaliser
from models.ood_score import OODScore
from twinheat import spec as specmod

ROOT = Path(__file__).resolve().parents[1]
SPEC = specmod.load(ROOT / "config" / "spec_v2.yaml")
DIM = P.DIM
TARGET = np.linspace(0.2, 0.8, DIM)  # the fake surrogate's coolest design


def fake_peak(x: np.ndarray) -> float:
    return 40.0 + 30.0 * float(np.mean((np.asarray(x) - TARGET) ** 2))


def fake_evaluate(xs):
    out = []
    for x in xs:
        x = np.asarray(x, dtype=np.float64)
        out.append(P.Prediction(x=x, x_hash=P.x_hash(x), peak_hat=fake_peak(x), r_hot=0.1,
                                sd_hot=0.2, r_hot_raw=0.01, ood_score=1.0, hot_cell=(0, 16, 16),
                                edge_hotspot=False))
    return out


def fake_oracle(xs: np.ndarray):
    xs = np.asarray(xs, dtype=np.float64)
    return (np.array([fake_peak(x) for x in xs]),
            30.0 * 2.0 * (xs - TARGET) / DIM)


def fake_group(pred, proposer):
    return f"proposer={proposer.name}"


def run_offline(proposer, cap=100_000):
    hist = P.History()
    P.run_offline(proposer, fake_evaluate, fake_group, hist, max_proposals=cap)
    return hist


# ------------------------------------------------------------------- history has no coins (B8)


def test_history_entries_have_no_coin_fields() -> None:
    names = {f.name for f in dataclasses.fields(P.HistoryEntry)}
    assert not names & P.FORBIDDEN_FIELDS
    assert {"x", "peak_hat", "decision", "B", "group"} <= names  # predictions and decisions
    e = P.HistoryEntry(t=0, design_id="t0", x=(0.5,) * DIM, x_hash="0" * 64, peak_hat=1.0,
                       r_hot=0.0, sd_hot=0.0, r_hot_raw=0.0, ood_score=0.0, group="g",
                       decision=None, B=None)
    with pytest.raises((AttributeError, TypeError, dataclasses.FrozenInstanceError)):
        e.audited = True  # frozen, slotted: no field can be added
    with pytest.raises(TypeError):
        P.History().append({"t": 0, "audited": True})  # only HistoryEntry objects go in
    assert P.FORBIDDEN_FIELDS >= {"audited", "audit", "coin", "A_t"}


def test_history_is_append_only() -> None:
    h = P.History()
    assert isinstance(h.entries, tuple)
    with pytest.raises(AttributeError):
        h.entries.append(None)


# ------------------------------------------------------ groups before the coin (call order, B8)


class RecordingCoin(AuditCoin):
    def __init__(self, log):
        super().__init__(seed=0)
        self.log = log

    def commit(self, design_id):
        self.log.append(("commit", design_id))
        super().commit(design_id)

    def draw(self, design_id, p):
        self.log.append(("draw", design_id))
        return super().draw(design_id, p)


class RecordingGate:
    gate_id = "G5"

    def __init__(self, log):
        self.log = log

    def decide(self, gi):
        self.log.append(("decide", gi.design_id))
        from gate.base import GateDecision
        return GateDecision(design_id=gi.design_id, B=gi.s, decision="accept", p_audit=0.5,
                            group=gi.group, q_t=1.0, alpha_t=0.1)


def test_groups_are_computed_before_the_coin_and_history_never_holds_it() -> None:
    log: list[tuple[str, str]] = []

    def grouper(pred, proposer):
        log.append(("group", pred.x_hash))
        return "proposer=P2"

    prop = P.SimulatedAnnealing(T0=1.0, step=0.05, steps=12, T_end_ratio=0.01, seed=3, level=2)
    seen_histories = []
    real_next = prop.next

    def spy_next(history):
        seen_histories.append(history)
        return real_next(history)

    prop.next = spy_next
    hist, audits = P.run_online(prop, fake_evaluate, grouper, RecordingGate(log),
                                RecordingCoin(log), scale=lambda pred: 0.5)
    ids = [e.design_id for e in hist.entries]
    assert ids and len(audits) == len(ids)
    for e in hist.entries:
        order = [k for k, (what, key) in enumerate(log)
                 if key in (e.design_id, e.x_hash)]
        kinds = [log[k][0] for k in order]
        # group (label-free) -> decide -> commit -> draw, for every design
        assert kinds.index("group") < kinds.index("decide") < kinds.index("commit") \
            < kinds.index("draw")
    for h in seen_histories:
        assert isinstance(h, P.History)
        for e in h.entries:
            assert not {f.name for f in dataclasses.fields(e)} & P.FORBIDDEN_FIELDS
    assert all(isinstance(a, bool) for _, a in audits)  # coins go back to the caller only


# ------------------------------------------------------------- same seed -> same trajectory


def _make(name: str, seed: int):
    if name == "P1":
        return P.TiltedMCMC(lam=3.0, step=0.05, n_proposals=40, seed=seed, level=2)
    if name == "P2":
        return P.SimulatedAnnealing(T0=1.0, step=0.05, steps=40, T_end_ratio=0.01, seed=seed,
                                    level=2)
    return P.BatchedGradient(fake_oracle, eta=0.03, starts=4, steps=5, halvings=5, seed=seed,
                             level=2)


@pytest.mark.parametrize("name", ["P1", "P2", "P3"])
def test_same_seed_gives_an_identical_trajectory(name: str) -> None:
    a = [e.x for e in run_offline(_make(name, 7)).entries]
    b = [e.x for e in run_offline(_make(name, 7)).entries]
    c = [e.x for e in run_offline(_make(name, 8)).entries]
    assert a == b and len(a) > 0
    assert a != c


@pytest.mark.parametrize("name", ["P1", "P2", "P3"])
def test_every_proposal_is_feasible_and_in_the_cube(name: str) -> None:
    for e in run_offline(_make(name, 1)).entries:
        x = np.array(e.x)
        assert x.shape == (DIM,) and (x >= 0).all() and (x <= 1).all()
        design_board(x)  # raises if infeasible


# ------------------------------------------------------------------------------- P1 (MCMC)


def test_p1_counts_proposals_and_is_more_aggressive_with_larger_lambda() -> None:
    def mean_peak(lam):
        p = P.TiltedMCMC(lam=lam, step=0.05, n_proposals=400, seed=0, level=1)
        h = run_offline(p)
        assert len(h.entries) == 400
        return np.mean([e.peak_hat for e in h.entries[200:]])

    assert mean_peak(10.0) < mean_peak(0.01)


def test_p1_metropolis_acceptance_is_exact() -> None:
    p = P.TiltedMCMC(lam=2.0, step=0.05, n_proposals=3, seed=0, level=1)
    hist = P.History()
    xs = p.next(hist)  # the start
    P.record(hist, fake_evaluate(xs), fake_group, p)
    start_peak = hist.entries[-1].peak_hat
    xs = p.next(hist)  # first candidate
    P.record(hist, fake_evaluate(xs), fake_group, p)
    p.next(hist)  # decides on the candidate
    cand_peak = hist.entries[-1].peak_hat
    accepted = p.last_accepted
    prob = min(1.0, float(np.exp(-2.0 * (cand_peak - start_peak))))
    assert p.last_accept_prob == pytest.approx(prob)
    assert (p.current_peak == cand_peak) is accepted


# ------------------------------------------------------------------------------- P2 (SA)


def test_p2_cools_geometrically_and_runs_its_steps() -> None:
    p = P.SimulatedAnnealing(T0=3.0, step=0.02, steps=1000, T_end_ratio=0.01, seed=0, level=1)
    assert p.temperature(0) == pytest.approx(3.0)
    assert p.temperature(999) == pytest.approx(0.03)
    assert p.temperature(500) == pytest.approx(3.0 * 0.01 ** (500 / 999))
    h = run_offline(p)
    assert p.iterations == 1000
    assert 0 < len(h.entries) <= 1000
    assert len(h.entries) + p.infeasible == 1000
    assert p.best_peak < h.entries[0].peak_hat


# ------------------------------------------------------------------------------- P3 (gradient)


def test_p3_proposes_starts_times_steps_and_descends() -> None:
    p = P.BatchedGradient(fake_oracle, eta=0.03, starts=64, steps=50, halvings=5, seed=0,
                          level=2)
    h = run_offline(p)
    assert len(h.entries) == 64 * 50
    first = np.mean([e.peak_hat for e in h.entries[:64]])
    last = np.mean([e.peak_hat for e in h.entries[-64:]])
    assert last < first


def test_p3_halves_an_infeasible_step_then_stays(monkeypatch) -> None:
    p = P.BatchedGradient(fake_oracle, eta=0.1, starts=2, steps=2, halvings=3, seed=0, level=3)
    hist = P.History()
    xs = p.next(hist)
    P.record(hist, fake_evaluate(xs), fake_group, p)
    calls = []
    monkeypatch.setattr(P, "is_feasible", lambda x: calls.append(1) or False)
    xs2 = p.next(hist)
    assert len(calls) == 2 * (3 + 1)  # eta, eta/2, eta/4, eta/8 for each start
    for a, b in zip(xs, xs2, strict=True):
        np.testing.assert_array_equal(a, b)  # every step infeasible: each start stays put
    assert p.stuck == 2


# ------------------------------------------------------------------------------- P4 (replay)


def test_p4_replays_logged_design_vectors_in_order(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    xs = []
    while len(xs) < 3:
        x = rng.random(DIM)
        if P.is_feasible(x):
            xs.append(x)
    f = tmp_path / "p4.jsonl"
    f.write_text("".join(json.dumps({"design_vector": x.tolist()}) + "\n" for x in xs), "utf-8")
    h = run_offline(P.Replay(f))
    assert [e.x for e in h.entries] == [tuple(x) for x in xs]
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"design_vector": [0.5] * 35}) + "\n", "utf-8")
    with pytest.raises(ValueError, match="36"):
        P.Replay(bad)


def test_bad_settings_raise() -> None:
    with pytest.raises(ValueError, match="lam"):
        P.TiltedMCMC(lam=0.0, step=0.05, n_proposals=10, seed=0, level=1)
    with pytest.raises(ValueError, match="T_end_ratio"):
        P.SimulatedAnnealing(T0=1.0, step=0.05, steps=10, T_end_ratio=1.5, seed=0, level=1)
    with pytest.raises(ValueError, match="eta"):
        P.BatchedGradient(fake_oracle, eta=0.0, starts=2, steps=2, halvings=1, seed=0, level=1)
    with pytest.raises(ValueError, match="level"):
        P.TiltedMCMC(lam=1.0, step=0.05, n_proposals=10, seed=0, level=4)


def test_reflection_keeps_points_in_the_cube() -> None:
    x = np.array([-0.3, 1.2, 2.7, -1.6, 0.4, 1.0, 0.0])
    y = P.reflect(x)
    np.testing.assert_allclose(y, [0.3, 0.8, 0.7, 0.4, 0.4, 1.0, 0.0])


# --------------------------------------------------------------- real evaluator, tiny ensemble


N = 32


def tiny_ensemble(members: int = 2) -> Ensemble:
    torch.manual_seed(0)
    models = []
    for _ in range(members):
        m = zoo.build({"arch": "fno", **zoo.IO, "width": 8, "modes": 4, "n_layers": 1,
                       "proj_hidden": 8})
        m.eval()
        models.append(m)
    off = np.zeros(inp.N_CHANNELS)
    scale = np.ones(inp.N_CHANNELS)
    off[0:4], scale[0:4] = -2.0, 1.0
    scale[inp.Q_CHANNEL] = 1e4
    norm = Normaliser(off, scale, theta_scale=10.0, dataset_hash="0" * 64, N=N, n_boards=1)
    return Ensemble(models, norm, torch.device("cpu"), members=[])


@pytest.fixture(scope="module")
def ens() -> Ensemble:
    return tiny_ensemble()


@pytest.fixture(scope="module")
def designs() -> list[np.ndarray]:
    rng = np.random.default_rng(5)
    out = []
    while len(out) < 4:
        x = rng.random(DIM)
        if P.is_feasible(x):
            out.append(x)
    return out


def test_evaluator_reports_label_free_quantities(ens, designs) -> None:
    from gate.scale import hot_residual_raw, label_free_scale
    from solvers.fvm import coefficients

    ood = OODScore([np.zeros(8)] * 2, [np.eye(8)] * 2, threshold=1.0)
    ev = P.SurrogateEvaluator(ens, ood, SPEC, n=N, m=16, batch=3)
    preds = ev(designs)
    raw = np.stack([inp.build(rasterise(design_board(x), SPEC, N), SPEC) for x in designs])
    mu, sd = ens.predict(raw)
    for i, p in enumerate(preds):
        assert p.x_hash == P.x_hash(designs[i])
        assert p.peak_hat == pytest.approx(float(mu[i].max()))
        rec = design_board(designs[i])
        c = coefficients(SPEC, rasterise(rec, SPEC, N), rec["dirichlet_edges"])
        _, r_hot, sd_hot = label_free_scale(mu[i], sd[i], c, c["rhs"], m=16, lam=0.0, eps=1.0)
        assert p.r_hot == pytest.approx(r_hot) and p.sd_hot == pytest.approx(sd_hot)
        assert p.r_hot_raw == pytest.approx(hot_residual_raw(mu[i], c, c["rhs"], m=16))
        assert np.isfinite(p.ood_score) and p.ood_score >= 0
        l_, y, x = p.hot_cell
        assert mu[i][l_, y, x] == pytest.approx(float(mu[i].max()))
    with pytest.raises(InfeasibleDesignError):
        ev([np.zeros(DIM)])  # all chips in one corner: the evaluator refuses, never guesses


def test_differentiable_inputs_match_the_rasteriser(designs) -> None:
    di = DesignInputs(SPEC, n=64)
    got = di(torch.tensor(np.stack(designs))).detach().numpy()
    for k, x in enumerate(designs):
        want = inp.build(rasterise(design_board(x), SPEC, 64), SPEC)
        for ch in (1, 2, 3, 8, 9, 10, 11, 12):  # design-independent: exactly the same
            np.testing.assert_allclose(got[k, ch], want[ch], rtol=0, atol=1e-6)
        for ch in (0, 4, 5, 6, 7):  # area fractions vs 4x4 supersampling: edges only
            d = np.abs(got[k, ch] - want[ch])
            scale = max(1e-12, float(np.abs(want[ch]).max()))
            assert np.mean(d < 1e-4 * scale) > 0.85, f"channel {ch}"
            assert d.mean() < 0.02 * scale, f"channel {ch}"


def test_differentiable_inputs_gradient_is_exact(designs) -> None:
    di = DesignInputs(SPEC, n=N, dtype=torch.float64)
    W = torch.tensor(np.random.default_rng(1).normal(size=(13, N, N)))
    x = designs[0]
    xt = torch.tensor(x[None], requires_grad=True)
    (g,) = torch.autograd.grad((di(xt) * W).sum(), xt)

    def f(v: np.ndarray) -> float:
        return float((di(torch.tensor(v[None])) * W).sum())

    for j in (0, 1, 8, 20, 32, 35):  # chip x, chip y, rect corner, rect size, via densities
        e = np.zeros(DIM)
        e[j] = 1e-6
        assert float(g[0, j]) == pytest.approx((f(x + e) - f(x - e)) / 2e-6, rel=1e-5, abs=1e-6)


def _away_from_cell_edges(n: int, clearance_mm: float) -> np.ndarray:
    """A feasible design whose chip-0 and rect-0 edges are all >= clearance from a cell boundary:
    the area fractions are piecewise linear in the edges, with kinks at cell boundaries, so a
    finite difference must not straddle one."""
    from data import sampler

    cell = 100.0 / n
    m = float(sampler._S1.chip_edge_margin_min.value)
    task = sampler._design_task()
    cw, chh = task["chips"][0]["w"], task["chips"][0]["h"]
    lo, hi = task["mapping"]["rect_side"]
    rng = np.random.default_rng(11)
    for _ in range(20_000):
        x = rng.random(DIM)
        if not P.is_feasible(x):
            continue
        cx0, cy0 = m + x[0] * (100 - 2 * m - cw), m + x[1] * (100 - 2 * m - chh)
        rw = lo + x[10] * (hi - lo)
        rx0 = x[8] * (100 - rw)
        edges = (cx0, cx0 + cw, cy0, cy0 + chh, rx0, rx0 + rw)
        if all(min(e % cell, cell - e % cell) >= clearance_mm for e in edges):
            return x
    raise RuntimeError("no design with clear edges")


def test_gradient_oracle_is_the_gradient_of_the_hottest_cell(ens, designs) -> None:
    # T̂_peak is a max over cells, so its gradient is that of the current hottest cell (cells
    # near the peak can tie within float32); compare against finite differences of that cell.
    di = DesignInputs(SPEC, n=N)
    oracle = P.GradientOracle(ens, di, chunk=2)
    x = _away_from_cell_edges(N, clearance_mm=0.3)
    peak, g = oracle(x[None])
    assert g.shape == (1, DIM) and np.isfinite(g).all() and np.abs(g).max() > 0

    def field(v: np.ndarray) -> np.ndarray:
        raw = di(torch.tensor(v[None], dtype=torch.float32)).detach().numpy()
        return ens.predict(raw)[0][0]

    mu = field(x)
    assert peak[0] == pytest.approx(float(mu.max()), rel=1e-5)
    cell = np.unravel_index(int(np.argmax(mu)), mu.shape)
    h = 2e-3  # x units: 0.16 mm of chip travel, well inside a 3.125 mm cell
    for j in (0, 1, 8, 32):
        e = np.zeros(DIM)
        e[j] = h
        fd = (field(np.clip(x + e, 0, 1))[cell] - field(np.clip(x - e, 0, 1))[cell]) / (2 * h)
        assert g[0, j] == pytest.approx(fd, rel=0.05, abs=2e-4)
