"""Proposers P1–P4 over the 36-dim design space, and the loop that feeds them (M3.3; CLAUDE.md
§11.2, §8.1 steps 1–4, §5 B8).

Every proposer exposes next(history) -> list of design vectors x in [0, 1]^36. `history` is a
History of HistoryEntry: the label-free prediction of every design proposed so far and, in an
online run, the gate's decision (accept / flag, and B). It never holds an audit coin: HistoryEntry
has no such field, it is frozen and slotted, and History takes nothing else (tests/
test_proposers.py). Coins are drawn by run_online after commit and returned to its caller only.
Proposers only emit feasible designs (data/sampler.design_board does not raise); an infeasible
candidate is a rejected move inside the proposer and is never proposed.

  P1 TiltedMCMC         random-walk Metropolis on p(x) ∝ p₀(x)·exp(−λ·T̂_peak(x)), p₀ uniform on the
                        feasible cube (as data/sampler's "design" mode); Gaussian steps of sd
                        `step`, reflected into the cube (a symmetric proposal). λ in 1/K.
  P2 SimulatedAnnealing same moves; accept downhill always, uphill with exp(−ΔT̂/T_k), geometric
                        cooling T_k = T0·r^(k/(steps−1)), r = T_end_ratio; `steps` iterations.
  P3 BatchedGradient    `starts` designs from p₀, then `steps − 1` projected steps
                        x ← reflect(x − η·g/‖g‖∞), g = ∇T̂_peak from GradientOracle (autograd through
                        models/diff_inputs and the ensemble mean); a step that makes a design
                        infeasible is halved up to `halvings` times, after which that start stays.
                        starts × steps proposals.
  P4 Replay             design vectors logged from human sessions, in order (one JSON object per
                        line with "design_vector": 36 floats; written by experiments/p4_import.py).

The P1 chain length, P2 presets and P3 step sizes are M's choices of 18 Nov 2026
(docs/decisions.md, experiments/configs/trajectories.yaml), to be frozen under D4. Every
proposer draws from its own numpy Generator(PCG64(seed)), so a seed fixes the trajectory.
"""

from __future__ import annotations

import abc
import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

from data import sampler
from data.raster import rasterise
from gate.base import GateInput
from gate.scale import hot_residual_raw, label_free_scale
from models import inputs as inp
from models.ood_score import pooled_features
from solvers.fvm import coefficients

__all__ = [
    "DIM", "FORBIDDEN_FIELDS", "BatchedGradient", "GradientOracle", "History", "HistoryEntry",
    "Prediction", "Proposer", "Replay", "SimulatedAnnealing", "SurrogateEvaluator", "TiltedMCMC",
    "is_feasible", "record", "reflect", "run_offline", "run_online", "sample_p0", "x_hash",
]

DIM = int(sampler._S2.design_space.dimension.value)  # 36 (§7.4)
LEVELS = (1, 2, 3)
# names a coin or an audit outcome could hide under; HistoryEntry must have none of them (B8)
FORBIDDEN_FIELDS = frozenset({"audited", "audit", "coin", "A_t", "a_t", "draw", "T_lo", "T_hi",
                              "S_hi", "err", "oracle_peak"})
_MAX_P0_TRIES = 10_000


# ----------------------------------------------------------------------------- design space


def x_hash(x: np.ndarray) -> str:
    """SHA-256 of x as little-endian float64 (the §10.4 x_hash)."""
    return hashlib.sha256(np.asarray(x, dtype="<f8").tobytes()).hexdigest()


def reflect(x: np.ndarray) -> np.ndarray:
    """Fold x back into [0, 1] by mirror reflection at 0 and 1 (any distance)."""
    y = np.mod(np.asarray(x, dtype=np.float64), 2.0)
    return np.where(y > 1.0, 2.0 - y, y)


def is_feasible(x: np.ndarray) -> bool:
    try:
        sampler.design_board(np.asarray(x, dtype=np.float64))
    except sampler.InfeasibleDesignError:
        return False
    return True


def sample_p0(rng: np.random.Generator) -> np.ndarray:
    """One x ~ p₀: uniform on the cube, rejected until feasible (data/sampler "design" mode)."""
    for _ in range(_MAX_P0_TRIES):
        x = rng.random(DIM)
        if is_feasible(x):
            return x
    raise RuntimeError(f"no feasible design vector in {_MAX_P0_TRIES} draws")


def _check_level(level: int | None, allow_none: bool = False) -> int | None:
    if level is None and allow_none:
        return None
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    return level


def _positive(name: str, v: float) -> float:
    v = float(v)
    if not (math.isfinite(v) and v > 0):
        raise ValueError(f"{name} must be finite and > 0, got {v}")
    return v


def _count(name: str, v: int, lo: int = 1) -> int:
    if not (isinstance(v, int) and not isinstance(v, bool) and v >= lo):
        raise ValueError(f"{name} must be an int >= {lo}, got {v!r}")
    return v


# ---------------------------------------------------------------------------------- history


@dataclass(frozen=True)
class Prediction:
    """Label-free summary of one design's surrogate prediction (K unless stated)."""

    x: np.ndarray
    x_hash: str
    peak_hat: float  # max of the ensemble mean over all layers
    r_hot: float  # mean |(Aμ − b)/diag A| over the m hottest cells (gate/scale.py)
    sd_hot: float  # mean ensemble sd over the same cells
    r_hot_raw: float  # W, DeepOHeat-v2's unscaled residual (G1 variant, D13)
    ood_score: float  # Mahalanobis OOD score of the pooled FNO features (models/ood_score.py)
    hot_cell: tuple[int, int, int]  # (layer, y, x) of the peak
    edge_hotspot: bool  # peak within 5 mm of an edge (gate/groups.py)


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    """What a proposer may see about one proposed design: prediction and gate decision only."""

    t: int
    design_id: str
    x: tuple[float, ...]
    x_hash: str
    peak_hat: float
    r_hot: float
    sd_hot: float
    r_hot_raw: float
    ood_score: float
    group: str
    decision: str | None  # "accept" | "flag" online; None in an offline (gate-free) run
    B: float | None


assert not {f.name for f in fields(HistoryEntry)} & FORBIDDEN_FIELDS


class History:
    """Append-only sequence of HistoryEntry."""

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: list[HistoryEntry] = []

    def append(self, e: HistoryEntry) -> None:
        if type(e) is not HistoryEntry:
            raise TypeError(f"History holds HistoryEntry objects only, got {type(e).__name__}")
        if e.t != len(self._entries):
            raise ValueError(f"entry t = {e.t} but the history has {len(self._entries)} entries")
        self._entries.append(e)

    @property
    def entries(self) -> tuple[HistoryEntry, ...]:
        return tuple(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


# -------------------------------------------------------------------------------- proposers


class Proposer(abc.ABC):
    name: str  # "P1".."P4"
    level: int | None

    @property
    @abc.abstractmethod
    def done(self) -> bool: ...

    @abc.abstractmethod
    def next(self, history: History) -> list[np.ndarray]:
        """Designs to propose now; the previous round's predictions are already in history."""


def _last(history: History, k: int, pending: Sequence[np.ndarray]) -> list[HistoryEntry]:
    got = history.entries[-k:] if k else ()
    if len(got) != k or any(tuple(np.asarray(p, dtype=np.float64)) != e.x
                            for p, e in zip(pending, got, strict=True)):
        raise RuntimeError("history does not end with this proposer's last proposals")
    return list(got)


class _Walker(Proposer):
    """Shared single-chain logic of P1 and P2: one candidate per round, reflected Gaussian moves."""

    def __init__(self, step: float, seed: int) -> None:
        self.step = _positive("step", step)
        self.rng = np.random.Generator(np.random.PCG64(_count("seed", seed, 0)))
        self.current: np.ndarray | None = None
        self.current_peak: float | None = None
        self.best_peak = math.inf
        self.pending: np.ndarray | None = None
        self.proposed = 0
        self.iterations = 0
        self.infeasible = 0
        self.last_accepted: bool | None = None
        self.last_accept_prob: float | None = None

    @abc.abstractmethod
    def _accept_prob(self, delta: float) -> float: ...

    @abc.abstractmethod
    def _more(self) -> bool: ...

    def _observe(self, history: History) -> None:
        if self.pending is None:
            return
        (e,) = _last(history, 1, [self.pending])
        self.best_peak = min(self.best_peak, e.peak_hat)
        if self.current is None:
            self.current, self.current_peak = self.pending, e.peak_hat
        else:
            prob = self._accept_prob(e.peak_hat - self.current_peak)
            self.last_accept_prob = prob
            self.last_accepted = bool(self.rng.random() < prob)
            if self.last_accepted:
                self.current, self.current_peak = self.pending, e.peak_hat
        self.pending = None

    def next(self, history: History) -> list[np.ndarray]:
        self._observe(history)
        if not self._more():
            return []
        if self.current is None:
            x = sample_p0(self.rng)
            self.iterations += 1
        else:
            while True:
                x = reflect(self.current + self.step * self.rng.standard_normal(DIM))
                self.iterations += 1
                if is_feasible(x):
                    break
                self.infeasible += 1  # density 0: a rejected move, the chain stays
                if not self._more():
                    return []
        self.pending = x
        self.proposed += 1
        return [x]


class TiltedMCMC(_Walker):
    """P1: Metropolis on p₀(x)·exp(−λ T̂_peak(x)); `n_proposals` feasible candidates."""

    name = "P1"

    def __init__(self, *, lam: float, step: float, n_proposals: int, seed: int,
                 level: int) -> None:
        super().__init__(step, seed)
        self.lam = _positive("lam", lam)
        self.n_proposals = _count("n_proposals", n_proposals)
        self.level = _check_level(level)

    def _accept_prob(self, delta: float) -> float:
        return 1.0 if delta <= 0 else math.exp(-self.lam * delta)

    def _more(self) -> bool:
        return self.proposed < self.n_proposals

    @property
    def done(self) -> bool:
        return not self._more() and self.pending is None


class SimulatedAnnealing(_Walker):
    """P2: `steps` iterations (start included) with geometric cooling from T0 (K)."""

    name = "P2"

    def __init__(self, *, T0: float, step: float, steps: int, T_end_ratio: float, seed: int,
                 level: int) -> None:
        super().__init__(step, seed)
        self.T0 = _positive("T0", T0)
        self.steps = _count("steps", steps, 2)
        if not (math.isfinite(T_end_ratio) and 0 < T_end_ratio < 1):
            raise ValueError(f"T_end_ratio must lie in (0, 1), got {T_end_ratio}")
        self.T_end_ratio = float(T_end_ratio)
        self.level = _check_level(level)

    def temperature(self, k: int) -> float:
        return self.T0 * self.T_end_ratio ** (k / (self.steps - 1))

    def _accept_prob(self, delta: float) -> float:
        # the candidate was drawn at iteration self.iterations - 1
        return 1.0 if delta <= 0 else math.exp(-delta / self.temperature(self.iterations - 1))

    def _more(self) -> bool:
        return self.iterations < self.steps

    @property
    def done(self) -> bool:
        return not self._more() and self.pending is None


class BatchedGradient(Proposer):
    """P3: `starts` designs from p₀, then projected normalised-gradient steps; starts × steps."""

    name = "P3"

    def __init__(self, oracle: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]], *,
                 eta: float, starts: int, steps: int, halvings: int, seed: int,
                 level: int) -> None:
        self.oracle = oracle
        self.eta = _positive("eta", eta)
        self.starts = _count("starts", starts)
        self.steps = _count("steps", steps)
        self.halvings = _count("halvings", halvings, 0)
        self.level = _check_level(level)
        self.rng = np.random.Generator(np.random.PCG64(_count("seed", seed, 0)))
        self.X: np.ndarray | None = None
        self.round = 0
        self.stuck = 0  # start-steps where every halving was infeasible

    @property
    def done(self) -> bool:
        return self.round >= self.steps

    def next(self, history: History) -> list[np.ndarray]:
        if self.done:
            return []
        if self.X is None:
            self.X = np.stack([sample_p0(self.rng) for _ in range(self.starts)])
        else:
            _last(history, self.starts, list(self.X))
            _, g = self.oracle(self.X)
            g = np.asarray(g, dtype=np.float64)
            if g.shape != self.X.shape or not np.isfinite(g).all():
                raise FloatingPointError(f"gradient oracle returned {g.shape}, finite = "
                                         f"{np.isfinite(g).all()}")
            for i in range(self.starts):
                gmax = float(np.abs(g[i]).max())
                if gmax == 0:
                    continue  # flat: nothing to follow
                d = g[i] / gmax
                for h in range(self.halvings + 1):
                    y = reflect(self.X[i] - self.eta * 0.5 ** h * d)
                    if is_feasible(y):
                        self.X[i] = y
                        break
                else:
                    self.stuck += 1
        self.round += 1
        return [x.copy() for x in self.X]


class Replay(Proposer):
    """P4: logged design vectors, replayed in order."""

    name = "P4"
    level = None

    def __init__(self, log_path: Path | str) -> None:
        self.path = Path(log_path)
        xs = []
        for k, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            v = json.loads(line).get("design_vector")
            x = np.asarray(v, dtype=np.float64) if v is not None else None
            if x is None or x.shape != (DIM,) or not np.isfinite(x).all() \
                    or (x < 0).any() or (x > 1).any():
                raise ValueError(f"{self.path}:{k}: design_vector must be {DIM} floats in [0, 1]")
            if not is_feasible(x):
                raise ValueError(f"{self.path}:{k}: design_vector is infeasible")
            xs.append(x)
        self.xs, self.i = xs, 0

    @property
    def done(self) -> bool:
        return self.i >= len(self.xs)

    def next(self, history: History) -> list[np.ndarray]:
        if self.done:
            return []
        self.i += 1
        return [self.xs[self.i - 1]]


# ------------------------------------------------------------------- surrogate and gradient


class SurrogateEvaluator:
    """Exact path for every proposed design: design_board → rasterise → inputs → ensemble."""

    def __init__(self, ensemble, ood, spec, *, n: int, m: int, batch: int) -> None:
        self.ens, self.ood, self.spec = ensemble, ood, spec
        self.n, self.m, self.batch = _count("n", n, 2), _count("m", m), _count("batch", batch)

    def __call__(self, xs: Sequence[np.ndarray]) -> list[Prediction]:
        from gate.groups import hotspot_near_edge

        out: list[Prediction] = []
        for s in range(0, len(xs), self.batch):
            chunk = [np.asarray(x, dtype=np.float64) for x in xs[s:s + self.batch]]
            recs = [sampler.design_board(x) for x in chunk]  # raises InfeasibleDesignError
            rasters = [rasterise(r, self.spec, self.n) for r in recs]
            raw = np.stack([inp.build(r, self.spec) for r in rasters])
            mu, sd = self.ens.predict(raw)
            xn = torch.from_numpy(self.ens.norm.normalise_x(raw)).to(self.ens.device)
            feats = [pooled_features(mdl, xn).cpu().numpy() for mdl in self.ens.models]
            ood = self.ood.score(feats)
            for k, (x, rec, ras) in enumerate(zip(chunk, recs, rasters, strict=True)):
                c = coefficients(self.spec, ras, rec["dirichlet_edges"])
                # lam and eps do not enter r_hot or sd_hot; s is formed later with D4's values
                _, r_hot, sd_hot = label_free_scale(mu[k], sd[k], c, c["rhs"], m=self.m,
                                                    lam=0.0, eps=1.0)
                hot = tuple(int(v) for v in np.unravel_index(int(np.argmax(mu[k])), mu[k].shape))
                out.append(Prediction(
                    x=x, x_hash=x_hash(x), peak_hat=float(mu[k].max()), r_hot=r_hot,
                    sd_hot=sd_hot, r_hot_raw=hot_residual_raw(mu[k], c, c["rhs"], m=self.m),
                    ood_score=float(ood[k]), hot_cell=hot,
                    edge_hotspot=hotspot_near_edge(hot, self.n)))
        return out


class GradientOracle:
    """(T̂_peak, ∇_x T̂_peak) of the ensemble mean through models/diff_inputs.DesignInputs."""

    def __init__(self, ensemble, design_inputs, *, chunk: int) -> None:
        self.ens, self.di, self.chunk = ensemble, design_inputs, _count("chunk", chunk)
        dt = design_inputs.dtype
        dev = ensemble.device
        self.off = torch.tensor(ensemble.norm.offset, dtype=dt, device=dev).view(1, -1, 1, 1)
        self.scale = torch.tensor(ensemble.norm.scale, dtype=dt, device=dev).view(1, -1, 1, 1)
        self.di.to(dev)

    def __call__(self, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xs = np.asarray(xs, dtype=np.float64)
        peaks, grads = [], []
        dt = self.di.dtype
        for s in range(0, len(xs), self.chunk):
            xt = torch.tensor(xs[s:s + self.chunk], dtype=dt, device=self.ens.device,
                              requires_grad=True)
            xn = ((self.di(xt) - self.off) / self.scale).float()  # the models run in float32
            mu = torch.stack([m(xn) for m in self.ens.models]).mean(dim=0)
            peak = mu.amax(dim=(1, 2, 3)) * self.ens.norm.theta_scale  # K
            (g,) = torch.autograd.grad(peak.sum(), xt)
            peaks.append(peak.detach().cpu().numpy())
            grads.append(g.cpu().numpy())
        return np.concatenate(peaks), np.concatenate(grads)


# ----------------------------------------------------------------------------------- loops


def record(history: History, preds: Sequence[Prediction], grouper, proposer, *,
           decisions: Sequence[tuple[str | None, float | None]] | None = None) -> list[str]:
    """Append predictions (and decisions) to history; returns the new design ids."""
    ids = []
    for k, p in enumerate(preds):
        t = len(history)
        dec, B = decisions[k] if decisions is not None else (None, None)
        history.append(HistoryEntry(
            t=t, design_id=f"t{t}", x=tuple(float(v) for v in p.x), x_hash=p.x_hash,
            peak_hat=p.peak_hat, r_hot=p.r_hot, sd_hot=p.sd_hot, r_hot_raw=p.r_hot_raw,
            ood_score=p.ood_score, group=grouper(p, proposer), decision=dec, B=B))
        ids.append(f"t{t}")
    return ids


def run_offline(proposer: Proposer, evaluate, grouper, history: History, *,
                max_proposals: int) -> History:
    """Gate-free trajectory (§11.5: one oracle-labelled set scores every gate offline)."""
    while not proposer.done and len(history) < max_proposals:
        xs = proposer.next(history)
        if not xs:
            break
        record(history, evaluate(xs), grouper, proposer)
    return history


def run_online(proposer: Proposer, evaluate, grouper, gate, coin, *, scale,
               max_proposals: int = 10**9) -> tuple[History, list[tuple[str, bool]]]:
    """§8.1 steps 1–4 with a live gate: propose → predict → group → decide → commit → coin.

    scale(pred) -> s (K, label-free; needs D4's λ, ε). Returns the history (no coins) and the
    (design_id, audited) pairs, which go to the verifier and the gate's update, never back to the
    proposer."""
    history, audits = History(), []
    while not proposer.done and len(history) < max_proposals:
        xs = proposer.next(history)
        if not xs:
            break
        for p in evaluate(xs):
            t = len(history)
            design_id = f"t{t}"
            group = grouper(p, proposer)  # label-free, before any coin
            gi = GateInput(design_id=design_id, x_hash=p.x_hash, peak_hat=p.peak_hat,
                           s=scale(p), r_hot=p.r_hot, sd_hot=p.sd_hot, proposer=proposer.name,
                           aggressiveness=proposer.level, group=group)
            dec = gate.decide(gi)
            coin.commit(design_id)
            audits.append((design_id, coin.draw(design_id, dec.p_audit)))
            history.append(HistoryEntry(
                t=t, design_id=design_id, x=tuple(float(v) for v in p.x), x_hash=p.x_hash,
                peak_hat=p.peak_hat, r_hot=p.r_hot, sd_hot=p.sd_hot, r_hot_raw=p.r_hot_raw,
                ood_score=p.ood_score, group=group, decision=dec.decision, B=dec.B))
    return history, audits

