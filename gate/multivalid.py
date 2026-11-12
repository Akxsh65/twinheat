"""G6: audited multivalid gate, one threshold per (intersecting) group (CLAUDE.md §8.5 Extension A,
§11.1, §18.3; decision D9). The threshold rule is MVP, Algorithm 1 of Bastani et al., "Practical
Adversarial Multivalid Conformal Prediction", arXiv 2206.01067v1 (docs/refs/bastani2022_mvp.txt),
read with the [sic] corrections listed at the end of that file.

Score map (paper footnote 10): S̃ = S⁺/(1 + S⁺) ∈ [0, 1), threshold back by q = q̃/(1 − q̃)
(q̃ = 1 ↔ q = ∞). With smoothing noise (D9 rule 4) S̃ = (S⁺ + U)/(1 + S⁺ + U), U ~ U[0, w].

Per committed design x_t with groups 𝒢(x_t) (label-free, from `membership`):
    C_i = Σ_{G ∈ 𝒢(x_t)} [exp(η V^{G,i}/f(n^{G,i})) − exp(−η V^{G,i}/f(n^{G,i}))] / f(n^{G,i}),
          i = 1..m, with (n, V) the counts and coverage errors so far (Definitions 3.2-3.3)
    all C_i > 0 → q̃ = 0;  all C_i < 0 → q̃ = 1;  else the first i* ∈ [m − 1] with
    C_{i*}·C_{i*+1} ≤ 0, p = |C_{i*+1}| / (|C_{i*+1}| + |C_{i*}|) (0/0 = 1), and
    q̃ = i*/m − 1/(rm) with probability p, i*/m with probability 1 − p.
Per audit (only audited designs, B8): bucket i = B⁻¹(q̃_t); for each G ∈ 𝒢(x_t):
    n^{G,i} += 1,  V^{G,i} += Cover(q̃_t, S̃_t) − (1 − α)      (MVP's δ = our α)

Thresholds live on the grid P^{rm} as integers k (q̃ = k/(rm)), so bucket lookup is exact.
f(n) = sqrt((n+1) ln^{1+ϵ}(n+2)), K_ϵ = Σ_{n≥0} 1/f(n)², η = sqrt(ln(|𝒢| m) / (2 K_ϵ |𝒢| m))
(Theorem 3.1). The paper writes "log" in f and "ln" in η; we use the natural log in both
(UNSURE: the reviewer should confirm; K_ϵ is computed with the same log, so η stays consistent).

D9 rules: decide() (a `query` preview) shows the deterministic upper candidate from the current
state and never draws; commit() runs after coin.commit(id), draws MVP's coin and the noise once,
from this gate's own PCG64 stream, and never re-rolls; update() takes audited designs only.
The numbers m, r, ϵ, w are D4 settings: no defaults.

What this does NOT give (CLAUDE.md §18.3): Theorem 3.1 is in expectation, per (group, bucket),
on the rounds MVP is fed (here: audited steps); the transfer to all steps and the one-sided
accepted-error statement are the week-7 proof target, else H8 is reported as empirical.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Sequence
from typing import Any

import numpy as np

from gate.base import AuditOutcome, AuditRecord, BaseGate, GateDecision, GateInput
from gate.coin import AuditCoin, CoinError

__all__ = ["MultiValidGate", "eta", "f", "k_eps", "q_from_grid", "smoothed_score"]

_K_TERMS = 1_000_000


def f(n: np.ndarray, eps: float) -> np.ndarray:
    """f(n) = sqrt((n + 1) ln^{1+ϵ}(n + 2)) (paper p. 7)."""
    n = np.asarray(n, dtype=np.float64)
    return np.sqrt((n + 1) * np.log(n + 2) ** (1 + eps))


def k_eps(eps: float) -> tuple[float, float, float]:
    """(K_ϵ, lower, upper) for K_ϵ = Σ_{n≥0} 1/f(n)². Sums the first 10⁶ terms and brackets the
    tail between the integrals 1/(ϵ ln^ϵ(N+3)) and 1/(ϵ ln^ϵ(N+1)) (the terms decrease; compare
    with 1/((x+a) ln^{1+ϵ}(x+a)) for a = 2 and a = 1). Returns the midpoint and the bracket."""
    if not (math.isfinite(eps) and eps > 0):
        raise ValueError(f"eps must be finite and > 0, got {eps}")
    n = np.arange(_K_TERMS, dtype=np.float64)
    head = float(np.sum(1.0 / f(n, eps) ** 2))
    N = _K_TERMS - 1
    lo = head + 1 / (eps * math.log(N + 3) ** eps)
    hi = head + 1 / (eps * math.log(N + 1) ** eps)
    return (lo + hi) / 2, lo, hi


def eta(n_groups: int, m: int, eps: float) -> float:
    """η of Theorem 3.1: sqrt(ln(|𝒢| m) / (2 K_ϵ |𝒢| m))."""
    gm = n_groups * m
    if gm < 2:
        raise ValueError(f"|G|·m must be >= 2 (ln(|G|m) > 0), got {gm}")
    return math.sqrt(math.log(gm) / (2 * k_eps(eps)[0] * gm))


def q_from_grid(k: int, rm: int) -> float:
    """Threshold on the S scale for q̃ = k/(rm): q = q̃/(1 − q̃), ∞ at q̃ = 1."""
    if not (0 <= k <= rm):
        raise ValueError(f"grid index {k} outside [0, {rm}]")
    return math.inf if k == rm else k / (rm - k)


def smoothed_score(S: np.ndarray | float, U: np.ndarray | float) -> np.ndarray:
    """S̃ = (S + U)/(1 + S + U) for S ≥ 0, U ≥ 0 (D9 rule 4; U = 0 is footnote 10's map)."""
    S, U = np.asarray(S, dtype=np.float64), np.asarray(U, dtype=np.float64)
    if (S < 0).any() or (U < 0).any() or not (np.isfinite(S).all() and np.isfinite(U).all()):
        raise ValueError("S and U must be finite and >= 0")
    return (S + U) / (1 + S + U)


class MultiValidGate(BaseGate):
    gate_id = "G6"

    def __init__(self, *, alpha: float, tau: float, p: float, groups: Sequence[str], m: int,
                 r: int, eps: float, noise_width: float, coin: AuditCoin, seed: int,
                 membership: Callable[[GateInput], Collection[str]]) -> None:
        if not (math.isfinite(alpha) and 0 < alpha < 1):
            raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
        if not (math.isfinite(tau) and tau > 0):
            raise ValueError(f"tau must be finite and > 0 (K), got {tau}")
        if not (math.isfinite(p) and 0 < p <= 1):
            raise ValueError(f"p must lie in (0, 1] (uniform audit probability), got {p}")
        groups = list(groups)
        if not groups or len(set(groups)) != len(groups) or \
                not all(isinstance(g, str) and g for g in groups):
            raise ValueError(f"groups must be distinct non-empty strings, got {groups!r}")
        for name, v in (("m", m), ("r", r)):
            if not (isinstance(v, int) and not isinstance(v, bool) and v >= 1):
                raise ValueError(f"{name} must be an int >= 1, got {v!r}")
        if m < 2:
            raise ValueError(f"m must be >= 2 buckets, got {m}")
        if not (math.isfinite(noise_width) and noise_width >= 0):
            raise ValueError(f"noise_width must be finite and >= 0, got {noise_width}")
        if not isinstance(coin, AuditCoin):
            raise TypeError(f"coin must be an AuditCoin, got {type(coin).__name__}")
        if not (isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0):
            raise ValueError(f"seed must be a non-negative int, got {seed!r}")
        if not callable(membership):
            raise TypeError("membership must be callable: GateInput -> group names")
        self.alpha, self.tau, self.p = float(alpha), float(tau), float(p)
        self.groups, self.m, self.r = groups, m, r
        self.eps, self.noise_width, self.seed = float(eps), float(noise_width), seed
        self.eta = eta(len(groups), m, self.eps)
        if not 0 < self.eta < 0.5:
            raise ValueError(f"eta = {self.eta} outside (0, 1/2) (Definition 3.4, Lemma 3.1)")
        self._gidx = {g: i for i, g in enumerate(groups)}
        self.coin, self._membership = coin, membership
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._n = np.zeros((len(groups), m), dtype=np.int64)
        self._V = np.zeros((len(groups), m), dtype=np.float64)
        self._pending: dict[str, dict[str, Any]] = {}  # committed, waiting for a possible audit
        self._committed: set[str] = set()
        self._used: set[str] = set()

    # ------------------------------------------------------------------ the threshold rule

    def _groups_of(self, inp: GateInput) -> list[int]:
        names = sorted(set(self._membership(inp)))
        if not names:
            raise ValueError(f"design {inp.design_id!r} is in no group: G6 needs every design in "
                             "at least one group (include an 'all' group)")
        bad = [g for g in names if g not in self._gidx]
        if bad:
            raise ValueError(f"design {inp.design_id!r}: unknown group(s) {bad}; known: "
                             f"{self.groups}")
        return [self._gidx[g] for g in names]

    def _candidates(self, group_names: Collection[str]) -> tuple[int, int, float]:
        """(k_low, k_high, P(k_low)) on the grid P^{rm} for a design in these groups."""
        gi = [self._gidx[g] for g in group_names]
        fn = f(self._n[gi], self.eps)  # [g, m]
        x = self.eta * self._V[gi] / fn
        C = np.sum((np.exp(x) - np.exp(-x)) / fn, axis=0)  # [m]
        rm = self.r * self.m
        if np.all(C > 0):
            return 0, 0, 1.0
        if np.all(C < 0):
            return rm, rm, 1.0
        prod = C[:-1] * C[1:]
        i_star = int(np.flatnonzero(prod <= 0)[0]) + 1  # first sign change, 1-indexed
        a, b = abs(C[i_star - 1]), abs(C[i_star])  # |C_{i*}|, |C_{i*+1}|
        p_low = 1.0 if a + b == 0 else b / (a + b)
        return i_star * self.r - 1, i_star * self.r, p_low

    def _bucket(self, k: int) -> int:
        """0-indexed bucket of q̃ = k/(rm): B_m(i) = [(i−1)/m, i/m), the last one closed."""
        return min(self.m - 1, k // self.r)

    def _decision(self, inp: GateInput, names: list[str], k: int) -> GateDecision:
        q = q_from_grid(k, self.r * self.m)
        B = math.inf if math.isinf(q) else q * inp.s
        return GateDecision(design_id=inp.design_id, B=B,
                            decision="accept" if B <= self.tau else "flag", p_audit=self.p,
                            group="|".join(names), q_t=q, alpha_t=None)

    # ------------------------------------------------------------------ decide / commit

    def decide(self, inp: GateInput) -> GateDecision:
        """Preview (D9 rule 1): the deterministic upper candidate; no draw, no state change."""
        names = [self.groups[i] for i in self._groups_of(inp)]
        return self._decision(inp, names, self._candidates(names)[1])

    def commit(self, inp: GateInput) -> GateDecision:
        """Binding decision (D9 rule 2): after coin.commit, one MVP draw and one noise draw."""
        if inp.design_id in self._committed:
            raise ValueError(f"design {inp.design_id!r} already committed to G6: no re-draws")
        if not self.coin.committed(inp.design_id):
            raise ValueError(f"G6.commit({inp.design_id!r}) before coin.commit: MVP's "
                             "randomisation is drawn only after the design is committed (D9)")
        names = [self.groups[i] for i in self._groups_of(inp)]
        k_lo, k_hi, p_lo = self._candidates(names)
        u, noise = self._rng.random(), self._rng.uniform(0.0, self.noise_width)  # always both
        k = k_lo if u < p_lo else k_hi
        self._committed.add(inp.design_id)
        self._pending[inp.design_id] = {"groups": names, "k": k, "noise": float(noise)}
        return self._decision(inp, names, k)

    # ------------------------------------------------------------------ update (audits only)

    def update(self, records: list[AuditRecord]) -> list[AuditOutcome]:
        seen: set[str] = set()
        for i, r in enumerate(records):
            if not isinstance(r, AuditRecord):
                raise TypeError(f"records[{i}] is a {type(r).__name__}, not an AuditRecord")
            if r.audited is not True:
                raise ValueError(f"records[{i}] ({r.design_id!r}) is not audited: only audited "
                                 "designs update G6 (B8)")
            if r.design_id in self._used or r.design_id in seen:
                raise ValueError(f"records[{i}] ({r.design_id!r}) was already used")
            try:
                drawn = self.coin.audited(r.design_id)
            except CoinError as e:
                raise ValueError(f"records[{i}] ({r.design_id!r}): no audit coin ({e})") from e
            if not drawn:
                raise ValueError(f"records[{i}] ({r.design_id!r}) claims audited but its coin "
                                 "came up 0")
            if r.design_id not in self._pending:
                raise ValueError(f"records[{i}] ({r.design_id!r}) was not committed to G6")
            seen.add(r.design_id)
        out = []
        rm = self.r * self.m
        for r in records:
            P = self._pending.pop(r.design_id)
            S = r.S_hi
            s_tilde = float(smoothed_score(S, P["noise"]))
            covered = s_tilde <= P["k"] / rm
            b = self._bucket(P["k"])
            gi = [self._gidx[g] for g in P["groups"]]
            self._n[gi, b] += 1
            self._V[gi, b] += float(covered) - (1 - self.alpha)
            self._used.add(r.design_id)
            out.append(AuditOutcome(design_id=r.design_id, T_lo=float(r.T_lo),
                                    T_hi=float(r.T_hi), S_hi=S, err=int(not covered),
                                    q_t=q_from_grid(P["k"], rm), alpha_t=self.alpha))
        return out

    # ------------------------------------------------------------------ state

    _SETTINGS = ("alpha", "tau", "p", "groups", "m", "r", "eps", "noise_width", "seed")

    def state_dict(self) -> dict[str, Any]:
        return {"gate": self.gate_id, **{k: getattr(self, k) for k in self._SETTINGS},
                "n": self._n.tolist(), "V": self._V.tolist(), "rng": self._rng.bit_generator.state,
                "pending": {k: dict(v) for k, v in sorted(self._pending.items())},
                "committed": sorted(self._committed), "used": sorted(self._used)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("gate") != self.gate_id:
            raise ValueError(f"state is for gate {state.get('gate')!r}, not {self.gate_id}")
        for k in self._SETTINGS:
            if state[k] != getattr(self, k):
                raise ValueError(f"state has {k} = {state[k]!r} but this gate has "
                                 f"{getattr(self, k)!r} (D4, D9)")
        n = np.asarray(state["n"], dtype=np.int64)
        V = np.asarray(state["V"], dtype=np.float64)
        if n.shape != self._n.shape or V.shape != self._V.shape or (n < 0).any():
            raise ValueError("G6 state arrays have the wrong shape or negative counts")
        rng = np.random.Generator(np.random.PCG64())
        rng.bit_generator.state = state["rng"]
        self._n, self._V, self._rng = n, V, rng
        self._pending = {str(k): dict(v) for k, v in state["pending"].items()}
        self._committed = {str(x) for x in state["committed"]}
        self._used = {str(x) for x in state["used"]}
