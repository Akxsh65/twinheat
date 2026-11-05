"""Independent verification of a generated dataset (prompt N2.2; CLAUDE.md §5 data hygiene, B2, B3).

    python -m data.verify_dataset --root DIR [--report data/verify_report.md] [--figs data/figs]
                                  [--resolve-frac 0.01] [--seed 0] [--phi-N 64] [--allow-dirty]

DIR holds one folder per split, each written by data/generate.py (manifest.json + shard_XXXX/).
Every check runs on every split; all failures are collected, written to the report, and the exit
status is 1 if there is any (0 otherwise):

1. Files: every shard file's SHA-256, row count and field shape against its manifest
   (data.generate.verify).
2. Provenance: generated from a committed tree (no "-dirty", unless --allow-dirty for test data),
   with spec files whose SHA-256 still match config/ today, and the split complete.
3. Records: each record's SHA-256 (sorted-key JSON) equals its stored record_sha256, and no record
   hash appears in two splits (or twice in one).
4. Re-solve: a seeded random fraction (default 1 %, at least one board) of each split is rebuilt
   from its record (after the envelope) with an independent solver, PyAMG-GMRES at relative
   residual 1e-10 (solvers.amg_baseline), and must satisfy max|θ − stored| / max θ < 2e-3
   (the stored field is float16).
5. Statistics: count, envelope rescale fraction (≤ 20 %, CLAUDE.md §6.5, except splits exempted
   by a logged decision), θ_max P5/P50/P95, chip-count histogram, hotspot-location heat map, and
   mean φ per layer (rasterised at --phi-N; coverage fractions barely depend on N).

The report and figures carry the CLAUDE.md §4 footer (verifier commit, dataset hash, seed,
hardware, date). The dataset hash is the SHA-256 of all split manifests' shard hashes.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from data import generate as gen
from data.raster import rasterise
from solvers.amg_baseline import amg_solve
from solvers.fvm import assemble_board

__all__ = ["FIELD_TOL", "RESCALE_EXEMPT", "Report", "SplitStats", "cross_split_duplicates",
           "current_spec_hashes", "main", "provenance_problems", "rescale_problem",
           "resolve_indices", "verify_dataset"]

ROOT = gen.ROOT
FIELD_TOL = 2e-3  # prompt N2.2: max|θ − stored| / max θ, the stored field being float16
RESOLVE_TOL = gen.TOL  # 1e-10, the data-generation tolerance (CLAUDE.md §6.4)
# Splits allowed above the §6.5 20 % rescale limit, each by a logged decision (docs/decisions.md):
RESCALE_EXEMPT = {"ood_2": "docs/decisions.md, 'Dataset v1: OOD-2 envelope rescale above 20 %'"}


@dataclasses.dataclass
class SplitStats:
    split: str
    count: int = 0
    shards: int = 0
    commit: str = ""
    rescaled: int = 0
    theta_max: list[float] = dataclasses.field(default_factory=list)
    chip_hist: Counter = dataclasses.field(default_factory=Counter)
    hotspot: np.ndarray | None = None  # [N, N] counts of the peak cell's (y, x)
    phi_sum: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(4))
    resolved: int = 0
    worst_rel_err: float = 0.0
    worst_board: int = -1

    @property
    def rescale_fraction(self) -> float:
        return self.rescaled / self.count if self.count else math.nan

    @property
    def mean_phi(self) -> np.ndarray:
        return self.phi_sum / self.count if self.count else np.full(4, np.nan)


@dataclasses.dataclass
class Report:
    root: Path
    splits: dict[str, SplitStats]
    failures: list[str]
    duplicates: dict[str, list[str]]
    dataset_hash: str


# ------------------------------------------------------------------------------- rules


def current_spec_hashes() -> dict[str, str]:
    return gen._spec_hashes()


def provenance_problems(man: dict[str, Any], allow_dirty: bool) -> list[str]:
    """Problems with where a split came from (empty list if none)."""
    out = []
    if str(man.get("commit", "")).endswith("-dirty") and not allow_dirty:
        out.append(f"generated from a dirty tree ({man['commit']}); released splits must come "
                   "from a committed tree (CLAUDE.md §5)")
    now = current_spec_hashes()
    for f, h in sorted(man.get("spec_sha256", {}).items()):
        if now.get(f) != h:
            out.append(f"{f} changed since generation (manifest {h[:12]}, now "
                       f"{str(now.get(f))[:12]})")
    if not man.get("complete", False):
        out.append("split is incomplete (manifest complete = false)")
    return out


def rescale_problem(split: str, fraction: float) -> str | None:
    """None if the envelope rescale fraction is acceptable, else the reason."""
    if fraction <= gen.RESCALE_LIMIT or split in RESCALE_EXEMPT:
        return None
    return (f"{fraction:.1%} of boards rescaled by the envelope, above {gen.RESCALE_LIMIT:.0%} "
            "(CLAUDE.md §6.5) and not exempted by a logged decision")


def cross_split_duplicates(hashes: dict[str, set[str]]) -> dict[str, list[str]]:
    """{record hash: sorted splits} for every hash found in more than one split."""
    seen: dict[str, list[str]] = {}
    for split, hs in hashes.items():
        for h in hs:
            seen.setdefault(h, []).append(split)
    return {h: sorted(s) for h, s in seen.items() if len(s) > 1}


def resolve_indices(count: int, frac: float, seed: int, split: str) -> np.ndarray:
    """Seeded sample of board indices to re-solve: ceil(frac·count), at least one, sorted."""
    if not 0 < frac <= 1:
        raise ValueError(f"resolve fraction must be in (0, 1], got {frac}")
    k = min(count, max(1, math.ceil(frac * count)))
    rng = np.random.Generator(np.random.PCG64(gen.board_seed(split, seed, 0)))
    return np.sort(rng.choice(count, size=k, replace=False))


# ------------------------------------------------------------------------------- checks


def _split_dirs(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root {root} does not exist")
    dirs = {}
    for d in sorted(p for p in root.iterdir() if (p / gen.MANIFEST).is_file()):
        man = json.loads((d / gen.MANIFEST).read_text(encoding="utf-8"))
        dirs[man["split"]] = d
    if not dirs:
        raise ValueError(f"no split folder with a {gen.MANIFEST} under {root}")
    return dirs


def _resolve(record: dict[str, Any], N: int) -> np.ndarray:
    r = rasterise(record, gen.SPEC, N)
    A, b = assemble_board(gen.SPEC, r, record["dirichlet_edges"])
    return amg_solve(A, b, tol=RESOLVE_TOL).theta.reshape(4, N, N)


def _check_split(split: str, d: Path, *, resolve_frac: float, seed: int, phi_N: int,
                 allow_dirty: bool, failures: list[str]) -> tuple[SplitStats, set[str]]:
    st = SplitStats(split)
    hashes: set[str] = set()
    try:
        man = gen.verify(d)
    except (gen.ManifestError, OSError, KeyError, ValueError) as e:
        failures.append(f"{split}: files do not match the manifest: {e}")
        return st, hashes
    st.commit, st.shards, N = man["commit"], len(man["shards"]), man["N"]
    failures.extend(f"{split}: {p}" for p in provenance_problems(man, allow_dirty))
    st.hotspot = np.zeros((N, N), dtype=np.int64)
    todo = set(resolve_indices(man["count"], resolve_frac, seed, split).tolist())
    for sh in man["shards"]:
        rows = (d / sh["name"] / "records.jsonl").read_text(encoding="utf-8").splitlines()
        fields = np.load(d / sh["name"] / "fields.npy", mmap_mode="r")
        for row, field in zip(rows, fields, strict=True):
            r = json.loads(row)
            i, rec = r["i"], r["record"]
            h = gen.record_hash(rec)
            if h != r["record_sha256"]:
                failures.append(f"{split}: board i = {i}: record_sha256 {r['record_sha256'][:12]} "
                                f"!= recomputed {h[:12]}")
            if h in hashes:
                failures.append(f"{split}: board i = {i}: record appears twice in the split")
            hashes.add(h)
            theta = np.asarray(field, dtype=np.float64)
            peak = float(theta.max())
            st.count += 1
            st.rescaled += int(r["scaled"])
            st.theta_max.append(peak)
            st.chip_hist[len(rec["chips"])] += 1
            _, y, x = np.unravel_index(int(theta.argmax()), theta.shape)
            st.hotspot[y, x] += 1
            st.phi_sum += rasterise(rec, gen.SPEC, phi_N).phi.mean(axis=(1, 2))
            if i in todo:
                exact = _resolve(rec, N)
                err = float(np.abs(exact - theta).max() / exact.max())
                st.resolved += 1
                if err > st.worst_rel_err:
                    st.worst_rel_err, st.worst_board = err, i
                if not err < FIELD_TOL:
                    failures.append(f"{split}: board i = {i}: re-solve differs from the stored "
                                    f"field by {err:.2e} of max θ (limit {FIELD_TOL})")
    if st.count != man["count"]:
        failures.append(f"{split}: {st.count} boards read, manifest says {man['count']}")
    problem = rescale_problem(split, st.rescale_fraction)
    if problem:
        failures.append(f"{split}: {problem}")
    return st, hashes


def verify_dataset(root: Path, *, report: Path | None = None, figs: Path | None = None,
                   resolve_frac: float = 0.01, seed: int = 0, phi_N: int = 64,
                   allow_dirty: bool = False, make_figures: bool = True) -> Report:
    """Run every check on every split under root; write the report (and figures) if asked."""
    root = Path(root)
    dirs = _split_dirs(root)
    failures: list[str] = []
    splits, hashes = {}, {}
    for split, d in dirs.items():
        splits[split], hashes[split] = _check_split(
            split, d, resolve_frac=resolve_frac, seed=seed, phi_N=phi_N, allow_dirty=allow_dirty,
            failures=failures)
    dups = cross_split_duplicates(hashes)
    failures.extend(f"record {h[:12]} appears in splits {', '.join(s)}" for h, s in dups.items())
    dh = hashlib.sha256()
    for split, d in sorted(dirs.items()):
        man = json.loads((d / gen.MANIFEST).read_text(encoding="utf-8"))
        for sh in man["shards"]:
            for f, h in sorted(sh["files"].items()):
                dh.update(f"{split}/{sh['name']}/{f}={h}\n".encode())
    rep = Report(root, splits, failures, dups, dh.hexdigest())
    if report is not None:
        footer = _footer(rep, seed)
        fig_names = _figures(rep, Path(figs), footer) if (make_figures and figs) else []
        _write_report(rep, Path(report), footer, fig_names, resolve_frac, phi_N)
    return rep


# ------------------------------------------------------------------------------- output


def _footer(rep: Report, seed: int) -> str:
    return (f"verifier commit {gen.git_commit()} · dataset {rep.dataset_hash[:16]} · seed {seed} · "
            f"{platform.platform()} {platform.processor()} · {dt.date.today().isoformat()}")


def _figures(rep: Report, figs: Path, footer: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs.mkdir(parents=True, exist_ok=True)
    names = list(rep.splits)
    cols = 4
    rows = math.ceil(len(names) / cols)
    out = []
    for kind in ("hotspots", "chips"):
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows + 0.4),
                                 squeeze=False, constrained_layout=True)
        for ax, s in zip(axes.flat, names, strict=False):
            st = rep.splits[s]
            if kind == "hotspots" and st.hotspot is not None:
                ax.imshow(st.hotspot, origin="lower", extent=(0, 100, 0, 100), cmap="inferno",
                          interpolation="nearest")
                ax.set(xlabel="x (mm)", ylabel="y (mm)")
            else:
                ks = sorted(st.chip_hist)
                ax.bar(ks, [st.chip_hist[k] for k in ks])
                ax.set(xlabel="chips per board", ylabel="boards")
            ax.set_title(f"{s} (n = {st.count})", fontsize=9)
        for ax in list(axes.flat)[len(names):]:
            ax.axis("off")
        fig.get_layout_engine().set(rect=(0, 0.03, 1, 0.97))
        fig.text(0.01, 0.005, footer, fontsize=6, color="0.35")
        name = f"verify_{kind}.png"
        fig.savefig(figs / name, dpi=110)
        plt.close(fig)
        out.append(name)
    return out


def _write_report(rep: Report, path: Path, footer: str, figs: list[str], frac: float,
                  phi_N: int) -> None:
    verdict = "PASS" if not rep.failures else f"FAIL ({len(rep.failures)} problems)"
    L = [f"# Dataset verification (N2.2): {verdict}", "",
         f"Root `{rep.root}` · dataset hash `{rep.dataset_hash}`", "",
         f"Checks: file hashes and counts; provenance; record hashes and cross-split duplicates; "
         f"re-solve of {frac:.0%} of each split (at least one board) with PyAMG-GMRES to "
         f"{RESOLVE_TOL:g} (limit {FIELD_TOL:g} of max θ); statistics (mean φ rasterised at N = "
         f"{phi_N}).", "",
         "| split | boards | shards | commit | rescaled | θ_max P5 / P50 / P95 (K) | "
         "re-solved | worst rel. error | mean φ L1 / L2 / L3 / L4 | chips per board |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for s, st in rep.splits.items():
        q = (np.percentile(st.theta_max, [5, 50, 95]) if st.theta_max
             else [math.nan] * 3)
        hist = ", ".join(f"{k}: {st.chip_hist[k]}" for k in sorted(st.chip_hist))
        exempt = " (exempt)" if s in RESCALE_EXEMPT else ""
        L.append(f"| {s} | {st.count} | {st.shards} | {st.commit} | "
                 f"{st.rescale_fraction:.1%}{exempt} | {q[0]:.1f} / {q[1]:.1f} / {q[2]:.1f} | "
                 f"{st.resolved} | {st.worst_rel_err:.2e} | "
                 f"{' / '.join(f'{v:.3f}' for v in st.mean_phi)} | {hist} |")
    L += ["", f"Cross-split duplicate records: {len(rep.duplicates)}."]
    for s, why in RESCALE_EXEMPT.items():
        if s in rep.splits:
            L.append(f"Rescale exemption for {s}: {why}.")
    L += ["", "## Problems", ""]
    L += [f"- {m}" for m in rep.failures] or ["None."]
    if figs:
        L += ["", "## Figures", ""] + [f"![{f}](figs/{f})" for f in figs]
    L += ["", "---", "", footer, ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, required=True, help="folder with one folder per split")
    ap.add_argument("--report", type=Path, default=ROOT / "data" / "verify_report.md")
    ap.add_argument("--figs", type=Path, default=ROOT / "data" / "figs")
    ap.add_argument("--resolve-frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phi-N", type=int, default=64)
    ap.add_argument("--allow-dirty", action="store_true", help="test data only, never a release")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    rep = verify_dataset(a.root, report=a.report, figs=a.figs, resolve_frac=a.resolve_frac,
                         seed=a.seed, phi_N=a.phi_N, allow_dirty=a.allow_dirty,
                         make_figures=not a.no_figures)
    print(f"{'PASS' if not rep.failures else 'FAIL'}: {len(rep.splits)} splits, "
          f"{sum(s.count for s in rep.splits.values())} boards, {len(rep.failures)} problems; "
          f"report {a.report}")
    for m in rep.failures[:20]:
        print("  -", m)
    return 1 if rep.failures else 0


if __name__ == "__main__":
    sys.exit(main())
