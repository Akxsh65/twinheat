"""Figures of the gate experiments (E7, E8, E9, E14, E15, E16). Every figure carries the
CLAUDE.md §4 footer (commit, frozen-config commit, input hash, seeds, hardware, date) that the
E-script passes in; a figure without a footer is refused."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, footer: str, path: Path) -> Path:
    if not footer:
        raise ValueError("figures need a footer (CLAUDE.md §4)")
    fig.text(0.01, 0.005, footer, fontsize=6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    _plt().close(fig)
    return path


def m1_vs_level(ci: list[dict[str, Any]], bound_by_level: dict[int, float], footer: str,
                path: Path, *, title: str) -> Path:
    """M1 (all proposals) vs aggressiveness, one panel per proposer, one line per gate, 95 % CI
    over seeds, and the Lemma 1 bound at the headline p."""
    plt = _plt()
    rows = [r for r in ci if r["metric"] == "M1_all"]
    props = sorted({r["proposer"] for r in rows})
    fig, axes = plt.subplots(1, len(props), figsize=(3.4 * len(props), 3.0), squeeze=False)
    for ax, prop in zip(axes[0], props, strict=True):
        for gate in sorted({r["gate"] for r in rows}):
            sel = sorted((r for r in rows if r["proposer"] == prop and r["gate"] == gate),
                         key=lambda r: r["level"] or 0)
            if not sel:
                continue
            x = [r["level"] or 0 for r in sel]
            m = np.array([r["mean"] for r in sel])
            err = np.array([[r["mean"] - r["ci_lo"] for r in sel],
                            [r["ci_hi"] - r["mean"] for r in sel]])
            ax.errorbar(x, m, yerr=err, marker="o", ms=3, capsize=2, label=gate)
        lv = sorted(bound_by_level)
        ax.plot(lv, [bound_by_level[k] for k in lv], "k--", lw=1, label="Lemma 1 bound")
        ax.set_title(prop)
        ax.set_xlabel("aggressiveness level")
        ax.set_xticks([1, 2, 3])
    axes[0][0].set_ylabel("M1: accepted large errors / proposals")
    axes[0][-1].legend(fontsize=6)
    fig.suptitle(title, fontsize=9)
    return _save(fig, footer, path)


def sweep(rows: list[dict[str, Any]], x: str, ys: tuple[str, str], footer: str, path: Path, *,
          title: str, group: str = "gate") -> Path:
    """Two panels: ys[0] and ys[1] (means over runs) against a swept setting x."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, y in zip(axes, ys, strict=True):
        for g in sorted({str(r[group]) for r in rows}):
            sel = [r for r in rows if str(r[group]) == g and r.get(x) is not None]
            xs = sorted({r[x] for r in sel})
            ax.plot(xs, [np.mean([r[y] for r in sel if r[x] == v]) for v in xs], marker="o",
                    ms=3, label=g)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
    axes[1].legend(fontsize=6)
    fig.suptitle(title, fontsize=9)
    return _save(fig, footer, path)


def group_bars(groups: list[dict[str, Any]], footer: str, path: Path, *, title: str) -> Path:
    """Per-group M1g of G5 and G6 (mean over runs) against α + slack_g (empirical target)."""
    plt = _plt()
    names = sorted({r["group"] for r in groups})
    fig, ax = plt.subplots(figsize=(max(5.0, 0.5 * len(names)), 3.0))
    w = 0.38
    for k, gate in enumerate(("G5", "G6")):
        vals = [np.mean([r["M1g"] for r in groups if r["gate"] == gate and r["group"] == n]
                        or [np.nan]) for n in names]
        ax.bar(np.arange(len(names)) + (k - 0.5) * w, vals, w, label=gate)
    tgt = [np.mean([r["alpha"] + r["slack_g"] for r in groups if r["group"] == n]) for n in names]
    ax.plot(np.arange(len(names)), tgt, "k_", ms=14, label="α + slack_g (empirical target)")
    ax.set_xticks(np.arange(len(names)), names, rotation=60, fontsize=6)
    ax.set_ylabel("M1g")
    ax.legend(fontsize=6)
    fig.suptitle(title, fontsize=9)
    return _save(fig, footer, path)


def best_verified_curves(curves: list[dict[str, Any]], footer: str, path: Path, *,
                         title: str) -> Path:
    """E8 / M3: best verified peak vs cumulative exact solves, per gate (all runs overlaid)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    colours = {"G1": "tab:orange", "G5": "tab:blue"}
    for c in curves:
        ax.plot(c["solves"], c["best"], lw=0.8, alpha=0.5, color=colours.get(c["gate"], "k"))
    for g, col in colours.items():
        ax.plot([], [], color=col, label=g)
    ax.set_xlabel("cumulative exact solves")
    ax.set_ylabel("best verified peak rise (K)")
    ax.legend(fontsize=6)
    fig.suptitle(title, fontsize=9)
    return _save(fig, footer, path)
