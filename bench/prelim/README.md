# bench/prelim: preliminary certificate prototypes (CLAUDE.md §8.6, prompt N1.3)

## Status (2026-10-30)

`stop.js` and `cert.js`, the scripts behind the 5 Oct 2026 four-variant table in CLAUDE.md §8.6, are **not
found**. The plan (`TwinHeat_plan_v3_1.html`, 5 Oct) names them but does not contain them. They were most
likely run in a claude.ai chat around 4–5 Oct; the search is still open.

What survives is the in-browser solver and Lemma 2 certificate widget of the 5 Oct explainer. It is kept
here as provenance. **It is not `stop.js` / `cert.js`, and its output is not a reproduction of the
four-variant table.**

## Files

| File | What it is |
|---|---|
| `explainer_cert_widget.js` | The explainer's 64² solver (FV assembly, Jacobi-PCG, multigrid V-cycles) and certificate code (`certConst`, `resInf`), extracted verbatim from `TwinHeat_Pitch.html`, plus a new headless harness at the bottom that repeats the widget's `prepare()` logic. |
| `explainer_cert_widget_results.csv` | Output of `node explainer_cert_widget.js` (node v22.19.0). Running it again gives this file byte for byte (checked 2026-10-30). |

Checked on 2026-10-30: every code block above the harness appears verbatim (whitespace-normalised) in
`TwinHeat_Pitch.html`.

Source files (kept outside git, large; SHA-256):

| File | Date | SHA-256 |
|---|---|---|
| `TwinHeat_Pitch.html` | 2026-10-22 | `f76b44dc01529ac23b58ebdce96352747bf9d3d37e10cc4be659ef09cd3fa0fc` |
| `twinheat_offline.html` (the explainer) | 2026-10-05 | `ca37b7880372e10b8c5bd9bad242b8e20b9bbd17e2bd90030a585a119dcf215b` |
| `TwinHeat_plan_v3_1.html` (names the scripts) | 2026-10-05 | `04b6f740d0d7854c023b5fdb76050d441683a6e470a94c645ad691ae3c886297` |

## What the widget reproduces, and what it does not

- **It reproduces** the "noisier emulated errors" sentence at the end of CLAUDE.md §8.6 (and Phase 0 §3.0).
- **It does not reproduce** the four-variant table or the pointwise (Gauss–Seidel sweep) table. According to the pitch, those
  used a milder error model ("smooth blobs with mild ringing and noise"). That model's settings and the
  other three board variants are not written down anywhere we have.
- **Board.** The widget's board is `TH.defaultBoard()`, which is not a spec_v1 sampler board:
  - 64² cells, k_FR4 = 0.9 (spec_v1 nominal is 0.81)
  - total face loss H = 24 W/(m²·K)
  - 2 oz painted copper
  - left edge heat-sunk
  - three chips (1.2, 0.8 and 0.5 W)
- **Error model.** Seed `TH.rng(7)`. The emulated prediction is
  μ = θ + lvl·peak·(sin(X/7 + 1.3)·cos(Y/9 + 1) + 0.4·g·sin(1.7X + 0.9Y) + 0.08·gauss), where g is the copper-edge
  indicator.
- **Rounding margin.** The certificate uses no margin, and its bound is checked against the true error at every iterate.

These numbers are preliminary, from an untagged prototype. Per CLAUDE.md §0 rule 11 they are not report numbers.

## If stop.js / cert.js stay lost

Whether to fall back is N and M's decision, logged in `docs/decisions.md`. The honest fallback:
1. Record that the scripts are lost.
2. Write the milder error model and the four board variants down explicitly, and commit them before any run.
3. Run them on this solver or on `certify/`.
4. Report the result as new preliminary evidence, not as a reproduction.
