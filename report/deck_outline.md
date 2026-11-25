# TwinHeat: viva deck outline, demo script and question prep (F13)

Every number on a slide must already be in a chapter table (report/tables/, generated from the
committed results). Every figure keeps the footer the script wrote. Tags show which subject a
slide mainly serves: **[DL]**, **[PAGP]**, **[CGVR]**, **[all]**. Target: 14 slides, about 15 min,
plus the live demo.

## Slides

| # | Title | Content | Source | Tag |
|---|---|---|---|---|
| 1 | TwinHeat | Title, team, one line: "certified neural surrogates for interactive circuit-board thermal design" | intro.tex | all |
| 2 | The problem | Exact solve takes seconds; a surrogate takes milliseconds but can be silently wrong. Picture: the viewer's board with a hotspot | cgvr.tex screenshot | all |
| 3 | The pipeline | Diagram: propose → FNO ensemble → gate (accept / flag) → coin after commit → certified PCG → viewer band | intro.tex §pipeline | all |
| 4 | Surrogate | 5-member FNO, 13→4 channels, 256²×4. E5: hotspot error met, rel. L2 target missed (stated plainly), OOD flags | dl.tex, Table e5 | DL |
| 5 | The gate and its guarantee | B = q·s ≤ τ; only audits move q; Lemma 1 in one line; what is cited vs ours | dl.tex, lemmas.tex | DL |
| 6 | Does the gate hold? | E7 at τ = 0.2 K: no gate 45/45 runs above α; G5 0 of 135; G4 (flagged-only) fails, as theory predicts. τ = 1 K too loose to rank (honest) | Table e7-tau02, e7-tau1 | DL |
| 7 | Groups and adaptive audits | E15 (G6 fails per group) and E16 (≈10 % audit saving): reported as negatives | Tables e15, e16 | DL |
| 8 | The certificate | Lemma 2 in one line: any z > 0 with Az > 0 bounds ‖A⁻¹‖∞; FP64 margin; never the FP32 recursive residual | pagp.tex, lemmas.tex | PAGP |
| 9 | Stopping early, with proof | E13: 1,000/1,000 certified, 0 violations, H6 passes with the AMG-CG pointwise certificate; the V-cycle finding | Table e13 | PAGP |
| 10 | GPU kernels | Bandwidth: one board 5–11 % of peak, 64 boards 79 %: why batching | Table bw | PAGP |
| 11 | Certify on demand | One batched launch, per-board stop; H5 passes at 34.9×; the FP32 ceiling | Table e4, Fig. pipeline_bench | PAGP |
| 12 | Seeing a certified field | Badge Predicted → Verifying → Verified; band rigorous for the float16 field drawn; WBOIT, JFA bevel, isolines | cgvr.tex | CGVR |
| 13 | Live and latency | E10 60 fps; E11: first band 4.96 s → 0.86 s with the warm-started z; verified time still CPU-bound: GPU path is the fix | Tables e10, e11 | CGVR |
| 14 | What we showed, what we did not | Hypothesis table (H1–H10): passed / not supported / not run, and why | conclusion.tex | all |

## Live demo (3–4 min, during or after slide 12)

Run Option A (laptop: `python -m server.app --n 64 --verify ...`) for reliability, or Option B
(rvcse FNO server with `--gate-frozen experiments/configs/frozen_tau0.2.yaml ... --z-warm-iters 5`
through the SSH tunnel) if the network allows. Fallback: the recorded stream
(`python -m server.ws_mock_server server/demo/verify_fno.jsonl --rate 5 --loop`).

1. Open the viewer (`?records=/demo/records.json`). Peel the layers apart; point at the via columns
   glowing with vertical heat flux; switch the LUT; show an isoline.
2. Load a design: badge amber, "Predicted". With the gate: "ACCEPTED by the gate (B ≤ τ)" or
   "FLAGGED (B > τ)".
3. Move a chip three times: each move is a preview ("PREVIEW while dragging"), never verified.
4. Commit: blue "VERIFYING", the board visibly converges while the guaranteed band shrinks;
   green "VERIFIED · guaranteed band ±0.09 K in every cell".
5. One sentence: "the band is a proof, not an estimate: the true temperature of every cell lies
   inside it, including the rounding of what you are looking at".

## Demo video (2–3 min; record the screen of the steps above)

| Time | Shot | Voice-over |
|---|---|---|
| 0:00 | Board at rest, layers peeling | "A four-layer board, 262,144 unknowns." |
| 0:20 | Load a design, amber badge | "The network predicts in milliseconds. The gate decides if we can use it." |
| 0:45 | Drag a chip | "While I drag, we only preview. Nothing is verified, nothing is gated." |
| 1:10 | Commit, blue badge, band shrinking | "Committed: the exact solver starts from the prediction, and every few iterations sends a certified band." |
| 1:50 | Green badge | "Verified: every cell is within the band of the true solution." |
| 2:10 | Cut to slide 11 figure | "In batch on the GPU this runs 35 times faster than one board at a time." |

## Questions to prepare (with the honest answer)

- *Why did H1 fail at τ = 1 K?* The surrogate's peak error exceeds 1 K on 0.1 % of proposals, so no
  gate is ever tested; the τ = 0.2 K run was added after seeing that, logged before running, and
  both are reported (docs/decisions.md, D4 amendment).
- *Is the Lemma 1 bound useful?* Not at our run lengths (it exceeds 1 at p = 0.1); the evidence is
  the measured rate against α over 45 runs per gate. A tighter constant (C⁺) awaits sign-off.
- *Why change the certificate's construction?* Three V-cycles gave no valid z on boards without a
  heat-sunk edge; Lemma 2(a) allows any positive z, and the check before use keeps it rigorous.
- *Why FP64 iterations on the GPU, when the plan said FP32?* With κ̂ ≈ 1.5e6 K/W the residual of an
  FP32 iterate cannot certify below about 1 K.
- *Why is the live verification still 15 s?* About 800 PCG iterations on the CPU of the live server;
  the batched GPU path is 35× faster but not wired into the server.
- *What is yours and what is prior work?* intro.tex "What is not claimed" and lemmas.tex remarks.
- *Who checked the proofs?* The sign-off table in lemmas.tex: answer with its current state.
