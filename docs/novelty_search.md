# Novelty search

Prompt M0.3 (plan, Phase 0 · M). Question checked:

> Has "pointwise, rigorous error certificates for a neural PDE surrogate's prediction, used to decide when a
> design is settled and stop the exact solver early" been published since our 4–5 Oct 2026 search?

Contribution this protects: CLAUDE.md §1, contribution 1 (M-matrix certificate with certified early stopping;
Lemma 2, H6, H7), and the "rigorous per-iterate bands" part of contribution 4.

**Source of the classifications below.** Only the eight papers in `docs/grand_guide.pdf` (24 Oct 2026), each read
from its own provided PDF. No paper is cited from memory (CLAUDE.md §0 rule 10). Anything not checked against a
provided text is labelled UNVERIFIED.

**Status (27 Oct 2026): complete.** The provided papers are classified, and none is type (a). The post-October
sweep (§1) was run by the team and found no relevant new publication since the 4–5 Oct 2026 search. M0.3's
checklist item "no type-(a) hit" holds.

## 1 · Search log

Databases: Google Scholar (GS), arXiv (AX), Semantic Scholar (SS). Restrict to 2026-10-04 onwards. Record one
row per query per database, including "0 relevant hits".

| # | Query string | Targets |
|---|---|---|
| Q1 | `"neural operator" "a posteriori" error bound "maximum principle"` | certificate for a surrogate's output |
| Q2 | `"M-matrix" "a posteriori" "error bound" surrogate` | Lemma 2 applied to ML predictions |
| Q3 | `"pointwise" "error certificate" "neural PDE" OR "neural operator"` | pointwise certificate (Lemma 2c) |
| Q4 | `supersolution "error bound" "neural network" "elliptic"` | supersolution w with Aw ≥ \|r\| |
| Q5 | `"certified early stopping" "conjugate gradient" OR PCG` | decision-sufficient stopping |
| Q6 | `"warm start" PCG "neural operator" "error bound" stopping` | surrogate warm start + certified stop |
| Q7 | `"decision-sufficient" OR "decision-aware" stopping iterative solver` | stop once the decision is settled |
| Q8 | `"surrogate certification" PDE "infinity norm" residual` | ∞-norm residual certificate |
| Q9 | `"inverse-positive" OR "monotone matrix" "error bound" "neural"` | A⁻¹ ≥ 0 used on ML output |
| Q10 | `"discrete maximum principle" "neural operator" verification` | maximum-principle verification of an operator |
| Q11 | `"a posteriori" "error estimate" "physics-informed" "guaranteed upper bound"` | guaranteed bounds for PINN-type models |
| Q12 | `"thermal" surrogate "certified" "exact solver" "early stopping"` | same idea in the thermal / chip domain |
| Q13 | `"norm of the inverse" "positive vector" "M-matrix" bound` | is general-z Lemma 2(a) classical (see §3) |
| Q14 | `"verified" OR "rigorous" "error bound" "neural operator" "floating point"` | FP-rigorous certificate (`certify/margin.py`) |
| Q15 | `"conformal" "neural operator" "exact solver" "certificate" gate` | combination of the two layers (§8) |

| Date | Query | Database | Relevant hits | Verdict | Checked by |
|---|---|---|---|---|---|
| 2026-10-27 | Q1–Q15 | GS / AX / SS | 0 new since 2026-10-04 | no (a) or (b) hit | akash (reported) |

## 2 · Closest prior work

Verdicts: **(a)** same idea · **(b)** related but different (says how) · **(c)** unrelated to the question.
"Affects" names the CLAUDE.md §1 contribution concerned. "Source" is the PDF behind the guide, with its guide tag.

| Paper (source) | Date | Verdict | How it differs from our claim | Affects |
|---|---|---|---|---|
| Yu et al., DeepOHeat-v2, arXiv 2608.16080v1 [DOH] | 17 Aug 2026 | **(b)**, closest | Prop. 7 bounds the hotspot error using the M-matrix property (A⁻¹ ≥ 0). But it needs the exact entries of A⁻¹, plus hotspot coincidence (A1) and hotspot-row diagonal dominance (A2). The authors call it "numerically loose": r_hot ranks predictions and certifies none. Their solver runs to a loose tolerance (ε = 0.5) with no certified stopping rule. Ours is computable (κ̂ from one positive z), needs neither A1 nor A2, is checked at every PCG iterate in FP64, and is used as the stopping rule. | 1 (wording, see §3); also 3 (their gate-to-warm-solver is a non-claim) |
| Varah 1975, ‖A⁻¹‖∞ lower bound on the smallest singular value [VA] | 1975 | **(b)** | Theorem 1, ‖A⁻¹‖∞ ≤ 1/α for strictly diagonally dominant A, is the z = 1 case of Lemma 2(a) on an M-matrix. It gives nothing on our 4-layer board (α = 0 on L2/L3, since h₂ = h₃ = 0). It has no surrogate and no stopping rule. | 1: the z = 1 case is not ours |
| Meurant & Tichý 2018, A-norm upper bounds in CG [MT] | 2018 | **(b)** | A per-iterate CG error bound, but in the A-norm and needing a rigorous µ ≤ λ_min. The practical Ritz estimate is not a bound. It is not a pointwise or ∞-norm certificate and has no surrogate. It is already our named fallback (§8.4). | 1 (A-norm bounds are a non-claim) |
| Shewchuk 1994, CG without the agonizing pain [CG] | 1994 | **(c)** | Eq. 52 is a convergence-rate bound that cannot be computed during a run (it needs ‖e⁽⁰⁾‖_A). It covers warm starts and residual drift, with no certificate and no surrogate. | — (background for 3) |
| Angelopoulos & Bates, conformal intro, arXiv 2107.07511v6 [CP] | Dec 2022 | **(c)** | Statistical. It gives no deterministic error certificate. | 2 (G2/G3 baselines; non-claim) |
| Gibbs & Candès, ACI, arXiv 2106.00170v3 [GC] | Oct 2021 | **(c)** | Statistical online calibration. It gives no solver certificate. | 2 (Lemma 1 step 2, cited) |
| Tu & Giesecke, arXiv 2605.17705v2 [TG] | 5 Oct 2026 | **(c)** | Statistical, with intermittent labels and an MCAR transfer. It gives no solver certificate. | 2 (Lemma 1 step 3, cited) |
| Bastani et al., multivalid CP, arXiv 2206.01067v1 [MVP] | Jun 2022 | **(c)** | Statistical, per-group calibration. It gives no solver certificate. | 2 (G6 algorithm, cited) |

**Result for the question.** No type-(a) hit among the provided papers, and none in the post-October sweep. DeepOHeat-v2 is the only paper that
pairs a surrogate with an M-matrix error bound, and it presents that bound as loose and not as a certificate or
stopping rule.

## 3 · Claims narrowed

No claim is narrowed. The G1 narrowing in CLAUDE.md §15 is not triggered. Two wording notes come from the guide
(Part I §7, "Everyone", item 4) and are recorded here for N and M to act on:

1. **DeepOHeat-v2 Prop. 7 already uses A⁻¹ ≥ 0 in an error bound** (their Appendix B). So the report must not
   present "using the M-matrix property to bound surrogate error" as ours. The claim stays as CLAUDE.md §1 words
   it: the certificate used for *decision-sufficient early stopping* of verifications and audits, inside one
   batched warm-started solve, with a derived FP64 margin. This is consistent with the §1 non-claims, which
   already exclude ‖e‖ ≤ ‖A⁻¹‖·‖r‖.
2. **The general-z form of Lemma 2(a) may be classical M-matrix theory.** UNVERIFIED: this is from the Varah
   guide author's memory, and no source has been provided. Query Q13 targets it. Until a source is in `docs/refs/`,
   the report presents Lemma 2(a) as standard (or cites it once found), never as new.

Open actions:

| Action | Owner | Status |
|---|---|---|
| Run Q1–Q15 on GS / AX / SS from 2026-10-04, fill the search log and §2 | M | done 2026-10-27, no hits |
| Re-run Q1–Q15 before the report is submitted (new work may appear) | M | open |
| Find a source for general-z Lemma 2(a) and put it in `docs/refs/` | N (Lemma 2 checker: M) | open |
| Confirm report wording for Lemma 2 against notes 1–2 | N + M | open |
