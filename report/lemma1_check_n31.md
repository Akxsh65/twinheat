# N3.1 · Check of the Lemma 1 / 1′ / G7 proof

**For N, who signs** (CLAUDE.md §2: N checks Lemmas 1 and 1′). This is a report only: the proof is
not changed here. Everything below is a draft until N fills in the verdict column of §1 and signs
§6.

**Material checked.** `report/lemma1_review.md`, which holds the proof that is to go into
`lemmas.tex`:
- §3 (steps 1–3 and the remarks);
- §4 (Lemma 1′);
- §7 (G7);
- §10 (the proposed statement).

No `lemmas.tex` exists yet.

**Theorem text used.** Only the two transcriptions made for this check:
- `docs/refs/gibbs_candes_prop41.txt` (arXiv 2106.00170v3, pp. 3 and 6);
- `docs/refs/tu_giesecke_thm54_56.txt` (arXiv 2605.17705v2, pp. 3–6 and 18–24).

Both are still "checked against the PDF by: (reviewer)", so N should confirm them against the PDFs
first.

**Independence.** The proof under check was drafted by Claude in M3.4, and this check is also
Claude's. It is a second pass by the same author, so it can catch slips but not shared blind
spots. N's own reading of each row is what makes the check independent.

---

## 1 · Step table

U1–U6 are the assumptions of `lemma1_review.md` §2:
- U1: no audit delay;
- U2: the coin is independent of everything before it;
- U3: B7 holds;
- U4: the audit does not depend on the coin;
- U5: α ∈ (0, 1) and α₁ ∈ [0, 1];
- U6: T is fixed in advance.

| # | Step | Claim | Justification given | Sufficient? | What is missing |
|---|---|---|---|---|---|
| 1 | §3 step 1, (1) | accepted ∧ e_t > τ ⇒ m⁺_t = 1, so W_T ≤ M⁺_T | case split on α(t): q = ∞ ⇒ flagged; α(t) ≥ 1 ⇒ m⁺ = 1 by definition; else e > τ ≥ q s ⇒ S > q ⇒ S⁺ > q (U3) | **yes** | nothing. The case "α(t) ∈ (0, 1) but q_t = ∞" (rank above n under D8) is covered by the first case, since B = ∞ > τ |
| 2 | §1, definition of m⁺_t on unaudited steps | m⁺_t is well defined for every t | "S⁺_t is the score the audit *would* return", via U4 | **yes, if U4 is written as a function** | state U4 as: S⁺_t = F(x_t, model_t, q_t, ε_S, cap) for a fixed deterministic F (or F with randomness independent of A_t). As written ("would return") it is counterfactual wording, not a definition |
| 3 | §3 step 2, (2) | pathwise \|Σ_{k≤n_a}(err_k − α)\| ≤ C/γ for the audited stream | "Lemma 4.1's proof uses only three facts" + Prop. 4.1's telescoping | **yes** | ⚑ **not supported by the statement as pasted.** Prop. 4.1 is stated for ACI's err_t = 1{Y_t ∉ Ĉ_t(α_t)}. Using it for an arbitrary 0/1 sequence with the two boundary properties applies the *proof* (p. 6), not the statement. The attribution should read "by the argument of Prop. 4.1", and §8's table "cited" should become "cited argument" |
| 4 | §3 step 2, C⁺ remark | α_k ≥ −γ(1 − α), hence Σ(err − α) ≤ (α₁ + γ(1 − α))/γ | level rises when α_k ≤ 0 (err = 0), falls by at most γ(1 − α) otherwise | **yes** | nothing. It needs α₁ ≥ 0 (true: α₁ = α) and is the mirror of Tu & Giesecke (28). Re-derived in §2 |
| 5 | §3 step 3, identification | err_k (the recursion's input) = m⁺_t on the k-th audited step | U1: the audit updates α before step t + 1 is gated, so the q scoring it is the q that gated it | **yes under U1; no without it** | counterexample C2 (§3) shows the identity fails for batched updates. §10's statement includes U1 as (iii), so the *statement* is fine; the code is not (review §5) |
| 6 | §3 step 3, martingale difference | E[D_t ∣ 𝒢_t] = 0 for D_t = (A_t − p)(m⁺_t − α) | m⁺_t is 𝒢_t-measurable (U4); E[A_t − p ∣ 𝒢_t] = 0 (U2) | **yes** | 𝒢_t must be defined explicitly as σ(everything before A_t): x_{≤t}, model_{≤t}, labels_{≤t}, A_{<t}, any proposer randomness. Then D_{<t} is 𝒢_t-measurable. It is described in words in §2 U2 but should be written as a σ-field |
| 7 | §3 step 3, Azuma–Hoeffding | P(−Σ D_t ≥ ε) ≤ exp(−2ε²/T) | D_t lies, given 𝒢_t, in an interval of length \|m⁺_t − α\| ≤ 1 | **yes** | the inequality itself is uncited (CLAUDE.md §0 rule 10: no source in `docs/refs/`; the review already says so). The interval form needs the endpoints to be 𝒢_t-measurable: they are −p(m⁺ − α) and (1 − p)(m⁺ − α) ✓ |
| 8 | §3 step 3, (4) | W_T ≤ α + C/(γpT) + (1/p)√(ln(1/δ)/(2T)) | (3) + (2) + Azuma, divided by pT | **yes** | re-derived in §2. The bound holds for a fixed T (U6) |
| 9 | §3 "implies §8.2" | (1/p)√(ln(1/δ)/(2T)) ≤ (2/p)√(ln(4/δ)/T) | squared: ln(1/δ)/2 ≤ 4 ln(4/δ) | **yes** | nothing |
| 10 | §0/§2 "α ≤ ½ not needed" | the proof never uses α ≤ ½ | inspection | **yes** | nothing; \|m⁺ − α\| ≤ max(α, 1 − α) ≤ 1 for any α ∈ (0, 1) |
| 11 | §3 "in expectation (cited)" | \|(1/T)Σ E m⁺_t − α\| ≤ C/(γpT) | expectations in (3); "same identity as T&G eq. (40)" | **yes** | supported by B.2.4: (40) uses only (30), the F_t-measurability of C_t and Assumption 5.8. Our version uses (2) for both sides, which replaces Assumption 5.6 |
| 12 | §2 "retraining is fine" | the model may change between steps if fixed before A_t | the argument never uses a fixed score | **yes** | T&G state their results for one fixed ŝ (Algorithm 1 "Require"). The claim stands only because the argument does not use their theorem, so cite nothing for it |
| 13 | §4 Lemma 1′, valid input | err = 1[S⁺ > q] satisfies Prop. 4.1's needs | the three boundary facts, D7 for α ≥ 1 | **yes** | same attribution note as row 3 |
| 14 | §4 "decided stops lose nothing" | interval entirely ≤ q or > q ⇒ 1[S⁺ > q] = 1[S > q] | S ∈ [S⁻, S⁺] because T ∈ [T⁻, T⁺] (U3) | **yes** | S⁻ = 0 when T̂ ∈ [T⁻, T⁺], else the distance to the nearer end over s; S⁺ is the farther end over s (checked) |
| 15 | §10 (ii) "coin stream unknown to the proposer" | stands in for U2 | U2 | **unclear** | the proof needs A_t to be a random variable independent of 𝒢_t. A PCG64 stream is a deterministic function of its seed. The statement should model the seed as drawn at random, independently of everything, and never revealed; then (ii) is a property of that model. Counterexample C1 (§3) shows the gap is real if the seed leaks |
| 16 | §10 statement, other hypotheses | (i), (iii), (iv) cover the proof | — | **yes** | s_t > 0 is implicit in "B_t = q_t s_t" (gate/scale.py's ε > 0); say it. "q_t computed from the past" is implied by (i) |
| 17 | §7 (2′), G7 level range | α ∈ [−γp̄/p_min, 1 + γp̄/p_min]; \|Σ_audited w(err − α)\| ≤ C′/γ | Lemma 4.1's argument with increments < γp̄/p_min | **yes** | the review marked it UNSURE. The argument needs only bounded increments (w_t ≤ p̄/p_min because p_t ≥ p_min) and the two boundary implications (unchanged by w_t > 0). It holds |
| 18 | §7 (3′), G7 transfer | p̄(m⁺ − α) = wA(m⁺ − α) − p̄(A/p_t − 1)(m⁺ − α), second term a bounded martingale difference | E[A_t ∣ 𝒢_t] = p_t with p_t 𝒢_t-measurable | **yes** | requires A_t ~ Bernoulli(p_t) given 𝒢_t (not constant p), p_t computed from label-free quantities before the coin. Range length p̄\|m⁺ − α\|/p_t ≤ p̄/p_min (checked) |
| 19 | §7 bound | W_T ≤ α + C′/(γp̄T) + (1/p_min)√(ln(1/δ)/(2T)) | (2′) + (3′) + Azuma | **yes** | the code's `_p` caps p_t at 1 and floors it at p_min ✓. Whether its running mean s̄_t is label-free is a code question (it is computed from committed s values, so yes), not a proof question |
| 20 | §6 G6 per-group | per-group accepted-error bound | "Lemma 1's transfer applies per group" | **no (no proof)** | the per-group analogue of (2) is missing. Bastani et al. Thm 3.1 is in expectation, per (group, bucket), and two-sided. No counterexample to the *claim* is known; it is unproved, not refuted. The review already reports it as a sketch, and E15 labels it empirical |
| 21 | §5 delayed version | C → C + (D − 1)γ·max(α, 1 − α) | "Prop. B.15 suggests" | **unclear (not claimed)** | B.15 is for T&G's delayed variant, which scores each label against the *stored* threshold (eq. (43)). Our code scores against the current one. The review marks this UNSURE and claims nothing; keep it that way |

---

## 2 · Constants re-derived independently

Notation: α₁ = α (code); C = max(α₁, 1 − α₁) + γ.

- **(2), two-sided** (Gibbs & Candès Prop. 4.1 argument).
  - From α_{n+1} = α₁ + γΣ_k(α − err_k) and α_{n+1} ∈ [−γ, 1 + γ]:
    Σ(err − α) = (α₁ − α_{n+1})/γ ∈ [(α₁ − 1 − γ)/γ, (α₁ + γ)/γ].
  - Both ends are ≤ C/γ in absolute value. ✓
- **One-sided C⁺.**
  - α_{k+1} ≥ min(α_k + γα, α_k − γ(1 − α)) when α_k > 0, and α_{k+1} = α_k + γα when α_k ≤ 0.
  - So inf_k α_k ≥ −γ(1 − α) (it can dip below 0 only from a positive level, by at most γ(1 − α)).
  - Hence Σ(err − α) ≤ (α₁ + γ(1 − α))/γ = C⁺/γ. ✓
- **Transfer.**
  - (3): p Σ_t(m⁺ − α) = Σ_aud(err − α) − Σ_t D_t.
  - With Σ_aud ≤ C/γ and −ΣD ≤ √(T ln(1/δ)/2) on an event of probability ≥ 1 − δ, dividing by pT gives
    M⁺_T − α ≤ C/(γpT) + (1/p)√(ln(1/δ)/(2T)). ✓
  - The √ term with the exact interval length: (max(α, 1 − α)/p)√(ln(1/δ)/(2T)). ✓
- **G7.**
  - α_{n+1} = α₁ + γΣ w_k(α − err_k) with α_{n+1} ∈ [−γp̄/p_min, 1 + γp̄/p_min], so
    \|Σ w(err − α)\| ≤ (max(α₁, 1 − α₁) + γp̄/p_min)/γ = C′/γ.
  - Transfer: divide by p̄T. The √ term has (p̄/p_min)/p̄ = 1/p_min. ✓

**Why this matters (illustrative arithmetic, not results).** γ, p and T are not frozen (D4); this
uses α = 0.1, δ = 0.05.

| γ | T | p | §8.2 bound | (4) | (4) with C⁺ |
|---|---|---|---|---|---|
| 0.01 | 2,000 | 0.1 | 1.49 | 0.83 | 0.43 |
| 0.01 | 3,200 | 0.2 | 0.61 | 0.35 | 0.23 |
| 0.05 | 2,000 | 0.1 | 1.13 | 0.47 | 0.39 |
| 0.05 | 3,200 | 0.2 | 0.50 | 0.24 | 0.21 |

This bears directly on the M3.5 anomaly, where H1 is meaningless against the §8.2 bound at these
run lengths. Once signed, (4) with C⁺ is about 2–3.5× tighter than §8.2 in these examples.

---

## 3 · Counterexamples for the "no" / "unclear" rows

**C1, the coin is predictable (row 15; U2 fails).**
- Setup:
  - the proposer knows the PCG64 seed, so it knows A_t before it commits;
  - it has two kinds of design: "safe" ones (error 0) and "bad" ones (error 3τ, with a small
    label-free scale s, so that B = q s ≤ τ at the calibrated q).
- Behaviour: on steps where A_t = 1 it proposes a safe design; where A_t = 0, a bad one.
- What the gate sees: every audit covers, so α_k rises and q_k falls. Bad designs stay accepted:
  their B only shrinks.
- Result: W_T → 1 − p (every unaudited step is an accepted large error), against a bound α + slack
  that tends to α. With p = 0.1 that is 0.9 against 0.1.
- So (ii) must be a modelling assumption on the seed, not a statement about the code.

**C2, batched updates scored against a moved threshold (row 5; U1 fails).**
- Setup: designs 1 and 2 are gated in one batch with q = 1.0 and s = 1. Both are audited, with true
  scores S₁ = 1.5 (a miss at gating) and S₂ = 1.05 (also a miss at gating, since 1.05 > 1.0).
- Update 1: err₁ = 1, so α falls and q rises, say to 1.1.
- Update 2: scored against the current q = 1.1, err₂ = 0. But m⁺₂ = 1[1.05 > 1.0] = 1.
- So Σ_aud(err − α) undercounts the misses that (3) needs. Repeated over B audits per batch, the
  deficit can grow linearly in the number of batched audits per threshold move, and the step-3
  identity has no fixed bound.
- The statement is safe because (iii) excludes this case; the batched code is not (review §5).

**Row 20 (G6).** No counterexample: the claim is unproved, not shown false. The missing ingredient
is a pathwise or high-probability bound on the audited steps of each group.

---

## 4 · Assumptions in the order the proof uses them

1. **U5:** α ∈ (0, 1), α₁ ∈ [0, 1]. Needed for Lemma 4.1's bounded increments (step 2).
2. **Boundary rules:** q = ∞ when α_k ≤ 0, and err = 1 when α_k ≥ 1 (D7). Needed for Lemma 4.1
   (step 2) and for m⁺'s definition (§1).
3. **s_t > 0, finite, label-free** (gate/scale.py). Needed for B_t, S_t, S⁺_t (step 1).
4. **U3 (B7):** T⁻ ≤ T_peak ≤ T⁺. Needed for S⁺ ≥ S (step 1, Lemma 1′).
5. **U4:** the audit is a fixed function of 𝒢_t-quantities. Needed for m⁺_t to be defined and
   𝒢_t-measurable (§1, step 3).
6. **U1:** no delay. Needed for err_k = m⁺_t (step 3).
7. **U2:** A_t ~ Bernoulli(p) independent of 𝒢_t; for G7, Bernoulli(p_t) with p_t
   𝒢_t-measurable and p_t ≥ p_min. Needed for the martingale difference (step 3).
8. **U6:** T fixed. Needed for Azuma at a fixed horizon (step 3).

---

## 5 · Steps the pasted theorem text does not support

- **Row 3 (and 13).** The use of Gibbs & Candès Prop. 4.1 for our err stream rests on its *proof*,
  not its statement. Not an error, but the attribution must say "by the argument of".
- **Row 12.** "Retraining is fine" has no support in Tu & Giesecke, whose results fix ŝ. It is
  supported by our own argument only.
- **Row 7.** Azuma–Hoeffding has no source in `docs/refs/`.
- **Everything cited from Tu & Giesecke** (Prop. 5.5, (28)–(30), (40), Prop. B.14, B.15) matches the
  transcription word for word. Nothing else is cited.

---

## 6 · Sign-off (N)

| Row(s) | N's verdict | Note | Initials, date |
|---|---|---|---|
| 1–2 | | | |
| 3–4 | | | |
| 5 (U1) | | | |
| 6–10 | | | |
| 11–12 | | | |
| 13–14 (Lemma 1′) | | | |
| 15–16 (statement) | | | |
| 17–19 (G7) | | | |
| 20 (G6) | | | |
| Refs transcriptions checked against the PDFs | | | |
