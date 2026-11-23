# Lemma 1, Lemma 1′, G6 and G7: hostile review (M3.4)

Drafted by Claude for the checker of Lemmas 1 and 1′ (CLAUDE.md §2: M; N signs this review). It
checks the statements as they stand in CLAUDE.md §8.2–§8.5 and §18.1, because `report/lemmas.tex`
does not exist yet. Nothing here is signed off: every "ours" step below needs the checker's initials
in the table of §9 before it goes into the report.

**Sources actually read** (every quotation and theorem number is from these PDFs, not from memory or
from the study guides):

- Gibbs & Candès, *Adaptive Conformal Inference Under Distribution Shift*, arXiv 2106.00170**v3**
  (`references/03_uncertainty_conformal/all2_Gibbs_AdaptiveConformalInference.pdf`): eq. (2) p. 3,
  Lemma 4.1 and Proposition 4.1 p. 6.
- Tu & Giesecke, *Online Conformal Prediction for Non-Exchangeable Panel Data*, arXiv
  2605.17705**v2** (`references/03_uncertainty_conformal/all3_Tu_OnlineConformalPanelData.pdf`):
  filtration p. 3, eqs. (2)–(5) pp. 4–5, Proposition 5.5, Assumptions 5.6 and 5.8, Proposition 5.7
  and Theorem 5.9 p. 6, App. B.2.1 pp. 18–19, Lemma B.13 pp. 20–21, B.2.4 pp. 21–22, Proposition
  B.14 p. 22, B.2.6 and Proposition B.15 pp. 23–24.
- Bastani et al., arXiv 2206.01067v1, through the committed transcription
  `docs/refs/bastani2022_mvp.txt` (Definition 3.1, Remark 3.1, Theorem 3.1).

The two refs files the prompt names, `docs/refs/gibbs_candes_prop41.txt` and
`docs/refs/tu_giesecke_thm54_56.txt`, do not exist. The PDFs above were used instead. CLAUDE.md
§16.2 still asks for those files.

---

## 0 · Verdict in one paragraph

Lemma 1 as stated in §8.2 is **true but proved by a different route than the sketch says**, and
its constant is loose. Step (1) is correct. Step (2) is correctly cited only after the D12 fix, and
the right citation is Gibbs & Candès Prop. 4.1 itself, applied to the audited subsequence, which is
what Tu & Giesecke's Prop. 5.5 does. Step (3), "transfer is Tu & Giesecke Thm 5.6 [5.9]", cannot be
cited as written: Theorem 5.9 is stated for their W-TQA algorithm under their Assumption 5.6, not for
our gate. And it is not needed. A direct argument, which is ours, gives the all-step bound in one
martingale step, with no control of n_a at all:

> **W_T ≤ α + C/(γpT) + (1/p)·√(ln(1/δ)/(2T))** with probability ≥ 1 − δ, C = max(α₁, 1 − α₁) + γ.

This implies the §8.2 bound, since (1/p)√(ln(1/δ)/(2T)) ≤ (2/p)√(ln(4/δ)/T), and it does not need
α ≤ ½. It does need four assumptions that §8.2 leaves unstated: no audit delay, a coin the proposer
cannot predict, B7, and an audit procedure that does not use the coin (§2).

Lemma 1′ holds, given B7. The G6 per-group statement is a sketch, not a proof (§6). For G7 there is a
candidate proof, ours and UNSURE, by the same martingale argument (§7).

---

## 1 · Setting and notation (what the proof is about)

Step t = 1, …, T (T fixed in advance):

- the proposer commits x_t;
- the gate computes the label-free scale s_t > 0 and the threshold q_t ∈ [0, ∞] from the past;
- it decides "accept" iff B_t = q_t·s_t ≤ τ (B_t = ∞ when q_t = ∞);
- the server draws A_t ~ Bernoulli(p).

Let e_t = |T̂_peak,t − T_peak,t| be the true peak error against the exact discrete solution, and
S_t = e_t/s_t. If A_t = 1, an audit returns a certified interval [T⁻_t, T⁺_t] ∋ T_peak,t and
S⁺_t = max(|T̂ − T⁻|, |T̂ − T⁺|)/s_t.

The level α_k moves only on audited steps (D7 forces err = 1 when α_k ≥ 1):

  α_{k+1} = α_k + γ(α − err_k),  err_k = 1 if α_k ≥ 1; 0 if α_k ≤ 0; else 1[S⁺ > q].

Define for **every** step t, audited or not, the potential indicator

  m⁺_t := 1 if α(t) ≥ 1; 0 if α(t) ≤ 0; else 1[S⁺_t > q_t],

where α(t) is the level in force when x_t is gated. S⁺_t is the score the audit *would* return.
That is well defined because the audit procedure is a deterministic function of the committed
design, the model, q_t and ε_S (assumption U4 in §2). On audited steps m⁺_t is the err fed to the
recursion. That identification holds only if the threshold in force at the update equals q_t
(assumption U1).

W_T = (1/T)·#{t : accepted and e_t > τ}.

---

## 2 · Unstated assumptions (each one is needed below)

| # | Assumption | Why it is needed | Where it stands |
|---|---|---|---|
| U1 | **No audit delay**: the audit of step t updates α before step t + 1 is gated. Then err for t is scored against the same q_t that gated x_t | identifies err_k with m⁺_t (step 3); without it the revealed-round identity and the gating use different thresholds | **Not true of the code in a batch.** `gate/aci_audit.py::update` and `gate/active_audit.py::update` score each record against the *current* q, which in a batch has already moved. See §5 |
| U2 | The coin is independent of everything before it: E[A_t − p ∣ 𝒢_t] = 0, where 𝒢_t = all history, x_t, the model at t, q_t, s_t and the label of t | the martingale step. This is Tu & Giesecke's Assumption 5.8 (MCAR, with ℋ_t = ℱ_t ∨ σ(Y_t)) and slightly more, because we also condition on past coins and labels | true by construction (`gate/coin.py`, drawn after commit) **only if the proposer cannot predict the PCG64 stream**: the seed must stay on the server and never be logged where a proposer sees it |
| U3 | **B7**: T⁻_t ≤ T_peak,t ≤ T⁺_t for every audit | S⁺ ≥ S (Lemma 1′) | a hard invariant (CLAUDE.md §5). Any certificate violation voids Lemma 1′ for the run |
| U4 | The audit (and so S⁺_t) is a deterministic function of quantities fixed before A_t: design, model, q_t, ε_S, a label-independent iteration cap. It does not depend on A_t or on later steps | m⁺_t must be 𝒢_t-measurable, i.e. predictable | design requirement for N3.3 `audit_until_decided`; any randomness in it must be independent of the coin |
| U5 | α ∈ (0, 1) and α₁ ∈ [0, 1] | Gibbs & Candès's Lemma 4.1 proof uses sup |α − err| < 1 | true (α₁ = α) |
| U6 | T is fixed in advance | Azuma–Hoeffding at a fixed horizon | for a run that stops on a data-dependent rule, use the maximal form (Hoeffding's inequality via Doob / Ville), still with the horizon T_max in the slack. UNSURE whether the report needs this |

What is **not** needed, despite the sketch:
- α ≤ ½: nowhere used below.
- A fixed score function, or a model held fixed between steps: retraining is fine, provided the model
  used at step t is fixed before A_t. This settles the UNSURE in §18.1 for our argument, because the
  argument never uses Tu & Giesecke's theorem.
- Tu & Giesecke's Assumption 5.6 (feasibility of the fallback set): it exists only because W-TQA
  replaces the infinite threshold by a finite one ("finitization", p. 4). We keep q = ∞, which gives
  Gibbs & Candès's deterministic lower bound α ≥ −γ instead.

---

## 3 · The proof, step by step

### Step 1: accepted large errors are miscoverages (ours, elementary) — CORRECT

If x_t is accepted with e_t > τ:
- q_t < ∞, since otherwise B_t = ∞ > τ and x_t would have been flagged;
- e_t > τ ≥ B_t = q_t s_t, so S_t > q_t.

If α(t) ≥ 1, q_t = 0 and m⁺_t = 1 by definition. If α(t) ≤ 0, q_t = ∞ and nothing is accepted.
In every case accepted ∧ e_t > τ ⇒ 1[S_t > q_t] = 1 ⇒ m⁺_t = 1, using S⁺_t ≥ S_t (U3) when
α(t) ∈ (0, 1). Hence

  **W_T ≤ M⁺_T := (1/T) Σ_t m⁺_t.**  (1)

### Step 2: the audited steps alone (cited) — CORRECT after the citation fix

Gibbs & Candès's Lemma 4.1 proof (p. 6) uses only three facts:
- err ∈ {0, 1};
- α_t < 0 ⇒ err_t = 0;
- α_t > 1 ⇒ err_t = 1 (the other case "is identical");

together with |α − err| < 1. Proposition 4.1 is then the telescoping

  α_{T+1} = α₁ + Σ γ(α − err_t) ∈ [−γ, 1 + γ].

Both hold for **any** sequence of 0/1 errors with those boundary properties. Our audited stream has
them: q = ∞ when α_k ≤ 0 gives err = 0, and D7 gives err = 1 when α_k ≥ 1. Index the audited steps
k = 1, …, n_a. Then, pathwise, for every n_a:

  **|Σ_{k ≤ n_a} (err_k − α)| ≤ C/γ, C = max(α₁, 1 − α₁) + γ.**  (2)

Tu & Giesecke's Proposition 5.5 is exactly this on revealed rounds. Their proof (App. B.2.1, p. 19)
says: "The argument is the telescoping of Proposition 4.1 of Gibbs and Candès (2021), applied only on
rounds where target feedback is observed".
- Their eq. (6) is the identity.
- Their eq. (7) bounds the miscoverage from below using the upper level bound α_t ≤ 1 + γα (their
  eq. (28)).
- They have no deterministic upper bound on the miscoverage. That side needs a lower bound on the
  level, which their finitization removes and Assumption 5.6 restores in probability.
- Our q = ∞ rule keeps that side deterministic.

*Sharper one-sided constant (ours, elementary; same reasoning as their (28)).* Only the upper bound
on miscoverage matters for W_T. If α_k ≤ 0 then err_k = 0 and the level rises; otherwise it falls by
at most γ(1 − α). So α_k ≥ −γ(1 − α) for all k, and

  Σ_{k ≤ n_a} (err_k − α) = (α₁ − α_{n_a+1})/γ ≤ C⁺/γ, C⁺ := α₁ + γ(1 − α).

With α₁ = α, C⁺ = α + γ(1 − α) ≤ C. C⁺ may replace C in (4) below; the checker should confirm.

*Citation fix (D12):* §8.2 step (2) cites "Tu & Giesecke Thm 5.4". In v2, Thm 5.4 is the spatial
oracle-profile bound (p. 5). The right citations are Gibbs & Candès Prop. 4.1, applied to the
audited subsequence, as in Tu & Giesecke Prop. 5.5.

### Step 3: from audited steps to all steps (ours) — REPLACES the sketch's step (3)

U1 makes err_k = m⁺_t on the k-th audited step t. So Σ_t A_t (m⁺_t − α) is the left side of (2).
Write

  p(m⁺_t − α) = A_t(m⁺_t − α) − (A_t − p)(m⁺_t − α),

and sum over t:

  p Σ_t (m⁺_t − α) = Σ_{audited} (err_k − α) − Σ_t D_t,  D_t := (A_t − p)(m⁺_t − α).  (3)

**The martingale difference.**
- m⁺_t is 𝒢_t-measurable (U4) and D_1, …, D_{t−1} are 𝒢_t-measurable (past coins are in 𝒢_t).
- By U2, E[D_t ∣ 𝒢_t] = (m⁺_t − α)·E[A_t − p ∣ 𝒢_t] = 0.
- Given 𝒢_t, D_t takes one of two values, −p(m⁺_t − α) and (1 − p)(m⁺_t − α), so it lies in an
  interval of length |m⁺_t − α| ≤ max(α, 1 − α) ≤ 1. That length is predictable.

The one-sided Azuma–Hoeffding inequality (interval form) gives
P(−Σ_t D_t ≥ ε) ≤ exp(−2ε²/Σ_t |m⁺_t − α|²) ≤ exp(−2ε²/T). With ε = √(T ln(1/δ)/2), on an event
of probability ≥ 1 − δ:

  −Σ_t D_t ≤ √(T ln(1/δ)/2).

Insert this and (2) into (3), divide by pT, and use (1):

  **W_T ≤ M⁺_T ≤ α + C/(γpT) + (1/p)·√(ln(1/δ)/(2T)).**  (4)

The slack can be sharpened to (max(α, 1 − α)/p)·√(ln(1/δ)/(2T)) by keeping the interval lengths.

**Re-deriving the §8.2 terms.**
- *C/(γpT)* is (2) divided by pT. It is the deterministic ACI slack C/(γ n_a) with n_a replaced by
  its mean pT, and it comes out exactly so, with no concentration of n_a.
- *(2/p)√(ln(4/δ)/T)* is not needed. The sketch appears to control n_a/(pT) by Hoeffding (n_a ≥
  pT − √(T ln(2/δ′)/2)) and split δ between that and the transfer, hence ln(4/δ). But controlling
  n_a alone does not transfer anything from audited to unaudited steps; the martingale (3) does that,
  and it absorbs n_a automatically.
- (4) is strictly smaller than the §8.2 bound: √(ln(1/δ)/2) ≤ 2√(ln(4/δ)) for every δ ∈ (0, 1). So
  §8.2's statement is **true but loose**, and its sketch does not prove it as written.
- n_a still matters for the *revealed-round* statement: (2) divided by n_a gives the observed audit
  miscoverage within C/(γ n_a) of α, deterministically. Report it beside W_T, as §8.2 suggests.

**In expectation (cited).** Taking expectations in (3), without Azuma, gives
|(1/T) Σ_t E m⁺_t − α| ≤ C/(γpT). This is the same identity as Tu & Giesecke eq. (40), proof of
Thm 5.9 (App. B.2.4, p. 22), which uses only their update (5) and Assumption 5.8. Cite it as "as in
the proof of Tu & Giesecke (2026, Thm 5.9)", not as their theorem: the theorem's hypotheses (W-TQA,
Assumption 5.6) are not ours.

### Corollary — CORRECT

Among accepted predictions, the large-error rate is W_T / (accepted fraction). That is a ratio of
two observed counts, with no extra probability statement. Report both, as §8.2 says.

---

## 4 · Lemma 1′ (certified scores) — CORRECT given U3 and U4

- *Is err = 1[S⁺ > q_t] valid input to the ACI recursion?* Yes. Proposition 4.1 needs nothing of
  err except the three properties listed in step 2, and those hold for the S⁺ stream with q = ∞ for
  α ≤ 0 and D7 for α ≥ 1. **D7 is what makes this true**: §8.2's original "q = 0 when α ≥ 1, so
  err = 1[S⁺ > 0]" fails the third property if S⁺ = 0 exactly.
- *Does W_T's bound carry over?* Yes, through (1): S ≤ S⁺ (U3) gives 1[S > q] ≤ 1[S⁺ > q]. W_T is
  bounded by the S⁺-miscoverage, which is what the recursion controls.
- *What early stopping costs.* If an audit stops because its score interval [S⁻, S⁺] lies entirely
  on one side of q_t, then 1[S⁺ > q_t] = 1[S > q_t] exactly, and nothing is lost. Only a stop for
  width < ε_S that straddles q_t gives err = 1 where the exact err may be 0. q_t then grows slightly
  conservative, as §8.3 says. This only holds if the audit stops relative to the **same** q_t that
  scores it (U1).
- A certificate violation (B7) can make S⁺ < S and break (1) for that step. That is why every E13 and
  E14 run must check B7 against the oracle.

---

## 5 · Finding in the code (U1): audits in a batch are scored against a moved threshold

`AuditedACIGate.update` (and the G7 subclass) scores record k with `err = int(S > q)`, where
`q = self._q` is the threshold **after** the updates of records 1, …, k − 1 of the same call. In the
§9.2 batched pipeline, every design in a batch was gated with the q_t in force at gating time. So the
second and later audits of a batch are scored against a threshold that did not gate them:
- step 3's identification err_k = m⁺_t fails;
- the early-stop "decided relative to q_t" of N3.3 refers to a different q than the one that scores
  the audit.

This is the delayed-feedback situation of Tu & Giesecke App. B.2.6. Their delayed variant computes
each delayed ℓ_s "from the stored threshold q̂_s" (p. 23, eq. (43)), and Proposition B.15 then keeps
the guarantees with constants γα and γ(1 − α) multiplied by D (the maximal delay), for their
algorithm.

**Proposed fix, for M (owner of `gate/`), not made here:**
1. Score each audit against the q stored in its `GateDecision` / `AuditRecord`. The M3.1 tests are
   sequential (one design, one update), so they would not notice.
2. Then either keep U1 by running gate experiments sequentially, or prove the delayed version of
   (2) for our gate. Prop. B.15 proves it for theirs, which suggests C → C + (D − 1)γ·max(α, 1 − α).
   That constant is UNSURE; deriving it is ours, not cited. This ties into D14 (audit delay bound).

Until one of the two is done, Lemma 1 covers **sequential** gating (batch size 1 between updates)
only.

---

## 6 · G6 per-group (§8.5 Extension A) — a SKETCH, not a proof

§8.5's argument is: "membership fixed before the coin ⇒ each group's audits are an unbiased
subsample ⇒ Lemma 1's transfer applies per group". The transfer half survives. Replace m⁺_t − α by
1[t ∈ g](m⁺_{t,g} − α) in (3); the indicator is predictable, so Azuma applies unchanged with T
replaced by a bound on the group size. What is missing is a **per-group analogue of (2)**, a bound on
the audited steps of group g, and Bastani et al. do not provide one in the form needed:

1. *In expectation, not pathwise.* Theorem 3.1 gives (c_exp α(·), m)-multivalidity "in expectation
   over the randomness of π_{1:T}", with c_exp ≤ √(4K_ϵ|𝒢|m ln(|𝒢|m)) + ρT. Our step 3 needs a bound
   that holds on every path, or with high probability.
2. *Per (group, bucket), two-sided coverage.* The guarantee is on the subsequence where x ∈ G and
   q_τ ∈ B_m(i). A per-group rate needs the sum over the m buckets: the slack adds up across buckets,
   and coverage must become a one-sided accepted-error rate through an analogue of (1) per group.
3. *Smoothness.* Theorem 3.1 holds "against any (ρ, rm)-smooth adversary". A deterministic solver
   gives a point-mass score for each design, so ρ = 1 and the bound is void. Remark 3.1's
   noise-perturbation fix must be drawn after commit and hidden. That the noise keeps the Lemma 1′
   direction is D9, unchecked.
4. *The score depends on the learner's draw.* In the protocol the adversary fixes (x_t, y_t) and the
   learner then randomises q_t. An audit that stops "when decided relative to q_t" makes S⁺_t depend
   on the learner's own draw. For G6 runs, audits must stop on width ε_S (or full certification)
   only, independently of q_t.

**Weakest correct statement now:**
- *What can be claimed:* with smoothing noise and q-independent audit stopping, Theorem 3.1 gives, in
  expectation over MVP's randomisation, multivalid coverage of the S̃ = S⁺/(1 + S⁺) stream on the
  audited steps for every (group, bucket) of 𝒢. The extension to a per-group accepted-error rate over
  all steps is not proved.
- *How to report H8:* empirically, as §8.5's week-8 fallback already allows. Name gaps 1–4 in the
  limitations.

---

## 7 · G7 risk-adaptive audits — a candidate proof (ours, UNSURE)

§8.5 and §18.1 say G7 has no cited validity argument. That is still true: nothing below is cited.
But the argument of §3 extends, provided:
- p_t is fixed before the coin;
- p_t ≥ p_min;
- A_t ~ Bernoulli(p_t) given 𝒢_t.

Here w_t = p̄/p_t ≤ p̄/p_min.

- *(2′) Pathwise bound on the weighted audited sum.* The update α ← α + γ w_t(α − err) has
  increments of size < γ p̄/p_min, and the same two boundary implications hold. Lemma 4.1's proof
  then gives α ∈ [−γp̄/p_min, 1 + γp̄/p_min], and telescoping gives
  |Σ_audited w_t(err_t − α)| ≤ C′/γ, with C′ = max(α₁, 1 − α₁) + γ p̄/p_min.
- *(3′) Transfer.* p̄(m⁺_t − α) = w_t A_t(m⁺_t − α) − p̄(A_t/p_t − 1)(m⁺_t − α). The second term has
  conditional mean 0 and lies in an interval of length p̄/p_t ≤ p̄/p_min.
- *Result.* Azuma as before gives

  W_T ≤ α + C′/(γ p̄ T) + (1/p_min)·√(ln(1/δ)/(2T)).

The price of risk-adaptive audits is 1/p_min in place of 1/p in the statistical slack. That favours
a p_min that is not too small (E16 sweeps it).

**UNSURE:**
- Whether the code's π_t (`gate/active_audit.py`, which uses a running mean s̄_t) is computed from
  label-free quantities only. It appears to be.
- Whether Lemma 4.1's step-size argument goes through with w_t > 1. It needs γ w_t |α − err| to stay
  bounded, which p_min ensures.

The checker must verify both before G7 is claimed as anything more than an ablation.

---

## 8 · What is cited and what is ours

| Piece | Status | Exact source |
|---|---|---|
| ACI update (2) and the miscoverage definition | cited | Gibbs & Candès (2021), eq. (2), p. 3 |
| α_t ∈ [−γ, 1 + γ] | cited | Gibbs & Candès, Lemma 4.1, p. 6 |
| deterministic \|mean err − α\| ≤ C/(Tγ) for any data | cited | Gibbs & Candès, Prop. 4.1, p. 6 |
| applying it to the revealed (audited) subsequence | cited | Tu & Giesecke (2026), Prop. 5.5, eq. (6); proof App. B.2.1 |
| MCAR assumption on the reveal mechanism | cited (we satisfy a stronger version) | Tu & Giesecke, Assumption 5.8 |
| in-expectation all-rounds identity | cited as an argument | Tu & Giesecke, eq. (40), proof of Thm 5.9 (App. B.2.4) |
| no transfer without an assumption on revelation | cited | Tu & Giesecke, Prop. 5.7 |
| selection bias of flagged-only updates (G4) | cited | Tu & Giesecke, Prop. B.14 |
| delayed labels scored with the stored threshold, constants × D | cited (their algorithm) | Tu & Giesecke, App. B.2.6, Prop. B.15 |
| accepted large error ⇒ miscoverage, (1) | **ours** | elementary |
| D7 boundary rule making S⁺ a valid ACI input | **ours** | decision D7 |
| high-probability all-step bound (4) via the martingale (3) | **ours** | Azuma–Hoeffding, a textbook inequality (UNVERIFIED which textbook to cite; ask before citing) |
| Lemma 1′ (S⁺ ≥ S, decided stops lose nothing) | **ours** | needs B7 (Lemma 2) |
| delayed-feedback version for our gate | **ours, not done** | §5 |
| G6 per-group claim | **ours, sketch only** | §6; MVP coverage cited from Bastani et al. Thm 3.1 |
| G7 bound | **ours, candidate, UNSURE** | §7 |

Already listed in CLAUDE.md §1 as not ours and must stay so: online conformal inference with
intermittently revealed labels, including the all-round transfer in expectation (Gibbs & Candès;
Tu & Giesecke).

---

## 9 · Checking table (for N to fill in; Claude does not sign)

| # | Item | Claude's verdict | Checker verdict | Initials, date |
|---|---|---|---|---|
| 1 | Step 1, (1) | correct | | |
| 2 | Step 2, (2): Gibbs & Candès Prop. 4.1 on the audited subsequence; boundary properties from q = ∞ and D7 | correct | | |
| 3 | Citation fix: Thm 5.4 → Prop. 5.5; "Thm 5.6" → "as in the proof of Thm 5.9" (D12) | needed | | |
| 4 | Step 3, martingale (3), predictability of m⁺_t, interval length ≤ 1 | correct (UNSURE until checked) | | |
| 5 | Bound (4) implies the §8.2 bound; α ≤ ½ not needed | correct | | |
| 6 | U1 (no delay) stated in the lemma; code finding §5 | needed | | |
| 7 | U2 (coin seed secret), U4 (audit does not use the coin) stated | needed | | |
| 8 | Lemma 1′ incl. "decided stops are exact" | correct given B7 | | |
| 9 | G6: report as empirical; gaps 1–4 | sketch only | | |
| 10 | G7 candidate bound | UNSURE | | |

---

## 10 · Proposed text for `lemmas.tex`

The weakest correct statement, i.e. what can be claimed today. Replaces the §8.2 statement once rows
1–8 are signed.

```latex
\begin{lemma}[Proposer-agnostic control of accepted errors]\label{lem:1}
Run the audited gate (G5) for a fixed horizon $T$, gating each step with $B_t=q_t s_t$ and updating
$\alpha_k$ only on audited steps by $\alpha_{k+1}=\alpha_k+\gamma(\alpha-\mathrm{err}_k)$, with
$q=\infty$ when $\alpha_k\le 0$ and $\mathrm{err}_k=1$ when $\alpha_k\ge 1$. Assume
(i) the design $x_t$, the model, $s_t$ and $q_t$ are fixed before the audit coin $A_t$;
(ii) $A_t\sim\mathrm{Bernoulli}(p)$ independently of everything before it, including $x_t$ and its
     label (the coin stream is unknown to the proposer);
(iii) each audit returns $T^-_t\le T_{\mathrm{peak},t}\le T^+_t$ (invariant B7) by a procedure that
     does not use $A_t$, and updates $\alpha$ before step $t+1$ is gated;
(iv) $\alpha\in(0,1)$, $\alpha_1\in[0,1]$.
Let $W_T$ be the fraction of the $T$ steps accepted while $|\hat T_{\mathrm{peak},t}-T_{\mathrm{peak},t}|>\tau$.
Then for every $\delta\in(0,1)$, with probability at least $1-\delta$,
\[
  W_T \;\le\; \alpha + \frac{\max\{\alpha_1,1-\alpha_1\}+\gamma}{\gamma p T}
        + \frac{1}{p}\sqrt{\frac{\ln(1/\delta)}{2T}} .
\]
\end{lemma}
\noindent The audited-step part is Gibbs and Cand\`es (2021, Prop.~4.1) applied to the revealed
subsequence, as in Tu and Giesecke (2026, Prop.~5.5); the transfer to all steps, in expectation, follows
the proof of Tu and Giesecke (2026, Thm~5.9); the high-probability transfer and the use of certified
scores are ours.
```

Open before the report uses any of this:
- the checker's sign-off (§9);
- the U1 code fix or a sequential-only scope (§5);
- the refs files of CLAUDE.md §16.2;
- a citation for Azuma–Hoeffding that the user supplies (CLAUDE.md §0 rule 10).
