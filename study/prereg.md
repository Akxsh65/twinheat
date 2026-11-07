# Pre-registration: does a calibrated trust signal change thermal-design decisions? (TwinHeat pilot, CG-1)

OSF-style pre-registration. It is filed (OSF time-stamp, or a tagged commit if OSF is not used) **before any study data
exist**, and after ethics approval (CLAUDE.md §13: the study runs only after approval). It holds no results. After
filing, any change is a dated deviation in `docs/decisions.md` (§9 below).

| Field | Value |
|---|---|
| Study | TwinHeat pilot CG-1 (CLAUDE.md §13), hypothesis H4 (§11.4) |
| Investigators | {{G}} (lead), {{N}}, {{M}}; supervisor {{name}} |
| Ethics | RVCE department committee, reference {{ref}}, approved {{date}} |
| Filed | {{date}}, commit {{sha}}, OSF {{url or "not used"}} |
| Status of data at filing | none collected; the dry run (§4.4) uses non-study boards and is not analysed |

---

## Fill before filing (decided now, never after seeing data)

These are not fixed in CLAUDE.md or the frozen configs. Each is decided by the owner and written here before filing.

| # | Item | Owner | Proposed (struck out or confirmed by the owner) |
|---|---|---|---|
| F1 | τ, the accept/flag tolerance on peak error (K); the same value as `experiments/configs/frozen.yaml` (open decision D4) | M | value from `frozen.yaml` |
| F2 | Task B: temperature limit and solve budget (number of exact verifications) | G | {{limit}} °C, {{k}} verifications |
| F3 | Bootstrap seed | G | {{seed}}, written in `study/analysis.py` at filing |
| F4 | NASA-TLX form: raw (unweighted mean of 6 subscales) or weighted (pairwise comparisons) | G | raw TLX |
| F5 | Order scheme: 3 orders (one Latin square, as in the plan), or 6 (a Williams design, which also balances which condition comes immediately before which) | G | 3, per the plan; see §3.3 |
| F6 | Target n inside 8–10 | G | 9 (a multiple of 3, so every order is used equally often) |
| F7 | Stimulus sets: "no repeats across conditions" needs three sets of 30 (90 boards, each 15 right / 15 wrong), but the G3.3 prompt says "30 boards". Confirm three matched sets | G + M | three matched sets of 30 |

---

## 1. Hypotheses

**Context.** In the TwinHeat design tool a neural surrogate predicts a board's temperature field. A prediction is
*wrong* when its true peak error against the exact solver is more than τ (F1). The calibrated condition shows a bound
B on that error from the audited conformal gate, together with a verification field that converges inside a guaranteed
band (CLAUDE.md §8.1, §13 T1.6–T1.7).

**H4 (directional, primary).** Participants **accept fewer wrong predictions** in the *calibrated* condition than in
(a) the *none* condition and (b) the *residual-flag* condition. That is, the mean paired difference in accepted-wrong
rate (calibrated − comparison) is **negative** for both contrasts.

No directional hypothesis is made for the secondary outcomes (§5.2). Their effect sizes are reported in full, but they
are not used to support or reject H4.

As in CLAUDE.md §11.4, this is a pilot and H4 is **exploratory in strength**. The result informs the design of a larger
study; it is not a test of generalisation to designers in general.

## 2. Participants

- 8–10 adult students with PCB-design or heat-transfer coursework, recruited voluntarily under the approved ethics
  protocol (target n: F6). Eligibility is checked by a single question before consent.
- Team members, anyone who saw the study stimuli, and dry-run participants are not eligible.
- Participants are identified by a code only (CLAUDE.md §10.6); no name enters any log.

## 3. Design

### 3.1 Conditions (within-subjects, every participant does all three)

| Condition | What the participant sees with each prediction |
|---|---|
| **none** | the predicted field and peak only |
| **flag** | prediction + a binary residual flag in the style of DeepOHeat-v2 (gate G1: hotspot residual above a sliding-window percentile) |
| **calibrated** | prediction + the calibrated bound B and the Predicted → Verifying → Verified badge; on a verification request, the anytime field with its guaranteed band (T1.6, T1.7) |

The interface is otherwise identical: same boards (within a condition, §3.3), same controls, same time limits.

### 3.2 Device

Seated VR by default. A participant who prefers the desktop, or who stops VR for discomfort, uses the desktop version,
which has every action (CLAUDE.md §13: desktop parity). The device is recorded for each block. Device is not a factor
in the primary analysis (§6.4).

### 3.3 Order and stimulus sets

- Condition order follows a 3 × 3 Latin square, `study/orders.json` (G3.3): (none, flag, calibrated), (flag,
  calibrated, none), (calibrated, none, flag). Participants are assigned to orders cyclically in order of arrival.
- With 3 orders each condition appears equally often in each position, but which condition comes immediately before
  which is not balanced. A 6-order Williams design would balance that too (F5). With n < 12 neither is fully balanced
  unless n is a multiple of the number of orders.
- Task-A stimuli come from `study/stimuli.json` (G3.3), selected by a fixed-seed script **before** filing:
  - 30 boards: 15 with true peak error ≤ τ (*right*) and 15 with error > τ (*wrong*);
  - at least 8 boards where the residual flag and the calibrated bound disagree;
  - the boards are spread over OOD terciles.
  - No board is shown twice to the same participant across conditions. Stimulus sets are counterbalanced with the
    condition order as in `orders.json`.
- The hidden truth (oracle peak, solved to 1e-8) is stored with each stimulus and is never shown during the session.

### 3.4 Session (≈ 45 min; exact script in G5.5)

Consent → 5-minute desktop training on non-study boards → three condition blocks in the assigned order. Each block is
Task A, then Task B, then NASA-TLX. A debrief ends the session.

## 4. Tasks

### 4.1 Task A: accept or verify (30 decisions per block)

For each of 30 predictions the participant chooses **accept** or **request verification**, with the condition's display
(§3.1). There is no correctness feedback during Task A. There is no time limit per decision; decision time is recorded.

### 4.2 Task B: cool a board (5 minutes per block)

Starting from a fixed board, the participant edits the design (move chips, paint copper, stamp vias) to bring the
verified peak temperature below the limit (F2). They have 5 minutes and a fixed budget of exact verifications (F2).
Only verified designs count.

### 4.3 NASA-TLX

Completed after each block (form: F4).

### 4.4 Dry run

One or two dry-run sessions with non-participants, on non-study boards, check the procedure and the logging. Their data
are not analysed. If the dry run shows that the stimuli fail to separate the conditions, stimuli are re-selected and the
change is logged **before** the first real session (plan, G3 fallbacks). Re-selection after the first real session is
not allowed.

## 5. Measures

All come from the session log (CLAUDE.md §10.6: `participant, condition, t_ms, action, design_hash, server_decision`),
the stimulus truth in `stimuli.json`, and the TLX form. They are computed per participant × condition.

### 5.1 Primary outcome

**Accepted-wrong rate.** The number of *wrong* Task-A stimuli the participant accepted, divided by the number of wrong
stimuli in that block (15). Range 0–1. Lower is better.

### 5.2 Secondary outcomes (no directional hypothesis)

| Outcome | Definition |
|---|---|
| Unnecessary-verification rate | *right* Task-A stimuli for which verification was requested ÷ right stimuli in the block (15) |
| Verified peak (Task B) | lowest verified peak temperature rise reached within 5 min and the budget (K); a participant with no verified design in a block gets that board's verified starting peak |
| Time | Task-A median decision time per block (s, from stimulus shown to accept / request_verify); Task-B time to the first verified design below the limit (s), censored at 300 s and reported with the number censored |
| Workload | NASA-TLX score per block (0–100, F4) |

## 6. Analysis plan (implemented in `study/analysis.py`, G5.6)

### 6.1 Contrasts

There are two paired contrasts for every outcome: **calibrated − none** and **calibrated − flag**. Flag − none is
exploratory (§7).

### 6.2 Effect size and interval

- **Effect size:** the mean of the per-participant differences (calibrated − comparison), in the outcome's own units
  (rate points, K, s, TLX points). This is the primary effect size. The standardised paired effect d_z (mean difference
  ÷ SD of the differences) is reported beside it, with a note that it is unstable at this n.
- **95 % CI:** percentile bootstrap over participants. Each resample draws n participants with replacement and keeps
  each participant's three conditions together, so pairing is preserved. 10,000 resamples, NumPy
  `Generator(PCG64(seed))` with the seed fixed in F3. d_z gets a CI from the same resamples. If a resample has all
  differences identical (SD 0), d_z is undefined for that resample; the count of such resamples is reported and they
  are dropped from the d_z interval only.
- **No p-values or significance tests.** There is no correction for the two contrasts: both are reported, and neither
  is selected afterwards.

### 6.3 Reading H4 (fixed now)

For each H4 contrast:
- **supported:** the point estimate is negative and the 95 % CI lies wholly below 0;
- **contradicted:** the CI lies wholly above 0;
- **inconclusive:** otherwise.

H4 overall is *supported* only if both contrasts are supported. These words describe this pilot only.

### 6.4 What does not change the primary analysis

Order, device and stimulus set are not modelled in the primary analysis; with 8–10 participants such a model cannot be
estimated reliably. They are reported descriptively and examined in §7.

### 6.5 Missing data

The primary analysis uses only participants who are complete under §8. There is no imputation.

## 7. Exploratory (labelled as such wherever reported)

Everything not in §6.1–§6.3, including:
- flag − none contrasts;
- order and position effects, and carry-over;
- VR versus desktop;
- per-stimulus results: boards where flag and bound disagree, OOD terciles;
- NASA-TLX subscales;
- learning across blocks;
- the relation between decision time and correctness;
- Task-B trajectories;
- any outcome, contrast, subgroup or exclusion not written here.

Exploratory results go to `study/results/exploratory.csv` and are never described as confirming H4.

## 8. Exclusion rules (decided now)

1. **Withdrawal.** A participant who withdraws consent is removed and their data deleted, as in the ethics protocol.
   This is counted in the participant flow only.
2. **Incomplete session.** A participant who does not finish all three blocks (for example, stopped for discomfort with
   no desktop continuation) is excluded from the primary analysis. Completed blocks are reported descriptively.
3. **Logging failure.** A participant missing any of the 90 Task-A decisions, or any Task-B or TLX record, from the
   logs is excluded from the primary analysis for that outcome only. Each exclusion is listed with its cause.
4. **Protocol breach.** A participant shown the wrong condition, order or stimulus set, or given correctness feedback
   in Task A, is excluded from the primary analysis. The breach is logged at the time by the experimenter.
5. **Ineligibility.** A participant found after the session not to meet §2 is excluded.

There are no performance-based exclusions: accepting all or verifying all is a valid behaviour and stays in. No other
exclusion is allowed in the primary analysis. Any other exclusion is exploratory and is reported beside the
unexcluded result.

## 9. Sample size and deviations

**Why 8–10.** This is a feasibility pilot, sized by what one week of sessions (week 9) and the eligible student pool
allow, not by a power analysis. It is not powered to detect an effect of any particular size. Its aims are:
- to check that the procedure, logging and stimuli work;
- to estimate the direction and rough size of the H4 effect and the spread of paired differences, for planning a
  larger study.

The CIs will be wide and are reported as such. Stopping is fixed: recruitment ends when n reaches the target (F6) or
the week-9 window closes, whichever comes first. It never depends on the data.

**Deviations.** Any change after filing is recorded in `docs/decisions.md` with its date, its reason, and whether any
data had been seen. It is reported in the pilot section.

## 10. Data and reproducibility

- Session logs follow CLAUDE.md §10.6 (participant code only) and are validated against
  `contracts/schemas/session_log_line.json`.
- Storage, back-up and deletion follow the approved data plan.
- `study/analysis.py`, `stimuli.json` and `orders.json` are committed and tagged before the first session. The analysis
  is run unchanged on the real logs, and the figure footer carries the commit, seed and date (CLAUDE.md §4).
