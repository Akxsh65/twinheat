# TwinHeat — Full Implementation Plan (Phase 0 to Phase 6)

Version 1.0 · 23 October 2026 · Team of 3 (N numerics/PAGP, M ML/DL, G graphics/VR/hardware) · 12 weeks
Builds on: plan v3.1 (5 Oct 2026), Phase 0 v1.2 (22 Oct 2026). Lives in the repo as `docs/TwinHeat_Implementation_Plan.md`.

---

## How this plan and CLAUDE.md work together

| File | What it holds | How Claude gets it |
|---|---|---|
| `CLAUDE.md` (repo root) | **Standing context**: physics spec, constants, equations, contracts, the maths (Lemmas 1, 1′, 2), reference code, gates, hypotheses, invariants, open decisions, rules for Claude | **Claude Code** loads it automatically every session. **Claude chat**: add it to the Project's knowledge (or paste it as the first message) |
| This plan | **The work**: every phase, every role, every task, with the prompt to send, how to check it, and what to do when it breaks | You copy one prompt at a time into Claude |
| `docs/refs/` | Material only a person can supply: paper algorithm text, datasheets, tables (CLAUDE.md §16.2) | Claude Code reads the files; in chat, attach them |

**Prompts are not stored in CLAUDE.md.** CLAUDE.md is what Claude *always knows*. A prompt is what you ask for *this time*. Every prompt below points to the exact CLAUDE.md sections it needs ("CLAUDE.md §7.2"), so nothing has to be pasted from the plan PDFs.

**What you still fill in.** Only things that exist at run time, marked `{{RUNTIME: …}}`: test output, a results CSV, a pasted abstract, a fab quote, your names. Everything else is in the prompt or in CLAUDE.md.

**Prompt anatomy.** Every task prompt has the same blocks:

```
READ FIRST      – CLAUDE.md sections and repo files Claude must read before writing anything
GOAL            – one paragraph: what and why
DELIVERABLES    – exact files and functions
SPEC            – rules that are specific to this task (the rest is in CLAUDE.md)
TESTS FIRST     – tests to write before the implementation
CONSTRAINTS     – what not to do
DONE WHEN       – objective completion criteria
REPORT BACK     – changed-file tree, test summary, 3 ways it could still be wrong, missing information
```

**Using Claude Code (recommended for code tasks).** Run `claude` in the repo root. Paste the prompt. Let it read the files. Review the diff. Run the tests yourself. Commit. Use chat Claude for maths checking, writing and literature, with CLAUDE.md in the Project's knowledge.

**Rules that apply to every task**
1. Spec before code. Tests before implementation. Small diffs.
2. Claude never has the last word on maths. The named checker signs (N checks Lemma 1/1′, M checks Lemma 2).
3. No number goes into the report unless a script on a tagged commit produced it.
4. A person opens every paper before it is cited.
5. Log useful prompts and outputs in `report/claude_log.md` (date, author, purpose).

---

## Team, ownership and parallel working

| Role | Owns | Examiner | Load (PW) | Checks |
|---|---|---|---|---|
| **N** numerics and parallel lead | `solvers/ certify/ pipeline/ bench/ data/generate.py` | PAGP | ≈ 11.8 | Lemma 1 and 1′ |
| **M** ML lead | `data/sampler.py raster.py models/ gate/ experiments/ release/ server/` (shared) | DL | ≈ 11.3 | Lemma 2 + FP margin |
| **G** graphics, VR and hardware lead | `viewer/ hardware/ study/ server/` (shared), deck | CGVR | ≈ 11.8 | contract compliance, figure readability |

**Stubs that keep the lanes independent (week 1)**
| Stub | Built by | Used by | Replaced by |
|---|---|---|---|
| `solve_stub(board)`: SciPy direct on the 64² plate | N | M, G | wk 3 4-layer solver |
| `fake_dataset/`: 200 boards, 64², 4-layer tensor shapes | N + M | M, G | wk 5 real data |
| `server/model_stub.py`: solver field + 2 % smooth noise, fake σ | M | G, N | wk 5 FNO ensemble |
| `server/ws_mock_server.py`: replays recorded messages | G + M | G | wk 7 real server |
| `gate_stub`: fixed q, uniform audits | M | N, G | wk 6 G5 |

**Cadence.** Daily async post (done / next / blocked; a block older than 24 h is raised). Monday 30-min sync (timeline, gates, `pytest` live). Thursday integration hour (merge, end-to-end smoke test board → solver → model → gate → server → viewer). Gate reviews at the end of weeks 3, 5, 6 and 8, recorded in `docs/decisions.md`.

**Git.** `main` always green; branches `n/…`, `m/…`, `g/…`; PR = CI + one reviewer; tags `v0-contracts`, `v1-solver`, `v2-data`, `v3-freeze`, `v1.0-release`. No data or checkpoints in git.

---

## Safety barriers (summary; full table in CLAUDE.md §5)

B1 M-matrix · B2 solver correctness (MMS slope 2, energy ≤ 0.1 %) · B3 solver agreement · B4 linearity · B5 symmetry · B6 contract validation · **B7 certificate is a true upper bound: any violation blocks results** · **B8 gate isolation (only audits move q; coin after commit; `query` never touches the gate)**.
Data hygiene: hashed shards, locked calibration split, train-only normalisation, never edit a released split.
Backups: code on GitHub + local clones; data on a Kaggle Dataset + Drive; checkpoints every epoch to Drive; oracle labels and audit logs on Drive + an external disk; hardware frames to Drive the same day; report on Overleaf with git sync.
Reproducibility: figure footers, frozen configs, 5 seeds, decision log.

---

## Timeline and hand-offs

| Week | N | M | G | Gate |
|---|---|---|---|---|
| 1 | repo, CI, env, spec YAMLs, M-matrix test | contracts, sampler v0, novelty search | viewer skeleton, ethics, equipment | **G0** |
| 2 | 64² + 4-layer FVM, certificate reproduction | rasteriser, OOD generators | KiCad boards, thermal shading | |
| 3 | CUDA kernels, PCG, AMG baseline, **cost go/no-go** | Darcy FNO validation | order boards | **G1** |
| 4 | data generation; certificate kernels start | input pipeline, FNO 128² | layer peel, realism | |
| 5 | data verified + uploaded; certified PCG | ensemble 256², E5, **accuracy go/no-go** | pre-registration filed, server v0 | **G2** |
| 6 | **certificate tightness decision**; fused gate kernel | gate G0–G7, Lemma 1 write-up | trust display, replay | cert |
| 7 | scan compaction, oracle labelling (~1e5) | G6 proof, E7/E8/E14–E16 | VR interaction, server integration, rig | |
| 8 | E4, E13, iterate streaming | E9 Darcy, gate bridge, **H1 + H6 in hand** | anytime rendering | **G3, G4** |
| 9 | server profiling, particle twin, DDP | server v1, convection fit, P4 import | pilot (dry run + 8–10), hardware measurements | |
| 10 | **freeze**, clean-room re-run | P4 in E7/E15 | features frozen | **G5** |
| 11 | PAGP figures + chapter | release with DOI, DL chapter | pilot analysis, CGVR chapter | |
| 12 | rehearsals | workshop draft | deck, backup video, rehearsals | **G6** |

Hand-offs: wk 1 contracts (M+G → all) · wk 3 verified solver (N → M) · wk 4 14k boards (N → M) · wk 5 ensemble `predict()` (M → N, G) · wk 6 G5 gate (M → N) · wk 7 oracle labels (N → M) · wk 8 `iterate` stream (N → G) · wk 9 P4 traces + hardware frames (G → M) · wk 10 `v3-freeze` (all).

---

# Phase 0 · Close-out and foundations (week 1)

**Goal.** Turn the finished research into an engineering base that all three can build on in parallel from day 2.
**Gate 0 (end of week 1):** repo + CI green · `spec_v1.yaml`, `spec_v2.yaml` merged · `docs/contracts.md` signed by all three, tag `v0-contracts` · novelty search logged · GPU access confirmed · ethics submitted · camera and fab answer requested · D1, D3, D6 decided (CLAUDE.md §17).

## All three · core reading and GPU check (days 1–2)

**In simple words.** Before code, each of you must understand the three ideas TwinHeat rests on well enough to explain them to an examiner. (1) A neural network guesses board temperature fast but is sometimes badly wrong. (2) A gate decides when to trust it, and random audits keep the gate honest. (3) The solver's matrix structure (M-matrix) lets us prove how wrong any answer can be, so solves can stop early. You also confirm you have GPUs to run on.

**What it actually does.** Produces a shared survey sheet (one row per paper: claim, assumptions, what we reuse, what we don't claim), which becomes the related-work chapter and the viva answer to "what is new?". Also produces `docs/compute.md` with real GPU quotas.

| Person | Must read | Must explain |
|---|---|---|
| All | DeepOHeat-v2; Gibbs & Candès 2021; Tu & Giesecke 2026 §5; Ma & Zhao 2026; Xu et al. 2024; Meurant & Tichý 2018 | Lemma 1 in 2 minutes; why the audit removes selection bias |
| N | + Varah 1975 / M-matrix background; Shewchuk on CG | Lemma 2(a)–(c), why rounding matters |
| M | + Bastani et al. 2022; Angelopoulos & Bates tutorial | ACI update; multivalid groups |
| G | + Cresswell et al. 2024; Correll et al. VSUP | why pre-register; the three trust encodings |

**Prompt 0.A · paper explainer (chat Claude, one chat per paper; CLAUDE.md in the Project's knowledge)**
```
READ FIRST: CLAUDE.md §1 (what TwinHeat is and what we do NOT claim), §8.1–§8.5 (the loop and lemmas).
GOAL: Help me understand one paper well enough to (a) fill one row of our survey sheet and (b) answer viva questions on it.
PAPER: {{RUNTIME: title, authors, year, section}}
TEXT: {{RUNTIME: paste the abstract, the main theorem statements with their assumptions, and the key method paragraph from the PDF}}
DO, IN ORDER:
1. Plain-language summary for a final-year CS student, ≤ 150 words.
2. The main theorem stated precisely, every assumption as a numbered list, quoting the text I pasted.
3. Map each assumption to TwinHeat (CLAUDE.md §8.1–§8.2: coin A_t ~ Bernoulli(p) drawn after commit; q_t updated only on
   audited steps; any proposer incl. humans; model retrained between steps). For each: holds / may fail / why.
4. Survey row: Claim | Method | Data/setting | What TwinHeat reuses | What TwinHeat does NOT claim (consistent with CLAUDE.md §1).
5. Ask me 3 viva questions, one at a time; after each answer, grade it (0–3) and give a model answer.
CONSTRAINTS: Use only the text I pasted. If something isn't there, write "not in the provided text". Never invent theorem
numbers, equations or citations.
REPORT BACK: the survey row as a single markdown table row I can paste.
```

**Prompt 0.B · GPU check script (Claude Code)**
```
READ FIRST: CLAUDE.md §4 (environment).
GOAL: One script that tells us exactly what each GPU environment (Kaggle 2×T4, Colab, laptop) can run.
DELIVERABLES: scripts/check_gpu.py
SPEC: print GPU names and count, CUDA driver/runtime versions, torch and cupy versions and whether each sees the GPU,
free/total memory; run 3 micro-tests, each printing PASS/FAIL + error text: (1) torch fp16 matmul 4096×4096 timing,
(2) cupy SpMV on a random 262,144-row matrix with 7 diagonals timing, (3) a trivial numba.cuda kernel launch.
Write everything also as one JSON object to stdout (last line) for docs/compute.md.
CONSTRAINTS: no installs inside the script; catch each test's exception separately so one failure doesn't hide the others.
DONE WHEN: runs on Kaggle and Colab with all PASS, or prints a clear FAIL reason.
REPORT BACK: the script and the exact commands to run it on Kaggle and Colab.
```

**Check it worked**
- [ ] Survey sheet has a row per core paper per person, each with initials and "checked against PDF"
- [ ] Each person explains Lemma 1 aloud in under 2 minutes, no notes
- [ ] Each person answers "Is the audited gate's guarantee new?" correctly (no: an application of Gibbs & Candès + Tu & Giesecke; the novelty is around it)
- [ ] `check_gpu.py` passes on Kaggle and Colab for all 3 accounts; JSON in `docs/compute.md`; weekly GPU-hours written down

**If it goes wrong**
- Claude summarises a paper wrongly → the barrier is the "checked against PDF" initials. A row without them is invalid.
- CuPy doesn't see the GPU → install the wheel matching `nvcc --version` (`cupy-cuda12x`), restart, rerun, and record the combination in `requirements.txt`.
- Someone can't explain Lemma 1 by day 3 → 30-minute whiteboard session at the Monday sync. Don't move on.

## N · repo, CI, environment, spec files, first M-matrix test

**In simple words.** You build the workshop everyone works in: one shared repository with an automatic checker (CI), one environment file so code runs the same everywhere, and one file holding every physical number so nobody types a constant by hand.

**What it actually does.** `twinheat/` with the CLAUDE.md §3 layout and CLAUDE.md at the root · GitHub Actions (ruff + fast tests per push, slow tests nightly) · pinned `environment.yml` and `requirements.txt` · `config/spec_v1.yaml`, `config/spec_v2.yaml` + `twinheat/spec.py` (validated loader, derived quantities, JSON export for the viewer) · `solvers/mmatrix_test.py` (B1).

**Prompt N0.1 · repo scaffold and CI**
```
READ FIRST: CLAUDE.md §3 (layout), §4 (environment, markers, git).
GOAL: Create the repository skeleton and CI so every later task has a place to live and is tested on every push.
DELIVERABLES: pyproject.toml; .github/workflows/ci.yml; tests/test_smoke.py; .gitignore; Makefile; empty __init__.py in
every Python package folder of §3; docs/decisions.md, docs/compute.md, docs/equipment.md, docs/novelty_search.md,
docs/future_work.md, report/claude_log.md as templates (headings only).
SPEC:
- pyproject: package 'twinheat'; ruff config; pytest config with markers slow, gpu and --strict-markers.
- ci.yml: on push and pull_request: Python 3.11, pip cache, install requirements.txt, `ruff check .`,
  `pytest -m "not slow and not gpu" -q`; fail if fewer than 1 test is collected. Nightly cron: `pytest -m "slow and not gpu"`.
- .gitignore: data/**/shard_*, checkpoints/, *.npy, *.pt, *.parquet, results/**/raw, node_modules/, .ipynb_checkpoints.
- Makefile targets: env, test, test-slow, lint, gpu-test (prints the command to run on Kaggle).
- test_smoke.py imports every package.
CONSTRAINTS: no implementation code beyond the templates; don't add dependencies not in CLAUDE.md §4.
DONE WHEN: `make test` passes locally; pushing to GitHub shows a green CI run; a PR with a deliberately failing test shows red.
REPORT BACK: file tree, then each file in full.
```

**Prompt N0.2 · pinned environment**
```
READ FIRST: CLAUDE.md §4. RUNTIME: {{RUNTIME: JSON output of scripts/check_gpu.py from Kaggle}}
GOAL: One environment that works identically on our laptops and on Kaggle.
DELIVERABLES: environment.yml (conda-forge + pip), requirements.txt (pip only, for Kaggle), scripts/verify_env.py.
SPEC: packages exactly as in CLAUDE.md §4; pick exact versions mutually compatible with the CUDA version in the Kaggle
JSON; explain in a comment every pin chosen for CUDA compatibility. verify_env.py imports each package, prints versions,
checks torch.cuda.is_available() and cupy device count when a GPU is present, and exits non-zero on any failure.
CONSTRAINTS: don't pin versions you cannot justify; say UNSURE where you are guessing compatibility.
DONE WHEN: verify_env.py passes on Kaggle (pip route) and on one laptop (conda route).
REPORT BACK: both files, verify_env.py, and the install commands for each route.
```

**Prompt N0.3 · spec YAMLs and loader**
```
READ FIRST: CLAUDE.md §6 (spec v1), §7 (spec v2), §17 (open decisions D3).
GOAL: Every physical constant in one validated place that the solver, sampler, viewer and KiCad generator all import.
DELIVERABLES: config/spec_v1.yaml, config/spec_v2.yaml, twinheat/spec.py, tests/test_spec.py.
SPEC:
- Each quantity: {value, unit, range: [min, max], source: "CLAUDE.md §x.y"}; SI units (geometry of the stack in m).
- spec_v1: everything in §6.3–§6.7 incl. sampler distributions and OOD ranges. spec_v2: §7.1 stack, §7.2 constants
  (D = 0.3e-3 m, t_p = 25e-6 m), §7.3 grid, §7.4 sampler additions and OOD names.
- Via convention comment: "we use 190 K/W per via for a 1.6 mm board (our formula), not 407 K/W".
- spec.py: pydantic v2 models; load(path) validates every range and raises naming the field; derived properties with
  unit docstrings: A_v = π(D·t_p − t_p²); sheet conductance G for the v1 cases; G_l and g_lm (per unit via density) for v2;
  export_json(spec, path) for the viewer.
- D3 (outer-layer dielectric term) is open: implement it as written in §7.2 (missing outer term = 0) behind a named
  config flag `outer_dielectric_term: "zero"` and add a TODO referencing D3.
TESTS FIRST: A_v == 2.16e-8 m² (rel 1 %); v1 sheet conductances 1.30e-3, 1.50e-2, 8.32e-2 W/K (rel 2 %); a YAML with
k_Cu = −1 raises; a YAML missing a field raises; export_json round-trips.
CONSTRAINTS: list every constant you could not find in CLAUDE.md instead of inventing it.
DONE WHEN: tests pass; a second person checked 10 random YAML values against CLAUDE.md.
REPORT BACK: files, test output, missing information.
```

**Prompt N0.4 · M-matrix checker (B1)**
```
READ FIRST: CLAUDE.md §5 (B1), §8.4.
GOAL: A checker that proves an assembled matrix satisfies the hypotheses Lemma 2 needs, used on every assembly forever.
DELIVERABLES: solvers/mmatrix_test.py with check_m_matrix(A: scipy.sparse.csr_matrix) -> dict; tests/test_mmatrix.py.
SPEC: check (1) all off-diagonals ≤ 0; (2) all diagonals > 0; (3) every row sum ≥ −1e-12·|diag| and at least one row
sum > 1e-12·|diag| (strict dominance somewhere); (4) the off-diagonal adjacency graph is connected
(scipy.sparse.csgraph.connected_components). Return {"ok": bool, "n_pos_offdiag", "n_nonpos_diag", "n_neg_rowsum",
"n_strict_rows", "n_components", "worst_rows": [...]}. Also assert_m_matrix(A) raising with that dict.
TESTS FIRST: 1D Laplacian + positive diagonal shift (ok); same with one positive off-diagonal (fail); block-diagonal of
two valid blocks (fail: disconnected); pure Neumann Laplacian with H = 0 (fail: no strict row). A @pytest.mark.slow
test placeholder that will run on 50 sampled boards once fvm.py exists (skip if missing).
DONE WHEN: all tests pass in CI.
REPORT BACK: file, tests, and one sentence per check saying which part of Lemma 2 needs it.
```

**Check it worked**
- [ ] Fresh clone + `make env && make test` passes on all three laptops; CI green; a deliberately broken PR shows red
- [ ] `verify_env.py` passes on Kaggle
- [ ] Spec tests pass; 10 values spot-checked by a second person
- [ ] M-matrix checker passes and fails the four hand-built cases correctly

**If it goes wrong**
- CI green but no tests ran → `--strict-markers` + the minimum-collected check catch it.
- Torch/CuPy CUDA conflict → pin to Kaggle's CUDA major and use the pip route there. Never "upgrade everything" mid-project; version changes go through a PR with `verify_env.py` output attached.
- Claude invents a constant → every YAML value needs a `source` pointing at CLAUDE.md, and the reviewer rejects values without one.

## M · frozen contracts, sampler skeleton, novelty search

**In simple words.** You write the agreement that lets three people build separately and still fit together: what a board looks like as data, what goes into and out of the model, what the server and viewer say to each other. You start the random board generator, and you check nobody has published our main idea since October.

**What it actually does.** `docs/contracts.md` + `contracts/schemas/*.json` + validators in Python and JS (B6), tagged `v0-contracts` · `data/sampler.py` v0 (seed → board record, incl. the 36-dim design vector) · `docs/novelty_search.md` · U items in the Phase 0 file confirmed.

**Prompt M0.1 · contracts, schemas, validators (with G)**
```
READ FIRST: CLAUDE.md §10 (all contracts), §7.5 (model I/O), §5 (B6, B8).
GOAL: Freeze machine-checkable contracts so N, M and G can work independently and integrate without surprises.
DELIVERABLES: contracts/schemas/{board_record,query,design,pred,iterate,verified,batch_progress,error,audit_log_row,
hardware_record,session_log_line}.json (JSON Schema draft 2020-12); contracts/validate.py (validate(obj, kind) raising
with the JSON path of the first error); viewer/src/contracts.js (ajv, same schemas); contracts/f16.py and
viewer/src/f16.js (encode/decode base64 little-endian float16 arrays with shape [layer, y, x]);
contracts/examples/<kind>/{valid_1..3,invalid_1..3}.json; tests/test_contracts.py and viewer/test/contracts.test.js;
docs/contracts.md (human-readable, with a signature table for N, M, G).
SPEC: exactly the fields, enums and units of CLAUDE.md §10; schema_version "1.0" required everywhere; pred.decision enum
preview|accept|flag|audit; additionalProperties false on every object.
TESTS FIRST: every valid example passes and every invalid example fails, in both languages; a float16 array encoded in
Python decodes in JS to identical values (and the reverse); shape mismatches are rejected.
CONSTRAINTS: no field that is not in CLAUDE.md §10. If you think one is missing, list it under "Proposed additions";
do not add it.
DONE WHEN: both test suites pass in CI; N, M, G sign docs/contracts.md; tag v0-contracts.
REPORT BACK: file tree, a summary of every ambiguity you resolved, proposed additions.
```

**Prompt M0.2 · board sampler v0**
```
READ FIRST: CLAUDE.md §6.5–§6.7 (sampler v1, BCs, OOD), §7.4 (sampler v2), §10.1 (board record), §17 D2.
GOAL: Deterministic random boards (and design-space boards) as JSON records; rasterisation comes in Phase 1.
DELIVERABLES: data/sampler.py with sample_board(seed: int, mode: str) -> dict, design_board(x: np.ndarray[36]) -> dict,
and transform_record(record, sym: int) for the 8 square symmetries; tests/test_sampler.py.
SPEC: numpy Generator(PCG64(seed)); distributions exactly as CLAUDE.md §6.5 + §7.4 (v2 layers L2 cut-outs 0–3, L3
2–3 split rectangles, L4 coverage 5–40 %, via arrays under chips p = 0.6 density U[0.2, 1]/mm², background 0–0.05/mm²,
outer 1 or 2 oz, inner 1 oz); chip placement by rejection (≥ 2 mm gap, ≥ 3 mm from edges, max 100 tries then resample
the count); total power ≤ 6 W; Dirichlet edges per §6.6. design_board maps x: 8 numbers → 4 chip centres, 24 → 6 L1
copper rectangles (x, y, w, h), 4 → via densities; all other parameters from config/design_task.yaml (create it with
clearly marked placeholder values and a TODO "D2: to be decided by M"). Output validates against the board_record schema.
transform_record rotates/flips all geometry AND maps dirichlet_edges (e.g. 90° CCW: W→S, S→E, E→N, N→W).
TESTS FIRST: same seed → byte-identical JSON; 1,000 records validate; chip constraints hold for all; chip count
histogram uniform on 1–4 (chi-square p > 0.01); design_board is deterministic in x; applying a symmetry and its
inverse returns the original record.
CONSTRAINTS: no rasterisation, no solver calls. OOD modes are added in Phase 1 (raise NotImplementedError for them now).
DONE WHEN: tests pass in CI.
REPORT BACK: file, tests, and the list of D2 placeholders.
```

**Prompt M0.3 · novelty search assistant (chat)**
```
READ FIRST: CLAUDE.md §1 (contributions, non-claims), §8.4 (Lemma 2), §16.
GOAL: Check whether "pointwise, rigorous error certificates for a neural PDE surrogate's prediction, used to decide when
a design is settled and stop the exact solver early" has been published since our 4–5 Oct 2026 search.
DO:
1. Give 15 search strings for Google Scholar, arXiv and Semantic Scholar combining: pointwise a posteriori bound, maximum
   principle, M-matrix, supersolution, neural operator, surrogate certification, certified early stopping, PCG,
   decision-sufficient.
2. I will paste abstracts: {{RUNTIME: abstracts with titles and links}}. Classify each: (a) same idea, (b) related but
   different (say exactly how), (c) unrelated. For (a) or (b), say which contribution in CLAUDE.md §1 it affects.
CONSTRAINTS: never claim a paper exists unless I pasted it; label anything from memory UNVERIFIED.
REPORT BACK: a table I can paste into docs/novelty_search.md (date, query, database, hit, verdict).
```

**Check it worked**
- [ ] Python and JS validators accept all valid and reject all invalid examples (CI runs both)
- [ ] Float16 round-trip Python ↔ JS exact
- [ ] Contracts signed by N, M, G; tag `v0-contracts`
- [ ] 1,000 sampled records validate; same seed gives identical JSON
- [ ] `novelty_search.md` complete; no type-(a) hit, or the G1 narrowing is logged
- [ ] U items updated to V or B

**If it goes wrong**
- Contract argument past day 4 → M decides, the objection is logged; v1.1 later needs a migration note and both validators updated in one PR.
- Wrong data reaches another lane later → B6 fails in CI; the server validates outgoing messages in dev mode and crashes loudly.
- A close paper turns up → read it fully, write a "they do / we do" note, narrow G1 (CLAUDE.md §15), log it. Never hide it.

## G · viewer skeleton, ethics and pre-registration, equipment

**In simple words.** You start the three things with the longest waits: the 3D viewer (so it can grow every week), the ethics paperwork for the user study (approval takes weeks), and the real-world kit (camera, boards), since ordering is slow.

**What it actually does.** `viewer/` (Vite + Three.js + WebXR, 4 stacked planes, WebSocket client validating messages, fps overlay) + `server/ws_mock_server.py` · `study/protocol.md`, consent form, data plan, submitted; `study/prereg.md` draft · `docs/equipment.md`.

**Prompt G0.1 · viewer skeleton + mock server**
```
READ FIRST: CLAUDE.md §10 (messages, float16 encoding), §13 (viewer spec, targets), §4 (Node 20, pinned three).
GOAL: A running viewer that renders 4 layer fields from contract-valid messages, so G never waits for N or M.
DELIVERABLES: viewer/ (Vite, vanilla JS, no React): index.html, src/main.js, src/scene.js, src/net.js, src/devOverlay.js,
src/lut.js (inferno LUT texture), shaders/field.vert, shaders/field.frag; uses existing src/contracts.js and src/f16.js.
server/ws_mock_server.py (python websockets): replays a .jsonl of messages at a set rate (--rate Hz, --loop).
SPEC: board shown 1 m wide; 4 planes stacked with an adjustable gap; each plane has an R16F DataTexture of its layer;
fragment shader maps (θ − θmin)/(θmax − θmin) through the 1D LUT; OrbitControls on desktop; VRButton,
renderer.xr.enabled, 'local-floor' or seated reference space. net.js: auto-reconnect; validate each message; on
pred/verified decode [4, n, n] float16 and update textures (needsUpdate only on change); invalid messages are logged and
never rendered. Dev overlay: fps, frame ms, last message type, last validation error.
TESTS FIRST: a node test that feeds 3 valid and 3 invalid messages to the handler and checks textures update only for valid
ones; a Python test that the mock server sends N messages in order.
CONSTRAINTS: no rendering libraries other than three; no new message types.
DONE WHEN: `npm run dev` + mock server shows 4 changing layers at ≥ 60 fps; Enter VR works on a headset or the WebXR emulator.
REPORT BACK: file tree, files, run commands.
```

**Prompt G0.2 · ethics application (chat)**
```
READ FIRST: CLAUDE.md §13 (pilot design), §10.6 (session log).
GOAL: A complete minimal-risk ethics application for RVCE's department committee, submitted this week.
FACTS: purpose = test whether a calibrated trust signal changes design decisions in a PCB thermal design tool (desktop + VR);
participants 8–10 adult students with PCB or heat-transfer coursework, voluntary, {{RUNTIME: payment or none}};
within-subjects, 3 counterbalanced conditions; Task A (30 accept-or-verify decisions), Task B (5 min, fixed solve budget);
~45 min; risks: VR discomfort (seated, teleport only, breaks, stop any time, desktop alternative); data: participant codes
only, interaction logs + NASA-TLX, stored at {{RUNTIME: location}}, deleted by {{RUNTIME: date}}; investigators
{{RUNTIME: names, supervisor}}.
PRODUCE: (1) 2-page protocol; (2) 1-page participant information sheet; (3) consent form with checkboxes; (4) data
management paragraph; (5) likely committee questions with our answers.
CONSTRAINTS: plain formal English; don't overstate risks or benefits; no claims about results.
```

**Prompt G0.3 · equipment request (chat)**
```
READ FIRST: CLAUDE.md §14 (hardware validation).
GOAL: A one-page request to {{RUNTIME: lab/department head}} for E17 equipment.
ITEMS: thermal camera (≥ 160×120, accuracy ±2 °C or better, adjustable emissivity) for weeks 5–9; bench DC supply with V
and I readout and current limit; 6–8 NTC thermistors + logger; matte black paint of known emissivity; fabrication of 2–3
four-layer 100×100 mm 1.6 mm boards with SMD power resistors: {{RUNTIME: fab, quote, lead time}}.
INCLUDE: purpose, dates, safety measures (current-limited supply, supervised operation, non-flammable standoffs), total
cost, fallback if no camera (thermistor array). Formal, concise, one page.
```

**Check it worked**
- [ ] Viewer renders 4 changing layers from the mock server; invalid messages logged and not rendered
- [ ] Enter VR works (headset or emulator)
- [ ] Ethics submitted; date + reference number in `decisions.md`
- [ ] Pre-registration draft exists (hypotheses, measures, analysis)
- [ ] Equipment request sent; reply or chase date in `docs/equipment.md`

**If it goes wrong**
- Viewer lags → `needsUpdate` set every frame. The fps overlay is always on in dev, and < 60 fps on desktop is a bug.
- Committee asks for changes → reply within 48 h. **The study never runs before approval.** If there's no approval by week 8, H4 becomes future work and P4 is dropped (logged).
- No camera → order the thermistor array now; ask Mechanical/EEE labs.
- Fab over budget → 2 boards (fit + held-out).

## Decide this week (all)
D1 (H6 judging rule) · D3 (outer dielectric term) · D6 (week-1 start date), recorded in `docs/decisions.md` (CLAUDE.md §17).

---

# Phase 1 · Solvers, M-matrix check and cost go/no-go (weeks 1–3)

**Goal.** An exact 4-layer solver that is proven correct, fast GPU kernels, and a measured solve cost that justifies the project.
**Gate 1 (end of week 3):** B1–B4 green on 1- and 4-layer boards · CLAUDE.md §8.6 tables reproduced on 64² (±10 %) · solver table v1 · median PyAMG solve ≥ 0.3 s at 256²×4 (else D5: 6 layers or 384² before any data) · boards ordered · tag `v1-solver`.

## N · solvers, M-matrix, certificate reproduction, CUDA kernels, cost go/no-go

**In simple words.** You build the ground-truth machine that computes a board's exact temperature. Every training label, every check of the network and every certificate comes from it, so it must be provably right. Build it small first (64×64 plate) to find bugs, then the real 4-layer board, then make it fast on the GPU and measure whether a solve is slow enough (≥ 0.3 s) that a surrogate is worth having.

**What it actually does.** The finite-volume method gives one equation per cell (heat from neighbours + heat lost to air = chip power), a sparse system Aθ = b with 262,144 unknowns. You solve it four ways that must agree (SciPy direct, PyAMG-GMRES, Jacobi/red-black, matrix-free GPU PCG), prove A is an M-matrix, and reproduce the preliminary certificate numbers.

**Prompt N1.1 · 64² plate solver**
```
READ FIRST: CLAUDE.md §6.2–§6.4 (PDE, constants, discretisation), §5 (B1, B2, B4, B5), §0 rules 6–7 (units, array order);
files: twinheat/spec.py, solvers/mmatrix_test.py, data/sampler.py.
GOAL: A provably correct cell-centred FV solver for the single plate; the reference for every later solver.
DELIVERABLES: solvers/fvm.py with assemble_plate(G, H, q, dx, dirichlet_edges) -> (A: csr, b), solve_direct(A, b),
energy_balance(theta, G, H, q, dx, dirichlet_edges) -> relative error; tests/test_fvm_plate.py.
SPEC: inputs float64 [N, N] in W/K, W/(m²·K), W/m²; dirichlet_edges ⊆ {"W","E","S","N"}; harmonic-mean faces; Dirichlet
face adds 2·G_P to the diagonal; Neumann adds nothing; cell equation in watts exactly as CLAUDE.md §6.4; index
k = y·N + x; fully vectorised (build COO from shifted index arrays, convert to CSR); energy balance compares Σ q·Δx² with
Σ H·Δx²·θ + Σ_Dirichlet faces 2·G_P·θ_P.
TESTS FIRST:
 1. Manufactured solution θ* = sin(πx/L)·sin(πy/L)-type with uniform G, H and all-Dirichlet edges (θ* = 0 there):
    derive q analytically in the test; N ∈ {32, 64, 128, 256}; fitted log-log slope of max error in [1.9, 2.1].
 2. Energy balance ≤ 1e-3 on 20 sampled boards (rasterise with a simple test rasteriser in the test file if
    data/raster.py doesn't exist yet).
 3. assert_m_matrix(A) on those boards.
 4. A symmetric: ‖A − Aᵀ‖ = 0.
 5. Linearity: q×3 → θ×3 (rel 1e-10).
 6. One centred chip, all Neumann → θ symmetric under x-flip (rel 1e-10).
CONSTRAINTS: no Python loops over cells; no changes outside solvers/ and tests/.
DONE WHEN: all 6 pass; bench/figs/plate_check.png (θ for one centred chip) saved and looks symmetric and smooth.
REPORT BACK: files, test output (incl. the fitted slope), 3 ways it could still be wrong.
```

**Prompt N1.2 · 4-layer assembly**
```
READ FIRST: CLAUDE.md §7.1–§7.3 (stack, equations, Δx² rule), §17 D3; files: solvers/fvm.py, twinheat/spec.py,
data/raster.py (if not ready, use a minimal Raster dataclass in the test with phi[4,N,N], q, nu[3,N,N], h_top, h_bot).
GOAL: Assemble the real 4-layer system with through-plane coupling and vias, as an M-matrix.
DELIVERABLES: assemble_board(spec, raster, dirichlet_edges) -> (A, b) in solvers/fvm.py; coefficients(spec, raster, edges)
-> dict of arrays [L,N,N] (Ge, Gw, Gn, Gs, gup, gdn, diag, rhs) for the GPU kernels; tests/test_fvm_board.py.
SPEC: index k = l·N² + y·N + x; G_l per §7.2 using t_Cu per layer from the record's copper weights and the D3 flag; in-plane
harmonic faces per layer; through-plane coupling g_lm·Δx² [W/K] between (l,y,x) and (m,y,x) with ν in vias/m²; h_top·Δx²
on L1 and h_bot·Δx² on L4 diagonals; q·Δx² on L1 only; Dirichlet edges on all layers.
TESTS FIRST:
 1. Analytic 1-D through-thickness check: uniform q on L1, uniform φ and ν, all Neumann edges → closed-form series
    resistances (derive in the docstring) vs the solve, rel 1e-8. (This is the test that catches a missing Δx² on g_lm.)
 2. With g_lm → 0, L1 equals assemble_plate with H = h_top (rel 1e-10).
 3. @slow: assert_m_matrix on 50 sampled 4-layer boards.
 4. Energy balance ≤ 1e-3 on 20 boards.
 5. coefficients() reproduces A @ x for random x (rel 1e-12).
 6. κ₂(A) by scipy.sparse.linalg.eigsh (Lanczos) on 3 boards, appended to bench/results/conditioning.csv.
CONSTRAINTS: vectorised; do not change assemble_plate's behaviour.
DONE WHEN: all pass; κ₂ logged (expect > 1e4).
REPORT BACK: files, test output, κ₂ values, 3 ways it could still be wrong.
```

**Prompt N1.3 · reproduce the preliminary certificate**
```
READ FIRST: CLAUDE.md §8.4 (Lemma 2 + stopping), §8.6 (target tables), §9.1 (reference code); files:
bench/prelim/stop.js, bench/prelim/cert.js, solvers/fvm.py.
GOAL: Show on the 64² plate that our Python certificate reproduces the 5 Oct preliminary evidence before any GPU work.
DELIVERABLES: certify/kappa.py (kappa_hat(A, n_vcycles=3) using pyamg V-cycles applied to ones; min(z) > 0 and
min(Az) > 0 checked in float64; returns max(z)/min(Az)·(1+1e-6) or None); certify/pointwise.py (pointwise w with
A w ≥ |r| via 4 V-cycles + two-SpMV fix-up, validity asserted); certify/stopping.py (CPU pcg_certified as §9.1);
bench/prelim/reproduce.py writing bench/results/prelim_stop.csv and prelim_pointwise.csv; tests/test_certificate_cpu.py.
SPEC: port the emulated-error models and seeds from stop.js/cert.js exactly (smooth blobs 1–5 % of peak, ringing at copper
edges, small noise); compute the true ‖A⁻¹‖∞ by solving A X = I in column blocks (64² only); same 4 board variants
and same error levels as §8.6.
TESTS FIRST: every reported bound ≥ the true error vs a 1e-12 direct solve, for every iterate (raise with board, iterate,
bound, error on violation); kappa_hat returns None when z has a non-positive entry (inject one).
DONE WHEN: κ̂ ÷ true within 1.0–1.07; iterations to certified 1 K within ±10 % of 29–43; zero violations.
REPORT BACK: both CSVs side by side with §8.6, differences explained.
```

**Prompt N1.4 · CUDA stencil, PCG, batched PCG**
```
READ FIRST: CLAUDE.md §9.1 (PCG), §9.4 (tiled Jacobi + its boundary bug), §5 (B3), §4 (gpu marker); files:
solvers/fvm.py (coefficients()).
GOAL: Fast, verified GPU kernels for the 4-layer operator, single and batched.
DELIVERABLES: solvers/cuda_kernels.py (Numba): stencil_apply_naive, stencil_apply_tiled (16×16 shared tiles + halos,
ALL cells incl. boundaries), red_black_gs_sweep, dot_warp_shuffle; solvers/pcg.py (CuPy): pcg(Aop, b, x0, Minv, tol,
maxit) and batched_pcg(Aop_b, b[B,...], x0, Minv, tol, maxit) with per-board convergence masks; preconditioners
jacobi(diag) and line_z_block_jacobi (4×4 tridiagonal solve per (y,x) column); tests/test_gpu_solvers.py (@gpu).
SPEC: coefficient arrays [L,N,N] float32 for iterations; an FP64 operator variant for residual checks; Neumann faces have
coefficient 0; Dirichlet terms live in diag.
TESTS FIRST (@gpu): stencil_apply_{naive,tiled} == scipy A @ x (rel 1e-6) on 20 boards incl. Dirichlet edges (B3);
pcg solution vs spsolve rel 1e-6; batched_pcg on 16 boards == 16 single solves; warm start from the exact solution
converges in ≤ 1 iteration; dot_warp_shuffle == cupy.vdot (rel 1e-6).
DONE WHEN: all GPU tests pass on a Kaggle T4 (paste log into the PR); achieved GB/s for naive vs tiled stencil recorded
in bench/results/stencil_bw.csv vs ~320 GB/s peak.
REPORT BACK: files, Kaggle test log, bandwidth table, 3 ways it could still be wrong.
```

**Prompt N1.5 · CPU baseline, benchmark harness, go/no-go**
```
READ FIRST: CLAUDE.md §7.3 (go/no-go), §11.5 (E3), §17 D5; files: solvers/*.
GOAL: Measure every solver fairly and decide whether 256²×4 is expensive enough (≥ 0.3 s).
DELIVERABLES: solvers/amg_baseline.py (pyamg.ruge_stuben_solver + GMRES acceleration; returns theta, iters, seconds);
bench/solver_bench.py; bench/results/solver_table_v1.csv; bench/figs/solver_table_v1.png.
SPEC: 20 fixed seeded boards at 256²×4; solvers: scipy spsolve, PyAMG-GMRES (tol 1e-8), Jacobi, red-black GS, PCG cold,
PCG warm (x0 = exact + 2 % smooth noise); GPU timing with CUDA events, 10 warm-ups, median of 20; host↔device copies as a
separate column; CPU model, core count and GPU model in every CSV row; figure footer per CLAUDE.md §4.
TESTS FIRST: all solvers agree to 1e-6 rel on the 20 boards.
DONE WHEN: table + figure exist; median PyAMG time measured ON THE MACHINE THAT WILL RUN ORACLE LABELS.
REPORT BACK: the table, and a one-paragraph go/no-go recommendation for docs/decisions.md (D5).
```

**Check it worked**
- [ ] N1.1: 6 tests pass, slope printed in [1.9, 2.1]
- [ ] N1.2: analytic slab test passes; M-matrix on 50 boards; κ₂ logged
- [ ] N1.3: tables match §8.6 within tolerance; zero violations
- [ ] N1.4: B3 passes on a T4; batched == single; bandwidth logged
- [ ] N1.5: table v1; D5 recorded; if < 0.3 s, `spec_v2.yaml` bumped and B1–B4 re-run; tag `v1-solver`

**If it goes wrong**
- MMS slope ≈ 1 → Dirichlet term wrong (2G_P, not G_P). Shrink to N = 4, print A dense, and check one boundary row by hand.
- Energy balance off by a constant → a missing Δx² (H, q, or g_lm). The slab test isolates the vertical coupling.
- M-matrix fails → a negative conductance (φ slightly > 1, G ≤ 0). Clip φ in the rasteriser, assert G > 0. **Never relax B1.**
- CUDA differs at edges → boundary kernel missing or Dirichlet applied twice. Plot |difference| as an image.
- Certificate violation → stop all certificate work. Usually float32 min(Az) or a non-positive z. Fix before anything else.
- Late at week 3 → M and G continue on 64² stubs; drop red-black and warp-shuffle (keep naive, tiled, PCG). The gate slips at most a week, logged.

## M · rasterisation, OOD sets, FNO validated on Darcy

**In simple words.** Records are lists of shapes; the solver and network need images. You write the code that paints records into grids identically at every resolution, the generators for the 10 deliberately unusual test sets, and you prove your training code reproduces a published result before trusting it on TwinHeat.

**What it actually does.** `data/raster.py` (record → φ_l, q, ν, h maps, mask at N ∈ {64, 128, 256}) · OOD modes in `sampler.py` · `data/envelope.py` · `models/fno.py` + `models/train.py` validated on PDEBench Darcy.

**Prompt M1.1 · rasteriser**
```
READ FIRST: CLAUDE.md §6.4 (supersampling rule), §6.6 (mask), §7.1–§7.4, §10.1 (record units: mm), §0 rules 6–7;
files: data/sampler.py, twinheat/spec.py.
GOAL: Convert any board record into the exact per-cell fields the solver and network consume, consistently at any N.
DELIVERABLES: data/raster.py with rasterise(record, spec, N) -> Raster (dataclass: phi[4,N,N], q[N,N] W/m²,
nu[3,N,N] vias/m², h_top[N,N], h_bot[N,N], dirichlet_mask[N,N], N, dx); tests/test_raster.py.
SPEC: mm → m conversion happens here only; 4×4 supersampling at sub-cell centres (i+0.5)/4; rect, disc, Manhattan trace
(segments with width, square caps) as analytic inside-tests; φ = area fraction, clipped to [0,1] and the board; L2 = 1
minus cut-outs; L3 = union of split rectangles; chip q = P/A spread by area fraction over its footprint (side "top" → L1;
"bottom" → raise NotImplementedError until OOD work); via density per gap: under-chip arrays on all three gaps, background
density everywhere; forced-air zone raises h_top inside it; mask = 1 on cells adjacent to a Dirichlet edge.
TESTS FIRST:
 1. Copper area from φ at N = 64, 128, 256 agrees within 1 % of the analytic area of the shapes.
 2. Σ q·Δx² == total chip power (rel 1e-12) at every N.
 3. For all 8 symmetries: rasterise(transform_record(r, s)) == transform_array(rasterise(r), s) EXACTLY, incl. mask.
 4. φ ∈ [0,1], no NaN, dtype float64.
 5. A 3 mm-radius disc at N = 64 has non-zero coverage.
CONSTRAINTS: vectorised over cells (loops over shapes are fine).
DONE WHEN: tests pass; data/figs/raster_check.png (12 random boards: φ per layer, q, ν) reviewed by a second person.
REPORT BACK: files, test output, 3 ways it could still be wrong.
```

**Prompt M1.2 · OOD generators and envelope rule**
```
READ FIRST: CLAUDE.md §6.7 (OOD table), §7.4 (v2 OOD additions), §6.5 (envelope rule); files: data/sampler.py.
GOAL: Ten test sets that each push exactly one property outside training, plus the power-rescale rule.
DELIVERABLES: sample_board(seed, mode) for mode ∈ {ood_1_chip_count, ood_2_power, ood_3_geometry, ood_4_thin,
ood_5_contrast, ood_6_boundary, ood_7_convection, ood_bottom_chips, ood_3oz, ood_dense_vias}; data/envelope.py with
apply_envelope(record, theta, max_rise=100.0) -> (record', theta', scaled: bool); rasteriser support for L-shapes,
annuli, 45° traces and bottom-side chips (q on L4); tests/test_ood.py, tests/test_envelope.py.
SPEC: each mode changes ONE axis to the OOD range in §6.7/§7.4 and samples everything else from ID; docstring per mode
naming the axis and range.
TESTS FIRST: per mode, 200 records: the moved axis is outside the ID range and a sample of other axes is inside it;
envelope: max θ' ≤ 100 + 1e-9 and θ'/θ == P'/P for every board.
DONE WHEN: tests pass.
REPORT BACK: files, a table mode → axis → range, test output.
```

**Prompt M1.3 · FNO + trainer validated on PDEBench Darcy**
```
READ FIRST: CLAUDE.md §12 (surrogate config), §5 (data hygiene), §11.6 (Darcy); files: docs/refs/pdebench_darcy_fno.txt.
GOAL: Prove our FNO and training loop are correct by reproducing a published number before using them on TwinHeat.
DELIVERABLES: models/fno.py (FNO2d(in_ch, out_ch, width=48, modes=24, n_layers=4): SpectralConv2d with complex weights
using rfft2/irfft2 and BOTH positive and negative y-frequencies kept for the low modes; 1×1 pointwise path; GELU;
lifting/projection MLPs); models/train.py (AdamW, cosine with 5 % warm-up, AMP with FP32 loss, grad clip 1.0,
relative-L2 loss, checkpoint best+last every epoch to --ckpt_dir AND copy to --backup_dir, resume from last, CSV log);
models/data_guard.py (a loader wrapper raising if a split named "cal" or "test" is requested without allow_eval=True);
experiments/darcy_validate.py; tests/test_fno.py.
SPEC: PDEBench 2D DarcyFlow β = 1.0 at 128², 9,000 train / 1,000 test, path {{RUNTIME: hdf5 path}}; compare test relative
L2 with the value in docs/refs/pdebench_darcy_fno.txt.
TESTS FIRST: SpectralConv2d passes a low-frequency sine through with identity-like weights; forward works at 64² and 128²;
resume after a simulated kill continues the loss curve; data_guard raises on "cal"/"test".
DONE WHEN: Darcy test relative L2 within ~20 % of the published FNO number with a healthy learning curve.
REPORT BACK: files, final metric vs published, the curve PNG path.
```

**Check it worked**
- [ ] Raster tests pass incl. symmetry with Dirichlet edges; visual check reviewed
- [ ] Every OOD mode moves exactly one axis; 200 records per mode generate cleanly
- [ ] Darcy FNO within ~20 % of the published value; resume works

**If it goes wrong**
- Symmetry test fails only with edges → wrong edge mapping in `transform_record`. Fix it; never skip the test, because a wrong augmentation teaches wrong physics.
- Area differs across N by > 1 % → sample points not at (i+0.5)/4.
- Darcy far off → spectral mode slicing (missing negative y-frequencies), normalisation, learning rate, in that order.
- Kaggle session dies → that's why resume and backup copies exist.

## G · real boards ordered, viewer on solver fields

**In simple words.** You design the 2–3 real boards that will be heated and filmed, and order them by the end of week 3. The viewer starts showing N's real solver fields with proper thermal colour maps.

**What it actually does.** `hardware/kicad/gen_board.py` (record → KiCad 4-layer PCB), `hardware/boards/board{1,2,3}.json`, fab order with the stack-up checked; viewer LUTs (inferno, ironbow, white-hot), isolines, hover probe.

**Prompt G1.1 · KiCad generator**
```
READ FIRST: CLAUDE.md §14 (boards, fiducials, resistors), §7.1 (stack), §10.1 (record); files: data/sampler.py,
config/spec_v2.yaml.
GOAL: Generate manufacturable boards from the SAME records the solver uses, so simulation and hardware match by construction.
DELIVERABLES: hardware/kicad/gen_board.py (KiCad 8 pcbnew Python API, run inside KiCad's Python); hardware/boards/board{1,2,3}.json;
outputs per board: board.kicad_pcb, fiducials.csv, heaters.csv (ref, x_mm, y_mm, nominal_W, max_W).
SPEC: 4 copper layers F.Cu, In1.Cu, In2.Cu, B.Cu; 100×100 mm outline; stack-up from spec_v2; every record shape as a
filled zone on its layer (disc as a 64-gon; traces as polygons of the given width); L2 plane minus cut-outs; L3 split
rectangles; each chip → 2512 resistor footprints, enough in parallel that each runs ≤ 50 % of rated power, with pads on F.Cu
and two supply test pads; thermal via arrays 0.3 mm drill / 0.6 mm pad on a grid giving the requested density; fiducials
1 mm copper dot with 2 mm mask opening at (5,5), (95,5), (95,95), (5,95) mm. The 3 board records differ in pours and via
arrays; board 1 is the fit board.
CONSTRAINTS: never hand-edit the PCB; anything unmanufacturable is reported, not silently changed.
DONE WHEN: all 3 boards pass KiCad DRC; a script confirms fiducial and heater CSVs match the records.
REPORT BACK: script, the 3 records, DRC summary, list of unmanufacturable features and how they were handled.
```

**Prompt G1.2 · fab stack-up check (chat)**
```
READ FIRST: CLAUDE.md §7.1. FAB TEXT: {{RUNTIME: paste the fab's standard 4-layer 1.6 mm stack-up page}}
GOAL: Make sure the solver models the board as it will actually be manufactured.
DO: compare layer by layer (copper weights, prepreg and core thickness, material); list every mismatch; give the exact
spec_v2.yaml values to change; list questions for the fab (copper tolerance, via plating thickness, solder-mask colour
and emissivity, lead time).
```

**Prompt G1.3 · thermal shading, isolines, probe**
```
READ FIRST: CLAUDE.md §13 (T1.3), §0 rule 7 (array order, y up); files: viewer/src/*.
GOAL: Show real solver fields correctly and readably.
DELIVERABLES: viewer/src/lut.js (256-entry LUT textures: inferno, ironbow, white-hot; runtime switch);
shaders/thermal.frag (LUT mapping, isolines every ΔT K via fract(T/ΔT) anti-aliased with fwidth, colour bar overlay);
viewer/src/probe.js (raycast, read θ from the CPU-side Float32 copy at the hit cell, tooltip in °C = 25 + θ).
TESTS FIRST: an asymmetric test field (hot corner at x = N−1, y = 0) appears bottom-right in the viewer (catches flipY and
[y,x] mix-ups); the probe at 5 known cells equals the field values.
DONE WHEN: N's 4-layer solutions for 3 sample boards display correctly; isolines smooth at 1 m in VR.
REPORT BACK: files and a 3-sentence paragraph for the report on why the default map is perceptual, not rainbow.
```

**Check it worked**
- [ ] DRC clean for all boards; CSVs match records
- [ ] `spec_v2.yaml` = as-fabricated stack; decision logged
- [ ] Order placed by end of week 3 (number, cost, ETA in `docs/equipment.md`)
- [ ] Viewer orientation and probe tests pass

**If it goes wrong**
- DRC errors → set zone clearances from the fab's rules and regenerate. Any manual change goes back into the record.
- Overheating risk → ≤ 50 % rated power per resistor; current-limited supply; max power in `heaters.csv`.
- Fab delay → order from a second fab now (2 boards minimum). Nothing by week 8 → E17 goes to the journal extension.
- Probe mismatch → `[y,x]` vs `[x,y]` or `flipY`. The asymmetric test catches it.

---

# Phase 2 · Data and surrogate (weeks 3–5)

**Goal.** 14,000 trustworthy labelled boards and a 5-member FNO ensemble meeting the week-5 targets (ID relative L2 < 3 %, hotspot < 1.5 K).
**Gate 2 (end of week 5):** all splits generated, verified, hashed, backed up (tag `v2-data`) · rescale fraction ≤ 20 % · 5 FNO members + U-Net trained, checkpoints on Drive · targets met or the 128² fallback logged · Darcy FNO reproduced.

## N · batched data generation, verification, upload

**In simple words.** The network learns from examples, so you mass-produce them: 14,000 random boards, each solved exactly. A bad example teaches the network something wrong, so you start small, check everything, protect the data from edits, and make interrupted jobs resume.

**What it actually does.** `data/generate.py` (records → raster → batched GPU PCG to 1e-10 → envelope → shards of 500, float16 fields) · `data/manifest.json` (seeds, commit, tolerance, rescale fraction, SHA-256 per shard) · `data/verify_dataset.py` (re-solve 1 % with PyAMG, duplicate check, distributions) · upload to a Kaggle Dataset + Drive.

**Prompt N2.1 · generator**
```
READ FIRST: CLAUDE.md §7.4 (splits, design-space half, float16 storage), §6.4 (tolerance 1e-10), §6.5 (envelope),
§5 (data hygiene); files: data/sampler.py, data/raster.py, data/envelope.py, solvers/fvm.py (coefficients),
solvers/pcg.py (batched_pcg).
GOAL: Produce every split reproducibly, resumably and verifiably.
DELIVERABLES: data/generate.py (CLI: --split {train,val,cal,test,ood_<name>} --count --seed --out --batch 32 --N 256);
tests/test_generate.py.
SPEC: per-board seed = SHA-256(split, seed, i) → 64-bit int; for --split train, boards with even i use mode "design" with
x ~ U[0,1]^36 from that seed, odd i use "id"; pipeline: sample → rasterise → coefficients → batched_pcg (line-z
preconditioner, tol 1e-10 rel) → apply_envelope → energy_balance ≤ 1e-3 (else raise naming the seed); shard of 500:
shard_XXXX/records.jsonl (record + scaled flag + iters) and fields.npy (float16 [n,4,N,N], C-order); write into
shard_XXXX.tmp then atomic rename; after each shard update manifest.json (seed, commit hash, tolerance, count, sha256 per
file, running rescale fraction, start/end time); resume = skip completed shards; refuse if --out holds another split's
manifest or a completed shard would be overwritten; log boards/s and ETA.
TESTS FIRST (N = 64, count 40, CPU fallback allowed): a simulated crash after shard 1 then resume yields byte-identical
shards to an uninterrupted run; manifest hashes verify; record hashes never collide across two splits.
DONE WHEN: tests pass; a 200-board pilot generated and its distributions reviewed by M and N together.
REPORT BACK: file, test output, pilot throughput (boards/s, GPU model).
```

**Prompt N2.2 · independent verification**
```
READ FIRST: CLAUDE.md §5 (B2, B3, data hygiene); files: data/generate.py, solvers/amg_baseline.py.
GOAL: An independent check that the released data is correct, complete and leak-free.
DELIVERABLES: data/verify_dataset.py (exit non-zero on any failure) + data/verify_report.md + data/figs/*.
SPEC: (1) recompute SHA-256 of every file vs the manifest; (2) re-solve a random 1 % per split with PyAMG-GMRES (tol 1e-10)
and require max|θ − stored| / max θ < 2e-3 (float16), reporting the worst; (3) hash each record JSON (sorted keys) and
require no hash in two splits; (4) per split: count, rescale fraction (≤ 0.20), θ_max P5/P50/P95, chip-count histogram,
hotspot-location heat map, mean φ per layer.
DONE WHEN: exits 0 on the full dataset; report committed.
REPORT BACK: file and the report.
```

**Check it worked**
- [ ] Pilot of 200 reviewed before the full run
- [ ] All splits complete; `verify_dataset.py` exits 0
- [ ] Rescale fraction ≤ 20 % (else lower power, regenerate)
- [ ] Throughput logged for the PAGP chapter
- [ ] Kaggle Dataset v1 + Drive copy; hashes match; tag `v2-data`

**If it goes wrong**
- Energy-balance raise → reproduce that seed alone, check PCG reached 1e-10 (raise maxit), check for a degenerate board. Fix the cause, then resume. Never skip boards silently.
- Rescale fraction > 20 % → lower the power range in `spec_v2.yaml`, log, regenerate.
- Kaggle disk full → write shards straight to the dataset output; `savez_compressed` if needed (update the loader + manifest).
- Bug found after release → new version `v2.1-data`, regenerate the affected splits, re-verify, retrain, log. Loaders check hashes, so stale data can't mix in.

## M · input pipeline, FNO ensemble, baselines, evaluation

**In simple words.** You train the fast guesser: board images in (13 channels), 4-layer temperatures out, in milliseconds. You train 5 copies; where they disagree, trust is lower (the gate uses this). Then you measure how good it is, including on unusual boards it never saw.

**What it actually does.** `models/inputs.py` + `models/dataset.py` (13 channels, train-only normalisation, augmentation, guards) · `models/train_ensemble.py` (128² pretrain → 256² fine-tune, 5 members) · `models/ensemble.py` (μ, σ) · U-Net and ablations · `experiments/e5_accuracy.py` · `models/ood_score.py`.

**Prompt M2.1 · inputs and dataset**
```
READ FIRST: CLAUDE.md §7.5 (13 channels, order, normalisation), §5 (data hygiene), §6.6 (mask), §12 (augmentation);
files: data/raster.py, data/sampler.py (transform_record), data/manifest.json, models/data_guard.py.
GOAL: Turn stored boards into exactly the tensors the contract defines, with no leakage.
DELIVERABLES: models/inputs.py (build(raster, spec) -> float32 [13,N,N]); models/normalise.py (Normaliser.fit on TRAIN
only, saves models/norm_stats.json with dataset hash; load refuses a hash mismatch; denormalise exactly inverts);
models/dataset.py (TwinHeatDataset(root, split, N, augment, allow_eval=False)); tests/test_inputs.py.
SPEC: channel order exactly §7.5; log10 G_l computed from spec + raster (assert G > 0); log(1+ν) with ν in vias/mm²;
dataset verifies shard hashes at init; re-rasterises inputs from records (optional on-disk cache keyed by record hash);
N = 128 → targets by 2×2 area averaging, inputs re-rasterised at 128; augmentation = one of 8 square symmetries applied to
the record (transform_record) and to the target field consistently; x/y channels regenerated after transform; splits
cal/test/ood_* raise unless allow_eval=True.
TESTS FIRST: round-trip normalisation; for 3 boards, transform(stored θ) equals the stored θ of the transformed record
re-solved (within 1e-3 K): physics-consistent augmentation; shapes/dtypes match the contract (B6); requesting "cal"
without allow_eval raises.
DONE WHEN: tests pass.
REPORT BACK: files, test output.
```

**Prompt M2.2 · ensemble training**
```
READ FIRST: CLAUDE.md §12 (config, loss, schedule, targets), §4 (compute), §5; files: models/fno.py, models/train.py,
models/dataset.py.
GOAL: Five independently seeded FNO members that meet the week-5 targets within free GPU quotas.
DELIVERABLES: models/train_ensemble.py (args: --member --seed --stage {pre128, ft256} --init --ckpt_dir --backup_dir
--epochs); models/losses.py (rel_l2, hotspot_l1 over top-64 hottest TRUE cells any layer); models/ensemble.py
(Ensemble.load(registry) → predict(x) -> mu, sd in kelvin); models/registry.json updater; a Kaggle notebook template
(notebooks/train_member.ipynb) that clones the repo at a tag and runs one member.
SPEC: FNO2d(13→4, width 48, modes 24, 4 layers); loss = rel_l2 + w_hot·hotspot_l1 (w_hot 0.1), both logged; AdamW lr 1e-3
(pre) / 3e-4 (ft); cosine with 5 % warm-up; AMP with FP32 loss; grad clip 1.0; pick batch size from a measured memory probe
on T4 16 GB (grad accumulation if needed); early stop on validation hotspot error (K), patience 10; save best+last every
epoch locally and to --backup_dir; resume; JSON summary per member (seed, epochs, best val rel-L2, best val hotspot K,
GPU-hours); registry stores checkpoint SHA-256 + norm_stats hash.
TESTS FIRST: 2-epoch smoke run on fake_dataset/ at 64²; resume continues the curve; Ensemble.predict returns sd = 0 when
all members are identical.
DONE WHEN: member 1 trained and its GPU-hours measured BEFORE launching members 2–5; all 5 trained with healthy curves.
REPORT BACK: files, member summaries, the 3-account training schedule you propose.
```

**Prompt M2.3 · baselines, E5 evaluation, OOD score**
```
READ FIRST: CLAUDE.md §11.5 (E5, E6), §12 (U-Net, ablations, go/no-go), §6.7 (runtime OOD flag), §5 (allow_eval only in
experiments/); files: models/*.
GOAL: Honest accuracy numbers on ID and every OOD set, and a learned OOD score for the gate's groups.
DELIVERABLES: models/unet.py (4 levels, 32 base channels, GroupNorm); models/fno_local.py (FNO + 3×3 conv branch);
experiments/energy_ablation.py (energy-form loss ½θᵀAθ − bᵀθ at 0/100/1k labels, matrix-free via coefficients());
experiments/e5_accuracy.py; models/ood_score.py; experiments/configs/e5.yaml.
SPEC E5: for ID test and each of the 10 OOD sets: ensemble mean, each member, U-Net, FNO+local → relative L2 (mean, P95),
hotspot temperature error (K), hotspot location error (mm), per layer and overall; 128-only model zero-shot at 256;
tidy CSV + LaTeX table + figure with footer. OOD score: pooled last-hidden-layer FNO features → Mahalanobis distance to
train features (Ledoit–Wolf covariance); threshold = 95th percentile on validation; report OOD-flag recall per set.
CONSTRAINTS: hyperparameters come only from config files committed before the run; no tuning on test/OOD.
DONE WHEN: E5 table produced by script; go/no-go recommendation written.
REPORT BACK: files, the E5 summary table, go/no-go recommendation with the numbers.
```

**Check it worked**
- [ ] Augmentation-consistency test passes
- [ ] Member 1 GPU-hours measured first; 3-account schedule written
- [ ] All 5 curves healthy; checkpoints + norm stats on Drive with hashes in `registry.json`
- [ ] E5: ID rel-L2 < 3 % and hotspot < 1.5 K, or fallback logged
- [ ] Ensemble σ larger on OOD than ID

**If it goes wrong**
- Slight miss (3–5 %) → design-space boards, FNO+local, one more week; Phase 3 starts on the current model anyway (the gate just flags more).
- Big miss → gate experiments at 128², stated in the report. Never lower targets afterwards.
- Errors at copper edges → expected FNO weakness; that's what U-Net and FNO+local are there to show.
- NaN with AMP → FP32 loss, clip, lower learning rate, assert G > 0 (no log10(0)).
- Leak guard bypassed → the reviewer rejects any `allow_eval=True` outside `experiments/`.

## G · layer peel, realism, server v0, pre-registration

**In simple words.** The viewer becomes a real 3D board: pull the four layers apart like pages, see vias glow as they carry heat, make it look like an actual PCB. Connect to a first real server with M. File the study's pre-registration before any data.

**What it actually does.** T1.1 layer peel with instanced via columns, OIT slab and cut plane · T1.4 JFA distance field → bevel → normals, PBR copper/clear-coat · `server/app.py` v0 with `model_stub` · OSF pre-registration.

**Prompt G2.1 · layer peel with OIT**
```
READ FIRST: CLAUDE.md §13 (T1.1, targets), §7.2 (vertical flow formula), §10 (pred fields); files: viewer/src/*.
GOAL: Make the inside of the board visible and physically meaningful.
DELIVERABLES: viewer/src/layerPeel.js, viewer/src/vias.js, viewer/src/oit.js, shaders/oit_accum.frag, shaders/oit_composite.frag,
viewer/src/cutPlane.js.
SPEC: separation s ∈ [0,1]: slider on desktop; in VR, distance between controllers while both grips are held (smoothed);
via columns = InstancedMesh of cylinders at via-array cells per gap; emissive intensity ∝ ν·k_Cu·A_v·(θ_l − θ_m)/d (tone-
mapped); FR-4 slabs with weighted blended OIT (McGuire & Bavoil 2013): accumulation (RGBA16F, additive) + revealage (R8,
multiplicative) targets, composite pass, blend states stated; grabbable cut plane showing a through-thickness section
(sample the 4 layer textures, interpolate in z).
TESTS FIRST: a board with vias under one chip only → that chip's columns are brightest; a board with no vias → no columns.
DONE WHEN: correct from every angle (no popping); desktop ≥ 60 fps, headset ≥ 72 fps.
REPORT BACK: files and a technique-checklist paragraph for the CGVR report.
```

**Prompt G2.2 · board realism**
```
READ FIRST: CLAUDE.md §13 (T1.4); files: viewer/src/*.
GOAL: A board that reads as a real PCB at a glance.
DELIVERABLES: viewer/src/jfa.js + shaders/jfa_{seed,step,dist}.frag; viewer/src/boardMaterial.js; viewer/src/packages.js.
SPEC: jump flooding on a 512² copper mask: seed pass, log2(512) flood passes (ping-pong RG32F), signed distance pass; bevel
height map over w mm at copper edges; normal map by central differences; MeshPhysicalMaterial-based layers: copper
(metalness 1, roughness 0.25, oxidation noise), solder mask as clearcoat with green tinted transmission, silkscreen from a
canvas texture; instanced QFN/SOT-23 packages with pins; IBL from a small HDRI. JFA baked once per edit, not per frame;
< 3 ms/frame after the bake. Debug toggle shows the SDF.
DONE WHEN: an outsider says "that looks like a real board"; frame budget met.
REPORT BACK: files and a paragraph on JFA for the report.
```

**Prompt G2.3 · server v0 (with M)**
```
READ FIRST: CLAUDE.md §10.3 (messages: query, design, pred), §5 (B6, B8); files: contracts/*, server/model_stub.py.
GOAL: The first real server, so the viewer talks to Python instead of a replay.
DELIVERABLES: server/app.py (FastAPI + websockets) v0; tests/test_server_v0.py.
SPEC: on query or design: validate; rasterise; model_stub.predict → mu, sd; peak; reply pred with decision "preview" for
query and "accept" for design (no gate yet), B null; inference in a worker thread with an asyncio queue; per client keep
only the newest queued query; validate every outgoing message in dev mode and raise on failure.
TESTS FIRST: TestClient websocket: 20 queries then a design → replies in order, all schema-valid; flooding 100 queries
leaves at most 1 queued.
DONE WHEN: viewer ↔ server round trip works with 10 drags/s and flat latency.
REPORT BACK: file, tests, run command.
```

**Prompt G2.4 · pre-registration (chat)**
```
READ FIRST: CLAUDE.md §13 (pilot), §11.4 (H4), §10.6 (session log).
GOAL: An OSF-style pre-registration filed before any study data.
INCLUDE: hypotheses (direction for H4); design (within-subjects, 3 conditions, Latin-square order); tasks A and B; measures
(accepted-wrong rate, unnecessary-verification rate, verified peak, time, NASA-TLX); exclusion rules (decided now);
primary analysis (paired effect sizes, bootstrap 95 % CIs, 10,000 resamples, fixed seed); what counts as exploratory;
sample size justification as a pilot.
CONSTRAINTS: modest claims; no results.
```

**Check it worked**
- [ ] Layer peel correct in desktop and VR; via brightness test passes
- [ ] Realism glance test passed; frame budget met
- [ ] Server v0 round trip; no queue build-up
- [ ] Pre-registration filed (link in `decisions.md`) before any study data

**If it goes wrong**
- OIT washed out → use the paper's depth weight and clamp; or sorted transparency for the 4 slabs (noted).
- Headset < 72 fps → profile first, then half-res post and the opaque layer mode.
- Server floods → drop-older rule + 30 Hz client throttle.
- Pre-registration late → results are labelled exploratory. Avoid this; it takes an hour.

---

# Phase 3 · Gate, baselines and guarantees (weeks 5–8)

**Goal.** Show the audited gate keeps accepted large errors at target for every proposer, where the published gate drifts.
**Gate 3 (end of week 8):** G0–G7 behind one interface, B8 green · P1–P3 trajectories oracle-labelled to 1e-8 · E7, E8, E9, E14, E15, E16 with 5 seeds and intervals · Lemmas 1 and 1′ signed by N · G6 proved or marked empirical · frozen hyperparameters committed before reported runs (D4).

## M · gates, proposers, groups, proofs, experiments

**In simple words.** For every proposed design, the gate decides: trust the network, or send it to the exact solver. It computes an error bound in kelvin; small bound → accept, large → flag. Separately, a coin flip audits about 10 % of designs at random (solved exactly regardless), and only audits adjust how strict the gate is. Because audits are random, the gate can't fool itself, and we can prove that badly wrong guesses are accepted at most about α of the time, for any source of designs, including a person. You build this gate, seven competitors, three automated designers and the experiments.

**What it actually does.** CLAUDE.md §8.1 loop and §9.3 code as `gate/`; G0–G7 (§11.1); P1–P4 and groups (§11.2, §8.5); `report/lemmas.tex`; experiments E7–E9, E14–E16 on oracle-labelled trajectories.

**Prompt M3.1 · gate interface, scale, coin, G5**
```
READ FIRST: CLAUDE.md §8.1–§8.3 (loop, Lemma 1, 1′), §9.2–§9.3 (reference code), §10.4 (audit-log columns), §5 (B8),
§17 D4; files: models/ensemble.py, solvers/fvm.py (coefficients), certify/peak_interval.py (if present; else mock it).
GOAL: The audited gate exactly as proved, with its assumptions enforced by code, not by convention.
DELIVERABLES: gate/base.py (BaseGate: decide(GateInput) -> GateDecision(B, decision ∈ {accept, flag}, p_audit, group);
update(list[AuditRecord]); state_dict/load_state_dict; log_row(...) producing one §10.4 row); gate/scale.py
(label_free_scale(mu, sd, coeffs, b, m=64, lam, eps) -> s, r_hot, sd_hot using r = (A mu − b)/diag in kelvin over the m
hottest cells of mu across all layers); gate/coin.py (AuditCoin(seed): commit(design_id); draw(design_id, p) raises if
commit wasn't called first; coins never returned to proposers); gate/aci_audit.py (AuditedACIGate(alpha, gamma, W, tau)
per §9.3: q from calibration scores; update() raises ValueError on any record with audited == False; q = inf if
alpha_t ≤ 0, 0 if alpha_t ≥ 1, else the (1 − alpha_t) quantile of the last W scores; uses S_hi from certified intervals);
tests/test_gate_isolation.py, tests/test_aci.py.
TESTS FIRST:
 1. B8: passing a flagged non-audited record to update() raises and q is unchanged.
 2. B8: draw() before commit() raises.
 3. i.i.d. synthetic scores: miscoverage on audited steps within C/(γ·n_a) of α (C per Lemma 1).
 4. Lemma 1′: S_hi ≥ S for 10,000 random (T̂, [T⁻, T⁺] ∋ T) cases.
 5. Adversarial synthetic drift: G5's accepted-large-error rate ≤ α + Lemma 1 slack; a static split-conformal gate exceeds it.
CONSTRAINTS: hyperparameters are constructor arguments (no defaults that silently become "the" values; D4 freezes them).
DONE WHEN: tests pass; N reviews test 1–2 code.
REPORT BACK: files, test output, and a table "Lemma 1 assumption A1/A2/A3 → the code line(s) that enforce it".
```

**Prompt M3.2 · baseline and extension gates**
```
READ FIRST: CLAUDE.md §11.1 (gates), §8.5 (G6, G7 rules); files: gate/base.py, docs/refs/deepoheat_v2_alg1.txt,
docs/refs/bastani2022_mvp.txt.
GOAL: Every comparison gate behind the same interface, each faithful to its source.
DELIVERABLES: gate/baselines.py (G0 NoGate, G1 DeepOHeatV2Gate, G2 StaticSplitConformal, G3 WeightedSplitConformal,
G4 ACIFlaggedOnly); gate/multivalid.py (G6); gate/active_audit.py (G7); tests/test_baselines.py.
SPEC: G1: flag if r_hot > (1−f) percentile of a FIFO window of the last 100 r_hot values, f tuned to a target solve
budget, exactly as docs/refs/deepoheat_v2_alg1.txt; G2: q from calibration once; G3: Tibshirani et al. 2019 weights,
exact density ratio for P1 (p ∝ p₀·exp(−λT̂)), classifier-estimated ratio (logistic regression on design vectors,
clipped to [0.05, 20]) otherwise; G4: ACI updated on flagged labels (the documented selection-bias ablation; it must
bypass the B8 guard explicitly via a named flag `ablation_flagged_only=True`); G6: per-group thresholds updated on audited
steps per docs/refs/bastani2022_mvp.txt, groups may intersect; G7: p_t = max(p_min, π(s, sd, g)) with mean p̄ (π mixes
uniform and uncertainty-proportional), update α += γ·(p̄/p_t)·A_t·(α − err).
TESTS FIRST: one synthetic test per gate demonstrating its defining behaviour (e.g. G1's flag rate ≈ f; G4 under-covers
on a distribution where flagged ≠ accepted errors; G6 covers each of 2 intersecting groups; G7's weights have mean ≈ 1).
CONSTRAINTS: if a refs file is missing, stop and ask; don't reconstruct algorithms from memory.
DONE WHEN: tests pass.
REPORT BACK: files, one docstring citation per gate, test output.
```

**Prompt M3.3 · proposers and groups**
```
READ FIRST: CLAUDE.md §11.2 (proposers), §8.5 (groups), §7.4 (design space), §5 (B8); files: data/sampler.py
(design_board), models/ensemble.py, models/ood_score.py, gate/coin.py.
GOAL: Automated designers of three aggressiveness levels that never see audit coins, plus label-free groups.
DELIVERABLES: gate/proposers.py (P1 TiltedMCMC(λ ∈ {1,3,10}), random-walk Metropolis step 0.05 on [0,1]^36 with
reflection; P2 SimulatedAnnealing (1,000 steps, 3 presets of T0 and step); P3 BatchedGradient (64 starts × 50 steps, 3
step-size presets) via finite differences on the ensemble mean OR a differentiable rasteriser (choose, justify); P4
Replay(log_path)); each proposer exposes next(history) where history contains predictions and gate decisions but NO
coins; gate/groups.py (proposer, aggressiveness, OOD tercile from ood_score thresholds on validation, hotspot within 5 mm
of an edge); experiments/run_trajectories.py writing parquet per proposer × level × seed with the §10.4 columns (minus
oracle_peak).
TESTS FIRST: history objects contain no coin fields (schema test); groups are computed before the coin (call-order
assertion with a mocked coin); same seed → identical trajectory.
DONE WHEN: P1–P3 × 3 levels × 5 seeds trajectories generated and handed to N for oracle labelling.
REPORT BACK: files, trajectory counts, estimated oracle solves needed.
```

**Prompt M3.4 · Lemma write-up check (chat; N signs)**
```
READ FIRST: CLAUDE.md §8.2–§8.5. MATERIAL: {{RUNTIME: our lemmas.tex draft}}; docs/refs/gibbs_candes_prop41.txt;
docs/refs/tu_giesecke_thm54_56.txt (paste both if in chat).
GOAL: A correct, honestly attributed statement of Lemmas 1 and 1′ (and the G6/G7 arguments) for the report and paper.
DO: act as a hostile reviewer: (1) check each step and name unstated assumptions; (2) re-derive C/(γpT) and
(2/p)√(ln(4/δ)/T), including the martingale difference (A_t − p)·m_t, its bound, and how n_a/(pT) is controlled;
(3) check that err_t = 1[S⁺ > q_t] is valid input to the ACI recursion and that W_T's bound carries over; (4) check the G6
per-group argument and say whether it is a proof or a sketch; (5) list exactly which parts are cited and which are ours.
CONSTRAINTS: no invented theorem numbers; if a pasted statement does not support a step, say so; propose the weakest
correct statement where something fails.
```

**Prompt M3.5 · experiment runners**
```
READ FIRST: CLAUDE.md §11.3 (metrics), §11.4 (H1, H2, H7, H8, H9), §11.5, §8.2 (bound formula), §4 (figures, frozen configs),
§17 D4; files: gate/*, experiments/oracle_label.py outputs.
GOAL: Offline replay of every gate on identical oracle-labelled trajectories, producing the DL headline results.
DELIVERABLES: experiments/run_gates.py; e7_validity.py; e8_efficiency.py; e14_cert_audit.py; e15_groups.py;
e16_active.py; e9_darcy.py; experiments/configs/frozen.yaml template; figure scripts.
SPEC: refuse to run unless frozen.yaml exists and its git commit predates the run; every gate sees the same proposals in
order; metrics M1 (all, and among accepted), M1g, M2; 5 seeds; 95 % CIs by bootstrap over seeds; report the Lemma 1 bound
α + C/(γpT) + (2/p)√(ln(4/δ)/T) beside M1; E8 = feedback runs of G1 vs G5 at equal solve budget with the 0.2 K
non-inferiority test (paired); E14 sweeps ε_S; E16 sweeps p_min at equal audit budget; E9 uses the same code path on the
Darcy task; tidy CSVs + figures with footers.
TESTS FIRST: on a tiny synthetic trajectory with known errors, M1 equals a hand count; frozen-config guard triggers when
the config is newer than HEAD~0 data.
DONE WHEN: all experiments run end to end; H1/H7/H8/H9 verdicts printed by script.
REPORT BACK: files, verdict table, any anomalies.
```

**Check it worked**
- [ ] B8 tests pass for every gate (G4's bypass is explicit and named)
- [ ] `frozen.yaml` committed (chosen on validation trajectories) before test runs
- [ ] E7: G5 M1 ≤ α + slack in ≥ 95 % of runs for P1–P3 at all levels
- [ ] E15: per-group rates with slack for groups with ≥ 100 audits
- [ ] Lemmas signed by N; G6 "proved" or "empirical" by week 8
- [ ] E9 Darcy run through the same code path

**If it goes wrong**
- G5 exceeds target → it's a theorem, so a violation means a bug or a broken assumption. Check B8, coin timing, frozen-config use, then whether the slack is large (small n_a). Report what you find.
- q oscillates → smaller γ, larger W, p = 0.2 early then 0.1 (on validation only).
- H2 fails → "validity at a stated price": trade-off over p ∈ {0.05, 0.1, 0.2}.
- Small groups → merge, or raise p for rare groups (G7). Never drop a group silently.
- Proof stuck → restate Lemma 1 in expectation exactly as Tu & Giesecke; G6 empirical; logged.

## N · proof checking, oracle labels, certified audit intervals

**In simple words.** You referee M's maths, and you supply ground truth: the exact answer for every one of ~100,000 proposed designs, so error rates are counted, not estimated. You also give the gate cheaper audits, with guaranteed intervals for the true peak from early-stopped solves.

**What it actually does.** A signed checking table for Lemmas 1, 1′, G6, G7 · `experiments/oracle_label.py` (1e-8, CPU pool + GPU batches, resumable, 0.5 % cross-solver check) · `certify/peak_interval.py` + `audit_until_decided`.

**Prompt N3.1 · proof checking aid (chat; you sign, not Claude)**
```
READ FIRST: CLAUDE.md §8.2–§8.5. MATERIAL: {{RUNTIME: lemmas.tex section}}; cited theorem statements from docs/refs/.
GOAL: Independent verification of M's proofs.
PRODUCE: a table step | claim | justification given | sufficient? (yes/no/unclear) | what is missing; then (1) independent
re-derivation of every constant; (2) the simplest counterexample for any "no"/"unclear" step; (3) the assumptions in the order
used. Report only; do not fix the proof. Flag any step the pasted theorem text does not support.
```

**Prompt N3.2 · oracle labelling job**
```
READ FIRST: CLAUDE.md §11.5 (oracle: ~1e5 solves at 1e-8), §10.4 (columns), §5 (B3); files:
experiments/run_trajectories.py outputs, data/sampler.py (design_board), solvers/amg_baseline.py, solvers/pcg.py.
GOAL: Exact ground truth for every proposal, so validity is counted, not estimated.
DELIVERABLES: experiments/oracle_label.py; experiments/oracle_report.md generator.
SPEC: input = directory of trajectory parquet; dedupe by record hash across files; rebuild each board from its design
vector; solve to 1e-8 relative residual; store oracle_peak (K), argmax (layer, y, x), iterations, solver, wall time,
SHA-256 of θ (float32); execution = multiprocessing CPU pool (PyAMG-GMRES, OMP_NUM_THREADS=1 per process) + optional GPU
worker (batched PCG, batch 64) pulling from one queue; append per input file; resume by skipping labelled hashes; after
completion re-solve a random 0.5 % with the OTHER solver and require |Δpeak| < 1e-4 K; report throughput (boards/s,
CPU-hours, GPU-hours). NEVER use certified early stopping here.
TESTS FIRST: on 50 boards, CPU and GPU labels agree < 1e-4 K; resume after kill produces no duplicates or gaps.
DONE WHEN: labelled count == unique proposals; cross-check passes; report committed.
REPORT BACK: file, counts, throughput.
```

**Prompt N3.3 · certified peak interval and certified audits**
```
READ FIRST: CLAUDE.md §8.3 (Lemma 1′), §8.4 (Lemma 2 b/c, stopping), §5 (B7); files: certify/kappa.py, certify/pointwise.py,
certify/margin.py (if not yet derived, use the §9.1 placeholder and mark results provisional), solvers/pcg.py.
GOAL: Cheaper audits that are still rigorous.
DELIVERABLES: certify/peak_interval.py with peak_interval(x, coeffs, b, kappa_hat=None, w=None) -> (T_lo, T_hi, method)
and audit_until_decided(mu, coeffs, b, s, q_t, eps_S, kappa_hat, check_every=5) -> (T_lo, T_hi, S_lo, S_hi, iters,
stop_reason); tests/test_peak_interval.py.
SPEC: with w (Lemma 2c, A w ≥ |r| verified in FP64): T_lo = max(x − w), T_hi = max(x + w); with κ̂ (Lemma 2b):
c = κ̂·‖b − A x‖∞ (FP64, × margin); T_lo = max(x) − c, T_hi = max(x) + c. Audit: certified PCG from μ, stop when
[S_lo, S_hi] (S = |T̂_peak − T_peak|/s) is entirely ≤ q_t or > q_t, or S_hi − S_lo < ε_S.
TESTS FIRST: on 200 boards T_lo ≤ oracle_peak ≤ T_hi (B7, zero violations); interval width shrinks on average along iterates.
DONE WHEN: zero violations; used by E14.
REPORT BACK: file, tests, violation count (must be 0).
```

**Check it worked**
- [ ] Checking table done by N, every "unclear" resolved; initials + date in `lemmas.tex`
- [ ] ~1e5 labelled; counts match; 0.5 % cross-check passes; report committed
- [ ] `peak_interval` zero violations in tests and in E14

**If it goes wrong**
- Oracle too slow → split by file across lab machines; add the GPU worker. Labels are resumable and deduped.
- Cross-solver disagreement → one didn't converge. Raise maxit, compare residuals, never average.
- Interval violation → treat as B7: stop E14, check the FP64 residual + margin, fix, re-run.
- A real proof gap → write it down, agree the weakest correct statement with M (and the mentor), update the claims everywhere.

## G · trust display, replay, pilot stimuli, human-trace logging

**In simple words.** Make the gate's decisions visible: three ways to show how much to trust the temperature, a Predicted/Verified badge, a replay of a whole design session, the study material, and the logging that turns a person's session into proposer P4.

**What it actually does.** T1.6 (shimmer/hatching/VSUP + badge) · T1.8 replay · `study/select_stimuli.py` + counterbalancing · `viewer/src/sessionLog.js` (§10.6).

**Prompt G3.1 · trust encodings and badge**
```
READ FIRST: CLAUDE.md §13 (T1.6, T1.7), §10.3 (pred, iterate, verified), §5 (B8); files: viewer/src/*.
GOAL: Show trust honestly and selectably, driven only by server messages.
DELIVERABLES: viewer/src/trust/{shimmer.frag, hatch.frag, vsup.js, vsup.frag, badge.js, conditions.js}.
SPEC: input U = band width texture (K, R16F) over the field T; (a) shimmer: animated 3D value noise (x, y, time) added to
the LUT lookup, amplitude ∝ U, capped, ≤ 2 Hz; (b) hatching: line density ∝ U, anti-aliased; (c) VSUP (Correll, Moritz &
Heer 2018): 4 uncertainty levels, value bins merge and desaturate toward grey as U grows, 2D LUT texture; badge: Predicted
(amber, shows B in K) → Verifying (blue) → Verified (green), state changed ONLY on server messages; conditions.js sets the
study condition: "none" hides bands and B, "flag" shows a binary residual flag, "calibrated" shows B + band + anytime
verification.
TESTS FIRST: badge never turns green from client-side logic (replay a log where iterates arrive without verified → stays blue).
DONE WHEN: all encodings switchable; band visibly tracks |error| on 10 sample boards.
REPORT BACK: files + 2 sentences per encoding for the report.
```

**Prompt G3.2 · proposer replay**
```
READ FIRST: CLAUDE.md §13 (T1.8), §10.4 (audit-log columns); files: viewer/src/*.
GOAL: The demo moment that explains the method to anyone.
DELIVERABLES: viewer/src/replay.js, viewer/src/replayChart.js.
SPEC: load an audit-log CSV + board records; scrubber, play/pause 1×–50×; board shown per step with decision colour
(accepted green, flagged amber, audited blue); if oracle_peak exists, red outline when accepted with |error| > τ; side
chart (canvas) of best verified peak vs cumulative exact solves for up to 3 gates (G1, G5, G6); precompute everything at load.
DONE WHEN: 10,000 steps at ≥ 60 fps; red outlines match the oracle at 5 spot-checked steps.
REPORT BACK: files.
```

**Prompt G3.3 · stimuli selection (with M)**
```
READ FIRST: CLAUDE.md §13 (pilot tasks), §8.1 (τ); files: oracle-labelled trajectories.
GOAL: Task-A stimuli that can actually separate the three conditions.
DELIVERABLES: study/select_stimuli.py → study/stimuli.json, study/orders.json.
SPEC: 30 boards: 15 with |error| ≤ τ and 15 with |error| > τ; at least 8 where the residual flag and the calibrated bound
disagree; spread over OOD terciles; fixed seed; store record, mu, sd, B per condition and the hidden truth; 3 Latin-square
condition orders; no stimulus repeats within a participant across conditions.
DONE WHEN: script reproducible; counterbalancing test passes.
REPORT BACK: file, summary table of the selected stimuli.
```

**Prompt G3.4 · session logging for P4**
```
READ FIRST: CLAUDE.md §10.6 (session log), §10.3 (query vs design), §8.2 A1, §5 (B8); files: viewer/src/*.
GOAL: Human sessions usable as proposer P4 without breaking the assumption that the proposer never sees the coin.
DELIVERABLES: viewer/src/sessionLog.js; tests/viewer/sessionLog.test.js.
SPEC: log every action per §10.6 with relative ms; participant code only; commit = chip released or brush stroke ended,
and only then a design message is sent; JSONL export convertible by experiments/p4_import.py.
TESTS FIRST: replaying a log against a mock server: no audit decision ever precedes its design message; dragging sends only query.
DONE WHEN: one internal session converts and replays through M's Replay proposer end to end.
REPORT BACK: file, test output.
```

**Check it worked**
- [ ] Encodings switchable; band tracks |error|; VSUP checked by a second person against the paper
- [ ] Replay smooth at 10k steps; outlines match the oracle
- [ ] Stimuli + orders reproducible
- [ ] P4 log end to end; coin-order test passes

**If it goes wrong**
- Shimmer causes discomfort in VR → cap at 2 Hz; hatching as the VR default (noted before the study).
- Stimuli too easy in the dry run → re-select borderline cases near τ; log it **before** the real pilot.
- The UI ever shows "audit" before commit → A1 is broken for P4; those sessions can't be used; fix before continuing.

---

# Phase 4 · Certify-on-demand pipeline and certificates (weeks 4–8, parallel to Phase 3)

**Goal.** Verify doubtful designs in batches on the GPU, each board stopping the moment its certificate settles its decision, with zero certificate violations.
**Gate 4 (end of week 8):** derived FP margin signed by M · B7 zero violations on 4 layers · E3, E4, E13 + rooflines + Nsight profile · crossover batch size reported · online/offline equivalence test passes.

## N · certificate kernels, certified PCG, fused gate, compaction, benchmarks

**In simple words.** When the gate flags hundreds of designs at once, checking them one by one on a CPU is too slow. You build a GPU assembly line: one kernel scores every design and makes the gate decision, a parallel scan packs the doubtful ones together, and one batched solver verifies them all. The certificate tells you, at every solver step, a guaranteed maximum error for each board, so each board leaves as soon as its answer is good enough.

**What it actually does.** `certify/kappa.py` (batched z ≈ A⁻¹1, FP64 checks) · `certify/margin.py` (derived rounding margin) · `certify/pcg_batched.py` · `pipeline/certify_batch.py` (reference) → `pipeline/certify.cu` (fused) → Philox coins → `pipeline/scan.cu` → batched certified PCG with stop masks and re-compaction → scatter · `bench/pipeline_bench.py`, `bench/roofline.py`.

**Prompt N4.1 · derive the rounding margin (chat, then code)**
```
READ FIRST: CLAUDE.md §8.4 (FP64, margin), §7.3 (grid), §9.1 (placeholder margin); coefficient ranges from config/spec_v2.yaml
{{RUNTIME: paste the min/max of diag and off-diagonal coefficients measured on 50 boards}}.
GOAL: A margin that makes every reported certificate a guaranteed upper bound despite floating point.
DO: (1) bound the rounding error of each computed row of A z and of b − A x for a 7-point stencil (diag + ≤ 6 off-diagonals)
in IEEE binary64 using fl(a op b) = (a op b)(1+δ), |δ| ≤ u = 2⁻⁵³, with and without FMA; (2) give ε_total (and an absolute
term if needed) such that fl(c)·(1+ε_total) + δ_abs ≥ κ̂_exact·‖r_exact‖∞, handling min(Az) near 0 (return "no certificate"
below a threshold you derive); (3) state every assumption (no overflow/underflow, z > 0, reduction order irrelevant for
max); (4) write certify/margin.py (margin(spec_stats) -> (eps_total, delta_abs, min_az_threshold)) and
tests/test_margin.py comparing against an 80-digit mpmath recomputation on 5 boards.
CONSTRAINTS: show every step; mark UNSURE where you are; M will check it line by line.
```

**Prompt N4.2 · batched certified PCG**
```
READ FIRST: CLAUDE.md §8.1 step 5, §8.4 (stopping rules), §9.1, §5 (B7); files: solvers/pcg.py, certify/kappa.py,
certify/margin.py, certify/peak_interval.py, pipeline/scan.py (Python wrapper; CUDA in N4.3).
GOAL: Solve many boards at once, each stopping exactly when its decision is certified.
DELIVERABLES: certify/kappa.py batched version kappa_hat_batched(coeffs[B], n_vcycles=3) (geometric V-cycles, in-plane
coarsening, layers kept; FP64 min(Az) and min(z)); certify/pcg_batched.py batched_pcg_certified(Aop32, Aop64, b, x0, Minv,
kappa[B], tau_v[B], mode[B] ∈ {flag, audit}, q, s[B], eps_S, check_every=5, recompact_every=4, maxit, stream_every=None)
returning per board x, iters, cert, stop_reason and optional iterates; tests/test_pcg_batched.py (@gpu).
SPEC: FP32 iterations; every check_every iterations compute ‖r_rec‖∞ per active board; boards whose κ̂·‖r_rec‖∞ suggests
done get an FP64 true residual and c = κ̂·‖r_true‖∞·(1+ε_total) + δ_abs; a flag board stops when c ≤ τ_v; an audit board
stops when its certified score interval is decided relative to q or narrower than ε_S; stopped boards are masked; every
recompact_every checks the active set is compacted; boards with kappa = None solve to 1e-8 and are counted.
TESTS FIRST (@gpu): 64 boards, every certificate ≥ oracle error (B7); a batch of 64 gives the same decisions and iterations
(±1) as 64 single solves; a mocked FP32 drift shows the recursive residual is never used alone to certify.
DONE WHEN: tests pass on a T4.
REPORT BACK: files, Kaggle log, violation count (must be 0).
```

**Prompt N4.3 · fused gate kernel and scan compaction**
```
READ FIRST: CLAUDE.md §9.2 (reference semantics), §9.5 (scan), §8.1 steps 2–4, §5 (B8); files: pipeline/certify_batch.py.
GOAL: The GPU front end of the pipeline, bit-for-bit consistent in decisions with the reference.
DELIVERABLES: pipeline/certify.cu + loader in pipeline/fused.py (CuPy RawModule): one thread block per board, inputs
[B,4,256,256] float32 mu, sd, coefficient arrays, b, diag; computes r = (A mu − b)/diag, block-wide top-m (m = 64) of mu over
all layers (two-pass: per-warp candidates then block merge; ties broken by lower linear index), r_hot, sd_hot,
s = r_hot + λ·sd_hot + ε, B = q·s, flag = B > τ; pipeline/coins.cu (curand Philox, seed = hash(run_seed, design_id), called
only after the host commits the batch); pipeline/scan.cu (work-efficient Blelloch exclusive scan with bank-conflict padding
and a block-sums pass up to 65,536) + compact(keep) -> idx; tests/test_fused.py (@gpu).
TESTS FIRST: decisions identical to pipeline/certify_batch.py on 512 random boards, B within 1e-5 rel; scan equals numpy
for all-0, all-1, random, non-power-of-two lengths; coins identical for identical (seed, id).
DONE WHEN: tests pass; occupancy, registers/thread and GB/s reported.
REPORT BACK: files, test log, performance numbers, one paragraph on the bank-conflict padding for the report.
```

**Prompt N4.4 · benchmarks (E3, E4, E13 raw runs, rooflines)**
```
READ FIRST: CLAUDE.md §11.4 (H5, H6), §11.5 (E2–E4, E13), §17 D1, §4 (figures); files: pipeline/*, certify/*, solvers/*.
GOAL: Measured evidence for the PAGP chapter and H5/H6.
DELIVERABLES: bench/pipeline_bench.py, bench/roofline.py, bench/e13_runs.py, results CSVs, figures.
SPEC pipeline_bench: batch ∈ {1,16,64,256,512}, flag rate ∈ {10 %, 30 %}; (a) sequential CPU loop: GPU surrogate + PyAMG-GMRES
per design to the same certified τ_v; (b) unfused GPU pipeline; (c) fused + compaction + certified stopping; certified
designs/s; per-stage time (CUDA events) + an nsys command; median of 10 after 3 warm-ups; crossover batch size.
e13_runs: 4-layer boards with the trained ensemble's predictions as warm starts; per board: iterations to 1e-8, to certified
τ_v ∈ {0.1, 1.0} K, c at stop, oracle error at stop, loose-tolerance field error at equal mean iterations, κ̂ vs true
‖A⁻¹‖∞ on a subset. roofline: stencil, SpMV, fused kernel — achieved bandwidth vs T4 peak, arithmetic intensity.
DONE WHEN: CSVs + figures with footers; violations = 0.
REPORT BACK: summary tables.
```

**Check it worked**
- [ ] Margin derived, mpmath test passes, signed by M (results provisional until then)
- [ ] B7 zero violations in tests and every E13 run
- [ ] Fused == reference on 512 boards; scan edge cases pass
- [ ] Week-6 decision logged (saving at 0.1/1 K, bound ÷ error, pointwise or not)
- [ ] E4: H5 met or crossover reported; E13 complete by week 8

**If it goes wrong**
- Any violation → freeze certificate results; check FP64 recheck → margin → z > 0 and Az > 0 in FP64 → B1 for that board; fix; re-run all of E13; log.
- Too loose (saving < 30 % or > 50× at week 6) → pointwise variant after smoothing sweeps; report the cost at which it pays; keep Lemma 2 for CG-2.
- Fused slower → split the kernel and report the trade-off (a clear negative result still earns marks).
- H5 < 10× → stage profile + crossover. Never handicap the baseline.
- Late in week 7 → drop the CUDA V-cycle (use CuPy/PyAMG for z) and mixed precision; keep scan, fused gate, certified stopping.

## M · Lemma 2 review, gate inside the pipeline, E13 analysis

**In simple words.** You referee N's maths (the certificate and the rounding margin), and make sure the fast GPU pipeline uses your gate exactly as proven: same thresholds, coins drawn at the same moment, audits the only thing that moves the threshold. A fast pipeline that quietly breaks the rules would make the guarantee false.

**What it actually does.** A signed checking table for Lemma 2 + margin · `pipeline/gate_bridge.py` · `tests/test_equivalence.py` (online == offline) · `experiments/e13_certificate.py` analysis using the D1 rule fixed in week 1.

**Prompt M4.1 · Lemma 2 checking aid (chat; you sign)**
```
READ FIRST: CLAUDE.md §8.4, §7.2 (M-matrix argument). MATERIAL: {{RUNTIME: certify/margin derivation text}}.
PRODUCE a step | claim | justification | sufficient? | missing table, checking specifically: (1) A⁻¹ ≥ 0 from a nonsingular
M-matrix, and that the 4-layer matrix meets the hypotheses in edge cases (no Dirichlet edges; h only on L1/L4; a layer
region with no vias and h = 0: is the matrix still irreducible?); (2) whether (a) needs z > 0 and Az > 0 strictly; (3) (c)
componentwise; (4) the peak interval; (5) whether the margin covers min(Az) and the max-norm. Report only; give the
simplest counterexample for any doubtful step.
```

**Prompt M4.2 · gate bridge and equivalence test**
```
READ FIRST: CLAUDE.md §8.1 (loop order), §9.2–§9.3, §5 (B8), §10.4; files: gate/*, pipeline/*, certify/pcg_batched.py,
experiments/run_gates.py.
GOAL: Guarantee that the deployed GPU loop makes exactly the decisions the proven offline gate makes.
DELIVERABLES: pipeline/gate_bridge.py; tests/test_equivalence.py; tests/test_bridge_isolation.py.
SPEC: per batch: (1) proposer commits all designs → coin.commit(ids); (2) bridge reads q (or per-group q for G6) from the gate;
(3) fused kernel → B, flag, s; (4) gate supplies label-free p_audit; (5) device coins (Philox, seed per id); (6) compact
flag | audit; (7) batched_pcg_certified; (8) audited results → gate.update(AuditRecord with T_lo, T_hi, S_hi, p_audit) ONLY;
flagged non-audited → retrain buffer ONLY; (9) one audit-log row per design. It must be impossible (by types) to send a
non-audited record to update().
TESTS FIRST: 2,000 P2 proposals at seed 0 through (a) the online GPU loop and (b) run_gates.py offline replay with the same
certified intervals → identical decision, coin, audited flag and q_t sequences; print the first divergence otherwise;
grep-style test that update() is called from exactly one place.
DONE WHEN: equivalence passes on a T4.
REPORT BACK: files, test log.
```

**Prompt M4.3 · E13 analysis**
```
READ FIRST: CLAUDE.md §11.4 (H6), §11.3 (M6), §17 D1, §4 (figures); files: bench/e13_runs.py outputs, docs/decisions.md.
GOAL: The H6 verdict, decided by a rule fixed before the data.
DELIVERABLES: experiments/e13_certificate.py; experiments/configs/h6_rule.yaml (copied from the D1 decision, committed
before the runs).
SPEC: read the rule from h6_rule.yaml and refuse to run if its commit is newer than the data's; compute median + IQR of
iterations saved vs 1e-8 at τ_v = 0.1 and 1 K; H6 verdict per the rule; distribution of c ÷ true error; violation count as an
exact integer; fraction of loose-tolerance fields that miss τ_v at equal mean iterations; LaTeX table + figures with footers.
DONE WHEN: verdict printed by the script.
REPORT BACK: verdict and table.
```

**Check it worked**
- [ ] Lemma 2 + margin signed by M
- [ ] Equivalence test passes (so E7's offline results apply to the deployed loop)
- [ ] `update()` called from exactly one place
- [ ] H6 rule committed before the data; verdict from the script; violations = 0

**If it goes wrong**
- Equivalence diverges → coin seeding/order or top-m tie-break. Fix the non-determinism, never the test.
- Hypothesis hole (e.g. an isolated region with h = 0) → add the case to B1, restrict the sampler or add a tiny h, log the scope change.
- H6 misses → report under the pre-fixed rule; the pointwise variant is a clearly labelled secondary result.

## G · anytime verified rendering, E11 latency, throughput panel

**In simple words.** While a design is being checked, the viewer shows the solver working: the map morphs from the guess to the exact answer inside a shrinking guaranteed band, and the badge turns green the moment the certificate says "good enough". You measure how fast it feels, and build a live panel showing boards leaving the GPU batch.

**What it actually does.** T1.7 (`iterate` → blended field + band = cert_k) · `experiments/e11_latency/` · `viewer/src/throughputPanel.js` (`batch_progress`).

**Prompt G4.1 · anytime verified rendering**
```
READ FIRST: CLAUDE.md §13 (T1.6, T1.7, targets), §10.3 (pred, iterate, verified), §8.4 (cert_k meaning); files: viewer/src/*.
GOAL: Make verification visible and honest in real time.
DELIVERABLES: viewer/src/anytime.js, shaders/anytime.frag, viewer/src/certChart.js (debug).
SPEC: on pred with decision flag/audit keep μ as T0; on each iterate upload θ to T1 and animate the blend over 150 ms on the
GPU; band uniform = cert_k (K) for Lemma 2(b) (or a per-cell texture if w is sent); shimmer amplitude = band; text
"guaranteed band ±X.XX K"; on verified blend to the final field; the badge reads ONLY the server's state; discard iterates for
designs that are no longer current; debug chart of cert_k vs k.
TESTS FIRST: band shown always equals the last received cert_k (log both, diff = 0); stale-design iterates are ignored.
DONE WHEN: with N's real stream the band shrinks and the badge goes green before verified on most flagged designs; VR ≥ 72 fps
while streaming.
REPORT BACK: files.
```

**Prompt G4.2 · latency measurement (E11)**
```
READ FIRST: CLAUDE.md §11.5 (E11), §13 (targets), §10.3 (t_server_ns); files: server/app.py, viewer/src/*.
GOAL: Measured, honest latency numbers.
DELIVERABLES: experiments/e11_latency/{client_probe.js, clock_sync.py, analyse.py}.
SPEC: viewer records performance.now() at receipt and at the first frame displaying each message (rAF after texture
upload); clock offset by an NTP-style exchange (min RTT of 20 pings); per design: gate→first iterate displayed,
gate→certified displayed, number of iterates, dropped iterates; 200 flagged designs on desktop and headset, localhost and
LAN; CSV + CDF plot with footer; medians and P95 vs 100 ms / 2 s.
DONE WHEN: numbers reported for all 4 configurations.
REPORT BACK: summary table.
```

**Prompt G4.3 · throughput panel**
```
READ FIRST: CLAUDE.md §10.3 (batch_progress), §13 (demo moments); files: viewer/src/*.
GOAL: The PAGP demo moment: boards visibly leaving the batch as they are certified.
DELIVERABLES: viewer/src/throughputPanel.js (desktop only).
SPEC: 8×8 tile grid (canvas 2D): grey active, green certified (with iterations), blue audited; tiles fade out on leaving;
live designs/s counter; stacked stage-time bar (gate, scan, pcg, scatter); throttled to 10 Hz.
DONE WHEN: numbers match pipeline_bench within noise.
REPORT BACK: file.
```

**Check it worked**
- [ ] Band == cert_k always; badge server-driven
- [ ] E11 medians vs targets reported; VR ≥ 72 fps while streaming
- [ ] Panel numbers match E4

**If it goes wrong**
- Iterates too fast → keep the latest per design; server streams at 10–20 Hz.
- Badge green too early → only the server decides; the replay test catches it.
- Latency misses → split server / network / decode / upload; move the float16 decode to a Web Worker.

---

# Phase 5 · Viewer, VR, pilot study and hardware (weeks 4–10)

**Goal.** An immersive viewer where a person designs a board, watches the guaranteed band shrink as verification runs, and compares with a real board under a thermal camera.
**Gate 5 (end of week 10, feature freeze):** T1.1–T1.9 on desktop and headset · ≥ 72 fps headset (E10) · E11 targets · pilot of 8–10 run after ethics approval, data anonymised · P4 handed to M · E17 on held-out boards (or formally moved).

## G · Tier 1 features, VR interaction, pilot study, hardware rig

**In simple words.** Your main phase, in three parts: finish the viewer (heat-flow particles, thermal-camera and glowing-metal looks, VR grabbing, probe pen, real thermal photo on the 3D board); run the user study; heat the real boards and compare with the solver.

**What it actually does.** T1.2, T1.3, T1.5, T1.9 · pilot session script, sessions, as-pre-registered analysis · rig, steady-state protocol, registration, comparison.

**Prompt G5.1 · flux particles + LIC**
```
READ FIRST: CLAUDE.md §13 (T1.2), §9.6 (advect.frag), §7.2 (vertical coupling); files: viewer/src/*.
GOAL: Show where heat flows, including down through vias.
DELIVERABLES: viewer/src/particles/{system.js, flux.frag, advect.frag, respawn_cdf.js, render.vert, render.frag, trails.js,
lic.frag}.
SPEC: positions in a 512×512 RGBA32F ping-pong texture (x, y, age, layer); flux per layer F = −k∇T in board-UV units from the
temperature texture and per-cell conductance (RG16F); RK2 midpoint as §9.6; respawn by inverse-CDF sampling of |F| near
chips (CDF texture rebuilt when the field changes) with a hash RNG; layer hop at via cells with probability ∝ vertical flux
g(θ_l − θ_m)·dt; points sized by |F|, coloured by T; trails via an accumulation buffer faded each frame; LIC mode (white noise
convolved along streamlines, 20 steps each way); debug view of F.
TESTS FIRST: with a uniform test flux, particles move in a straight line at the expected speed; with no vias, no hops.
DONE WHEN: particles drain through via arrays into L2 on a board with vias; < 2 ms/frame at 262k on a mid-range GPU.
REPORT BACK: files + a pass-by-pass explanation for the report.
```

**Prompt G5.2 · blackbody LUT and HDR pipeline**
```
READ FIRST: CLAUDE.md §13 (T1.3); files: docs/refs/cie1931_2deg_5nm.csv, docs/refs/planckian_locus_check.csv, viewer/src/*.
GOAL: A physically grounded (and honestly labelled) emissive mode plus our own post-processing.
DELIVERABLES: viewer/src/color/blackbody.js; viewer/src/post/{hdr.js, bloom.js, tonemap.js}; tests/viewer/blackbody.test.js.
SPEC: integrate Planck's law against the CIE 1931 2° CMFs from the CSV for T ∈ [800, 2000] K (256 entries); XYZ → linear sRGB
(IEC 61966-2-1 matrix); normalise luminance; document the negative-value handling; HalfFloat render target; bloom = bright
pass + separable Gaussian at 4 mip levels + additive composite; ACES (Narkowicz fit) and AgX tone mapping as our own passes;
on-screen note "Emissive colours are a stylised mapping of 25–125 °C to 800–2000 K".
TESTS FIRST: chromaticity (x, y) at 1000 K and 2000 K within 0.005 of planckian_locus_check.csv.
DONE WHEN: test passes; frame budget kept.
REPORT BACK: files.
```

**Prompt G5.3 · VR interaction**
```
READ FIRST: CLAUDE.md §13 (T1.5), §10.3 (query vs design), §6.5 (chip constraints), §5 (B8); files: viewer/src/*.
GOAL: Design a board with your hands, without ever touching the gate's assumptions.
DELIVERABLES: viewer/src/xr/{controllers.js, grab.js, snap.js, probe.js, brush.js, viaStamp.js, undo.js, comfort.js}.
SPEC: ray + grip; grab by ray at distance, direct within 10 cm; 1 mm snap on the board plane; ghost footprint; non-overlap
(≥ 2 mm) and edge clearance (≥ 3 mm); while dragging send query at ≤ 30 Hz; on release send design (commit); probe pen
shows T, band and a flux arrow, haptic intensity from T; copper brush per layer (radius from thumbstick); via-array stamp;
20-level undo; teleport, seated reference space, snap turning; desktop parity (mouse/keyboard) for every action.
TESTS FIRST: dragging emits only query messages; release emits exactly one design; constraints reject illegal placements.
DONE WHEN: works in a headset and on desktop; server log confirms message types.
REPORT BACK: module list with the events each emits.
```

**Prompt G5.4 · measured overlay (T1.9) and hardware analysis (with M)**
```
READ FIRST: CLAUDE.md §14 (rig, steady state, fit on board 1 only, H10), §10.5 (hardware record), §13 (T1.9);
files: hardware/boards/*, solvers/fvm.py, docs/refs/camera_datasheet.pdf.
GOAL: Compare the verified solver with real boards honestly, and show it in the viewer.
DELIVERABLES: hardware/analysis/{register.py, steady.py, fit_convection.py (M finishes), compare.py}; viewer/src/overlay/projector.js.
SPEC: register = fiducial detection (template match + manual-click fallback) → homography to board mm → bilinear resample onto
the 256² L1 grid with emissivity correction; steady = hotspot change < 0.1 K/min over 5 min; fit = h_top, h_bot in [2, 50] on
board 1 only (raise for any other id), solver in the loop, bootstrap uncertainty, write hardware/results/fit.json; compare =
held-out hotspot error (K), field RMSE, uncertainty budget (camera accuracy, emissivity, V·I, ambient, fit); viewer projects
the registered frame onto L1 with a difference map.
TESTS FIRST: a synthetic frame made from a known field + homography registers back within 1 mm at fiducials; fit raises on board 2.
DONE WHEN: E17 table produced; fit.json committed BEFORE compare.py runs on boards 2–3.
REPORT BACK: files, E17 summary.
```

**Prompt G5.5 · pilot session script (chat)**
```
READ FIRST: CLAUDE.md §13 (pilot), study/prereg.md {{RUNTIME: paste if in chat}}.
GOAL: Identical, safe, well-run sessions.
PRODUCE: a 45-minute script: consent signed before any data; 5-minute desktop training on non-study boards; 3 condition
blocks in the participant's Latin-square order, each Task A (30 decisions, no feedback) + Task B (5 min, fixed solve
budget) + NASA-TLX; debrief; exact experimenter wording; pre-session checklist (headset cleaned, logs empty, order loaded,
participant code assigned); stop rules (discomfort → stop, offer desktop); post-session data checklist (export, verify,
back up, delete local copy).
```

**Prompt G5.6 · pilot analysis exactly as pre-registered**
```
READ FIRST: CLAUDE.md §13 (measures); files: study/prereg.md, study/stimuli.json, session logs.
GOAL: Results that are credible because they follow the pre-registration exactly.
DELIVERABLES: study/analysis.py → study/results/{primary.csv, exploratory.csv, figs/}.
SPEC: outcomes per participant × condition: accepted-wrong rate, unnecessary-verification rate, verified peak (Task B), time,
NASA-TLX; primary = paired effect sizes (calibrated vs flag, calibrated vs none) with bootstrap 95 % CIs (10,000 resamples,
fixed seed); only the pre-registered exclusions; anything else goes to exploratory.csv; paired dot-plot figure with footer.
DONE WHEN: script runs end to end on the real logs.
REPORT BACK: primary table.
```

**Check it worked**
- [ ] Every T1 feature in the technique checklist with a screenshot; E10 frame-time log
- [ ] Particles drain through vias; blackbody test passes
- [ ] VR message-type tests pass
- [ ] Ethics approval on file before session 1; dry run; 8–10 sessions; logs backed up; analysis as pre-registered
- [ ] P4 delivered to M
- [ ] E17: steady state for every frame used; fit on board 1 only; held-out table + budget

**If it goes wrong**
- Headset < 72 fps → 131k particles, half-res bloom, opaque layers; measure after each change.
- No headset → desktop study, stated; VR becomes a demo only.
- No ethics approval by week 9 → no study data; H4 → future work; P4 dropped; logged.
- A participant feels unwell → stop at once; data excluded per the pre-registration.
- **Hardware safety** → current-limited supply below `heaters.csv` max, non-flammable standoffs, never unattended, camera check before raising power. Smell or smoke → cut the power first.
- Disagreement > max(2 K, 10 %) → check emissivity, V·I, ambient, fit; then report model-form error as a finding.
- No boards → T1.9 with a sample image; E17 to the journal extension.

## M · real server, convection fit, P4 traces into the gate

**In simple words.** Replace the fake server with the real one (network → gate → GPU verifier → streamed results), help G match the solver to real boards using board 1 only, and run the human sessions through every gate as proposer P4.

**What it actually does.** `server/app.py` v1 (query/design, gate bridge, streaming, back-pressure) · `hardware/analysis/fit_convection.py` · `experiments/p4_import.py` + P4 in E7/E15.

**Prompt M5.1 · server v1**
```
READ FIRST: CLAUDE.md §10.3 (all messages), §8.1 (loop), §5 (B6, B8); files: server/app.py v0, pipeline/gate_bridge.py,
certify/pcg_batched.py, models/ensemble.py.
GOAL: The production server for the viewer, the study and the demo.
DELIVERABLES: server/app.py v1, server/iterate_stream.py, server/health.py, tests/test_server_v1.py.
SPEC: query → ensemble → pred (decision "preview"), with NO gate call and NO coin; design → coin.commit(id) → ensemble →
gate_bridge → pred (accept | flag | audit) → if flag/audit stream iterate every k iterations then verified; audit-log row per
design; one GPU worker process with a priority queue (design > query); per client drop queued queries older than the newest;
if the client is slow drop intermediate iterates but never verified; t_server_ns on every message; dev mode validates every
outgoing message and crashes on failure; /health returns GPU memory and queue lengths.
TESTS FIRST: query×20 → design → pred, ≥ 1 iterate, verified in order; a query leaves gate.state_dict() unchanged; no coin
appears in any message before its design's pred; 10-minute soak with simulated dragging shows flat latency and no memory growth.
DONE WHEN: tests pass; G's viewer works against it.
REPORT BACK: files, soak-test summary.
```

**Prompt M5.2 · convection fit (with G)**
```
READ FIRST: CLAUDE.md §14, §11.4 (H10); files: hardware/analysis/{register.py, steady.py, fit_convection.py draft},
hardware/boards/board1.json, solvers/fvm.py.
GOAL: Fit the one unknown (convection) on board 1 only, so boards 2–3 stay an honest test.
DELIVERABLES: hardware/analysis/fit_convection.py (final), hardware/results/fit.json.
SPEC: parameters (h_top, h_bot) ∈ [2, 50] W/(m²·K); model = 4-layer solver L1 field for board 1 at each measured power (V·I);
data = registered frames (K rise over ambient), coated area only; weighted least squares (weights from camera noise);
scipy least_squares with cached solves; uncertainty by bootstrap over frames (200 resamples); residual map per power level;
raise if any board other than board 1 is passed. Commit fit.json before compare.py runs.
DONE WHEN: fit.json committed; values in a plausible natural-convection range (≈ 5–15) or the deviation explained.
REPORT BACK: fitted values with intervals, residual maps.
```

**Prompt M5.3 · P4 into the gate experiments**
```
READ FIRST: CLAUDE.md §11.2 (P4), §8.2 A1, §10.6, §11.4 (H1, H8); files: viewer session logs, gate/proposers.py (Replay),
experiments/run_gates.py, experiments/oracle_label.py.
GOAL: Show the guarantee for the human proposer.
DELIVERABLES: experiments/p4_import.py; updates to run_gates.py, e7_validity.py, e15_groups.py; figure "accepted-error rate,
human proposer alone: G1 vs G5 vs G6".
SPEC: one row per committed design in time order, participant code as a sub-id, condition recorded; assert every commit precedes
its coin in the log (else drop that session from P4 and report it); send to oracle_label.py; replay offline like P1–P3; add
"P4 only" rows to E7 and E15.
DONE WHEN: P4 rows and the figure exist.
REPORT BACK: P4 counts and results with intervals.
```

**Check it worked**
- [ ] Server tests + soak test pass; `query` never changes gate state
- [ ] `fit.json` committed before the held-out comparison (git history)
- [ ] P4 imported, labelled, in E7/E15 with its own figure

**If it goes wrong**
- Server stalls in demos → check the queue rules; never train on the demo GPU.
- Fitted h hits a bound → a measurement or model issue (emissivity, standoff contact, power). Investigate before widening; report as model-form error.
- Few P4 designs → report with wide intervals as indicative. Never pool humans with optimisers to hide it.

## N · server performance, CUDA particle twin, DDP scaling

**In simple words.** Keep the live pipeline fast under real use, build a CUDA twin of G's particle system for a clean GPU-vs-browser comparison, and measure multi-GPU training scaling.

**What it actually does.** NVTX/CUDA-event profile of the server path + load generator · `bench/particles_cuda.py` · `bench/ddp_scaling.py`.

**Prompt N5.1 · server profiling**
```
READ FIRST: CLAUDE.md §13 (latency targets), §11.5 (E11); files: server/*, pipeline/*.
GOAL: Find and remove the slow parts of the live path.
DELIVERABLES: NVTX ranges in server/app.py (rasterise, inputs, ensemble, fused gate, scan, PCG per check, encode, send);
bench/server_profile.py (asyncio load generator: 1, 4, 16 users, each dragging at 30 Hz and committing every 3 s, 5 minutes);
results CSV + stacked stage-time bar + latency CDF.
SPEC: P50/P95/P99 per stage, GPU utilisation (pynvml), queue lengths; then measure the top 3 optimisations (pinned memory,
CUDA streams overlap, GPU rasterisation / GPU-side float16 encoding) one at a time, before and after.
DONE WHEN: P95 design → first iterate < 100 ms at 1 user, or the bottleneck explained with numbers.
REPORT BACK: tables, before/after gains.
```

**Prompt N5.2 · CUDA particle twin**
```
READ FIRST: CLAUDE.md §9.6, §13 (T1.2); files: viewer/src/particles/* (G's), a readback export from G.
GOAL: The same algorithm in CUDA and WebGL, for a PAGP comparison.
DELIVERABLES: bench/particles_cuda.py (Numba): SoA state (x, y, age float32; layer int8); bilinear flux sampling from [L,N,N];
RK2 midpoint; inverse-CDF respawn with per-thread xoroshiro RNG; via hops ∝ vertical flux; benchmark 65k–1M particles ×
1,000 steps: particle-steps/s, achieved bandwidth, SoA vs AoS, float16 vs float32 flux.
TESTS FIRST: with the RNG disabled, one particle's path matches the shader's readback exactly; after 500 steps, per-layer density
histograms match the shader within 5 %.
DONE WHEN: table for the PAGP chapter.
REPORT BACK: table.
```

**Prompt N5.3 · DDP scaling**
```
READ FIRST: CLAUDE.md §12, §4 (Kaggle 2×T4); files: models/train_ensemble.py.
GOAL: Measured multi-GPU scaling for the PAGP chapter (P6).
DELIVERABLES: bench/ddp_scaling.py + Kaggle launch command.
SPEC: 200 steps of the 256² FNO on 1 and 2 GPUs (torch.distributed, NCCL, DistributedSampler, same global batch) after 20 warm-up
steps; samples/s, GPU utilisation, time in all-reduce (profiler); speed-up, efficiency, Amdahl serial fraction; weak scaling
with the global batch doubled; footer with GPU model and versions.
DONE WHEN: numbers reproducible by the script.
REPORT BACK: table + one-paragraph interpretation.
```

**Check it worked**
- [ ] Stage tables and CDFs for 1/4/16 users; at least one optimisation measured before/after
- [ ] Particle twin matches statistically; throughput table
- [ ] DDP speed-up and efficiency reported

**If it goes wrong**
- GPU-bound with many users → batch concurrent designs into one launch; previews may use one member (the gate always uses five).
- Twin mismatch → units (UV vs mm) or the RNG distribution; compare a deterministic path first.
- Poor DDP scaling → expected on T4 PCIe; report the all-reduce share.
- Profiler distorts timing → report numbers from runs without it.

---

# Phase 6 · Freeze, evaluation, release and report (weeks 10–12)

**Goal.** Every claim backed by a number that regenerates from one tagged release; each examiner sees a chapter, figures and a demo moment that are clearly theirs.
**Gate 6 (end of week 12):** all figures regenerated from `v3-freeze` with footers · novelty search repeated · TwinHeat-Traj v1 released (DOI, datasheet, licence, checksums) · three chapters + deck + backup video · two full rehearsals · workshop draft.

## N · clean-room re-run, PAGP figures and chapter

**In simple words.** Prove the results are real by re-running everything from scratch on a fresh machine from the frozen code and checking the same numbers come out. Then write the parallel-computing chapter.

**What it actually does.** `Makefile` + `scripts/run_all.sh` + `scripts/compare.py` · PAGP figures (roofline, solver comparison, certified stopping, designs/s vs batch, scan profile, DDP, particle twin) · `report/pagp.tex`.

**Prompt N6.1 · clean-room reproduction**
```
READ FIRST: CLAUDE.md §4 (figures, frozen configs, tags), §11.5 (E1–E18), §5 (hashes); files: Makefile, experiments/*,
bench/*, data/manifest.json, models/registry.json.
GOAL: One command reproduces every result from tag v3-freeze on a clean machine and flags any drift.
DELIVERABLES: Makefile targets e1 … e18 and all-experiments; scripts/run_all.sh; scripts/fetch_artifacts.py (downloads data +
checkpoints by hash from {{RUNTIME: Kaggle dataset / Drive locations}}, fails on mismatch); scripts/compare.py.
SPEC: create the env from environment.yml (or requirements.txt on Kaggle); verify_env.py; each target independent and resumable,
printing an estimated wall time up front; frozen configs only; 5 seeds; CSVs to results/<exp>/; figures with footers;
compare.py flags every metric outside the 95 % CI stored in results_reference/ and prints a pass/fail table.
DONE WHEN: a clean Kaggle/lab run passes compare.py, or each difference is explained and logged.
REPORT BACK: files and the compare table.
```

**Prompt N6.2 · PAGP chapter (chat)**
```
READ FIRST: CLAUDE.md §1, §9, §11.4 (H5, H6), §11.5. RESULTS: {{RUNTIME: E2, E3, E4, E13 tables, roofline numbers, DDP and
particle tables, server profile}}.
GOAL: A PAGP chapter (≤ 12 pages) where every syllabus topic is shown with a measured figure.
DO: (1) outline mapping sections to topics (data parallelism, memory hierarchy and tiling, reductions, scan/compaction,
batched solvers, streams/overlap, scaling laws), each naming its figure; (2) a 2-sentence caption per figure stating the
finding with its number; (3) a Limitations subsection from these facts: {{RUNTIME: list}}.
CONSTRAINTS: no numbers that are not in the pasted results; mark gaps [NEEDS DATA].
```

**Prompt N6.3 · PAGP viva drill (chat)**
```
READ FIRST: CLAUDE.md §8.4, §9. CHAPTER: {{RUNTIME: pagp.tex text}}.
Act as a strict PAGP examiner. Ask 10 questions, hardest first, including: why the stencil is memory-bound and what the
roofline shows; Blelloch scan work and depth; why batched PCG with stop masks needs re-compaction; the per-stage time split;
whether the CPU AMG baseline is fair; how FP64 rechecks interact with FP32 iterations. One question at a time; grade each answer
(0–3) and give a better 3-sentence answer.
```

**Check it worked**
- [ ] Clean-room run passes `compare.py`
- [ ] Every PAGP figure has a footer and a Makefile target
- [ ] Chapter has no `[NEEDS DATA]`; every number traced to a CSV by M or G
- [ ] Two viva drills done; weak answers rewritten into the notes

**If it goes wrong**
- The re-run differs → the re-run wins. Find the cause (seed, data version, dependency, nondeterministic kernel), fix, log, regenerate. Never report old numbers.
- Nondeterministic GPU results → report the variance over 5 runs, or use deterministic reductions.
- Short on time → cut prose, not evidence.

## M · TwinHeat-Traj v1 release, DL chapter, workshop draft

**In simple words.** Package ~100,000 designs with every gate's decision and the exact answer so anyone can test a trust gate against ours (with a permanent DOI). Write the DL chapter, precise about what is cited and what is ours, and draft a short workshop paper.

**What it actually does.** `release/build_release.py` + `verify_release.py` + datasheet + licences + DOI · `report/dl.tex` · workshop draft.

**Prompt M6.1 · release build and verification**
```
READ FIRST: CLAUDE.md §11.5 (E18), §10.4 (columns), §10.6 (anonymisation), §5; files: experiments outputs, gate/*.
GOAL: A benchmark anyone can reproduce our gate results from, with no private data.
DELIVERABLES: release/build_release.py; release/verify_release.py; release/traj_v1/{README.md, DATASHEET.md, LICENSE-DATA
(CC BY 4.0), LICENSE-CODE (MIT), CITATION.cff, SHA256SUMS}.
SPEC: parquet partitioned by task/proposer/aggressiveness/seed with board record, μ peak, σ_hot, r_hot, s, decisions G0–G7, coins
and probabilities, oracle peak, certified intervals, group labels; P4 participant codes remapped to P4-01…, absolute times →
relative ms; README quick start (load, replay G5, reproduce one figure); verify_release.py runs in a fresh directory
containing ONLY the release, re-runs all gates and reproduces E7/E15 summaries, failing on mismatch.
DATASHEET facts: {{RUNTIME: generation process, sizes, tolerances, limitations, intended and out-of-scope uses}} (Gebru et al.
questions; simulated vs measured stated clearly; no claim about real boards beyond E17).
DONE WHEN: verify_release.py passes; a second person checks 50 random P4 rows for identifying data.
REPORT BACK: files, verification output.
```

**Prompt M6.2 · DL chapter consistency check (chat)**
```
READ FIRST: CLAUDE.md §1 (claims and non-claims), §8 (lemmas), §11.4. DRAFT: {{RUNTIME: dl.tex text}}. RESULTS:
{{RUNTIME: E5, E7, E8, E9, E14, E15, E16 summaries}}.
GOAL: No mismatched numbers, no overclaims.
CHECK: (1) every quantitative claim against the results (list mismatches with line refs); (2) every novelty sentence against the
non-claims list (flag e.g. calling the audited ACI guarantee new); (3) Lemma 1 presented as an application of Gibbs & Candès +
Tu & Giesecke with step (1) and the high-probability form as ours; (4) G6 labelled "proved" or "empirical" consistently with
lemmas.tex. Propose safer wording for each flag. Do not add results.
```

**Prompt M6.3 · workshop draft (chat)**
```
READ FIRST: CLAUDE.md §1, §8, §11. MATERIAL: {{RUNTIME: final result tables, verified .bib keys}}.
GOAL: A 4-page NeurIPS ML4PS-style draft: Introduction, Related work, Method (two-layer certification: Lemma 1/1′, Lemma 2,
certified stopping), Experiments, Limitations. One main figure (validity curves G1 vs G5 vs G6 across proposers incl. the
human proposer) and one table (certified-stopping savings with the violation count). Claims exactly within CLAUDE.md §1;
cite only the given keys; mark missing items [TODO].
```

**Check it worked**
- [ ] `verify_release.py` passes in a clean directory; checksums; DOI minted (or scheduled right after submission)
- [ ] No identifying data in P4 rows (second-person check)
- [ ] DL chapter: zero mismatches, zero overclaims on the final PDF text; N re-read the lemma sections
- [ ] Workshop draft has no [TODO] in Method or Experiments

**If it goes wrong**
- A figure doesn't reproduce from the release → something outside it was used. Add it or fix the script. No publish until it reproduces.
- Licence or privacy doubts on human traces → release the optimiser trajectories now and P4 after ethics confirmation.
- No time → the workshop draft is the first cut.

## G · pilot results, CGVR chapter, deck, demo video, rehearsals

**In simple words.** Turn the viewer, study and real-board experiment into what examiners see: a CGVR chapter that names each technique and shows it working, a seven-minute live demo with a recorded backup, and a deck. You lead the rehearsals.

**What it actually does.** Pilot results section · `report/cgvr.tex` + technique checklist · deck · demo script + backup video · rehearsals.

**Prompt G6.1 · pilot results section (chat)**
```
READ FIRST: CLAUDE.md §13 (pilot), §11.4 (H4). MATERIAL: {{RUNTIME: prereg.md and analysis.py output}}.
GOAL: An honest pilot section (≤ 1.5 pages).
WRITE: participant flow; primary outcomes with effect sizes and 95 % CIs exactly as computed; one sentence per pre-registered
hypothesis (supported / not supported / inconclusive); a separate Exploratory paragraph; cautious language for n = 8–10;
suggest a paired dot-plot figure. No new numbers.
```

**Prompt G6.2 · CGVR chapter and technique checklist (chat)**
```
READ FIRST: CLAUDE.md §13 (T1.1–T1.9). MATERIAL: {{RUNTIME: file paths per feature, E10/E11 numbers, E17 summary}}.
GOAL: A CGVR chapter (≤ 12 pages) where each section shows one technique we implemented, with a figure and a measured number.
DO: (1) table technique | where | our file | what we wrote vs what three.js provides | evidence; (2) chapter outline;
(3) "why perceptual colour maps" and "why the blackbody mapping is stylised" paragraphs (3 sentences each).
CONSTRAINTS: label anything from a library as such; mark gaps [NEEDS DATA].
```

**Prompt G6.3 · demo script and backup video (chat)**
```
READ FIRST: CLAUDE.md §13 (demo moments), §1.
GOAL: A 7-minute demo for three examiners (CGVR, PAGP, DL) that survives failures.
SEQUENCE: pull the board apart in VR → drag a chip → bound flares → field converges inside its guaranteed band → real thermal
image on top → throughput panel with boards leaving the batch → proposer replay (G1 drifts, G5/G6 hold, incl. the human proposer).
PRODUCE: per segment: time, speaker, screen, the one sentence each examiner must hear, and the live-failure fallback; then a shot
list for the backup video (1080p screen + headset mirror) with captions.
```

**Prompt G6.4 · CGVR viva drill (chat)**
```
READ FIRST: CLAUDE.md §13, §8.4. CHAPTER + SCRIPT: {{RUNTIME: paste}}.
Act as a strict CGVR examiner. Ask 10 questions, hardest first, including: how weighted blended OIT works and when it fails;
RK2 vs Euler for particles; how JFA computes a distance field and its error; what VSUP does that a bivariate map doesn't; how
the homography is estimated from 4 fiducials; why the "guaranteed band" is guaranteed. One at a time; grade (0–3) and give a
better 3-sentence answer.
```

**Check it worked**
- [ ] Pilot numbers match `analysis.py` exactly; exploratory content separated
- [ ] Technique checklist complete; no [NEEDS DATA]
- [ ] Backup video recorded and checked offline on a second machine
- [ ] Two rehearsals under 7 minutes, including a failure drill
- [ ] Demo laptop works offline (local server, cached data); spare laptop ready

**If it goes wrong**
- Headset fails → desktop mode, "same features", backup video for the VR moment.
- Network fails → local server on the demo laptop (rehearsed once).
- Null pilot → report a well-run null with effect sizes and a power estimate.
- Short on time → cut deck polish, never the video or rehearsals.

---

# Mess-up recovery playbook

| Symptom | Likely cause | First move | Owner |
|---|---|---|---|
| `main` broken after a merge | untested PR / skipped GPU test | `git revert` the merge; fix on a branch | merger |
| Results changed, nobody knows why | dataset version, seed, dependency | compare figure footers; re-run from the tag | figure owner |
| Lost checkpoint | saved only to session disk | resume from Drive `last.pt`; else retrain member 1 and time it | M |
| Dataset wrong after training began | late sampler/solver bug | freeze training; regenerate as a new version; retrain; log | N + M |
| Certificate violation (B7) | FP32 residual, missing margin, z not positive, non-M-matrix | stop certificate results; check in that order; re-run E13 | N |
| Gate breaks its target | coin visible, flagged labels in q, tiny n_a | run B8; review the audit log; check frozen config | M |
| Claude code fails subtly | hallucinated API, units, hidden assumption | failing test first, then the debug prompt below | author |
| Claude cites a paper wrongly | model error | open the paper; mark U until checked | citer |
| A member out for a week | illness, exams | Monday re-plan with the cut list; others take gate-critical items only | all |
| Two people edited the same file | unclear ownership | ownership table decides; owner merges | owner |
| Demo hardware dies | — | backup video + desktop mode + spare laptop | G |
| Scope creep | enthusiasm | not before week 10 → `docs/future_work.md` | all |
| Behind overall | accumulated slips | cut list: LIC → procedural traces → Darcy validity-only → G7 offline → pilot of 6 → two boards | all |

Rules: fail loudly · one change at a time when debugging (commit the failing test first) · an incident line in `decisions.md` the same day · ask for help after 2 hours stuck.

**Debug prompt (Claude Code)**
```
READ FIRST: CLAUDE.md §0, §5 and the sections for the module involved: {{RUNTIME: e.g. §7.2, §9.1}}.
BUG: {{RUNTIME: one-line symptom}}. EXPECTED: {{RUNTIME}}. OBSERVED (numbers): {{RUNTIME}}.
REPRODUCTION: {{RUNTIME: smallest command}}. OUTPUT: {{RUNTIME: full error / test output}}. RECENT CHANGES: {{RUNTIME: git diff --stat}}.
DO NOT rewrite the module. (1) Rank the 3 most likely causes, each with a quick check that confirms or rules it out.
(2) After I report the checks, propose the smallest fix + a regression test that would have caught it.
(3) Say whether any invariant B1–B8 is implicated.
```

**Weekly planning prompt (chat, Monday)**
```
READ FIRST: CLAUDE.md §2, §15, §17. CONTEXT: week {{RUNTIME: k}}/12. Status N: {{RUNTIME}}; M: {{RUNTIME}}; G: {{RUNTIME}}.
Gates passed: {{RUNTIME}}. Open incidents: {{RUNTIME}}. This week's plan items: {{RUNTIME: paste from the phase sections}}.
PRODUCE: (1) per-person Mon–Fri tasks with hours (≤ 25 h each); (2) hand-off points with the file/contract used; (3) which gate is
at risk and the earliest signal we're ignoring; (4) which §15 fallback should trigger now; (5) the tasks to drop, in order, if
someone loses 2 days. Flag any task that silently depends on unfinished work. Be blunt.
```
