# Contracts v1.0

The data agreements between the solver/pipeline (N), the model/gate (M) and the viewer/hardware/study (G) lanes.
Source: CLAUDE.md §10, plus §7.5 for model I/O and §5 B6/B8. This page describes the machine-checkable files and
records every interpretation made while turning §10 into schemas. **It becomes binding when all three people sign
the table at the end, and the commit is tagged `v0-contracts`.**

| What | Where |
|---|---|
| Schemas (JSON Schema draft 2020-12) | `contracts/schemas/<kind>.json` |
| Python validator | `contracts/validate.py`: `validate(obj, kind)` raises `ContractError` (`.kind`, `.path`, `.message`) |
| JS validator (ajv 8.20.0, same schema files, imported not copied) | `viewer/src/contracts.js`: `validate(obj, kind)` throws `ContractError` (`.kind`, `.path`, `.detail`) |
| float16 field codec | `contracts/f16.py` and `viewer/src/f16.js` |
| Examples (3 valid + 3 invalid per kind) | `contracts/examples/<kind>/`; expected error paths in `contracts/examples/expected_errors.json` |
| Cross-language float16 vectors | `tests/fixtures/f16_vectors.json` |
| Tests | `tests/test_contracts.py` (pytest), `viewer/test/contracts.test.js` (`cd viewer && npm test`); CI runs both |

Kinds: `board_record`, `query`, `design`, `pred`, `iterate`, `verified`, `batch_progress`, `error`, `audit_log_row`,
`hardware_record`, `session_log_line`.

## 1 · Rules that apply everywhere

- Every record and message has `"schema_version": "1.0"`, and it is required.
- Every object has `additionalProperties: false`. An unknown field is an error, never ignored.
- Every field listed is **required**. Where a value can be absent, the field is present and `null` (the meaning of
  `null` is given per field below). The only optional field is `t_server_ns`, which any server message may carry.
- Errors report the JSON path of the **first** error, e.g. `$.record.layers[0].shapes[1].r`. For a missing or
  unexpected field the path is the object that holds it, and the message names the field.
- Units: geometry in **mm** in board and hardware records (origin lower-left, x right, y up, board 100 × 100 mm);
  temperatures in messages are a **rise above ambient in K**; hardware temperatures are absolute **°C**.
- Server dev mode validates every outgoing message and crashes on failure (plan, M section).

### Field arrays (`mu`, `sd`, `theta`)

- Base64 (standard alphabet, padded) of **little-endian IEEE float16**, row-major `[layer, y, x]`. y index 0 is the
  south (bottom) edge and x index 0 the west (left) edge (CLAUDE.md §0 rule 7).
- The message carries one `"shape": [4, n, n]` that applies to all of its field arrays. Both validators also check
  what JSON Schema cannot: `shape[1] == shape[2]`, and every field decodes to exactly `2 · 4 · n · n` bytes.
- Encoding rounds to nearest, ties to even, straight from the input's precision (the JS encoder does not round through
  float32, so it cannot double-round). Python and JS produce **byte-identical** base64, which the shared vectors check.
- Both encoders refuse NaN, ±inf and values that overflow float16 (|x| ≥ 65520). Both decoders refuse a byte length
  that doesn't match the shape, a malformed shape, invalid base64, and inf/NaN halves.
- Python `decode` returns `np.float16`; JS `decode` returns a `Float32Array`. Every float16 value is exact in float32.

```python
from contracts import f16
from contracts.validate import validate
msg = {"schema_version": "1.0", "type": "iterate", "id": "d-2", "k": 40,
       "theta": f16.encode(theta), "cert_k": 0.08, "shape": list(theta.shape)}   # theta: [4, n, n] in K
validate(msg, "iterate")
```
```js
import { validate } from "./contracts.js";
import { decode } from "./f16.js";
validate(msg, "iterate");
const theta = decode(msg.theta, msg.shape);   // Float32Array, row-major [layer, y, x]
```

### What the schemas check, and what they don't

The schemas check **structure and physical validity** (types, enums, positivity, mm within the board, design vector in
[0, 1]). They do **not** check sampler training ranges such as chip power 0.1–3 W or 1–4 chips. Those ranges belong to
the runtime OOD rule check (CLAUDE.md §6.7), and OOD records (`ood_3oz`, 5–8 chips, …) must validate. Cross-field
rules the schemas don't encode (a shape staying inside the board, chip spacing, `mode` vs `design_vector`) belong to
the sampler tests.

## 2 · Board record (`board_record`, §10.1)

| Field | Type | Unit / meaning |
|---|---|---|
| `id` | non-empty string | |
| `seed` | integer ≥ 0 | |
| `mode` | `id` \| `design` \| `ood_<name>` (`<name>` = lowercase letters, digits, `_`) | |
| `stack.outer_oz`, `stack.inner_oz` | number > 0 | oz copper |
| `stack.k_fr4_xy` | number > 0 | W/(m·K) |
| `layers` | exactly 4 items, in the order L1, L2, L3, L4 | |
| L1, L4 | `{layer: 1 \| 4, shapes: [rect \| disc \| trace]}` | |
| L2 | `{layer: 2, plane: bool, cutouts: [rect]}` | |
| L3 | `{layer: 3, split_rects: [rect]}` | |
| `rect` | `{type: "rect", x0, y0, w, h}` | mm; x0, y0 ∈ [0, 100]; w, h ∈ (0, 100] |
| `disc` | `{type: "disc", cx, cy, r}` | mm; r ∈ (0, 100] |
| `trace` | `{type: "trace", points: [[x, y], …] (≥ 2), width}` | mm; segments may be at any angle |
| `chips[]` (≥ 1) | `{x0, y0, w, h, P, side}` | mm, P in W (> 0), side `top` \| `bottom` |
| `via_arrays[]` | `{x0, y0, w, h, density_per_mm2}` | mm, vias/mm² (≥ 0) |
| `background_via_density_per_mm2` | number ≥ 0 | vias/mm² |
| `convection` | `{h_top, h_bot, forced_zone}` | W/(m²·K), both > 0 |
| `convection.forced_zone` | `null` or `{x0, y0, w, h, h_top}` | mm, W/(m²·K) |
| `dirichlet_edges` | unique subset of `W`, `E`, `S`, `N` | heat-sunk edges |
| `design_vector` | `null` or exactly 36 numbers in [0, 1] | |

## 3 · Model I/O (§10.2, §7.5)

These are tensors, not JSON, so they have no schema file. Input `[B, 13, 256, 256]` float32, in the channel order of
CLAUDE.md §7.5. Output `[B, 4, 256, 256]` float32 = θ per layer in K after denormalisation. When a field crosses the
wire it is encoded as in §1 above.

## 4 · WebSocket messages (§10.3)

Client → server:

| `type` | Fields | Notes |
|---|---|---|
| `query` | `id`, `record` (a full board record) | Prediction only. Never touches gate state, never draws a coin (B8). A query cannot carry gate fields because the schema rejects any extra field. |
| `design` | `id`, `record` | Commit. The server calls `coin.commit(id)` first, then gates. |

Server → client (each may also carry `t_server_ns`, an integer ≥ 0 from the server's monotonic clock in ns, for E11):

| `type` | Field | Type | Unit / meaning |
|---|---|---|---|
| `pred` | `id` | string | |
| | `mu`, `sd` | field arrays | K |
| | `shape` | `[4, n, n]` | |
| | `B` | number ≥ 0 or `null` | K; `null` = no finite bound (no gate yet, as in server v0, or q_t = +∞) |
| | `decision` | `preview` \| `accept` \| `flag` \| `audit` | `preview` only answers a `query` |
| | `group` | non-empty string or `null` | `null` = no group assigned |
| | `peak` | number | K |
| | `units` | `{mu: "K", sd: "K", B: "K", peak: "K"}` | |
| `iterate` | `id`, `k` (integer ≥ 0), `theta`, `shape` | | `k` = PCG iteration |
| | `cert_k` | number ≥ 0 or `null` | K, the Lemma 2 bound on every cell at iterate k; `null` = this board has no certificate (§8.4 check failed), so the viewer must not draw a "guaranteed band" |
| `verified` | `id`, `theta`, `shape` | | |
| | `cert` | number ≥ 0 or `null` | K; `null` = no certificate (solved to tolerance) |
| | `iters` | integer ≥ 0 | PCG iterations |
| | `ms` | number ≥ 0 | wall-clock ms |
| | `stop_reason` | non-empty string | see proposed addition A3 |
| `batch_progress` | `batch_id` | string | |
| | `active_ids` | unique strings | |
| | `done[]` | `{id, iters, cert (≥ 0 or null), reason}` | K for `cert` |
| | `stage_ms` | `{gate, scan, pcg, scatter}`, each ≥ 0 | ms |
| | `designs_per_s` | number ≥ 0 | certified designs per second |
| `error` | `id` | string or `null` | `null` = not tied to a request (e.g. the incoming message didn't parse) |
| | `message` | non-empty string | |

## 5 · Audit log row (`audit_log_row`, §10.4)

One CSV row per committed proposal. The schema validates a row **after parsing to typed values**: an empty cell is
`null`, `audited` is a boolean, and numbers are numbers. Column order is `schema_version` followed by the §10.4 order.
Every column is always present.

| Column | Type | Unit / meaning |
|---|---|---|
| `schema_version` | `"1.0"` | |
| `run_id` | string matching `<config>-<7–40 hex commit>-<seed>` | CLAUDE.md §4 |
| `gate` | `G0` … `G7` | |
| `seed`, `t` | integer ≥ 0 | |
| `proposer` | `P1` … `P4` | |
| `aggressiveness` | `1` \| `2` \| `3` \| `null` | `null` for P4, which has no knob |
| `group` | non-empty string or `null` | |
| `x_hash` | SHA-256, 64 lowercase hex characters | |
| `s` | number > 0 | K |
| `r_hot`, `sd_hot` | number ≥ 0 | K |
| `q_t` | number ≥ 0 or `null` | dimensionless; `null` = gate has no q (G0, G1) or q = +∞ |
| `B` | number ≥ 0 or `null` | K; same `null` meaning |
| `decision` | `accept` \| `flag` \| `audit` | |
| `p_t` | number in [0, 1] | |
| `audited` | boolean | |
| `T_lo`, `T_hi` | number or `null` | K; `null` on unaudited rows |
| `S_hi` | number ≥ 0 or `null` | dimensionless |
| `err` | `0` \| `1` \| `null` | |
| `alpha_t` | number or `null` | `null` for gates without ACI |
| `iters_used` | integer ≥ 0 | |
| `oracle_peak` | number or `null` | K; offline only |

## 6 · Hardware record (`hardware_record`, §10.5)

| Field | Type | Unit |
|---|---|---|
| `board_id`, `record_id`, `camera_frame_id`, `camera_model` | non-empty string | |
| `power_per_resistor[]` (≥ 1) | `{ref, V ≥ 0, I ≥ 0}` | V, A |
| `ambient_C` | number | °C |
| `emissivity` | number in (0, 1] | |
| `sensor_readings[]` | `{id, x_mm, y_mm, T_C}` | mm in [0, 100], °C |
| `steady_state` | `{criterion: "<0.1 K/min over 5 min", met: bool}` | the criterion string is fixed |
| `timestamp` | ISO 8601 with time zone, e.g. `2026-11-20T14:32:05+05:30` | |

## 7 · Session log line (`session_log_line`, §10.6)

| Field | Type | Meaning |
|---|---|---|
| `participant` | `P` + 2–3 digits, e.g. `P07` | a code, never a name; the pattern rejects names |
| `condition` | non-empty string | see proposed addition A3 |
| `t_ms` | number ≥ 0 | ms since session start |
| `action` | `grab` `move` `release` `brush` `via_stamp` `undo` `request_verify` `accept` `query` `commit` | |
| `design_hash` | SHA-256 hex or `null`; **required non-null when `action` is `commit`** | same hash as `x_hash`, so logs join |
| `server_decision` | `preview` \| `accept` \| `flag` \| `audit` \| `null` | `null` = no server decision for this action |

## 8 · Interpretations made while writing the schemas (need sign-off)

§10 leaves each of these points open. Each choice below adds no field and is the narrowest reading I could find.
Disagree here before signing.

1. **Where `shape` goes.** One top-level `shape` per message, shared by `mu` and `sd`. The first dimension is fixed to
   4. n can be any value ≥ 1 (examples use 4 and 8, production uses 256), and n × n squareness is checked in code.
2. **`units`** appears only on `pred`, the only message §10.3 gives it to. Its keys are the fields that carry units,
   all `"K"`. Other messages follow the units in this document (A1 below).
3. **Infinity.** JSON can't encode +∞, so `B` and `q_t` use `null` for "no finite bound" (q_t = +∞ when α_t ≤ 0,
   or no gate yet). The plan's server v0 already sends `B: null`. In the audit log, `null` also covers "this gate has
   no q" (G0, G1). See A2.
4. **No certificate.** `cert_k`, `cert` and `done[].cert` are nullable, because §8.4 has a "no certificate, solve to
   tolerance" path and the server must still send `iterate`, `verified` and `batch_progress` for those boards.
5. **Other nullable fields:** `pred.group`, `error.id`, `session_log_line.server_decision`, the audit-log columns
   that only exist on audited rows (`T_lo`, `T_hi`, `S_hi`, `err`), `oracle_peak` (online rows), and `alpha_t`, `q_t`,
   `B` (gates without them).
6. **Free strings, not enums,** for `verified.stop_reason`, `batch_progress.done[].reason` and
   `session_log_line.condition`. §10 names no values, so writing an enum would invent one (A3).
7. **`mode`** is a pattern, `id | design | ood_<name>`, because §6.7 names OOD-1…7 by number only (A3).
8. **Schema = structure + physics, not training ranges** (see §1). Consequences: `outer_oz`, `inner_oz` > 0 with no
   enum; ≥ 1 chip; coordinates in [0, 100] mm; sizes > 0. That makes the all-zero placeholder cut-out in the §10.1
   example invalid on purpose. `h_top`, `h_bot` > 0 (needed for well-posedness, §6.2); emissivity in (0, 1].
9. **Layers** are exactly 4, in the order L1–L4, and each layer only accepts its own fields (L2 `plane` + `cutouts`,
   L3 `split_rects`, L1/L4 `shapes`).
10. **Traces** may have diagonal segments (OOD-3 has 45° traces). Axis alignment is a sampler rule, not a contract
    rule.
11. **The audit log gets a `schema_version` column**, because §10's general rule says every record carries one. It
    is the first column. The schema validates typed rows (empty cell = `null`). `decision` excludes `preview`,
    since queries are never logged.
12. **Hashes** (`x_hash`, `design_hash`) are SHA-256 lowercase hex, the same algorithm as `data/manifest.json`.
13. **`aggressiveness`** is an integer level 1–3, or `null` for P4.
14. **`participant`** must match `P` + 2–3 digits, which is how "never a name" is enforced.
15. **`timestamp`** is checked with a regex for ISO 8601 with a time zone, not with `format: date-time`, because
    neither validator asserts `format` by default and the two would disagree.
16. **`id` on `query`/`design`** is the message id and isn't required to equal `record.id`.
17. **Error paths** point at the object for missing or extra fields, and at the value otherwise. Both validators
    report the same first path for every invalid example (the tests check this).

## 9 · Proposed additions (not in v1.0; each needs a §10 change and all three signatures)

- **A1 · Units on more messages.** Add `units` to `iterate` (`theta`, `cert_k`), `verified` (`theta`, `cert`, `ms`) and
  `hardware_record` (`V`, `I`). The field names alone don't state these units.
- **A2 · Distinguish +∞ from "not applicable".** For example, a string `"inf"` allowed in `B`/`q_t`, or a boolean
  `q_infinite`. Today `null` means both.
- **A3 · Enums for free strings.** `stop_reason`/`reason` ∈ {`cert_tau_v`, `audit_decided`, `audit_width`, `tol`,
  `maxit`} (from §8.1 and §8.4, plus the D14 cap); `condition` ∈ {`none`, `residual`, `calibrated`} (§13); OOD mode
  names for OOD-1…7.
- **A4 · Flagged and audited at once.** A design can be both (§8.1 step 5), but `pred.decision` holds one value.
  Either fix the precedence (suggestion: `audit` wins, since only audits update q) or add `audited: bool` to `pred`.
  Showing the coin in `pred` is consistent with B8 only because `pred` for a design is sent after commit.
- **A5 · Intersecting groups.** §8.5 allows a design to belong to several groups, but `pred.group` and the
  audit-log `group` hold one string. Suggest `groups: [string]`.
- **A6 · Annulus shape.** OOD-3 needs annuli, which `rect`/`disc`/`trace` can't express (L-shapes can be made from
  two rects).
- **A7 · `t_server_ns` precision.** JS numbers are exact only up to 2⁵³ ns (≈ 104 days of monotonic clock).
  Safer: send it as a decimal string, or as µs.
- **A8 · `mode` ↔ `design_vector`.** Require `design_vector` to be non-null exactly when `mode` is `design`.

None of the open decisions in CLAUDE.md §17 is settled here. D9 (what a `query` preview may show) does not affect v1.0,
because `pred` carries no q_t.

## 10 · Changing a contract

Per CLAUDE.md §10: all three signatures below, plus a version bump. Additive changes bump the minor version (1.1).
Existing fields never change meaning. One PR updates the schema, the examples, `expected_errors.json`, both validators'
tests and this page together. Both suites must pass in CI. Per the plan, if an argument runs past day 4, M decides and
the objection is logged in `docs/decisions.md`.

## Signatures (v1.0)

| Role | Name | Date | Signature |
|---|---|---|---|
| N · numerics and parallel lead | | | |
| M · ML lead | | | |
| G · graphics, VR and hardware lead | | | |
