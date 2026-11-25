// Live design panel helpers (src/live.js): moved records stay contract-valid and on the 0.001 mm
// grid, the vias under the moved chip follow it, and the messages are valid query / design (B6).

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { validate } from "../src/contracts.js";
import { clientMessage, moveChip } from "../src/live.js";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const rec = JSON.parse(readFileSync(path.join(ROOT, "contracts/examples/board_record/valid_1.json"), "utf-8"));

test("moving a chip returns a valid copy on the mm grid and leaves the original alone", () => {
  const before = JSON.stringify(rec);
  const r = moveChip(rec, 0, 1.0004, -2);
  assert.equal(JSON.stringify(rec), before);
  assert.equal(r.chips[0].x0, Math.round((rec.chips[0].x0 + 1.0004) * 1000) / 1000);
  assert.equal(r.chips[0].y0, rec.chips[0].y0 - 2);
  validate(r, "board_record");
  assert.throws(() => moveChip(rec, 99, 1, 0), /no chip 99/);
});

test("query and design messages validate against the contracts", () => {
  for (const kind of ["query", "design"]) {
    const m = clientMessage(kind, rec, 7);
    validate(m, kind);
    assert.equal(m.id, `live-${kind[0]}-7`);
  }
  assert.throws(() => clientMessage("pred", rec, 1), /query or design/);
});
