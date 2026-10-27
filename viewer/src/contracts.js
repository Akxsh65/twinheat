// Validate records and messages against contracts/schemas (CLAUDE.md §10, invariant B6).
//
// validate(obj, kind) throws ContractError carrying the JSON path ("$.layers[0].shapes[1].r") of
// the first error. After the schema it runs the two checks JSON Schema cannot express, identically
// to contracts/validate.py: field-array shapes are square ([4, n, n]) and each base64 field
// decodes to exactly 2 * prod(shape) bytes. The schemas are imported, not copied, so Python and JS
// always validate against the same files.

import Ajv2020 from "ajv/dist/2020.js";

import auditLogRow from "../../contracts/schemas/audit_log_row.json" with { type: "json" };
import batchProgress from "../../contracts/schemas/batch_progress.json" with { type: "json" };
import boardRecord from "../../contracts/schemas/board_record.json" with { type: "json" };
import design from "../../contracts/schemas/design.json" with { type: "json" };
import error from "../../contracts/schemas/error.json" with { type: "json" };
import hardwareRecord from "../../contracts/schemas/hardware_record.json" with { type: "json" };
import iterate from "../../contracts/schemas/iterate.json" with { type: "json" };
import pred from "../../contracts/schemas/pred.json" with { type: "json" };
import query from "../../contracts/schemas/query.json" with { type: "json" };
import sessionLogLine from "../../contracts/schemas/session_log_line.json" with { type: "json" };
import verified from "../../contracts/schemas/verified.json" with { type: "json" };

const SCHEMAS = {
  board_record: boardRecord,
  query,
  design,
  pred,
  iterate,
  verified,
  batch_progress: batchProgress,
  error,
  audit_log_row: auditLogRow,
  hardware_record: hardwareRecord,
  session_log_line: sessionLogLine,
};
export const KINDS = Object.freeze(Object.keys(SCHEMAS));
const FIELD_ARRAYS = { pred: ["mu", "sd"], iterate: ["theta"], verified: ["theta"] };

export class ContractError extends Error {
  constructor(kind, path, detail) {
    super(`${kind}: ${path}: ${detail}`);
    this.name = "ContractError";
    this.kind = kind;
    this.path = path;
    this.detail = detail;
  }
}

const ajv = new Ajv2020({ strict: true, allErrors: false, allowUnionTypes: true });
for (const schema of Object.values(SCHEMAS)) ajv.addSchema(schema);
const VALIDATORS = Object.fromEntries(KINDS.map((k) => [k, ajv.getSchema(SCHEMAS[k].$id)]));

function jsonPath(pointer) {
  let out = "$";
  for (const raw of pointer.split("/").slice(1)) {
    const part = raw.replace(/~1/g, "/").replace(/~0/g, "~");
    out += /^(0|[1-9][0-9]*)$/.test(part) ? `[${part}]` : `.${part}`;
  }
  return out;
}

function base64Bytes(s) {
  if (s.length % 4 !== 0) return null;
  return (3 * s.length) / 4 - (s.length - s.replace(/=+$/, "").length);
}

function checkFieldArrays(obj, kind) {
  const [, ny, nx] = obj.shape;
  if (ny !== nx) {
    throw new ContractError(kind, "$.shape", `field arrays must be [4, n, n], got ${JSON.stringify(obj.shape)}`);
  }
  const expected = 2 * 4 * ny * nx;
  for (const field of FIELD_ARRAYS[kind]) {
    const got = base64Bytes(obj[field]);
    if (got !== expected) {
      throw new ContractError(
        kind,
        `$.${field}`,
        `decodes to ${got} bytes but shape ${JSON.stringify(obj.shape)} needs ${expected}`,
      );
    }
  }
}

/** Throw ContractError if obj does not satisfy contract `kind`; return undefined otherwise. */
export function validate(obj, kind) {
  const check = VALIDATORS[kind];
  if (!check) throw new ContractError(kind, "$", `unknown contract kind "${kind}"; known: ${KINDS.join(", ")}`);
  if (!check(obj)) {
    const e = check.errors[0];
    const extra = e.params && e.params.additionalProperty ? ` ("${e.params.additionalProperty}")` : "";
    throw new ContractError(kind, jsonPath(e.instancePath), `${e.message}${extra}`);
  }
  if (FIELD_ARRAYS[kind]) checkFieldArrays(obj, kind);
}
