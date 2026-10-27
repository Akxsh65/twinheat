// Vite config. The viewer imports the shared schemas from ../contracts/schemas (one source of
// truth for Python and JS, invariant B6), so the dev server must be allowed to read the repo root.
import path from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig } from "vite";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

export default defineConfig({
  server: { fs: { allow: [repoRoot] } },
  worker: { format: "es" }, // src/net.worker.js is an ES module (imports net.js, contracts.js)
});
