"""Physical constants the viewer needs, exported from the spec YAMLs (G2.1; CLAUDE.md §7.2, §13).

    python -m server.viewer_constants      # rewrites viewer/src/spec_constants.json

The viewer computes the heat carried by each via, P = k_Cu · A_v · (θ_l − θ_m) / d_lm (§7.2, §13
T1.1), so it needs k_Cu, the via barrel area A_v and the gap thicknesses. They are never typed
into JavaScript: this script copies them from config/spec_v2.yaml (A_v as computed by
twinheat/spec.py), and tests/test_viewer_constants.py fails if the committed JSON drifts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from twinheat import spec as specmod

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "viewer" / "src" / "spec_constants.json"


def constants() -> dict[str, Any]:
    s = specmod.load(ROOT / "config" / "spec_v2.yaml")
    return {
        "source": "config/spec_v2.yaml via server/viewer_constants.py; do not edit by hand",
        "L_m": s.geometry.L.value,
        "k_cu_W_per_mK": s.materials.k_cu.value,
        "via_area_m2": s.via_area,
        "gap_thickness_m": [s.stack.d12.value, s.stack.d23.value, s.stack.d34.value],
    }


def main() -> None:
    OUT.write_text(json.dumps(constants(), indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
