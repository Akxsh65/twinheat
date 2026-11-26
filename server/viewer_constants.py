"""Physical constants the viewer needs, exported from the spec YAMLs (G2.1; CLAUDE.md §7.2, §13).

    python -m server.viewer_constants      # rewrites viewer/src/spec_constants.json

The viewer computes the heat carried by each via, P = k_Cu · A_v · (θ_l − θ_m) / d_lm (§7.2, §13
T1.1), so it needs k_Cu, the via barrel area A_v and the gap thicknesses. They are never typed
into JavaScript: this script copies them from config/spec_v2.yaml (A_v as computed by
twinheat/spec.py), and tests/test_viewer_constants.py fails if the committed JSON drifts.

The live optimiser demo (viewer/src/optimise.js) also needs the chip edge margin and gap of
CLAUDE.md §6.5 (config/spec_v1.yaml, as data/sampler.py uses them) and the P1-P3 settings of
experiments/configs/trajectories.yaml (lambda, step, T0, cooling ratio, eta, halvings), so the
demo runs the same rules with the same numbers as the offline trajectories.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from twinheat import spec as specmod

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "viewer" / "src" / "spec_constants.json"


TRAJECTORIES = ROOT / "experiments" / "configs" / "trajectories.yaml"


def proposer_settings() -> dict[str, Any]:
    """P1-P3 settings, per level 1..3, as experiments/run_trajectories.make_proposer reads them."""
    cfg = yaml.safe_load(TRAJECTORIES.read_text(encoding="utf-8"))
    p1, p2, p3 = cfg["P1"], cfg["P2"], cfg["P3"]
    return {
        "source": "experiments/configs/trajectories.yaml",
        "P1": {"lambda_per_K": list(p1["lambda_per_K"]), "step": p1["step"]},
        "P2": {"T0_K": [q["T0_K"] for q in p2["presets"]],
               "step": [q["step"] for q in p2["presets"]], "T_end_ratio": p2["T_end_ratio"]},
        "P3": {"eta": list(p3["eta"]), "halvings": p3["halvings"]},
    }


def constants() -> dict[str, Any]:
    s = specmod.load(ROOT / "config" / "spec_v2.yaml")
    s1 = specmod.load(ROOT / "config" / "spec_v1.yaml").sampler
    return {
        "source": "config/spec_v2.yaml via server/viewer_constants.py; do not edit by hand",
        "L_m": s.geometry.L.value,
        "k_cu_W_per_mK": s.materials.k_cu.value,
        "via_area_m2": s.via_area,
        "gap_thickness_m": [s.stack.d12.value, s.stack.d23.value, s.stack.d34.value],
        "chip_edge_margin_min_mm": s1.chip_edge_margin_min.value,
        "chip_gap_min_mm": s1.chip_gap_min.value,
        "proposers": proposer_settings(),
    }


def main() -> None:
    OUT.write_text(json.dumps(constants(), indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
