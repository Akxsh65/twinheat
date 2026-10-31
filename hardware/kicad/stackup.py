"""Stack-up of the hardware boards from config/spec_v2.yaml and the records (G1.1; CLAUDE.md §7.1).

    python -m hardware.kicad.stackup     # rewrites hardware/kicad/stackup.json

KiCad's bundled Python cannot read the YAML spec, so this normal-Python step writes the stack-up
gen_board.py applies: copper thickness per layer from the record's outer_oz / inner_oz (1 oz =
config/spec_v1.yaml t_cu) and dielectrics d12, d23, d34 from spec_v2. tests/test_kicad_plan.py
checks the committed JSON against the spec. If the fab's stack differs, CLAUDE.md §7.1 says the
YAML is changed to the as-fabricated stack (decision logged), then this file is regenerated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from twinheat import spec as specmod

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "hardware" / "kicad" / "stackup.json"


def stackup(rec: dict[str, Any]) -> dict[str, Any]:
    """Layers top to bottom with thicknesses in mm, for one board record."""
    v1 = specmod.load(ROOT / "config" / "spec_v1.yaml")
    v2 = specmod.load(ROOT / "config" / "spec_v2.yaml")
    oz_mm = v1.materials.t_cu.value * 1e3  # 1 oz copper, mm
    st = rec["stack"]
    cu = [st["outer_oz"], st["inner_oz"], st["inner_oz"], st["outer_oz"]]
    d = [v2.stack.d12.value * 1e3, v2.stack.d23.value * 1e3, v2.stack.d34.value * 1e3]
    names = ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"]
    kinds = ["prepreg", "core", "prepreg"]
    layers: list[dict[str, Any]] = []
    for i in range(4):
        layers.append({"name": names[i], "type": "copper", "oz": cu[i],
                       "thickness_mm": round(cu[i] * oz_mm, 6)})
        if i < 3:
            layers.append({"name": f"dielectric {i + 1}", "type": kinds[i],
                           "thickness_mm": round(d[i], 6)})
    return {"board_id": rec["id"], "layers": layers,
            "total_mm": round(sum(x["thickness_mm"] for x in layers), 6),
            "source": "config/spec_v2.yaml stack (CLAUDE.md §7.1) + record stack copper weights"}


def build(boards_dir: Path = ROOT / "hardware" / "boards") -> dict[str, Any]:
    paths = sorted(boards_dir.glob("board*.json"))
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    return {r["id"]: stackup(r) for r in recs}


def main() -> None:
    OUT.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
