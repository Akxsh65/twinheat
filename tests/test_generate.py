"""Dataset generator (N2.1; CLAUDE.md §7.4, §6.4, §6.5, §5 data hygiene).

N = 64, 40 boards, shards of 10, on the CPU (the prompt allows the CPU fallback): every run
here goes through the same sample -> rasterise -> coefficients -> batched_pcg (line-z, 1e-10)
-> envelope -> energy-balance pipeline as a GPU run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from data import generate as gen

N, COUNT, SHARD, BATCH = 64, 40, 10, 8
KW = {"N": N, "shard_size": SHARD, "batch": BATCH, "device": "cpu", "allow_dirty": True,
      "log": lambda *a, **k: None}


class Crash(RuntimeError):
    pass


def crash_at(event: str, shard: int):
    def hook(e: str, s: int) -> None:
        if (e, s) == (event, shard):
            raise Crash(f"simulated crash at {e} of shard {s}")
    return hook


def files(out: Path) -> dict[str, bytes]:
    """Every shard file's bytes, by relative path (the manifest is excluded: it holds times)."""
    return {str(p.relative_to(out)).replace("\\", "/"): p.read_bytes()
            for p in sorted(out.glob("shard_*/*"))}


@pytest.fixture(scope="module")
def reference(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("ref")
    gen.generate("train", COUNT, 7, out, **KW)
    return out


# ---------------------------------------------------------------------------- seeds, modes

def test_board_seed_is_64_bit_sha256_of_split_seed_and_index():
    s = gen.board_seed("train", 7, 3)
    want = int.from_bytes(hashlib.sha256(b"train\x007\x003").digest()[:8], "big")
    assert s == want and 0 <= s < 2**64
    assert len({gen.board_seed(sp, sd, i) for sp in ("train", "val") for sd in (0, 1)
                for i in range(100)}) == 400


def test_train_alternates_design_and_id_other_splits_are_one_mode():
    assert [gen.board_mode("train", i) for i in range(4)] == ["design", "id", "design", "id"]
    assert {gen.board_mode(s, 5) for s in ("val", "cal", "test")} == {"id"}
    assert gen.board_mode("ood_3oz", 0) == "ood_3oz"
    with pytest.raises(gen.GenerateError, match="ood_9"):
        gen.board_mode("ood_9", 0)
    with pytest.raises(gen.GenerateError, match="bad"):
        gen.board_mode("bad", 0)


def test_bottom_chip_boards_are_refused_until_the_solver_reads_q_bot(tmp_path):
    with pytest.raises(gen.GenerateError, match="q_bot"):
        gen.generate("ood_bottom_chips", 4, 0, tmp_path, **KW)


# ---------------------------------------------------------------------------- output

def test_shards_records_and_fields(reference):
    man = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
    assert man["complete"] and man["split"] == "train" and man["count"] == COUNT
    assert man["tolerance"] == 1e-10 and man["N"] == N
    assert [s["name"] for s in man["shards"]] == [f"shard_{k:04d}" for k in range(4)]
    for k, sh in enumerate(man["shards"]):
        d = reference / sh["name"]
        lines = (d / "records.jsonl").read_text(encoding="utf-8").splitlines()
        fields = np.load(d / "fields.npy")
        assert fields.dtype == np.float16 and fields.shape == (SHARD, 4, N, N)
        assert fields.flags["C_CONTIGUOUS"]
        for j, line in enumerate(lines):
            row = json.loads(line)
            i = k * SHARD + j
            assert row["i"] == i and row["seed"] == gen.board_seed("train", 7, i)
            assert row["record"]["mode"] == gen.board_mode("train", i)
            assert isinstance(row["scaled"], bool) and row["iters"] > 0
            assert row["energy_balance"] <= gen.ENERGY_TOL
            assert row["record_sha256"] == gen.record_hash(row["record"])
        assert 0 < float(fields.max()) <= 100.0 + 0.1  # envelope, float16 rounding
    assert "boards_per_s" in man["shards"][0] and man["end_time"] >= man["start_time"]


def test_fields_are_the_solver_solution(reference):
    """Re-solve one board directly from its stored record: the stored float16 field matches."""
    from data.raster import rasterise
    from solvers.fvm import assemble_board, solve_direct
    row = json.loads((reference / "shard_0001" / "records.jsonl").read_text(
        encoding="utf-8").splitlines()[3])
    rec = row["record"]
    A, b = assemble_board(gen.SPEC, rasterise(rec, gen.SPEC, N), rec["dirichlet_edges"])
    theta = solve_direct(A, b).reshape(4, N, N)
    stored = np.load(reference / "shard_0001" / "fields.npy")[3].astype(np.float64)
    np.testing.assert_allclose(stored, theta, rtol=2.0**-10, atol=1e-6)


def test_manifest_hashes_verify_and_tampering_is_caught(reference, tmp_path):
    assert gen.verify(reference)["complete"]
    import shutil
    bad = tmp_path / "bad"
    shutil.copytree(reference, bad)
    p = bad / "shard_0002" / "fields.npy"
    data = bytearray(p.read_bytes())
    data[-1] ^= 1
    p.write_bytes(bytes(data))
    with pytest.raises(gen.ManifestError, match="shard_0002"):
        gen.verify(bad)


# ---------------------------------------------------------------------------- resume

@pytest.mark.parametrize("event,shard", [("tmp_written", 1), ("renamed", 1),
                                         ("manifest_written", 1), ("tmp_written", 0)])
def test_crash_then_resume_gives_byte_identical_shards(reference, tmp_path, event, shard):
    with pytest.raises(Crash):
        gen.generate("train", COUNT, 7, tmp_path, hook=crash_at(event, shard), **KW)
    man = gen.generate("train", COUNT, 7, tmp_path, **KW)
    assert man["complete"]
    assert not list(tmp_path.glob("*.tmp"))
    assert files(tmp_path) == files(reference)
    assert gen.verify(tmp_path)["complete"]


def test_resume_skips_completed_shards(reference, tmp_path):
    import shutil
    shutil.copytree(reference, tmp_path / "out")
    solved = []
    gen.generate("train", COUNT, 7, tmp_path / "out", hook=lambda e, s: solved.append((e, s)),
                 **KW)
    assert not any(e == "tmp_written" for e, _ in solved)  # nothing recomputed


def test_refuses_another_splits_output_and_overwriting_completed_shards(reference, tmp_path):
    import shutil
    out = tmp_path / "out"
    shutil.copytree(reference, out)
    with pytest.raises(gen.GenerateError, match="train"):
        gen.generate("val", COUNT, 7, out, **KW)
    with pytest.raises(gen.GenerateError, match="seed"):
        gen.generate("train", COUNT, 8, out, **KW)
    # a completed shard that is on disk but not in the manifest, with different content
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    man["shards"] = man["shards"][:1]
    man["complete"] = False
    (out / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    p = out / "shard_0001" / "fields.npy"
    data = bytearray(p.read_bytes())
    data[-1] ^= 1
    p.write_bytes(bytes(data))
    with pytest.raises(gen.GenerateError, match="shard_0001"):
        gen.generate("train", COUNT, 7, out, **KW)


# ---------------------------------------------------------------------------- hygiene

def test_record_hashes_never_collide_across_two_splits(reference, tmp_path):
    gen.generate("val", COUNT, 7, tmp_path, **KW)

    def hashes(out: Path) -> set[str]:
        return {json.loads(line)["record_sha256"] for f in sorted(out.glob("shard_*/records.jsonl"))
                for line in f.read_text(encoding="utf-8").splitlines()}
    a, b = hashes(reference), hashes(tmp_path)
    assert len(a) == len(b) == COUNT
    assert not a & b


def test_energy_balance_failure_raises_naming_the_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "board_energy_balance", lambda *a, **k: 1.0)
    with pytest.raises(gen.GenerateError, match=str(gen.board_seed("val", 0, 0))):
        gen.generate("val", 4, 0, tmp_path, **{**KW, "shard_size": 4, "batch": 4})


def test_envelope_scaling_is_recorded_and_counted(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "THETA_MAX", 1.0)  # every board exceeds 1 K
    man = gen.generate("val", 8, 0, tmp_path, **{**KW, "shard_size": 4, "batch": 4})
    assert man["rescale_fraction"] == 1.0
    f = np.load(tmp_path / "shard_0000" / "fields.npy")
    assert float(f.max()) <= 1.0 + 1e-3
    rows = [json.loads(s) for s in (tmp_path / "shard_0000" / "records.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert all(r["scaled"] for r in rows)


def test_a_dirty_tree_is_refused_unless_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "git_commit", lambda: "abc1234-dirty")
    with pytest.raises(gen.GenerateError, match="dirty"):
        gen.generate("val", 4, 0, tmp_path, **{**KW, "allow_dirty": False})
