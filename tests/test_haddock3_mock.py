"""Mock tests for HADDOCK 3 adapter (no real haddock3 invocation).

Coverage:
  - write_haddock_config emits valid-looking TOML with absolute paths
  - extract_rna_chain_pdb filters non-RNA residues + renames multi-char
    chain IDs to single-char "B"
  - find_final_stage_dir picks the highest-numbered stage subdir
  - find_best_pose tries the documented patterns and falls back
  - prepare_input writes protein.pdb + rna.pdb + config.toml
  - run_tool wraps `haddock3 <toml>` in conda env, passes CNS_SOLVE
  - parse_output reads the best pose and emits the expected fields
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import gemmi  # noqa: E402

from step4_tool_adapters.adapters.haddock3_adapter import (  # noqa: E402
    Haddock3Adapter, find_best_pose, find_final_stage_dir,
    write_haddock_config,
)
from step4_tool_adapters.adapters.structure_utils import (  # noqa: E402
    _is_rna_residue, effective_rna_chain_id, extract_rna_chain_pdb,
)
from step4_tool_adapters.tool_runner import ToolRunResult  # noqa: E402


# ---- gemmi helpers (mirror the existing adapter tests) ------------------


def _add_atom(res, name, x, y, z, element, b_iso=80.0):
    a = gemmi.Atom()
    a.name = name
    a.pos = gemmi.Position(x, y, z)
    a.element = gemmi.Element(element)
    a.b_iso = b_iso
    a.occ = 1.0
    res.add_atom(a)


def _build_protein_rna_structure() -> "gemmi.Structure":
    """Tiny complex: protein chain A (2 residues) + RNA chain B (2
    residues), protein res 1 contacts RNA nt 1 within 4.5 Å."""
    s = gemmi.Structure()
    s.cell = gemmi.UnitCell()
    m = gemmi.Model("1")

    chA = gemmi.Chain("A")
    r1 = gemmi.Residue(); r1.name = "LYS"
    r1.seqid = gemmi.SeqId(1, " "); r1.label_seq = 1
    _add_atom(r1, "N",  0.0, 0.0, 0.0, "N")
    _add_atom(r1, "CA", 1.5, 0.0, 0.0, "C")
    chA.add_residue(r1)
    r2 = gemmi.Residue(); r2.name = "ALA"
    r2.seqid = gemmi.SeqId(2, " "); r2.label_seq = 2
    _add_atom(r2, "CA", 50.0, 50.0, 50.0, "C")
    chA.add_residue(r2)
    m.add_chain(chA)

    chB = gemmi.Chain("B")
    rB1 = gemmi.Residue(); rB1.name = "A"
    rB1.seqid = gemmi.SeqId(1, " "); rB1.label_seq = 1
    _add_atom(rB1, "N1", 3.0, 0.0, 0.0, "N")
    chB.add_residue(rB1)
    rB2 = gemmi.Residue(); rB2.name = "G"
    rB2.seqid = gemmi.SeqId(2, " "); rB2.label_seq = 2
    _add_atom(rB2, "N1", 60.0, 60.0, 60.0, "N")
    chB.add_residue(rB2)
    m.add_chain(chB)

    s.add_model(m)
    return s


def _make_protein_rna_cif(path: Path) -> None:
    """Write the synthetic complex as mmCIF (extension must be ``.cif``)."""
    _build_protein_rna_structure().make_mmcif_document().write_file(str(path))


def _make_protein_rna_pdb(path: Path) -> None:
    """Write the synthetic complex as real PDB (extension ``.pdb``).

    Used for the HADDOCK 3 output simulation — the docking tool emits
    real PDB files, and gemmi's reader picks the parser by extension,
    so an mmCIF-content-in-.pdb file would be silently rejected.
    """
    _build_protein_rna_structure().write_pdb(str(path))


def _make_raw_with_multi_char_rna_chain(path: Path) -> None:
    """CIF with a protein chain ``A`` and a multi-char RNA chain
    ``RNA1`` (some ribosome complexes have these). Used to verify the
    extractor renames the chain to single-char ``B`` on PDB output."""
    s = gemmi.Structure()
    m = gemmi.Model("1")
    chA = gemmi.Chain("A")
    r = gemmi.Residue(); r.name = "ALA"
    r.seqid = gemmi.SeqId(1, " "); r.label_seq = 1
    _add_atom(r, "CA", 0.0, 0.0, 0.0, "C")
    chA.add_residue(r)
    m.add_chain(chA)

    # Multi-char chain ID — only valid in mmCIF.
    chR = gemmi.Chain("RNA1")
    rR = gemmi.Residue(); rR.name = "A"
    rR.seqid = gemmi.SeqId(1, " "); rR.label_seq = 1
    _add_atom(rR, "N1", 3.0, 0.0, 0.0, "N")
    chR.add_residue(rR)
    rR2 = gemmi.Residue(); rR2.name = "U"
    rR2.seqid = gemmi.SeqId(2, " "); rR2.label_seq = 2
    _add_atom(rR2, "N1", 6.0, 0.0, 0.0, "N")
    chR.add_residue(rR2)
    m.add_chain(chR)
    s.add_model(m)
    s.make_mmcif_document().write_file(str(path))


# ---- RNA residue classifier + chain extractor ---------------------------


class TestRnaResidueClassifier(unittest.TestCase):
    def test_standard_rna_letters(self):
        for n in ("A", "U", "G", "C", "I"):
            self.assertTrue(_is_rna_residue(n))

    def test_common_modified_bases(self):
        # PSU (pseudouridine) is the dominant modification in rRNA;
        # the classifier must accept it.
        self.assertTrue(_is_rna_residue("PSU"))
        self.assertTrue(_is_rna_residue("OMG"))

    def test_protein_and_other_rejected(self):
        for n in ("ALA", "LYS", "HOH", "MG", ""):
            self.assertFalse(_is_rna_residue(n))


class TestEffectiveRnaChainId(unittest.TestCase):
    def test_single_char_passes_through(self):
        self.assertEqual(effective_rna_chain_id("B"), "B")
        self.assertEqual(effective_rna_chain_id("F"), "F")

    def test_multi_char_collapses_to_B(self):
        self.assertEqual(effective_rna_chain_id("RNA1"), "B")
        self.assertEqual(effective_rna_chain_id("23"), "B")


class TestExtractRnaChainPdb(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_extracts_rna_only(self):
        cif = self.tmp / "x.cif"
        _make_protein_rna_cif(cif)
        out = self.tmp / "rna.pdb"
        extract_rna_chain_pdb(cif, "B", out)
        text = out.read_text(encoding="utf-8")
        # ATOM records present, all on chain B.
        atom_lines = [l for l in text.splitlines() if l.startswith("ATOM")]
        self.assertGreater(len(atom_lines), 0)
        chain_ids = {l[21] for l in atom_lines}
        self.assertEqual(chain_ids, {"B"})

    def test_multi_char_chain_collapses_to_B(self):
        cif = self.tmp / "x.cif"
        _make_raw_with_multi_char_rna_chain(cif)
        out = self.tmp / "rna.pdb"
        extract_rna_chain_pdb(cif, "RNA1", out)
        text = out.read_text(encoding="utf-8")
        atom_lines = [l for l in text.splitlines() if l.startswith("ATOM")]
        chain_ids = {l[21] for l in atom_lines}
        self.assertEqual(chain_ids, {"B"})

    def test_missing_chain_raises(self):
        cif = self.tmp / "x.cif"
        _make_protein_rna_cif(cif)
        with self.assertRaises(ValueError):
            extract_rna_chain_pdb(cif, "Z", self.tmp / "rna.pdb")

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            extract_rna_chain_pdb(
                self.tmp / "no.cif", "B", self.tmp / "rna.pdb",
            )


# ---- TOML writer --------------------------------------------------------


class TestWriteHaddockConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_basic_layout(self):
        toml = self.tmp / "x.toml"
        write_haddock_config(
            toml,
            run_dir=self.tmp / "haddock_run",
            protein_pdb=self.tmp / "protein.pdb",
            rna_pdb=self.tmp / "rna.pdb",
            sampling=30,
            flexref_sampling_factor=2,
            emref_sampling_factor=2,
        )
        text = toml.read_text(encoding="utf-8")
        # Module headers present in correct order.
        for section in ("[topoaa]", "[rigidbody]", "[flexref]", "[emref]"):
            self.assertIn(section, text)
        # Per-module param names match HADDOCK 3's schema: rigidbody
        # uses ``sampling``, flexref / emref use ``sampling_factor``.
        # Passing ``sampling`` to flexref / emref makes haddock3 2026.5
        # abort on launch — pin this in the test so the rename doesn't
        # regress silently.
        self.assertIn("[rigidbody]\nsampling = 30", text)
        self.assertIn("[flexref]\nsampling_factor = 2", text)
        self.assertIn("[emref]\nsampling_factor = 2", text)
        self.assertNotIn("[flexref]\nsampling = ", text)
        self.assertNotIn("[emref]\nsampling = ", text)
        # rigidbody MUST carry cmrest = true so the ab-initio sampler
        # keeps the protein + RNA bound during pose generation.
        # Without it the two molecules drift apart and the run wastes
        # time on garbage poses.
        self.assertIn("cmrest = true", text)
        # Path entries quoted; molecules in order.
        self.assertIn("molecules = [", text)
        self.assertIn("protein.pdb", text)
        self.assertIn("rna.pdb", text)

    def test_cmrest_toggle(self):
        # cmrest=False emits ``cmrest = false`` (still present, just
        # off) so toggling via YAML doesn't accidentally drop the
        # parameter from the rendered TOML.
        toml = self.tmp / "x.toml"
        write_haddock_config(
            toml,
            run_dir=self.tmp / "haddock_run",
            protein_pdb=self.tmp / "protein.pdb",
            rna_pdb=self.tmp / "rna.pdb",
            cmrest=False,
        )
        text = toml.read_text(encoding="utf-8")
        self.assertIn("cmrest = false", text)
        self.assertNotIn("cmrest = true", text)

    def test_invalid_sampling_raises(self):
        toml = self.tmp / "x.toml"
        with self.assertRaises(ValueError):
            write_haddock_config(
                toml,
                run_dir=self.tmp / "haddock_run",
                protein_pdb=self.tmp / "protein.pdb",
                rna_pdb=self.tmp / "rna.pdb",
                sampling=0,
            )
        with self.assertRaises(ValueError):
            write_haddock_config(
                toml,
                run_dir=self.tmp / "haddock_run",
                protein_pdb=self.tmp / "protein.pdb",
                rna_pdb=self.tmp / "rna.pdb",
                flexref_sampling_factor=-1,
            )


# ---- output discovery ---------------------------------------------------


class TestFindFinalStageDir(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_picks_highest_numbered(self):
        run = self.tmp / "run"
        for name in ("0_topoaa", "1_rigidbody", "2_flexref", "3_emref"):
            (run / name).mkdir(parents=True)
        self.assertEqual(find_final_stage_dir(run).name, "3_emref")

    def test_ignores_unrecognised_subdirs(self):
        run = self.tmp / "run"
        (run / "0_topoaa").mkdir(parents=True)
        (run / "logs").mkdir()
        (run / "10_emref").mkdir()  # double-digit index still wins
        self.assertEqual(find_final_stage_dir(run).name, "10_emref")

    def test_no_stage_dirs_returns_none(self):
        run = self.tmp / "run"
        run.mkdir()
        (run / "logs").mkdir()
        self.assertIsNone(find_final_stage_dir(run))

    def test_missing_run_dir_returns_none(self):
        self.assertIsNone(find_final_stage_dir(self.tmp / "no_such"))


class TestFindBestPose(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_prefers_rank1_pattern(self):
        stage = self.tmp / "3_emref"
        stage.mkdir()
        (stage / "rigidbody_1.pdb").write_text("ATOM\n")
        (stage / "rigidbody_5.pdb").write_text("ATOM\n")
        # *_1.pdb pattern beats *.pdb fallback alphabetical order.
        self.assertEqual(find_best_pose(stage).name, "rigidbody_1.pdb")

    def test_falls_back_to_any_pdb(self):
        stage = self.tmp / "3_emref"
        stage.mkdir()
        # No matches for *_1.pdb / cluster_1_model_1.pdb / ranked_0.pdb;
        # *.pdb catches everything else, sorted alphabetically.
        (stage / "z.pdb").write_text("ATOM\n")
        (stage / "a.pdb").write_text("ATOM\n")
        self.assertEqual(find_best_pose(stage).name, "a.pdb")

    def test_missing_stage_dir_returns_none(self):
        self.assertIsNone(find_best_pose(self.tmp / "no_such"))


# ---- traceback.tsv parsing + .pdb.gz handling --------------------------


import gzip  # noqa: E402  (kept near the gz-related tests)


def _write_traceback_tsv(
    run_dir: Path, rows: list[tuple[str, str, int]],
) -> Path:
    """Write a minimal traceback.tsv at the path HADDOCK 3 emits it.

    Each row is ``(rigidbody_name, emref_name, emref_rank)`` — the
    columns the adapter actually reads are the LAST ``*_rank`` column
    and the column immediately to its left. We surround them with a
    plausible header so the column-detection logic exercises the
    "pick the last rank column" branch.
    """
    out = run_dir / "traceback" / "traceback.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\t".join((
            "00_topo1", "00_topo2",
            "1_rigidbody", "1_rigidbody_rank",
            "3_emref",      "3_emref_rank",
        )),
    ]
    for rigid_name, emref_name, emref_rank in rows:
        lines.append("\t".join((
            "topo1.pdb", "topo2.pdb",
            rigid_name, "27",
            emref_name, str(emref_rank),
        )))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _make_run_with_traceback(
    tmp: Path,
    *,
    rank1_pose_name: str = "emref_33.pdb",
    gzipped: bool = False,
) -> Path:
    """Lay out a run_dir with a 3_emref stage carrying ``rank1_pose_name``
    (optionally gzipped) and a traceback.tsv that points at it."""
    run = tmp / "haddock_run"
    stage = run / "3_emref"
    stage.mkdir(parents=True)

    # Write the actual pose PDB (real format so contact_extractor can
    # parse it in the end-to-end test). Reuse the existing helper.
    pose_path = stage / rank1_pose_name
    _make_protein_rna_pdb(pose_path)
    if gzipped:
        gz_path = stage / f"{rank1_pose_name}.gz"
        with open(pose_path, "rb") as src, gzip.open(gz_path, "wb") as dst:
            dst.write(src.read())
        pose_path.unlink()  # only the .gz remains on disk

    _write_traceback_tsv(run, rows=[
        ("rigidbody_33.pdb", rank1_pose_name, 1),  # the winner
        ("rigidbody_41.pdb", "emref_41.pdb",   2),
    ])
    return run


class TestFindBestPoseTraceback(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_picks_rank1_from_traceback(self):
        run = _make_run_with_traceback(
            self.tmp, rank1_pose_name="emref_33.pdb",
        )
        out = find_best_pose(run)
        self.assertIsNotNone(out)
        self.assertEqual(out.name, "emref_33.pdb")
        self.assertEqual(out.parent.name, "3_emref")

    def test_gzipped_pose_is_decompressed(self):
        # rank=1 pose is on disk only as emref_33.pdb.gz — find_best_pose
        # must inflate it to emref_33.pdb next to the .gz so downstream
        # gemmi can read it as a real PDB.
        run = _make_run_with_traceback(
            self.tmp, rank1_pose_name="emref_33.pdb", gzipped=True,
        )
        out = find_best_pose(run)
        self.assertIsNotNone(out)
        self.assertEqual(out.name, "emref_33.pdb")
        self.assertTrue(out.is_file())
        # Decompressed file is real PDB content (starts with HEADER /
        # ATOM / CRYST etc. — not gzip magic bytes).
        head = out.read_bytes()[:2]
        self.assertNotEqual(head, b"\x1f\x8b")

    def test_picks_last_rank_column_when_multiple_present(self):
        # Multi-column traceback: rigidbody_rank vs emref_rank. The
        # adapter must pick the LAST *_rank column (emref) because
        # that's the final ranking. We make the rigidbody rank=1 row
        # disagree with the emref rank=1 row to verify the choice.
        run = self.tmp / "haddock_run"
        stage = run / "3_emref"
        stage.mkdir(parents=True)
        _make_protein_rna_pdb(stage / "emref_RIGHT.pdb")
        _make_protein_rna_pdb(stage / "emref_WRONG.pdb")
        (run / "traceback").mkdir()
        # Header lists rigidbody_rank BEFORE emref_rank; the
        # right answer is whichever pose has emref_rank == 1.
        (run / "traceback" / "traceback.tsv").write_text(
            "topo1\ttopo2\t1_rigidbody\t1_rigidbody_rank\t3_emref\t3_emref_rank\n"
            "t1\tt2\trigidbody_A.pdb\t1\temref_WRONG.pdb\t9\n"
            "t1\tt2\trigidbody_B.pdb\t9\temref_RIGHT.pdb\t1\n",
            encoding="utf-8",
        )
        out = find_best_pose(run)
        self.assertEqual(out.name, "emref_RIGHT.pdb")

    def test_falls_back_to_patterns_when_traceback_missing(self):
        # Build a run_dir with a stage + pattern-matchable pose, no
        # traceback/ subdir. Old code path still works.
        run = self.tmp / "haddock_run"
        stage = run / "3_emref"
        stage.mkdir(parents=True)
        _make_protein_rna_pdb(stage / "rigidbody_1.pdb")
        _make_protein_rna_pdb(stage / "rigidbody_5.pdb")
        out = find_best_pose(run)
        self.assertIsNotNone(out)
        self.assertEqual(out.name, "rigidbody_1.pdb")

    def test_pattern_fallback_handles_gzip(self):
        # No traceback, only *_1.pdb.gz on disk → fallback should
        # match the .gz pattern and decompress.
        run = self.tmp / "haddock_run"
        stage = run / "3_emref"
        stage.mkdir(parents=True)
        _make_protein_rna_pdb(stage / "rigidbody_1.pdb")
        with open(stage / "rigidbody_1.pdb", "rb") as src, \
                gzip.open(stage / "rigidbody_1.pdb.gz", "wb") as dst:
            dst.write(src.read())
        (stage / "rigidbody_1.pdb").unlink()
        out = find_best_pose(run)
        self.assertIsNotNone(out)
        self.assertEqual(out.name, "rigidbody_1.pdb")
        self.assertTrue(out.is_file())

    def test_no_rank_column_returns_none_and_falls_back(self):
        # traceback.tsv has no *_rank column at all → traceback path
        # returns None, fallback uses patterns. Stage carries a
        # rigidbody_1.pdb so the fallback can still resolve a pose.
        run = self.tmp / "haddock_run"
        stage = run / "3_emref"
        stage.mkdir(parents=True)
        _make_protein_rna_pdb(stage / "rigidbody_1.pdb")
        (run / "traceback").mkdir()
        (run / "traceback" / "traceback.tsv").write_text(
            "topo1\ttopo2\t1_rigidbody\t3_emref\n"
            "t1\tt2\trigidbody_A.pdb\temref_A.pdb\n",
            encoding="utf-8",
        )
        out = find_best_pose(run)
        # Pattern fallback fires.
        self.assertEqual(out.name, "rigidbody_1.pdb")

    def test_referenced_pose_missing_returns_none_from_traceback(self):
        # rank=1 row references a file the stage dir doesn't actually
        # carry. The traceback helper returns None internally, then
        # fallback kicks in with patterns. With no patterned files
        # present, the overall result is None — verifying both code
        # paths short-circuit cleanly.
        run = self.tmp / "haddock_run"
        stage = run / "3_emref"
        stage.mkdir(parents=True)  # empty stage
        _write_traceback_tsv(run, rows=[
            ("rigidbody_X.pdb", "emref_NOPE.pdb", 1),
        ])
        self.assertIsNone(find_best_pose(run))


# ---- end-to-end adapter -------------------------------------------------


def _sample_json(sample_id: str = "test_A_B") -> dict:
    return {
        "sample_id": sample_id,
        "source_pdb": "synth",
        "protein": {"chain_id": "A", "sequence": "MK", "length": 2},
        "rna": {"chain_id": "B", "sequence": "AG", "length": 2},
    }


def _config(work: Path, *, raw_dir: Path) -> dict:
    return {
        "tools": {
            "haddock3": {
                "conda_env": "haddock3",
                "timeout": 600,
                "cns_solve": "/fake/cns",
                "sampling": 20,
                "flexref_sampling_factor": 2,
                "emref_sampling_factor": 2,
                "distance_scale": 8.0,
            },
        },
        "structure_source": {"raw_dir": str(raw_dir)},
        "log_dir": str(work / "logs"),
        "contact_threshold": 4.5,
    }


def _ok_run() -> ToolRunResult:
    return ToolRunResult(
        command="fake", cwd=None, returncode=0,
        stdout="", stderr="", runtime_seconds=10.0, log_path=None,
    )


class TestPrepareInput(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.raw_dir = self.tmp / "raw"
        self.raw_dir.mkdir()
        # The adapter discovers raw via _find_raw_structure; synthesise
        # a CIF named after the sample's source_pdb.
        _make_protein_rna_cif(self.raw_dir / "synth.cif")
        self.work = self.tmp / "work"
        self.work.mkdir()

    def test_writes_protein_rna_and_toml(self):
        adapter = Haddock3Adapter()
        paths = adapter.prepare_input(
            _sample_json(), self.work, _config(self.work, raw_dir=self.raw_dir),
        )
        self.assertTrue(Path(paths["protein_pdb"]).is_file())
        self.assertTrue(Path(paths["rna_pdb"]).is_file())
        self.assertTrue(Path(paths["config_path"]).is_file())
        toml = Path(paths["config_path"]).read_text(encoding="utf-8")
        self.assertIn("[rigidbody]", toml)
        self.assertIn("sampling = 20", toml)

    def test_missing_chain_id_raises(self):
        adapter = Haddock3Adapter()
        sj = _sample_json()
        sj["rna"]["chain_id"] = ""
        with self.assertRaises(ValueError):
            adapter.prepare_input(
                sj, self.work, _config(self.work, raw_dir=self.raw_dir),
            )

    def test_missing_raw_structure_raises(self):
        adapter = Haddock3Adapter()
        empty_raw = self.tmp / "empty_raw"
        empty_raw.mkdir()
        with self.assertRaises(FileNotFoundError):
            adapter.prepare_input(
                _sample_json(), self.work,
                _config(self.work, raw_dir=empty_raw),
            )


class TestRunTool(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.work = self.tmp / "work"
        self.work.mkdir()
        self.toml = self.work / "x.toml"
        self.toml.write_text("run_dir = 'r'\nmolecules = []\n", encoding="utf-8")
        # run_dir must exist after the call to satisfy the rc=0 check.
        self.run_dir = (self.work / "haddock_run").resolve()
        self.run_dir.mkdir()

    def test_success_sets_cns_env(self):
        captured = {}

        def fake(env, cmd, **kwargs):
            captured["env"] = env
            captured["cmd"] = cmd
            captured["extra_env"] = kwargs.get("extra_env")
            captured["timeout"] = kwargs.get("timeout")
            return _ok_run()

        adapter = Haddock3Adapter()
        cfg = {
            "tools": {
                "haddock3": {
                    "conda_env": "haddock3", "timeout": 600,
                    "cns_solve": "/opt/biotools/cns_v1.3_r9",
                },
            },
            "log_dir": str(self.work / "logs"),
        }
        with patch(
            "step4_tool_adapters.adapters.haddock3_adapter.run_in_conda_env",
            side_effect=fake,
        ):
            out = adapter.run_tool(
                {"config_path": self.toml, "run_dir": self.run_dir},
                self.work, cfg,
            )
        self.assertEqual(captured["env"], "haddock3")
        self.assertIn("haddock3", captured["cmd"])
        self.assertIn("x.toml", captured["cmd"])
        self.assertEqual(
            captured["extra_env"],
            {"CNS_SOLVE": "/opt/biotools/cns_v1.3_r9"},
        )
        self.assertEqual(captured["timeout"], 600)
        self.assertEqual(out, self.run_dir)

    def test_failure_raises(self):
        bad = ToolRunResult(
            command="fake", cwd=None, returncode=2,
            stdout="", stderr="cns missing", runtime_seconds=1.0,
            log_path=None,
        )
        adapter = Haddock3Adapter()
        cfg = {"tools": {"haddock3": {"conda_env": "haddock3"}}}
        with patch(
            "step4_tool_adapters.adapters.haddock3_adapter.run_in_conda_env",
            return_value=bad,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                adapter.run_tool(
                    {"config_path": self.toml, "run_dir": self.run_dir},
                    self.work, cfg,
                )
        self.assertIn("rc=2", str(ctx.exception))

    def test_missing_run_dir_after_success_raises(self):
        # rc=0 but the run_dir wasn't created (haddock3 didn't actually
        # write where the TOML said) — surface as a clear adapter error
        # rather than letting parse_output crash on the empty path.
        adapter = Haddock3Adapter()
        cfg = {"tools": {"haddock3": {"conda_env": "haddock3"}}}
        ghost_run = self.work / "ghost_run"  # not created
        with patch(
            "step4_tool_adapters.adapters.haddock3_adapter.run_in_conda_env",
            return_value=_ok_run(),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                adapter.run_tool(
                    {"config_path": self.toml, "run_dir": ghost_run},
                    self.work, cfg,
                )
        self.assertIn("missing", str(ctx.exception))


class TestParseOutput(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.run_dir = self.tmp / "haddock_run"
        # Lay out one stage with a single best pose. Mirrors the
        # documented "0_topoaa/, 1_rigidbody/, 2_flexref/, 3_emref/"
        # convention.
        self.stage_dir = self.run_dir / "3_emref"
        self.stage_dir.mkdir(parents=True)
        # HADDOCK 3 writes real PDB into its stage dirs; use the PDB
        # writer here so gemmi's reader picks the right parser.
        _make_protein_rna_pdb(self.stage_dir / "rigidbody_1.pdb")

    def test_full_parse(self):
        adapter = Haddock3Adapter()
        cfg = {
            "tools": {"haddock3": {"distance_scale": 8.0}},
            "contact_threshold": 4.5,
        }
        pred = adapter.parse_output(self.run_dir, _sample_json(), cfg)
        self.assertTrue(pred.success)
        self.assertEqual(pred.tool_id, "haddock3")
        self.assertEqual(pred.category, "D")
        # Protein res 1 (CA ≈1.5Å from RNA atom) contacts RNA nt 1.
        self.assertIn(1, pred.binding_protein_residues or [])
        self.assertIn(1, pred.binding_rna_nucleotides or [])
        # per_residue_pae_score populated (distance-derived).
        self.assertIsNotNone(pred.per_residue_pae_score)
        self.assertGreater(pred.per_residue_pae_score[1],
                           pred.per_residue_pae_score[2])
        # Per spec: Cat D doesn't populate per_residue_confidence —
        # only per_residue_pae_score for eval-time correlation.
        self.assertIsNone(pred.per_residue_confidence)
        # Structure path points at the best pose.
        self.assertTrue(
            pred.predicted_structure_path.endswith("rigidbody_1.pdb")
        )

    def test_missing_stage_dir(self):
        empty = self.tmp / "empty_run"
        empty.mkdir()
        adapter = Haddock3Adapter()
        pred = adapter.parse_output(empty, _sample_json(), {})
        self.assertFalse(pred.success)
        self.assertIn("<N>_<module>", pred.error_message)

    def test_stage_dir_without_pdb(self):
        run = self.tmp / "run2"
        (run / "3_emref").mkdir(parents=True)  # exists but empty
        adapter = Haddock3Adapter()
        pred = adapter.parse_output(run, _sample_json(), {})
        self.assertFalse(pred.success)
        self.assertIn("no PDB found", pred.error_message)

    def test_traceback_gzip_end_to_end(self):
        # Mirror of test_full_parse but the rank=1 pose is gzipped
        # and discovery goes through traceback.tsv. parse_output must
        # decompress + extract contacts + populate per_residue_pae_score
        # exactly like the .pdb path.
        run = _make_run_with_traceback(
            self.tmp / "gz_case",
            rank1_pose_name="emref_33.pdb",
            gzipped=True,
        )
        adapter = Haddock3Adapter()
        cfg = {
            "tools": {"haddock3": {"distance_scale": 8.0}},
            "contact_threshold": 4.5,
        }
        pred = adapter.parse_output(run, _sample_json(), cfg)
        self.assertTrue(pred.success)
        self.assertIn(1, pred.binding_protein_residues or [])
        self.assertIsNotNone(pred.per_residue_pae_score)
        self.assertTrue(
            pred.predicted_structure_path.endswith("emref_33.pdb")
        )


if __name__ == "__main__":
    unittest.main()
