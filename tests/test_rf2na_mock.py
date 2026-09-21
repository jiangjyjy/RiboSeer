"""Mock tests for RoseTTAFold2NA adapter (no real RF2NA invocation).

What this covers:
  - target name sanitisation
  - FASTA writer (line-wrap, whitespace rejection, empty rejection)
  - prepare_input writes both protein + RNA FASTAs
  - run_tool wraps `bash run_RF2NA.sh ...` inside `conda run -n RF2NA2`
    with the P:/R: prefix convention, cwd=install_dir, and surfaces
    failures as RuntimeError
  - parse_output: model_00.pdb contacts + B-factor pLDDT + optional
    PAE npz, missing-PDB failure, npz-absent fallback
  - end-to-end predict() with stubbed run_in_conda_env writing a
    synthetic complex PDB into <out>/models/

NOTE: the RF2NA database (~230 GB) is still downloading on the server,
so test_rf2na_real.py is written but cannot actually run yet. These
mocks are the only thing exercising the adapter today.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import gemmi  # noqa: E402  — environment.yml dependency
import numpy as np  # noqa: E402

from step4_tool_adapters.adapters.rf2na_adapter import (  # noqa: E402
    RF2NAAdapter, _sanitize_target_name, make_single_seq_a3m,
    patch_msa_script, patch_run_script, write_fasta,
)
from step4_tool_adapters.tool_runner import ToolRunResult  # noqa: E402


# ---------- helpers --------------------------------------------------------


def _add_atom(res: gemmi.Residue, name: str, x: float, y: float, z: float,
              element: str, b_iso: float = 80.0) -> None:
    a = gemmi.Atom()
    a.name = name
    a.pos = gemmi.Position(x, y, z)
    a.element = gemmi.Element(element)
    a.b_iso = b_iso
    a.occ = 1.0
    res.add_atom(a)


def _make_synthetic_complex_pdb(path: Path) -> None:
    """Build a tiny RNA-protein complex PDB.

    Same shape as the boltz2 test fixture but written as PDB so we
    exercise gemmi's PDB reader path:
      Chain A — 3 protein residues 1..3 (B-factors 85, 70, 60),
                res1 LYS:N at origin.
      Chain B — 2 RNA nucleotides 1..2 (B-factors 55, 40),
                res1 A:N1 at (3,0,0) — ~3 Å from LYS:N.
    Expected contacts: protein 1 ↔ rna 1 (single pair).
    """
    s = gemmi.Structure()
    s.cell = gemmi.UnitCell()
    m = gemmi.Model("1")

    chA = gemmi.Chain("A")
    r1 = gemmi.Residue(); r1.name = "LYS"
    r1.seqid = gemmi.SeqId(1, " "); r1.label_seq = 1
    _add_atom(r1, "N",  0.0, 0.0, 0.0, "N", b_iso=85.0)
    _add_atom(r1, "CA", 1.5, 0.0, 0.0, "C", b_iso=85.0)
    _add_atom(r1, "C",  2.5, 1.0, 0.0, "C", b_iso=85.0)
    chA.add_residue(r1)

    r2 = gemmi.Residue(); r2.name = "ALA"
    r2.seqid = gemmi.SeqId(2, " "); r2.label_seq = 2
    _add_atom(r2, "N",  50.0, 50.0, 50.0, "N", b_iso=70.0)
    _add_atom(r2, "CA", 51.5, 50.0, 50.0, "C", b_iso=70.0)
    chA.add_residue(r2)

    r3 = gemmi.Residue(); r3.name = "SER"
    r3.seqid = gemmi.SeqId(3, " "); r3.label_seq = 3
    _add_atom(r3, "N",  52.0, 50.0, 50.0, "N", b_iso=60.0)
    _add_atom(r3, "CA", 53.5, 50.0, 50.0, "C", b_iso=60.0)
    chA.add_residue(r3)
    m.add_chain(chA)

    chB = gemmi.Chain("B")
    rB1 = gemmi.Residue(); rB1.name = "A"
    rB1.seqid = gemmi.SeqId(1, " "); rB1.label_seq = 1
    _add_atom(rB1, "N1", 3.0, 0.0, 0.0, "N", b_iso=55.0)
    _add_atom(rB1, "C2", 4.0, 1.0, 0.0, "C", b_iso=55.0)
    chB.add_residue(rB1)

    rB2 = gemmi.Residue(); rB2.name = "G"
    rB2.seqid = gemmi.SeqId(2, " "); rB2.label_seq = 2
    _add_atom(rB2, "N1", 60.0, 60.0, 60.0, "N", b_iso=40.0)
    _add_atom(rB2, "C2", 61.0, 61.0, 60.0, "C", b_iso=40.0)
    chB.add_residue(rB2)
    m.add_chain(chB)

    s.add_model(m)
    s.write_pdb(str(path))


def _write_pae_npz(path: Path) -> Path:
    arr = np.array([[0.0, 5.0, 12.0],
                    [5.0, 0.0, 7.0],
                    [12.0, 7.0, 0.0]])
    np.savez(str(path), pae=arr)
    return path


def _sample_json() -> dict:
    return {
        "sample_id": "1un6_B_F",
        "source_pdb": "1un6",
        "protein": {
            "chain_id": "B",
            "sequence": "MKTVLAQK",
        },
        "rna": {
            "chain_id": "F",
            "sequence": "GCCGGCCAU",
        },
    }


def _make_install_dir(tmp: Path, *, with_script: bool = True) -> Path:
    """Create a fake RF2NA install dir under ``tmp`` with a stub
    ``run_RF2NA.sh`` containing the upstream's hardcoded ``conda activate
    RF2NA`` line. When ``with_script=False`` the dir exists but the
    launcher is missing — used to exercise patch failure paths."""
    install_dir = tmp / "RoseTTAFold2NA"
    install_dir.mkdir(parents=True, exist_ok=True)
    if with_script:
        (install_dir / "run_RF2NA.sh").write_text(
            "#!/bin/bash\n"
            "set -e\n"
            "source ~/.bashrc\n"
            "conda activate RF2NA\n"
            "echo running RF2NA on $@\n",
            encoding="utf-8",
        )
    return install_dir


def _config(tmp: Path, *, install_dir: Optional[Path] = None,
            skip_script_patch: bool = False,
            single_seq: bool = False) -> dict:
    """Test config. ``single_seq`` defaults to False here so the legacy
    run_RF2NA.sh / MSA-patch tests keep exercising that path; the
    PRODUCTION default (configs/step4_config.yaml) is single_seq:true.
    Single-seq tests pass single_seq=True explicitly."""
    if install_dir is None:
        install_dir = _make_install_dir(tmp)
    cfg = {
        "tools": {
            "rosettafold2na": {
                "conda_env": "RF2NA2",
                "install_dir": str(install_dir),
                "timeout": 14400,
                "single_seq": single_seq,
            },
        },
        "contact_threshold": 4.5,
        "log_dir": str(tmp / "_logs"),
    }
    if skip_script_patch:
        cfg["tools"]["rosettafold2na"]["skip_script_patch"] = True
    return cfg


def _ok_run(stdout: str = "ok") -> ToolRunResult:
    return ToolRunResult(
        command="fake", cwd=None, returncode=0,
        stdout=stdout, stderr="", runtime_seconds=0.4, log_path=None,
    )


def _bad_run(stderr: str = "fail", rc: int = 2) -> ToolRunResult:
    return ToolRunResult(
        command="fake", cwd=None, returncode=rc,
        stdout="", stderr=stderr, runtime_seconds=0.1, log_path=None,
    )


# ---------- sanitisation / FASTA ------------------------------------------


class TestSanitize(unittest.TestCase):
    def test_alphanum(self):
        self.assertEqual(_sanitize_target_name("1un6_B_F"), "1un6_B_F")

    def test_dash_replaced(self):
        self.assertEqual(_sanitize_target_name("2wj8_A_-a"), "2wj8_A__a")

    def test_all_invalid(self):
        self.assertEqual(_sanitize_target_name("!!!"), "sample")


class TestWriteFasta(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_basic_wrap(self):
        seq = "A" * 130
        out = self.tmp / "x.fa"
        write_fasta(out, "tgt", seq, line_width=60)
        body = out.read_text(encoding="utf-8").splitlines()
        self.assertEqual(body[0], ">tgt")
        self.assertEqual(body[1], "A" * 60)
        self.assertEqual(body[2], "A" * 60)
        self.assertEqual(body[3], "A" * 10)

    def test_rejects_whitespace(self):
        with self.assertRaises(ValueError):
            write_fasta(self.tmp / "x.fa", "tgt", "ACDE FGH")

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            write_fasta(self.tmp / "x.fa", "tgt", "")


# ---------- prepare_input --------------------------------------------------


class TestPrepareInput(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_writes_both_fastas(self):
        adapter = RF2NAAdapter()
        paths = adapter.prepare_input(_sample_json(), self.tmp, _config(self.tmp))
        self.assertTrue(paths["protein_fa"].is_file())
        self.assertTrue(paths["rna_fa"].is_file())
        self.assertEqual(paths["target"], "1un6_B_F")
        prot_text = paths["protein_fa"].read_text(encoding="utf-8")
        rna_text = paths["rna_fa"].read_text(encoding="utf-8")
        self.assertIn(">1un6_B_F_protein", prot_text)
        self.assertIn("MKTVLAQK", prot_text)
        self.assertIn(">1un6_B_F_rna", rna_text)
        self.assertIn("GCCGGCCAU", rna_text)

    def test_missing_protein_seq(self):
        adapter = RF2NAAdapter()
        bad = _sample_json()
        bad["protein"]["sequence"] = ""
        with self.assertRaises(ValueError):
            adapter.prepare_input(bad, self.tmp, _config(self.tmp))

    def test_missing_rna_seq(self):
        adapter = RF2NAAdapter()
        bad = _sample_json()
        bad["rna"]["sequence"] = ""
        with self.assertRaises(ValueError):
            adapter.prepare_input(bad, self.tmp, _config(self.tmp))

    def test_dashed_sample_id(self):
        adapter = RF2NAAdapter()
        sj = _sample_json()
        sj["sample_id"] = "2wj8_A_-a"
        paths = adapter.prepare_input(sj, self.tmp, _config(self.tmp))
        self.assertEqual(paths["target"], "2wj8_A__a")
        self.assertTrue(paths["protein_fa"].name.startswith("2wj8_A__a"))


# ---------- run_tool -------------------------------------------------------


class TestPatchRunScript(unittest.TestCase):
    """Unit-level tests for the run_RF2NA.sh patcher itself."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.install = self.tmp / "install"
        self.work = self.tmp / "work"
        self.install.mkdir()
        self.work.mkdir()

    def _write_script(self, body: str) -> None:
        (self.install / "run_RF2NA.sh").write_text(body, encoding="utf-8")

    def test_replaces_conda_activate(self):
        self._write_script(
            "#!/bin/bash\n"
            "set -e\n"
            "conda activate RF2NA\n"
            "echo done\n"
        )
        out = patch_run_script(self.install, self.work, "RF2NA2")
        body = out.read_text(encoding="utf-8")
        self.assertIn("conda activate RF2NA2\n", body)
        self.assertNotIn("conda activate RF2NA\n", body)
        # File is in work_dir, not install_dir.
        self.assertEqual(out.parent, self.work)

    def test_replaces_source_activate(self):
        self._write_script(
            "#!/bin/bash\n"
            "source activate RF2NA\n"
        )
        out = patch_run_script(self.install, self.work, "RF2NA2")
        body = out.read_text(encoding="utf-8")
        # We canonicalise to `conda activate` regardless of source form.
        self.assertIn("conda activate RF2NA2", body)
        self.assertNotIn("source activate RF2NA\n", body)

    def test_does_not_touch_rf2na2_or_other_envs(self):
        """Word-boundary regex must not turn RF2NA2 into RF2NA22."""
        self._write_script(
            "#!/bin/bash\n"
            "conda activate RF2NA2\n"
            "conda activate RF2NA-cpu\n"
            "conda activate something_else\n"
        )
        out = patch_run_script(self.install, self.work, "TARGETENV")
        body = out.read_text(encoding="utf-8")
        # None of these matched the regex (none had bare `RF2NA` followed
        # by whitespace / EOL), so they should pass through unchanged.
        self.assertIn("conda activate RF2NA2\n", body)
        self.assertIn("conda activate RF2NA-cpu\n", body)
        self.assertIn("conda activate something_else\n", body)
        self.assertNotIn("conda activate TARGETENV", body)

    def test_replaces_multiple_occurrences(self):
        self._write_script(
            "conda activate RF2NA\n"
            "echo middle\n"
            "source activate RF2NA\n"
        )
        out = patch_run_script(self.install, self.work, "MYENV")
        body = out.read_text(encoding="utf-8")
        self.assertEqual(body.count("conda activate MYENV"), 2)

    def test_no_activate_line_passthrough(self):
        """When neither the activate line nor the PIPEDIR=$SCRIPT line is
        present, both regexes are no-ops on the body — but PIPEDIR
        override is still injected at the top as a safety net (so a
        future upstream that uses $PIPEDIR but assigns it differently
        still gets the install_dir pin)."""
        original = "#!/bin/bash\necho only this\n"
        self._write_script(original)
        out = patch_run_script(self.install, self.work, "MYENV")
        body = out.read_text(encoding="utf-8")
        # Original lines preserved verbatim, just appended-to.
        self.assertIn("#!/bin/bash\necho only this\n", body)
        # PIPEDIR fallback injection is present.
        self.assertIn(f'export PIPEDIR="{self.install.resolve()}"', body)

    def test_pipedir_override_appended_after_dirname_assign(self):
        """The standard upstream pattern
        ``export PIPEDIR=`dirname $SCRIPT``` should keep the original
        line and gain a NEW override line directly after it pinning
        $PIPEDIR to install_dir."""
        self._write_script(
            "#!/bin/bash\n"
            "SCRIPT=`realpath -s $0`\n"
            "export PIPEDIR=`dirname $SCRIPT`\n"
            "echo $PIPEDIR\n"
        )
        out = patch_run_script(self.install, self.work, "MYENV")
        body = out.read_text(encoding="utf-8")
        # Original derivation kept.
        self.assertIn("export PIPEDIR=`dirname $SCRIPT`", body)
        # Override comes immediately after.
        derived_idx = body.index("export PIPEDIR=`dirname $SCRIPT`")
        override_idx = body.index(
            f'export PIPEDIR="{self.install.resolve()}"'
        )
        self.assertGreater(override_idx, derived_idx)
        # The fallback "patch by riboseer" tag is NOT present —
        # we found the upstream line and used the targeted append.
        self.assertNotIn("upstream assignment style not recognised", body)
        # Fix marker is in the targeted append.
        self.assertIn("# patched by riboseer: pin to install_dir", body)

    def test_pipedir_override_handles_dollar_paren_variant(self):
        """``$(dirname $SCRIPT)`` should also be matched."""
        self._write_script(
            "#!/bin/bash\n"
            "SCRIPT=$(realpath -s $0)\n"
            'export PIPEDIR=$(dirname $SCRIPT)\n'
            "echo $PIPEDIR\n"
        )
        out = patch_run_script(self.install, self.work, "MYENV")
        body = out.read_text(encoding="utf-8")
        self.assertIn("export PIPEDIR=$(dirname $SCRIPT)", body)
        self.assertIn(
            f'export PIPEDIR="{self.install.resolve()}"', body,
        )

    def test_pipedir_unrelated_lines_untouched(self):
        """A literal `export PIPEDIR="foo"` (no $SCRIPT reference) should
        NOT trigger the targeted append — the regex anchors on $SCRIPT
        specifically. The fallback injection at the top still happens."""
        self._write_script(
            "#!/bin/bash\n"
            'export PIPEDIR="/some/other/path"\n'
            "echo $PIPEDIR\n"
        )
        out = patch_run_script(self.install, self.work, "MYENV")
        body = out.read_text(encoding="utf-8")
        # Original line preserved.
        self.assertIn('export PIPEDIR="/some/other/path"', body)
        # Targeted append did NOT fire (no marker), but fallback did.
        self.assertNotIn("# patched by riboseer: pin to install_dir", body)
        self.assertIn("upstream assignment style not recognised", body)

    def test_target_env_name_is_configurable(self):
        self._write_script(
            "#!/bin/bash\nconda activate RF2NA\n"
        )
        out = patch_run_script(self.install, self.work, "FooBarEnv")
        self.assertIn("conda activate FooBarEnv\n",
                      out.read_text(encoding="utf-8"))

    def test_missing_launcher_raises(self):
        # install dir exists but no run_RF2NA.sh
        with self.assertRaises(FileNotFoundError):
            patch_run_script(self.install, self.work, "X")


class TestPatchMsaScript(unittest.TestCase):
    """Unit tests for the BFD comment-out patcher on make_protein_msa.sh."""

    # Synthetic launcher that mimics the upstream BFD-using shape.
    UPSTREAM_BODY = (
        "#!/bin/bash\n"
        "set -e\n"
        "DB_UR30=$PIPEDIR/UniRef30_2020_06/UniRef30_2020_06\n"
        "DB_BFD=$PIPEDIR/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt\n"
        "HHBLITS_UR30=\"hhblits -d $DB_UR30 -n 2 -e 1e-3\"\n"
        "HHBLITS_BFD=\"hhblits -d $DB_BFD -n 1 -e 1e-3\"\n"
        "$HHBLITS_UR30 -i prot.fa -o ur30.a3m\n"
        "$HHBLITS_BFD -i prot.fa -o bfd.a3m\n"
        "echo done\n"
    )

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.install = self.tmp / "install"
        (self.install / "input_prep").mkdir(parents=True)
        self.msa = self.install / "input_prep" / "make_protein_msa.sh"
        self.msa.write_text(self.UPSTREAM_BODY, encoding="utf-8")

    def test_skip_bfd_true_comments_three_lines(self):
        out = patch_msa_script(self.install, skip_bfd=True)
        self.assertEqual(out, self.msa)
        body = self.msa.read_text(encoding="utf-8")
        # Three lines should now start with "# " (commented).
        self.assertIn("# DB_BFD=$PIPEDIR/bfd/", body)
        self.assertIn("# HHBLITS_BFD=", body)
        self.assertIn("# $HHBLITS_BFD -i prot.fa", body)
        # patch tag is present so a future reader can `grep` for it.
        self.assertIn("patched by riboseer", body)
        # UniRef30 lines should NOT be touched.
        self.assertIn("DB_UR30=", body)
        self.assertIn("HHBLITS_UR30=", body)
        self.assertIn("$HHBLITS_UR30 -i prot.fa", body)
        # NB: HHBLITS_BFD assign line happens to match the
        # `$HHBLITS_BFD` regex too (the literal `$HHBLITS_BFD` doesn't
        # appear there, only `HHBLITS_BFD=...`), so only one comment is
        # added on that line.
        for line in body.splitlines():
            self.assertFalse(line.startswith("# # "),
                             f"double-commented line: {line!r}")

    def test_creates_backup_on_first_call(self):
        backup = self.msa.with_name(self.msa.name + ".riboseer_original")
        self.assertFalse(backup.is_file())
        patch_msa_script(self.install, skip_bfd=True)
        self.assertTrue(backup.is_file())
        self.assertEqual(
            backup.read_text(encoding="utf-8"), self.UPSTREAM_BODY,
        )

    def test_idempotent_across_repeat_calls(self):
        """Patching twice with same args should produce identical content
        (no double-comments, no compounding)."""
        patch_msa_script(self.install, skip_bfd=True)
        first = self.msa.read_text(encoding="utf-8")
        patch_msa_script(self.install, skip_bfd=True)
        second = self.msa.read_text(encoding="utf-8")
        self.assertEqual(first, second)

    def test_skip_bfd_false_restores_original(self):
        """skip_bfd=False after a patched run should put back the
        original upstream content (BFD lines un-commented)."""
        patch_msa_script(self.install, skip_bfd=True)
        body_after_skip = self.msa.read_text(encoding="utf-8")
        self.assertIn("# DB_BFD=", body_after_skip)

        patch_msa_script(self.install, skip_bfd=False)
        restored = self.msa.read_text(encoding="utf-8")
        self.assertEqual(restored, self.UPSTREAM_BODY)
        # Backup should still be there for future re-patches.
        self.assertTrue(
            self.msa.with_name(
                self.msa.name + ".riboseer_original"
            ).is_file()
        )

    def test_missing_msa_script_returns_none(self):
        """Some forks don't ship input_prep/make_protein_msa.sh — no-op."""
        self.msa.unlink()
        result = patch_msa_script(self.install, skip_bfd=True)
        self.assertIsNone(result)

    def test_custom_msa_subpath(self):
        """msa_subpath override lets a fork relocate the helper."""
        custom = self.install / "scripts" / "msa.sh"
        custom.parent.mkdir()
        custom.write_text(self.UPSTREAM_BODY, encoding="utf-8")
        out = patch_msa_script(
            self.install, skip_bfd=True, msa_subpath="scripts/msa.sh",
        )
        self.assertEqual(out, custom)
        self.assertIn(
            "# DB_BFD=", custom.read_text(encoding="utf-8"),
        )

    def test_already_commented_lines_get_double_commented(self):
        """Re-patching content that already starts with `# DB_BFD=` is
        guaranteed not to happen because we always restart from the
        backup (verified by the idempotent test). But verify the regex
        itself does NOT skip already-commented lines if someone passes
        already-commented content directly: this guards us against
        relying on the regex's own idempotency."""
        # We test this via the backup-roundtrip rather than calling the
        # regex directly. The behaviour to assert: "after backup is
        # seeded with already-patched content (an unusual case), a
        # subsequent skip_bfd=True call would re-patch what's there —
        # but the backup is the fresh upstream so this never happens
        # in practice." This test documents the invariant.
        backup = self.msa.with_name(self.msa.name + ".riboseer_original")
        # Seed the backup with already-patched content.
        backup.write_text(
            "# DB_BFD=foo  # patched by riboseer: BFD not available\n"
            "echo ok\n",
            encoding="utf-8",
        )
        patch_msa_script(self.install, skip_bfd=True)
        body = self.msa.read_text(encoding="utf-8")
        # Regex sees `^DB_BFD=` not `^# DB_BFD=`, so it doesn't
        # re-comment — body matches backup.
        self.assertEqual(
            body,
            "# DB_BFD=foo  # patched by riboseer: BFD not available\n"
            "echo ok\n",
        )


class TestRunTool(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.work = self.tmp / "work"
        self.work.mkdir()
        self.protein_fa = self.work / "p.fa"
        self.protein_fa.write_text(">p\nMKTV\n", encoding="utf-8")
        self.rna_fa = self.work / "r.fa"
        self.rna_fa.write_text(">r\nGCCG\n", encoding="utf-8")

    def _input_paths(self) -> dict:
        return {
            "protein_fa": self.protein_fa,
            "rna_fa": self.rna_fa,
            "target": "1un6_B_F",
        }

    def test_success_returns_output_dir(self):
        adapter = RF2NAAdapter()
        with patch(
            "step4_tool_adapters.adapters.rf2na_adapter.run_in_conda_env",
            return_value=_ok_run(),
        ):
            out = adapter.run_tool(self._input_paths(), self.work, _config(self.tmp))
        self.assertEqual(out, (self.work / "rf2na_output").resolve())
        self.assertTrue(out.is_dir())

    def test_command_uses_p_r_prefixes_and_install_cwd(self):
        captured = {}
        def fake(env, cmd, **kw):
            captured["env"] = env
            captured["cmd"] = cmd
            captured["cwd"] = kw.get("cwd")
            captured["timeout"] = kw.get("timeout")
            return _ok_run()
        adapter = RF2NAAdapter()
        cfg = _config(self.tmp)
        install_dir = cfg["tools"]["rosettafold2na"]["install_dir"]
        with patch(
            "step4_tool_adapters.adapters.rf2na_adapter.run_in_conda_env",
            side_effect=fake,
        ):
            adapter.run_tool(self._input_paths(), self.work, cfg)
        self.assertEqual(captured["env"], "RF2NA2")
        # By default we run the patched copy living under work_dir, not
        # the upstream `bash run_RF2NA.sh`.
        self.assertIn("bash ", captured["cmd"])
        self.assertIn("run_RF2NA_patched.sh", captured["cmd"])
        self.assertIn(f"P:{self.protein_fa.resolve()}", captured["cmd"])
        self.assertIn(f"R:{self.rna_fa.resolve()}", captured["cmd"])
        # cwd must still be the install dir so the patched script's
        # relative paths to bundled tools resolve as if it were original.
        self.assertEqual(captured["cwd"], install_dir)
        self.assertEqual(captured["timeout"], 14400)
        # Patched file lives under work_dir and is missing the old env name.
        patched_path = self.work.resolve() / "run_RF2NA_patched.sh"
        self.assertTrue(patched_path.is_file())
        body = patched_path.read_text(encoding="utf-8")
        self.assertIn("conda activate RF2NA2", body)
        self.assertNotIn("conda activate RF2NA\n", body)
        self.assertNotIn("source activate RF2NA\n", body)

    def test_missing_install_dir(self):
        adapter = RF2NAAdapter()
        cfg = _config(self.tmp)
        cfg["tools"]["rosettafold2na"]["install_dir"] = ""
        with self.assertRaises(ValueError):
            adapter.run_tool(self._input_paths(), self.work, cfg)

    def test_missing_protein_fasta(self):
        adapter = RF2NAAdapter()
        self.protein_fa.unlink()
        with patch(
            "step4_tool_adapters.adapters.rf2na_adapter.run_in_conda_env",
            return_value=_ok_run(),
        ):
            with self.assertRaises(FileNotFoundError):
                adapter.run_tool(self._input_paths(), self.work, _config(self.tmp))

    def test_run_tool_patches_make_protein_msa_when_present(self):
        """run_tool should also patch make_protein_msa.sh in-place when
        it exists in install_dir/input_prep/."""
        cfg = _config(self.tmp)
        install_dir = Path(cfg["tools"]["rosettafold2na"]["install_dir"])
        msa_dir = install_dir / "input_prep"
        msa_dir.mkdir(parents=True, exist_ok=True)
        msa_path = msa_dir / "make_protein_msa.sh"
        msa_path.write_text(
            "#!/bin/bash\n"
            "DB_BFD=$PIPEDIR/bfd/foo\n"
            "HHBLITS_BFD=\"hhblits -d $DB_BFD\"\n"
            "$HHBLITS_BFD -i x\n",
            encoding="utf-8",
        )

        adapter = RF2NAAdapter()
        with patch(
            "step4_tool_adapters.adapters.rf2na_adapter.run_in_conda_env",
            return_value=_ok_run(),
        ):
            adapter.run_tool(self._input_paths(), self.work, cfg)

        body = msa_path.read_text(encoding="utf-8")
        self.assertIn("# DB_BFD=", body)
        self.assertIn("# HHBLITS_BFD=", body)
        self.assertIn("# $HHBLITS_BFD", body)
        # Backup created.
        self.assertTrue(
            msa_path.with_name(
                msa_path.name + ".riboseer_original"
            ).is_file()
        )

    def test_run_tool_skip_bfd_false_keeps_msa_intact(self):
        cfg = _config(self.tmp)
        cfg["tools"]["rosettafold2na"]["skip_bfd"] = False
        install_dir = Path(cfg["tools"]["rosettafold2na"]["install_dir"])
        msa_dir = install_dir / "input_prep"
        msa_dir.mkdir(parents=True, exist_ok=True)
        msa_path = msa_dir / "make_protein_msa.sh"
        original = (
            "#!/bin/bash\n"
            "DB_BFD=$PIPEDIR/bfd/foo\n"
            "echo done\n"
        )
        msa_path.write_text(original, encoding="utf-8")

        adapter = RF2NAAdapter()
        with patch(
            "step4_tool_adapters.adapters.rf2na_adapter.run_in_conda_env",
            return_value=_ok_run(),
        ):
            adapter.run_tool(self._input_paths(), self.work, cfg)

        # No BFD comment-out — the file matches what we wrote.
        self.assertEqual(msa_path.read_text(encoding="utf-8"), original)

    def test_run_tool_no_msa_script_is_silent_noop(self):
        """If install_dir/input_prep/make_protein_msa.sh doesn't exist,
        the MSA patch silently no-ops; run_tool still completes."""
        cfg = _config(self.tmp)  # _make_install_dir doesn't create input_prep
        adapter = RF2NAAdapter()
        with patch(
            "step4_tool_adapters.adapters.rf2na_adapter.run_in_conda_env",
            return_value=_ok_run(),
        ):
            out = adapter.run_tool(self._input_paths(), self.work, cfg)
        self.assertTrue(out.is_dir())

    def test_run_tool_skip_script_patch_skips_msa_too(self):
        """skip_script_patch: true should prevent BOTH run_RF2NA.sh
        rewrite AND make_protein_msa.sh BFD patch."""
        cfg = _config(self.tmp, skip_script_patch=True)
        install_dir = Path(cfg["tools"]["rosettafold2na"]["install_dir"])
        msa_dir = install_dir / "input_prep"
        msa_dir.mkdir(parents=True, exist_ok=True)
        msa_path = msa_dir / "make_protein_msa.sh"
        original = "#!/bin/bash\nDB_BFD=$PIPEDIR/bfd/foo\n"
        msa_path.write_text(original, encoding="utf-8")

        adapter = RF2NAAdapter()
        with patch(
            "step4_tool_adapters.adapters.rf2na_adapter.run_in_conda_env",
            return_value=_ok_run(),
        ):
            adapter.run_tool(self._input_paths(), self.work, cfg)

        # No BFD patching, no backup created.
        self.assertEqual(msa_path.read_text(encoding="utf-8"), original)
        self.assertFalse(
            msa_path.with_name(
                msa_path.name + ".riboseer_original"
            ).is_file()
        )

    def test_skip_script_patch_falls_back_to_raw_launcher(self):
        """``skip_script_patch: true`` runs ``bash run_RF2NA.sh ...``
        unchanged (cwd=install_dir handles the resolution). For sites
        whose launcher doesn't have the hardcoded activate line."""
        captured = {}

        def fake(env, cmd, **kw):
            captured["cmd"] = cmd
            return _ok_run()

        adapter = RF2NAAdapter()
        with patch(
            "step4_tool_adapters.adapters.rf2na_adapter.run_in_conda_env",
            side_effect=fake,
        ):
            adapter.run_tool(
                self._input_paths(), self.work,
                _config(self.tmp, skip_script_patch=True),
            )
        self.assertIn("bash run_RF2NA.sh ", captured["cmd"])
        self.assertNotIn("run_RF2NA_patched.sh", captured["cmd"])
        self.assertFalse((self.work / "run_RF2NA_patched.sh").is_file())

    def test_failure_propagated(self):
        adapter = RF2NAAdapter()
        with patch(
            "step4_tool_adapters.adapters.rf2na_adapter.run_in_conda_env",
            return_value=_bad_run("DB missing"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                adapter.run_tool(self._input_paths(), self.work, _config(self.tmp))
        msg = str(ctx.exception)
        self.assertIn("rc=2", msg)
        self.assertIn("DB missing", msg)


# ---------- parse_output ---------------------------------------------------


class TestParseOutput(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "rf2na_output"
        # Mimic upstream layout: <out>/models/model_00.{pdb,npz}
        self.models_dir = self.out / "models"
        self.models_dir.mkdir(parents=True)
        _make_synthetic_complex_pdb(self.models_dir / "model_00.pdb")
        _write_pae_npz(self.models_dir / "model_00.npz")

    def test_full_parse(self):
        adapter = RF2NAAdapter()
        pred = adapter.parse_output(self.out, _sample_json(), _config(self.tmp))
        self.assertTrue(pred.success, msg=pred.error_message)
        # Single contact pair: protein res 1 ↔ RNA nt 1
        self.assertEqual(pred.binding_protein_residues, [1])
        self.assertEqual(pred.binding_rna_nucleotides, [1])
        # plddt_mean: average of (85, 70, 60) = 71.667
        self.assertAlmostEqual(pred.plddt_mean, (85 + 70 + 60) / 3, places=2)
        # pae_mean: mean of the 3x3 matrix (sum = 48, n = 9)
        self.assertAlmostEqual(pred.pae_mean, 48 / 9, places=4)
        self.assertIsNone(pred.iptm_score)  # RF2NA doesn't emit it
        self.assertTrue(pred.predicted_structure_path.endswith("model_00.pdb"))
        self.assertEqual(set(pred.per_residue_confidence.keys()), {1, 2, 3})
        # Cat A populates structure / pLDDT, never pockets.
        self.assertIsNone(pred.pockets)

    def test_missing_pdb_failure(self):
        (self.models_dir / "model_00.pdb").unlink()
        adapter = RF2NAAdapter()
        pred = adapter.parse_output(self.out, _sample_json(), _config(self.tmp))
        self.assertFalse(pred.success)
        self.assertIn("model_00.pdb", pred.error_message)

    def test_no_npz_means_pae_is_none(self):
        (self.models_dir / "model_00.npz").unlink()
        adapter = RF2NAAdapter()
        pred = adapter.parse_output(self.out, _sample_json(), _config(self.tmp))
        self.assertTrue(pred.success, msg=pred.error_message)
        self.assertIsNone(pred.pae_mean)
        # pLDDT still comes out of the PDB.
        self.assertAlmostEqual(pred.plddt_mean, (85 + 70 + 60) / 3, places=2)

    def test_nested_models_dir_is_found(self):
        # Move the PDB into a deeper nesting; rglob should still find it.
        nested = self.out / "predictions" / "1un6_B_F"
        nested.mkdir(parents=True)
        (self.models_dir / "model_00.pdb").rename(nested / "model_00.pdb")
        adapter = RF2NAAdapter()
        pred = adapter.parse_output(self.out, _sample_json(), _config(self.tmp))
        self.assertTrue(pred.success, msg=pred.error_message)
        self.assertEqual(pred.binding_protein_residues, [1])


# ---------- end-to-end predict() ------------------------------------------


class TestPredictPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_full_predict_with_stub(self):
        captured = {}

        def fake(env, cmd, **kwargs):
            captured["env"] = env
            captured["cmd"] = cmd
            # Drop a synthetic complex PDB into <out>/models/. The output
            # dir is the first positional after the bash launcher path.
            tokens = cmd.split()
            # The launcher is now `<work_dir>/run_RF2NA_patched.sh`; we
            # locate it by suffix instead of exact name.
            launcher_idx = next(i for i, t in enumerate(tokens)
                                if t.endswith("run_RF2NA_patched.sh")
                                or t.endswith("run_RF2NA.sh"))
            out_dir = Path(tokens[launcher_idx + 1])
            models_dir = out_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            _make_synthetic_complex_pdb(models_dir / "model_00.pdb")
            _write_pae_npz(models_dir / "model_00.npz")
            return _ok_run()

        adapter = RF2NAAdapter()
        with patch(
            "step4_tool_adapters.adapters.rf2na_adapter.run_in_conda_env",
            side_effect=fake,
        ):
            pred = adapter.predict(
                _sample_json(), self.tmp / "work", _config(self.tmp),
            )
        self.assertEqual(captured["env"], "RF2NA2")
        self.assertTrue(pred.success, msg=pred.error_message)
        self.assertEqual(pred.binding_protein_residues, [1])
        self.assertEqual(pred.binding_rna_nucleotides, [1])
        self.assertIsNotNone(pred.runtime_seconds)
        self.assertGreaterEqual(pred.runtime_seconds, 0.0)

    def test_missing_sequence_yields_failure_record(self):
        adapter = RF2NAAdapter()
        bad = _sample_json()
        bad["rna"]["sequence"] = ""
        pred = adapter.predict(bad, self.tmp / "work", _config(self.tmp))
        self.assertFalse(pred.success)
        self.assertIn("rna.sequence", pred.error_message)


# ---------- single-sequence mode ------------------------------------------


class TestMakeSingleSeqA3m(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_writes_query_only_record(self):
        p = make_single_seq_a3m(self.tmp / "q.a3m", "MKTVRQ", name="prot")
        self.assertEqual(p.read_text(encoding="utf-8"), ">prot\nMKTVRQ\n")

    def test_default_name_is_query(self):
        p = make_single_seq_a3m(self.tmp / "r.afa", "GCCGGU")
        self.assertEqual(p.read_text(encoding="utf-8"), ">query\nGCCGGU\n")

    def test_rejects_whitespace_and_empty(self):
        with self.assertRaises(ValueError):
            make_single_seq_a3m(self.tmp / "a", "AC GT")
        with self.assertRaises(ValueError):
            make_single_seq_a3m(self.tmp / "b", "")


class TestPrepareInputSingleSeq(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_single_seq_emits_a3m_and_afa(self):
        adapter = RF2NAAdapter()
        paths = adapter.prepare_input(
            _sample_json(), self.tmp, _config(self.tmp, single_seq=True))
        self.assertIn("protein_a3m", paths)
        self.assertIn("rna_afa", paths)
        a3m = Path(paths["protein_a3m"]).read_text(encoding="utf-8")
        afa = Path(paths["rna_afa"]).read_text(encoding="utf-8")
        # query-only: exactly one header + one sequence line.
        self.assertEqual(a3m.count(">"), 1)
        self.assertEqual(afa.count(">"), 1)
        self.assertEqual(len(a3m.strip().splitlines()), 2)

    def test_legacy_default_omits_a3m(self):
        adapter = RF2NAAdapter()
        paths = adapter.prepare_input(
            _sample_json(), self.tmp, _config(self.tmp))  # single_seq=False
        self.assertNotIn("protein_a3m", paths)
        self.assertNotIn("rna_afa", paths)


class TestRunToolSingleSeq(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.work = self.tmp / "work"
        self.work.mkdir()

    def _inputs(self, cfg):
        return RF2NAAdapter().prepare_input(_sample_json(), self.work, cfg)

    def test_single_seq_calls_predict_directly(self):
        cfg = _config(self.tmp, single_seq=True)
        rcfg = cfg["tools"]["rosettafold2na"]
        install_dir = rcfg["install_dir"]
        inputs = self._inputs(cfg)
        captured = {}

        def fake(env, cmd, **kw):
            captured.update(env=env, cmd=cmd, cwd=kw.get("cwd"))
            return _ok_run()

        with patch(
            "step4_tool_adapters.adapters.rf2na_adapter.run_in_conda_env",
            side_effect=fake,
        ):
            out = RF2NAAdapter().run_tool(inputs, self.work, cfg)

        self.assertEqual(out, (self.work / "rf2na_output").resolve())
        self.assertEqual(captured["env"], "RF2NA2")
        cmd = captured["cmd"]
        # Direct predict.py call — NOT the run_RF2NA.sh launcher.
        self.assertNotIn("run_RF2NA", cmd)
        self.assertIn("network/predict.py", cmd.replace("\\", "/"))
        self.assertIn("-inputs", cmd)
        self.assertIn(f"P:{Path(inputs['protein_a3m']).resolve()}", cmd)
        self.assertIn(f"R:{Path(inputs['rna_afa']).resolve()}", cmd)
        self.assertIn("-prefix", cmd)
        self.assertIn("-model", cmd)
        self.assertIn("RF2NA_apr23.pt", cmd)
        self.assertEqual(captured["cwd"], str(Path(install_dir)))
        # No MSA-search patch artefacts in single-seq mode.
        self.assertFalse(
            (self.work.resolve() / "run_RF2NA_patched.sh").exists())

    def test_templates_db_appended_when_set(self):
        cfg = _config(self.tmp, single_seq=True)
        cfg["tools"]["rosettafold2na"]["templates_db"] = "pdb100/db"
        inputs = self._inputs(cfg)
        captured = {}
        with patch(
            "step4_tool_adapters.adapters.rf2na_adapter.run_in_conda_env",
            side_effect=lambda e, c, **k: (captured.update(cmd=c)
                                           or _ok_run()),
        ):
            RF2NAAdapter().run_tool(inputs, self.work, cfg)
        self.assertIn("-db", captured["cmd"])
        self.assertIn("pdb100", captured["cmd"].replace("\\", "/"))

    def test_missing_a3m_raises(self):
        cfg = _config(self.tmp, single_seq=True)
        # Hand it legacy-only inputs (no a3m/afa keys).
        bad = {"protein_fa": self.work / "p.fa",
               "rna_fa": self.work / "r.fa", "target": "t"}
        (self.work / "p.fa").write_text(">p\nMK\n", encoding="utf-8")
        (self.work / "r.fa").write_text(">r\nGC\n", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            RF2NAAdapter().run_tool(bad, self.work, cfg)


class TestPredictPipelineSingleSeq(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_full_single_seq_predict_with_stub(self):
        cfg = _config(self.tmp, single_seq=True)

        def fake(env, cmd, **kw):
            # single-seq writes -prefix <out>/model → <out>/model.pdb
            tokens = cmd.split()
            out_dir = Path(tokens[tokens.index("-prefix") + 1]).parent
            out_dir.mkdir(parents=True, exist_ok=True)
            _make_synthetic_complex_pdb(out_dir / "model.pdb")
            _write_pae_npz(out_dir / "model.npz")
            return _ok_run()

        with patch(
            "step4_tool_adapters.adapters.rf2na_adapter.run_in_conda_env",
            side_effect=fake,
        ):
            pred = RF2NAAdapter().predict(
                _sample_json(), self.tmp / "work", cfg)

        self.assertTrue(pred.success, msg=pred.error_message)
        self.assertEqual(pred.binding_protein_residues, [1])
        self.assertEqual(pred.binding_rna_nucleotides, [1])
        self.assertTrue(
            pred.predicted_structure_path.endswith("model.pdb"))
        # Cat A distance-based per-residue score is populated and in
        # [0, 1] for every protein residue.
        self.assertIsNotNone(pred.per_residue_pae_score)
        self.assertTrue(pred.per_residue_pae_score)
        for v in pred.per_residue_pae_score.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)
        # pLDDT still extracted from the PDB B-factors.
        self.assertAlmostEqual(
            pred.plddt_mean, (85 + 70 + 60) / 3, places=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
