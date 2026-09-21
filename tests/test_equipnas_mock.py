"""Mock tests for EquiPNAS adapter.

What this covers:
  - target name sanitisation for awkward sample IDs
  - FASTA writer (line-wrap, whitespace rejection)
  - input layout staging (PDB/FASTA/input.list/distmaps dir)
  - per-residue CSV parser: comma / whitespace / mixed columns / chain
    prefix tokens / out-of-range probabilities / missing file
  - find_equipnas_output: .csv / .pred / nested fallback
  - prepare_input full path: stages files and invokes the 3 preprocessing
    scripts in EquiPNAS env (subprocess mocked)
  - prepare_input with skip_preprocess=True does not call any subprocess
  - run_tool wraps EquiPNAS.py inside `conda run -n EquiPNAS` and
    surfaces failures as RuntimeError
  - parse_output: full parse / missing CSV / threshold override /
    chain filter
  - end-to-end predict() with stubbed run_in_conda_env that drops a
    synthetic CSV
  - preprocessing-script symlinking, per-residue feature
    array alignment (pad / trim / 1D), output dir auto-creation
"""
from __future__ import annotations

import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step4_tool_adapters.adapters.equipnas_adapter import (  # noqa: E402
    EquiPNASAdapter, PREPROCESS_SCRIPTS, _align_feature_arrays,
    _coerce_probability, _coerce_residue_index, _read_fasta_seq_len,
    _sanitize_target_name, _symlink_preprocessing_scripts,
    _try_parse_single_column_floats, find_equipnas_output,
    parse_equipnas_csv, parse_equipnas_output, stage_equipnas_input,
    write_fasta,
)
from step4_tool_adapters.tool_runner import ToolRunResult  # noqa: E402


# ---------- fixtures -------------------------------------------------------


RNA2P_DIR = Path("/path/to/raw_structures")  # local raw PDB store


def _sample_json() -> dict:
    """Lightweight sample JSON; enough fields for the adapter."""
    return {
        "sample_id": "1un6_B_F",
        "source_pdb": "1un6",
        "protein": {
            "chain_id": "B",
            "sequence": "MYVCHFENCGKAFKKHNQLKVHQFSHTQQLPYECPHEGCDKR",
        },
    }


def _config(raw_dir: Path, *, skip_preprocess: bool = False) -> dict:
    """Default fixture config.

    added PSSM / ESM-2 / MSA-dummy / distance-map generation.
    Tests that pre-date v7 don't mock those new subprocess calls, so this
    fixture flips every v7 ``skip_*`` flag on by default — preserving the
    existing call-count assertions. The dedicated v7 tests build their
    own configs with the flags off to exercise the new code paths.
    """
    return {
        "tools": {
            "equipnas": {
                "conda_env": "EquiPNAS",
                "install_dir": "/fake/EquiPNAS",
                "model_path": "models/EquiPNAS-RNA/E-l12-768.pt",
                "timeout": 600,
                "preprocess_timeout": 600,
                "residue_prob_threshold": 0.5,
                "skip_preprocess": skip_preprocess,
                # v7: opt out of every new generator + the psiblast probe.
                "skip_psiblast_check": True,
                "skip_pssm": True,
                "skip_esm2": True,
                "skip_msa_dummy": True,
                "skip_distance_map": True,
            },
        },
        "structure_source": {"raw_dir": str(raw_dir)},
        "log_dir": str(raw_dir / "_logs"),
    }


def _ok_run(stdout: str = "ok", **_kw) -> ToolRunResult:
    return ToolRunResult(
        command="fake", cwd=None, returncode=0,
        stdout=stdout, stderr="", runtime_seconds=0.4, log_path=None,
    )


def _bad_run(stderr: str = "fail", rc: int = 2) -> ToolRunResult:
    return ToolRunResult(
        command="fake", cwd=None, returncode=rc,
        stdout="", stderr=stderr, runtime_seconds=0.1, log_path=None,
    )


def _mock_csv(rows: list[tuple[int, float]],
              header: str = "residue_id,probability",
              sep: str = ",") -> str:
    lines = [header]
    for idx, p in rows:
        lines.append(f"{idx}{sep}{p}")
    return "\n".join(lines) + "\n"


# ---------- sanitisation ---------------------------------------------------


class TestSanitizeTargetName(unittest.TestCase):
    def test_alphanum_passes(self):
        self.assertEqual(_sanitize_target_name("1un6_B_F"), "1un6_B_F")

    def test_dash_replaced(self):
        self.assertEqual(_sanitize_target_name("2wj8_A_-a"), "2wj8_A__a")

    def test_strip_edge_underscores(self):
        # Leading "-" → "_" then stripped, trailing "!" → "_" then stripped.
        self.assertEqual(_sanitize_target_name("-foo!"), "foo")

    def test_all_invalid_falls_back(self):
        self.assertEqual(_sanitize_target_name("!!!"), "sample")


# ---------- FASTA writer ---------------------------------------------------


class TestWriteFasta(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_basic_wrap(self):
        seq = "A" * 130
        out = self.tmp / "x.fasta"
        write_fasta(out, "tgt", seq, line_width=60)
        body = out.read_text(encoding="utf-8").splitlines()
        self.assertEqual(body[0], ">tgt")
        # 60 + 60 + 10 chars
        self.assertEqual(body[1], "A" * 60)
        self.assertEqual(body[2], "A" * 60)
        self.assertEqual(body[3], "A" * 10)

    def test_no_wrap(self):
        seq = "ACDE" * 20
        out = self.tmp / "x.fasta"
        write_fasta(out, "tgt", seq, line_width=0)
        body = out.read_text(encoding="utf-8").splitlines()
        self.assertEqual(body[1], seq)

    def test_rejects_whitespace_in_sequence(self):
        with self.assertRaises(ValueError):
            write_fasta(self.tmp / "x.fasta", "tgt", "ACDE FGH")

    def test_empty_sequence_raises(self):
        with self.assertRaises(ValueError):
            write_fasta(self.tmp / "x.fasta", "tgt", "")


# ---------- residue index / probability coercion --------------------------


class TestCoerceResidueIndex(unittest.TestCase):
    def test_plain_int(self):
        self.assertEqual(_coerce_residue_index("14"), 14)

    def test_chain_underscore_token(self):
        self.assertEqual(_coerce_residue_index("A_14"), 14)

    def test_chain_glued_token(self):
        self.assertEqual(_coerce_residue_index("A14"), 14)

    def test_with_insertion_code(self):
        self.assertEqual(_coerce_residue_index("A_14B"), 14)

    def test_garbage(self):
        self.assertIsNone(_coerce_residue_index(""))
        self.assertIsNone(_coerce_residue_index("foo"))


class TestCoerceProbability(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(_coerce_probability("0.42"), 0.42)
        self.assertEqual(_coerce_probability("1.0"), 1.0)
        self.assertEqual(_coerce_probability("0"), 0.0)

    def test_drops_out_of_range(self):
        # logit-style outputs not allowed; we expect a probability
        self.assertIsNone(_coerce_probability("1.5"))
        self.assertIsNone(_coerce_probability("-0.1"))

    def test_garbage(self):
        self.assertIsNone(_coerce_probability(""))
        self.assertIsNone(_coerce_probability("foo"))


# ---------- per-residue CSV parser ----------------------------------------


class TestParseEquipnasCsv(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def _write(self, name: str, content: str) -> Path:
        p = self.tmp / name
        p.write_text(content, encoding="utf-8")
        return p

    def test_basic_comma(self):
        path = self._write(
            "x.csv",
            _mock_csv([(1, 0.10), (2, 0.95), (3, 0.50)]),
        )
        out = parse_equipnas_csv(path)
        self.assertEqual(out, {1: 0.10, 2: 0.95, 3: 0.50})

    def test_alternative_column_names(self):
        path = self._write(
            "x.csv",
            "res_id,score\n10,0.8\n11,0.2\n",
        )
        out = parse_equipnas_csv(path)
        self.assertEqual(out, {10: 0.8, 11: 0.2})

    def test_chain_prefix_tokens(self):
        path = self._write(
            "x.csv",
            "residue_id,probability\nA_5,0.7\nA_6,0.4\nB_5,0.9\n",
        )
        # Without chain_filter we keep all (B_5 collides with A_5;
        # later row wins — that's acceptable as a dirty-input guard).
        out = parse_equipnas_csv(path)
        self.assertIn(5, out)
        self.assertIn(6, out)

    def test_chain_filter_drops_other_chains(self):
        path = self._write(
            "x.csv",
            "chain,residue_id,probability\n"
            "A,5,0.7\nA,6,0.4\nB,5,0.9\nB,7,0.3\n",
        )
        out = parse_equipnas_csv(path, chain_filter="A")
        self.assertEqual(out, {5: 0.7, 6: 0.4})

    def test_whitespace_separated_no_header(self):
        # 2-col headerless layout; parser falls back to whitespace split.
        path = self._write("x.txt", "1 0.10\n2 0.95\n# comment\n3 0.50\n")
        out = parse_equipnas_csv(path)
        self.assertEqual(out, {1: 0.10, 2: 0.95, 3: 0.50})

    def test_drops_out_of_range_rows(self):
        path = self._write(
            "x.csv",
            "residue_id,probability\n1,0.5\n2,1.5\n3,-0.1\n4,0.9\n",
        )
        out = parse_equipnas_csv(path)
        self.assertEqual(out, {1: 0.5, 4: 0.9})

    def test_missing_file_returns_empty(self):
        self.assertEqual(parse_equipnas_csv(self.tmp / "no.csv"), {})

    def test_empty_file_returns_empty(self):
        path = self._write("x.csv", "")
        self.assertEqual(parse_equipnas_csv(path), {})


# ---------- find_equipnas_output -----------------------------------------


class TestFindOutput(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_direct_csv(self):
        p = self.tmp / "1un6_B_F.csv"
        p.write_text("x", encoding="utf-8")
        self.assertEqual(find_equipnas_output(self.tmp, "1un6_B_F"), p)

    def test_pred_fallback(self):
        p = self.tmp / "1un6_B_F.pred"
        p.write_text("x", encoding="utf-8")
        self.assertEqual(find_equipnas_output(self.tmp, "1un6_B_F"), p)

    def test_nested_search(self):
        nested = self.tmp / "predictions"
        nested.mkdir()
        p = nested / "1un6_B_F.csv"
        p.write_text("x", encoding="utf-8")
        self.assertEqual(find_equipnas_output(self.tmp, "1un6_B_F"), p)

    def test_missing_returns_none(self):
        self.assertIsNone(find_equipnas_output(self.tmp, "missing"))


# ---------- prepare_input --------------------------------------------------


class TestStageInput(unittest.TestCase):
    """stage_equipnas_input is a thin helper but it touches the real PDB
    extraction code, so verify file shapes directly."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    @unittest.skipUnless(
        (RNA2P_DIR / "1un6.pdb").is_file(),
        "rna2p_balanced/1un6.pdb not available — skip",
    )
    def test_stages_pdb_fasta_inputlist(self):
        staged = stage_equipnas_input(
            raw_path=RNA2P_DIR / "1un6.pdb",
            chain_id="B",
            sequence="MYV",  # short stub — content tested separately
            target="1un6_B_F",
            work_dir=self.tmp,
        )
        self.assertTrue(staged["pdb"].is_file())
        self.assertTrue(staged["fasta"].is_file())
        self.assertTrue(staged["input_list"].is_file())
        # No trailing newline — see equipnas_adapter.stage_equipnas_input
        # for the rationale (upstream split("\n") would otherwise see an
        # empty target).
        self.assertEqual(
            staged["input_list"].read_text(encoding="utf-8"),
            "1un6_B_F",
        )
        # distmaps dir is created (empty) so preprocessing can write there
        self.assertTrue((staged["preprocessed_dir"] / "distmaps").is_dir())


class TestPrepareInput(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.raw = self.tmp / "raw"
        self.raw.mkdir()

    def test_no_raw_file_raises(self):
        adapter = EquiPNASAdapter()
        with self.assertRaises(FileNotFoundError):
            adapter.prepare_input(_sample_json(), self.tmp, _config(self.raw))

    def test_missing_sequence_raises(self):
        adapter = EquiPNASAdapter()
        bad = _sample_json()
        bad["protein"]["sequence"] = ""
        with self.assertRaises(ValueError):
            adapter.prepare_input(bad, self.tmp, _config(self.raw))

    @unittest.skipUnless(
        (RNA2P_DIR / "1un6.pdb").is_file(),
        "rna2p_balanced/1un6.pdb not available — skip",
    )
    def test_skip_preprocess_does_not_call_subprocess(self):
        # Seed raw_dir with the real PDB.
        (self.raw / "1un6.pdb").write_bytes(
            (RNA2P_DIR / "1un6.pdb").read_bytes()
        )
        adapter = EquiPNASAdapter()
        with patch(
            "step4_tool_adapters.adapters.equipnas_adapter.run_in_conda_env",
        ) as m:
            paths = adapter.prepare_input(
                _sample_json(), self.tmp, _config(self.raw, skip_preprocess=True),
            )
        self.assertFalse(m.called)
        self.assertTrue(paths["pdb"].is_file())
        self.assertTrue(paths["fasta"].is_file())

    @unittest.skipUnless(
        (RNA2P_DIR / "1un6.pdb").is_file(),
        "rna2p_balanced/1un6.pdb not available — skip",
    )
    def test_runs_three_preprocessing_scripts_in_order(self):
        (self.raw / "1un6.pdb").write_bytes(
            (RNA2P_DIR / "1un6.pdb").read_bytes()
        )
        adapter = EquiPNASAdapter()

        seen: list[tuple[str, str]] = []

        def fake_run(env, cmd, **kwargs):
            seen.append((env, cmd))
            # The DSSP probe needs a non-empty stdout to count as success.
            if "command -v" in cmd:
                return _ok_run(stdout="/opt/conda/bin/mkdssp")
            if "import esm" in cmd:
                return _ok_run()
            # mkdssp writes a non-empty .dssp file on success.
            if cmd.startswith("mkdssp ") or cmd.startswith("mkdssp -i"):
                tokens = shlex.split(cmd)
                # Output path is either the second positional or after "-o".
                if "-o" in tokens:
                    out = Path(tokens[tokens.index("-o") + 1])
                else:
                    out = Path(tokens[2])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text("HEADER mock dssp\n", encoding="utf-8")
            return _ok_run()

        with patch(
            "step4_tool_adapters.adapters.equipnas_adapter.run_in_conda_env",
            side_effect=fake_run,
        ):
            adapter.prepare_input(_sample_json(), self.tmp, _config(self.raw))

        # 1 dssp probe + 1 esm probe + 1 mkdssp invocation + 3 preprocessing.
        dssp_probe = [(e, c) for e, c in seen if "command -v" in c]
        esm_probe = [(e, c) for e, c in seen if "import esm" in c]
        mkdssp_calls = [(e, c) for e, c in seen if c.startswith("mkdssp ")]
        script_calls = [(e, c) for e, c in seen
                        if "command -v" not in c
                        and "import esm" not in c
                        and not c.startswith("mkdssp ")]
        self.assertEqual(len(dssp_probe), 1)
        self.assertEqual(len(esm_probe), 1)
        self.assertEqual(len(mkdssp_calls), 1)
        self.assertEqual(len(script_calls), 3)
        # Probes run before mkdssp, mkdssp runs before the first script.
        first_mkdssp_idx = next(i for i, (_, c) in enumerate(seen)
                                if c.startswith("mkdssp "))
        first_script_idx = next(i for i, (_, c) in enumerate(seen)
                                if c.startswith("python ") and "import esm" not in c)
        esm_idx = next(i for i, (_, c) in enumerate(seen) if "import esm" in c)
        self.assertLess(esm_idx, first_mkdssp_idx)
        self.assertLess(first_mkdssp_idx, first_script_idx)
        for (env, cmd), expected in zip(script_calls, PREPROCESS_SCRIPTS):
            self.assertEqual(env, "EquiPNAS")
            self.assertIn(expected, cmd)
        # The middle script gets explicit -i / -o args.
        _, mid_cmd = script_calls[1]
        self.assertIn(" -i ", mid_cmd)
        self.assertIn(" -o ", mid_cmd)

    @unittest.skipUnless(
        (RNA2P_DIR / "1un6.pdb").is_file(),
        "rna2p_balanced/1un6.pdb not available — skip",
    )
    def test_preprocessing_failure_raises(self):
        (self.raw / "1un6.pdb").write_bytes(
            (RNA2P_DIR / "1un6.pdb").read_bytes()
        )
        adapter = EquiPNASAdapter()

        # DSSP probe + ESM probe + mkdssp succeed so we reach the real
        # scripts; the 3 preprocessing scripts then fail.
        def staged_run(env, cmd, **kwargs):
            if "command -v" in cmd:
                return _ok_run(stdout="/opt/conda/bin/mkdssp")
            if "import esm" in cmd:
                return _ok_run()
            if cmd.startswith("mkdssp "):
                tokens = shlex.split(cmd)
                out = Path(tokens[tokens.index("-o") + 1]
                           if "-o" in tokens else tokens[2])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text("HEADER mock\n", encoding="utf-8")
                return _ok_run()
            return _bad_run("missing PSSM")

        with patch(
            "step4_tool_adapters.adapters.equipnas_adapter.run_in_conda_env",
            side_effect=staged_run,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                adapter.prepare_input(_sample_json(), self.tmp, _config(self.raw))
        self.assertIn("rc=2", str(ctx.exception))

    @unittest.skipUnless(
        (RNA2P_DIR / "1un6.pdb").is_file(),
        "rna2p_balanced/1un6.pdb not available — skip",
    )
    def test_dssp_missing_raises_clear_error(self):
        """If `which mkdssp` returns empty stdout, we should fail fast with
        a message that names DSSP and gives the install command — not run
        any preprocessing scripts."""
        (self.raw / "1un6.pdb").write_bytes(
            (RNA2P_DIR / "1un6.pdb").read_bytes()
        )
        adapter = EquiPNASAdapter()

        seen_cmds: list[str] = []

        def fake_run(env, cmd, **kwargs):
            seen_cmds.append(cmd)
            if "command -v" in cmd:
                # Probe "succeeds" exit-wise but emits no path → DSSP missing.
                return _ok_run(stdout="")
            return _ok_run()

        with patch(
            "step4_tool_adapters.adapters.equipnas_adapter.run_in_conda_env",
            side_effect=fake_run,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                adapter.prepare_input(_sample_json(), self.tmp, _config(self.raw))
        msg = str(ctx.exception)
        self.assertIn("DSSP", msg)
        self.assertIn("mkdssp", msg)
        self.assertIn("conda install", msg)
        # No preprocessing script was launched after the probe.
        non_probe = [c for c in seen_cmds if "command -v" not in c]
        self.assertEqual(non_probe, [])

    @unittest.skipUnless(
        (RNA2P_DIR / "1un6.pdb").is_file(),
        "rna2p_balanced/1un6.pdb not available — skip",
    )
    def test_skip_dssp_check_bypasses_probe(self):
        """``skip_dssp_check: true`` should skip the probe entirely — no
        ``command -v`` call goes to ``run_in_conda_env``. ESM probe +
        mkdssp themselves still run."""
        (self.raw / "1un6.pdb").write_bytes(
            (RNA2P_DIR / "1un6.pdb").read_bytes()
        )
        adapter = EquiPNASAdapter()
        cfg = _config(self.raw)
        cfg["tools"]["equipnas"]["skip_dssp_check"] = True

        seen_cmds: list[str] = []

        def fake_run(env, cmd, **kwargs):
            seen_cmds.append(cmd)
            if cmd.startswith("mkdssp "):
                tokens = shlex.split(cmd)
                out = Path(tokens[tokens.index("-o") + 1]
                           if "-o" in tokens else tokens[2])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text("HEADER mock\n", encoding="utf-8")
            return _ok_run()

        with patch(
            "step4_tool_adapters.adapters.equipnas_adapter.run_in_conda_env",
            side_effect=fake_run,
        ):
            adapter.prepare_input(_sample_json(), self.tmp, cfg)
        self.assertFalse(any("command -v" in c for c in seen_cmds))
        # 1 esm probe + 1 mkdssp + 3 preprocessing scripts (dssp probe skipped).
        self.assertEqual(len(seen_cmds), 5)
        self.assertIn("import esm", seen_cmds[0])

    @unittest.skipUnless(
        (RNA2P_DIR / "1un6.pdb").is_file(),
        "rna2p_balanced/1un6.pdb not available — skip",
    )
    def test_mkdssp_falls_back_to_flag_form(self):
        """If positional `mkdssp <pdb> <dssp>` fails (older mkdssp 3.x),
        the adapter retries with `-i / -o` flags."""
        (self.raw / "1un6.pdb").write_bytes(
            (RNA2P_DIR / "1un6.pdb").read_bytes()
        )
        adapter = EquiPNASAdapter()
        seen: list[str] = []

        def fake_run(env, cmd, **kwargs):
            seen.append(cmd)
            if "command -v" in cmd:
                return _ok_run(stdout="/opt/conda/bin/mkdssp")
            if "import esm" in cmd:
                return _ok_run()
            if cmd.startswith("mkdssp ") and "-i" not in cmd:
                # Positional form: simulate failure (no output file).
                return _bad_run("unrecognized option", rc=1)
            if cmd.startswith("mkdssp -i"):
                # Flag form: succeed and write the file.
                tokens = shlex.split(cmd)
                out = Path(tokens[tokens.index("-o") + 1])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text("HEADER mock\n", encoding="utf-8")
                return _ok_run()
            return _ok_run()

        with patch(
            "step4_tool_adapters.adapters.equipnas_adapter.run_in_conda_env",
            side_effect=fake_run,
        ):
            adapter.prepare_input(_sample_json(), self.tmp, _config(self.raw))

        mkdssp_calls = [c for c in seen if c.startswith("mkdssp ")]
        self.assertEqual(len(mkdssp_calls), 2)
        self.assertNotIn("-i", mkdssp_calls[0])
        self.assertIn("-i", mkdssp_calls[1])

    @unittest.skipUnless(
        (RNA2P_DIR / "1un6.pdb").is_file(),
        "rna2p_balanced/1un6.pdb not available — skip",
    )
    def test_mkdssp_failure_raises(self):
        """If both mkdssp invocation forms fail, prepare_input raises
        with a message that names the target."""
        (self.raw / "1un6.pdb").write_bytes(
            (RNA2P_DIR / "1un6.pdb").read_bytes()
        )
        adapter = EquiPNASAdapter()

        def fake_run(env, cmd, **kwargs):
            if "command -v" in cmd:
                return _ok_run(stdout="/opt/conda/bin/mkdssp")
            if "import esm" in cmd:
                return _ok_run()
            if cmd.startswith("mkdssp "):
                return _bad_run("malformed PDB", rc=2)
            return _ok_run()

        with patch(
            "step4_tool_adapters.adapters.equipnas_adapter.run_in_conda_env",
            side_effect=fake_run,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                adapter.prepare_input(_sample_json(), self.tmp, _config(self.raw))
        msg = str(ctx.exception)
        self.assertIn("mkdssp", msg)
        self.assertIn("1un6_B_F", msg)
        self.assertIn("malformed PDB", msg)

    @unittest.skipUnless(
        (RNA2P_DIR / "1un6.pdb").is_file(),
        "rna2p_balanced/1un6.pdb not available — skip",
    )
    def test_skip_mkdssp_bypasses_dssp_generation(self):
        """``skip_mkdssp: true`` means mkdssp isn't invoked (only the
        probes + 3 preprocessing scripts run). For sites where DSSP
        files are pre-staged externally."""
        (self.raw / "1un6.pdb").write_bytes(
            (RNA2P_DIR / "1un6.pdb").read_bytes()
        )
        adapter = EquiPNASAdapter()
        cfg = _config(self.raw)
        cfg["tools"]["equipnas"]["skip_mkdssp"] = True

        seen_cmds: list[str] = []

        def fake_run(env, cmd, **kwargs):
            seen_cmds.append(cmd)
            if "command -v" in cmd:
                return _ok_run(stdout="/opt/conda/bin/mkdssp")
            if "import esm" in cmd:
                return _ok_run()
            return _ok_run()

        with patch(
            "step4_tool_adapters.adapters.equipnas_adapter.run_in_conda_env",
            side_effect=fake_run,
        ):
            adapter.prepare_input(_sample_json(), self.tmp, cfg)
        self.assertFalse(any(c.startswith("mkdssp ") for c in seen_cmds))
        # 1 dssp probe + 1 esm probe + 3 preprocessing scripts; no mkdssp.
        self.assertEqual(len(seen_cmds), 5)

    @unittest.skipUnless(
        (RNA2P_DIR / "1un6.pdb").is_file(),
        "rna2p_balanced/1un6.pdb not available — skip",
    )
    def test_esm_missing_raises_clear_error(self):
        """If `python -c "import esm"` fails (returncode != 0), prepare_input
        should raise with `pip install fair-esm` in the message and not
        proceed to mkdssp / preprocessing scripts."""
        (self.raw / "1un6.pdb").write_bytes(
            (RNA2P_DIR / "1un6.pdb").read_bytes()
        )
        adapter = EquiPNASAdapter()

        seen_cmds: list[str] = []

        def fake_run(env, cmd, **kwargs):
            seen_cmds.append(cmd)
            if "command -v" in cmd:
                return _ok_run(stdout="/opt/conda/bin/mkdssp")
            if "import esm" in cmd:
                return _bad_run("ModuleNotFoundError: No module named 'esm'",
                                rc=1)
            return _ok_run()

        with patch(
            "step4_tool_adapters.adapters.equipnas_adapter.run_in_conda_env",
            side_effect=fake_run,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                adapter.prepare_input(_sample_json(), self.tmp, _config(self.raw))
        msg = str(ctx.exception)
        self.assertIn("fair-esm", msg)
        self.assertIn("pip install", msg)
        # Nothing past the ESM probe should have been called.
        self.assertFalse(any(c.startswith("mkdssp ") for c in seen_cmds))
        self.assertFalse(any("python " in c and "import esm" not in c
                             for c in seen_cmds))

    @unittest.skipUnless(
        (RNA2P_DIR / "1un6.pdb").is_file(),
        "rna2p_balanced/1un6.pdb not available — skip",
    )
    def test_skip_esm_check_bypasses_probe(self):
        """``skip_esm_check: true`` should skip the `import esm` probe."""
        (self.raw / "1un6.pdb").write_bytes(
            (RNA2P_DIR / "1un6.pdb").read_bytes()
        )
        adapter = EquiPNASAdapter()
        cfg = _config(self.raw)
        cfg["tools"]["equipnas"]["skip_esm_check"] = True

        seen_cmds: list[str] = []

        def fake_run(env, cmd, **kwargs):
            seen_cmds.append(cmd)
            if "command -v" in cmd:
                return _ok_run(stdout="/opt/conda/bin/mkdssp")
            if cmd.startswith("mkdssp "):
                tokens = shlex.split(cmd)
                out = Path(tokens[tokens.index("-o") + 1]
                           if "-o" in tokens else tokens[2])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text("HEADER mock\n", encoding="utf-8")
            return _ok_run()

        with patch(
            "step4_tool_adapters.adapters.equipnas_adapter.run_in_conda_env",
            side_effect=fake_run,
        ):
            adapter.prepare_input(_sample_json(), self.tmp, cfg)
        self.assertFalse(any("import esm" in c for c in seen_cmds))
        # 1 dssp probe + 1 mkdssp + 3 preprocessing scripts (esm skipped).
        self.assertEqual(len(seen_cmds), 5)


# ---------- run_tool -------------------------------------------------------


class TestRunTool(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.work = self.tmp / "work"
        self.work.mkdir()
        self.preprocessed = self.work / "equipnas_in"
        self.preprocessed.mkdir()

    def _input_paths(self) -> dict:
        return {
            "preprocessed_dir": self.preprocessed,
            "target": "1un6_B_F",
        }

    def test_success_returns_output_dir(self):
        adapter = EquiPNASAdapter()
        with patch(
            "step4_tool_adapters.adapters.equipnas_adapter.run_in_conda_env",
            return_value=_ok_run(),
        ):
            out = adapter.run_tool(self._input_paths(), self.work, _config(self.tmp))
        self.assertEqual(out, (self.work / "equipnas_output").resolve())
        self.assertTrue(out.is_dir())

    def test_missing_install_dir(self):
        adapter = EquiPNASAdapter()
        cfg = _config(self.tmp)
        cfg["tools"]["equipnas"]["install_dir"] = ""
        with self.assertRaises(ValueError):
            adapter.run_tool(self._input_paths(), self.work, cfg)

    def test_failure_propagated(self):
        adapter = EquiPNASAdapter()
        with patch(
            "step4_tool_adapters.adapters.equipnas_adapter.run_in_conda_env",
            return_value=_bad_run("CUDA OOM"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                adapter.run_tool(self._input_paths(), self.work, _config(self.tmp))
        msg = str(ctx.exception)
        self.assertIn("rc=2", msg)
        self.assertIn("CUDA OOM", msg)

    def test_command_shape(self):
        captured = {}
        def fake(env, cmd, **kw):
            captured["env"] = env
            captured["cmd"] = cmd
            captured["cwd"] = kw.get("cwd")
            return _ok_run()
        adapter = EquiPNASAdapter()
        with patch(
            "step4_tool_adapters.adapters.equipnas_adapter.run_in_conda_env",
            side_effect=fake,
        ):
            adapter.run_tool(self._input_paths(), self.work, _config(self.tmp))
        self.assertEqual(captured["env"], "EquiPNAS")
        self.assertIn("python EquiPNAS.py", captured["cmd"])
        self.assertIn("--model_state_dict", captured["cmd"])
        self.assertIn("--indir", captured["cmd"])
        self.assertIn("--outdir", captured["cmd"])
        self.assertEqual(captured["cwd"], "/fake/EquiPNAS")


# ---------- parse_output ---------------------------------------------------


class TestParseOutput(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "equipnas_output"
        self.out.mkdir()
        (self.out / "1un6_B_F.csv").write_text(
            _mock_csv([(1, 0.10), (5, 0.85), (10, 0.55), (12, 0.30), (20, 0.92)]),
            encoding="utf-8",
        )

    def test_full_parse(self):
        adapter = EquiPNASAdapter()
        pred = adapter.parse_output(self.out, _sample_json(), _config(self.tmp))
        self.assertTrue(pred.success, msg=pred.error_message)
        self.assertEqual(pred.tool_id, "equipnas")
        self.assertEqual(pred.category, "C")
        # threshold 0.5 (strict >); 5 (0.85), 10 (0.55), 20 (0.92) qualify.
        self.assertEqual(pred.binding_protein_residues, [5, 10, 20])
        self.assertEqual(pred.per_residue_confidence[5], 0.85)
        # Cat C never sets RNA / structure / pockets fields.
        self.assertIsNone(pred.binding_rna_nucleotides)
        self.assertIsNone(pred.predicted_structure_path)
        self.assertIsNone(pred.pockets)

    def test_missing_csv_failure(self):
        (self.out / "1un6_B_F.csv").unlink()
        adapter = EquiPNASAdapter()
        pred = adapter.parse_output(self.out, _sample_json(), _config(self.tmp))
        self.assertFalse(pred.success)
        self.assertIn("no EquiPNAS output", pred.error_message)

    def test_threshold_override(self):
        adapter = EquiPNASAdapter()
        cfg = _config(self.tmp)
        cfg["tools"]["equipnas"]["residue_prob_threshold"] = 0.9
        pred = adapter.parse_output(self.out, _sample_json(), cfg)
        # only 20 (0.92) makes it past 0.9
        self.assertEqual(pred.binding_protein_residues, [20])

    def test_empty_csv_returns_failure(self):
        (self.out / "1un6_B_F.csv").write_text(
            "residue_id,probability\n", encoding="utf-8",
        )
        adapter = EquiPNASAdapter()
        pred = adapter.parse_output(self.out, _sample_json(), _config(self.tmp))
        self.assertFalse(pred.success)
        self.assertIn("0 per-residue rows", pred.error_message)


# ---------- end-to-end predict() ------------------------------------------


class TestPredictPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.raw = self.tmp / "raw"
        self.raw.mkdir()

    @unittest.skipUnless(
        (RNA2P_DIR / "1un6.pdb").is_file(),
        "rna2p_balanced/1un6.pdb not available — skip",
    )
    def test_full_predict_with_stub(self):
        # Seed raw_dir.
        (self.raw / "1un6.pdb").write_bytes(
            (RNA2P_DIR / "1un6.pdb").read_bytes()
        )

        # Stub run_in_conda_env: pass DSSP + ESM probes, run mkdssp,
        # succeed for the 3 preprocessing scripts, and for the main
        # EquiPNAS run drop a synthetic CSV into the parsed --outdir.
        call_log: list[str] = []

        def fake_run(env, cmd, **kwargs):
            call_log.append(cmd)
            if "command -v" in cmd:
                # DSSP probe: emit a non-empty path so _ensure_dssp_available
                # treats it as found.
                return _ok_run(stdout="/opt/conda/bin/mkdssp")
            if "import esm" in cmd:
                return _ok_run()
            if cmd.startswith("mkdssp "):
                # Materialise the .dssp file so the adapter accepts it.
                tokens = shlex.split(cmd)
                out = Path(tokens[tokens.index("-o") + 1]
                           if "-o" in tokens else tokens[2])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text("HEADER mock\n", encoding="utf-8")
                return _ok_run()
            if "EquiPNAS.py" in cmd:
                tokens = cmd.split()
                i = tokens.index("--outdir")
                outdir = Path(tokens[i + 1])
                outdir.mkdir(parents=True, exist_ok=True)
                (outdir / "1un6_B_F.csv").write_text(
                    _mock_csv([(2, 0.20), (6, 0.80), (14, 0.55)]),
                    encoding="utf-8",
                )
            return _ok_run()

        adapter = EquiPNASAdapter()
        with patch(
            "step4_tool_adapters.adapters.equipnas_adapter.run_in_conda_env",
            side_effect=fake_run,
        ):
            pred = adapter.predict(
                _sample_json(), self.tmp / "work", _config(self.raw),
            )

        self.assertTrue(pred.success, msg=pred.error_message)
        self.assertEqual(pred.binding_protein_residues, [6, 14])
        # 1 dssp probe + 1 esm probe + 1 mkdssp + 3 preprocess + 1 main = 7
        self.assertEqual(len(call_log), 7)
        # Windows time.monotonic() granularity; accept zero.
        self.assertIsNotNone(pred.runtime_seconds)
        self.assertGreaterEqual(pred.runtime_seconds, 0.0)

    def test_predict_with_missing_raw_returns_failure_record(self):
        adapter = EquiPNASAdapter()
        pred = adapter.predict(
            _sample_json(), self.tmp / "work", _config(self.raw),
        )
        self.assertFalse(pred.success)
        self.assertEqual(pred.tool_id, "equipnas")
        self.assertIn("no raw structure", pred.error_message)


# ---------- symlink preprocessing scripts ----------------------


class TestSymlinkPreprocessingScripts(unittest.TestCase):
    """``_symlink_preprocessing_scripts`` mirrors install_dir/Preprocessing/*.py
    into the staged dir so the upstream scripts can find sibling .py files
    via bare relative paths."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.install = self.tmp / "EquiPNAS"
        self.preproc_src = self.install / "Preprocessing"
        self.preproc_src.mkdir(parents=True)
        self.staged = self.tmp / "equipnas_in"
        self.staged.mkdir()

    def test_links_all_py_files(self):
        for name in ("a.py", "b.py", "extract_dssp_feat.py"):
            (self.preproc_src / name).write_text("# stub\n", encoding="utf-8")
        # A non-.py file should NOT be linked — only *.py.
        (self.preproc_src / "README.md").write_text("docs", encoding="utf-8")

        _symlink_preprocessing_scripts(self.staged, self.install)

        for name in ("a.py", "b.py", "extract_dssp_feat.py"):
            self.assertTrue(
                (self.staged / name).is_file(),
                f"{name} should exist in staged dir after linking",
            )
            self.assertEqual(
                (self.staged / name).read_text(encoding="utf-8"), "# stub\n",
            )
        self.assertFalse((self.staged / "README.md").exists())

    def test_idempotent_when_link_already_exists(self):
        (self.preproc_src / "a.py").write_text("# original\n", encoding="utf-8")
        # Pre-create a file with the same name in staged dir; helper should
        # leave it alone (idempotent) rather than crashing or overwriting.
        (self.staged / "a.py").write_text("# pre-existing\n", encoding="utf-8")
        _symlink_preprocessing_scripts(self.staged, self.install)
        self.assertEqual(
            (self.staged / "a.py").read_text(encoding="utf-8"),
            "# pre-existing\n",
        )

    def test_missing_preprocessing_dir_is_silent(self):
        # When install_dir/Preprocessing doesn't exist, the helper should
        # quietly no-op (let the downstream "script not found" error speak
        # for itself rather than crashing here).
        bad_install = self.tmp / "no_such_install"
        # Should not raise.
        _symlink_preprocessing_scripts(self.staged, bad_install)
        # And nothing should appear in the staged dir.
        self.assertEqual(list(self.staged.iterdir()), [])


# ---------- feature array alignment ----------------------------


class TestAlignFeatureArrays(unittest.TestCase):
    """``_align_feature_arrays`` pads or trims tmp/ feature arrays so all
    per-residue features are exactly seq_len long before
    gen_preprocessed_node_5461features_new stacks them."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.preprocess = self.tmp / "equipnas_in"
        self.tmp_dir = self.preprocess / "tmp"
        self.tmp_dir.mkdir(parents=True)
        self.target = "1un6_B_F"

    def _write(self, fname: str, arr: np.ndarray) -> Path:
        p = self.tmp_dir / fname
        np.save(p, arr)
        return p

    def test_pads_short_2d_array(self):
        # DSSP-derived feat.npy comes out at 60, but the protein has 87 res.
        arr = np.ones((60, 14), dtype=np.float32)
        p = self._write(f"{self.target}.feat.npy", arr)
        _align_feature_arrays(self.preprocess, self.target, seq_len=87)
        out = np.load(p)
        self.assertEqual(out.shape, (87, 14))
        # First 60 rows preserved, padding rows are zero.
        np.testing.assert_array_equal(out[:60], arr)
        np.testing.assert_array_equal(out[60:], np.zeros((27, 14)))

    def test_trims_long_2d_array(self):
        # imputed3.npy comes out at 86 but seq_len is 87 — also test the
        # opposite case where some arrays come out longer.
        arr = np.arange(100 * 3, dtype=np.float32).reshape(100, 3)
        p = self._write(f"{self.target}.imputed3.npy", arr)
        _align_feature_arrays(self.preprocess, self.target, seq_len=87)
        out = np.load(p)
        self.assertEqual(out.shape, (87, 3))
        np.testing.assert_array_equal(out, arr[:87])

    def test_equal_length_unchanged(self):
        # Already-aligned array should not be modified at all.
        arr = np.arange(87 * 5, dtype=np.float32).reshape(87, 5)
        p = self._write(f"{self.target}.feat22.npy", arr)
        mtime_before = p.stat().st_mtime_ns
        _align_feature_arrays(self.preprocess, self.target, seq_len=87)
        out = np.load(p)
        np.testing.assert_array_equal(out, arr)
        # File untouched (np.save would bump mtime).
        self.assertEqual(p.stat().st_mtime_ns, mtime_before)

    def test_aligns_1d_array(self):
        # concount.npy is 1D — pad/trim must work on rank-1 arrays too.
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        p = self._write(f"{self.target}.concount.npy", arr)
        _align_feature_arrays(self.preprocess, self.target, seq_len=8)
        out = np.load(p)
        self.assertEqual(out.shape, (8,))
        np.testing.assert_array_equal(out[:5], arr)
        np.testing.assert_array_equal(out[5:], np.zeros(3, dtype=np.int32))

    def test_missing_files_are_skipped(self):
        # Helper should silently skip files that don't exist (e.g. when an
        # earlier preprocessing script silently failed to write some).
        # No file → no-op, no exception.
        _align_feature_arrays(self.preprocess, self.target, seq_len=87)

    def test_missing_tmp_dir_is_silent(self):
        # When tmp/ doesn't exist (e.g. early failure), helper no-ops.
        empty = self.tmp / "no_tmp"
        empty.mkdir()
        _align_feature_arrays(empty, self.target, seq_len=87)


class TestReadFastaSeqLen(unittest.TestCase):
    """``_read_fasta_seq_len`` is the source-of-truth for seq_len used by
    the alignment helper. Verify it agrees with len(sequence) for various
    layouts."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_single_line(self):
        p = self.tmp / "x.fasta"
        seq = "ACDEFGHIKLMNPQRSTVWY"
        write_fasta(p, "tgt", seq, line_width=0)
        self.assertEqual(_read_fasta_seq_len(p), len(seq))

    def test_wrapped(self):
        p = self.tmp / "x.fasta"
        seq = "A" * 87
        write_fasta(p, "tgt", seq, line_width=60)
        self.assertEqual(_read_fasta_seq_len(p), 87)

    def test_extra_blank_lines(self):
        p = self.tmp / "x.fasta"
        p.write_text(">tgt\n\nMYV\n\nFENC\n", encoding="utf-8")
        self.assertEqual(_read_fasta_seq_len(p), 7)


# ---------- outdir auto-created in run_tool --------------------


class TestRunToolOutdirCreation(unittest.TestCase):
    """``run_tool`` must create the output dir before launching EquiPNAS.py
    — the upstream binary errors with 'No such output directory exists'
    when --outdir doesn't already exist."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.work = self.tmp / "work"
        self.work.mkdir()
        self.preprocessed = self.work / "equipnas_in"
        self.preprocessed.mkdir()

    def test_outdir_created_before_invocation(self):
        observed = {}

        def fake_run(env, cmd, **kwargs):
            # Capture whether --outdir exists *at the moment* the command
            # would run. This is the contract EquiPNAS.py needs.
            tokens = cmd.split()
            i = tokens.index("--outdir")
            observed["outdir"] = Path(tokens[i + 1])
            observed["exists_at_call_time"] = observed["outdir"].is_dir()
            return _ok_run()

        adapter = EquiPNASAdapter()
        with patch(
            "step4_tool_adapters.adapters.equipnas_adapter.run_in_conda_env",
            side_effect=fake_run,
        ):
            adapter.run_tool(
                {"preprocessed_dir": self.preprocessed, "target": "1un6_B_F"},
                self.work,
                _config(self.tmp),
            )
        self.assertTrue(observed["exists_at_call_time"],
                        "output dir must exist before EquiPNAS.py is launched")
        self.assertEqual(
            observed["outdir"], (self.work / "equipnas_output").resolve(),
        )


# ---------- integration with _run_preprocessing -----------------


class TestRunPreprocessingStage5(unittest.TestCase):
    """End-to-end check that _run_preprocessing wires the v5 helpers in:
    symlink before the scripts, alignment after the second script (so the
    third script sees uniformly shaped arrays)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.raw = self.tmp / "raw"
        self.raw.mkdir()
        # Build a fake install_dir with a Preprocessing/ that has a sibling
        # script — used to verify the symlink step actually fires.
        self.install = self.tmp / "EquiPNAS_install"
        (self.install / "Preprocessing").mkdir(parents=True)
        (self.install / "Preprocessing" / "extract_dssp_feat.py").write_text(
            "# sibling stub\n", encoding="utf-8",
        )

    def _config_with_install(self) -> dict:
        cfg = _config(self.raw)
        cfg["tools"]["equipnas"]["install_dir"] = str(self.install)
        return cfg

    @unittest.skipUnless(
        (RNA2P_DIR / "1un6.pdb").is_file(),
        "rna2p_balanced/1un6.pdb not available — skip",
    )
    def test_symlink_and_align_fire_during_preprocessing(self):
        (self.raw / "1un6.pdb").write_bytes(
            (RNA2P_DIR / "1un6.pdb").read_bytes()
        )

        # The fake "genpssmto20feat.py" run materialises a too-short feat
        # file in tmp/ — alignment should pad it up to len(sequence).
        sample = _sample_json()
        seq_len = len(sample["protein"]["sequence"])

        def fake_run(env, cmd, **kwargs):
            if "command -v" in cmd:
                return _ok_run(stdout="/opt/conda/bin/mkdssp")
            if "import esm" in cmd:
                return _ok_run()
            if cmd.startswith("mkdssp "):
                tokens = shlex.split(cmd)
                out = Path(tokens[tokens.index("-o") + 1]
                           if "-o" in tokens else tokens[2])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text("HEADER mock\n", encoding="utf-8")
                return _ok_run()
            if "genpssmto20feat.py" in cmd:
                # Drop a too-short feat.npy + 1D concount in tmp/ to exercise
                # the alignment path. The cwd is the staged dir.
                cwd = Path(kwargs["cwd"])
                tmp_dir = cwd / "tmp"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                short_2d = np.ones((seq_len - 5, 14), dtype=np.float32)
                np.save(tmp_dir / "1un6_B_F.feat.npy", short_2d)
                short_1d = np.arange(seq_len - 3, dtype=np.int32)
                np.save(tmp_dir / "1un6_B_F.concount.npy", short_1d)
                return _ok_run()
            return _ok_run()

        adapter = EquiPNASAdapter()
        with patch(
            "step4_tool_adapters.adapters.equipnas_adapter.run_in_conda_env",
            side_effect=fake_run,
        ):
            paths = adapter.prepare_input(sample, self.tmp / "work",
                                          self._config_with_install())

        # Symlink/copy step: extract_dssp_feat.py should now exist in staged.
        self.assertTrue(
            (paths["preprocessed_dir"] / "extract_dssp_feat.py").is_file(),
            "preprocessing-script symlink/copy should populate staged dir",
        )
        # Alignment step: both files should now match seq_len exactly.
        feat = np.load(paths["preprocessed_dir"] / "tmp" / "1un6_B_F.feat.npy")
        self.assertEqual(feat.shape, (seq_len, 14))
        concount = np.load(
            paths["preprocessed_dir"] / "tmp" / "1un6_B_F.concount.npy",
        )
        self.assertEqual(concount.shape, (seq_len,))


# ---------- .out format (one float per line) -------------------


class TestParseOutFormat(unittest.TestCase):
    """``.out`` is the EquiPNAS upstream default (as of 2026-05): one
    probability per line, no header, no separator. Line N (1-based)
    encodes residue N. Verify the parser detects and decodes it without
    needing the .csv code path."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def _write(self, name: str, content: str) -> Path:
        p = self.tmp / name
        p.write_text(content, encoding="utf-8")
        return p

    def test_helper_returns_indexed_dict(self):
        # Helper itself: bare-text → {1: f1, 2: f2, ...}.
        text = "0.10\n0.95\n0.50\n"
        self.assertEqual(
            _try_parse_single_column_floats(text),
            {1: 0.10, 2: 0.95, 3: 0.50},
        )

    def test_helper_returns_empty_on_internal_whitespace(self):
        # Two tokens per line → not single-column, fall through.
        self.assertEqual(
            _try_parse_single_column_floats("1 0.10\n2 0.95\n"),
            {},
        )

    def test_helper_returns_empty_on_comma(self):
        self.assertEqual(
            _try_parse_single_column_floats("0.10,0.95\n"),
            {},
        )

    def test_helper_returns_empty_on_header_line(self):
        # CSV header — first line "probability" doesn't parse as float
        # in [0,1], so the helper bails immediately.
        self.assertEqual(
            _try_parse_single_column_floats("probability\n0.5\n"),
            {},
        )

    def test_helper_returns_empty_on_out_of_range(self):
        # 1.5 isn't a probability → bail so caller can keep trying.
        self.assertEqual(
            _try_parse_single_column_floats("0.10\n1.5\n0.50\n"),
            {},
        )

    def test_helper_returns_empty_on_empty_input(self):
        self.assertEqual(_try_parse_single_column_floats(""), {})
        self.assertEqual(_try_parse_single_column_floats("\n\n  \n"), {})

    def test_helper_skips_blank_and_comment_lines(self):
        text = "# leading comment\n0.10\n\n0.95\n# trailing\n0.50\n"
        # Comments / blanks are ignored; remaining 3 lines map 1..3.
        self.assertEqual(
            _try_parse_single_column_floats(text),
            {1: 0.10, 2: 0.95, 3: 0.50},
        )

    def test_parse_equipnas_output_dot_out(self):
        # Real-world shape: 87 floats matching the 1un6_B_F sequence
        # length. Spot-check first/last rows to confirm 1-based indexing.
        lines = [f"{0.01 * i:.4f}" for i in range(1, 88)]
        path = self._write("1un6_B_F.out", "\n".join(lines) + "\n")
        out = parse_equipnas_output(path)
        self.assertEqual(len(out), 87)
        self.assertAlmostEqual(out[1], 0.01, places=4)
        self.assertAlmostEqual(out[87], 0.87, places=4)
        # Indices are dense 1..87 with no gaps.
        self.assertEqual(sorted(out.keys()), list(range(1, 88)))

    def test_parse_equipnas_output_chain_filter_ignored_for_out(self):
        # ``.out`` has no chain column, so chain_filter must not affect
        # the result (don't accidentally drop everything).
        path = self._write("x.out", "0.10\n0.95\n0.50\n")
        with_filter = parse_equipnas_output(path, chain_filter="A")
        without_filter = parse_equipnas_output(path)
        self.assertEqual(with_filter, without_filter)
        self.assertEqual(with_filter, {1: 0.10, 2: 0.95, 3: 0.50})

    def test_csv_path_still_works_after_v6(self):
        # Sanity: the CSV / whitespace fallbacks are unchanged. The .out
        # detector returns {} for these because the first line isn't a
        # bare float, so we fall through to the existing parsers.
        csv_path = self._write(
            "x.csv",
            "residue_id,probability\n10,0.8\n11,0.2\n",
        )
        self.assertEqual(parse_equipnas_output(csv_path), {10: 0.8, 11: 0.2})

        ws_path = self._write("y.txt", "1 0.10\n2 0.95\n")
        self.assertEqual(parse_equipnas_output(ws_path), {1: 0.10, 2: 0.95})

    def test_parse_equipnas_csv_alias_still_exported(self):
        # Older callers import parse_equipnas_csv; v6 keeps it as an alias
        # so we don't break the import surface.
        self.assertIs(parse_equipnas_csv, parse_equipnas_output)


class TestFindOutputDotOut(unittest.TestCase):
    """``find_equipnas_output`` must prefer ``.out`` over ``.csv`` after
    v6 — that's the upstream default and CSV files are now produced only
    by hand-converted dumps."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_direct_out(self):
        p = self.tmp / "1un6_B_F.out"
        p.write_text("0.5\n", encoding="utf-8")
        self.assertEqual(find_equipnas_output(self.tmp, "1un6_B_F"), p)

    def test_out_preferred_over_csv(self):
        # Both present → .out wins.
        out = self.tmp / "1un6_B_F.out"
        csv = self.tmp / "1un6_B_F.csv"
        out.write_text("0.5\n", encoding="utf-8")
        csv.write_text("residue_id,probability\n1,0.5\n", encoding="utf-8")
        self.assertEqual(find_equipnas_output(self.tmp, "1un6_B_F"), out)

    def test_nested_out(self):
        nested = self.tmp / "predictions"
        nested.mkdir()
        p = nested / "1un6_B_F.out"
        p.write_text("0.5\n", encoding="utf-8")
        self.assertEqual(find_equipnas_output(self.tmp, "1un6_B_F"), p)

    def test_final_fallback_includes_out(self):
        # Mismatched filename, but it's the only .out file → final
        # rglob fallback should pick it up.
        p = self.tmp / "weird_name_unrelated.out"
        p.write_text("0.5\n", encoding="utf-8")
        self.assertEqual(find_equipnas_output(self.tmp, "1un6_B_F"), p)


class TestParseOutputUsesOutFormat(unittest.TestCase):
    """End-to-end through the adapter: when EquiPNAS drops a ``.out``
    file, ``parse_output`` must surface a successful ToolPrediction with
    line-indexed probabilities."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out_dir = self.tmp / "equipnas_output"
        self.out_dir.mkdir()

    def test_parses_out_file_into_prediction(self):
        # 5 residues; threshold default 0.5 keeps {2: 0.85, 5: 0.92}.
        (self.out_dir / "1un6_B_F.out").write_text(
            "0.10\n0.85\n0.50\n0.30\n0.92\n", encoding="utf-8",
        )
        adapter = EquiPNASAdapter()
        # The fixture _config wants raw_dir; pass any tmp dir, not used here.
        cfg = {
            "tools": {
                "equipnas": {
                    "conda_env": "EquiPNAS",
                    "install_dir": "/fake/EquiPNAS",
                    "model_path": "models/EquiPNAS-RNA/E-l12-768.pt",
                    "residue_prob_threshold": 0.5,
                },
            },
            "structure_source": {"raw_dir": str(self.tmp)},
        }
        pred = adapter.parse_output(self.out_dir, _sample_json(), cfg)
        self.assertTrue(pred.success, msg=pred.error_message)
        self.assertEqual(pred.binding_protein_residues, [2, 5])
        self.assertAlmostEqual(pred.per_residue_confidence[1], 0.10)
        self.assertAlmostEqual(pred.per_residue_confidence[5], 0.92)


# ---------- PSSM / ESM-2 / MSA-dummy / distance-map ------------


def _config_v7(raw_dir: Path, install_dir: Path) -> dict:
    """Like _config() but with every v7 generator *enabled* (skip_* off).
    Tests exercise the new code paths via this builder."""
    cfg = _config(raw_dir)
    eq = cfg["tools"]["equipnas"]
    eq["install_dir"] = str(install_dir)
    eq["skip_pssm"] = False
    eq["skip_esm2"] = False
    eq["skip_msa_dummy"] = False
    eq["skip_distance_map"] = False
    eq["skip_psiblast_check"] = False
    eq["pssm_db"] = "/fake/blast_db/uniref50"
    eq["pssm_fallback_db"] = "/fake/blast_db/swissprot"
    return cfg


def _make_install_dir(root: Path) -> Path:
    """Build a minimal fake install_dir matching the EquiPNAS layout the
    v7 helpers expect (Preprocessing/pdb2rr.py, ESM-2 helper script)."""
    install = root / "EquiPNAS"
    (install / "Preprocessing").mkdir(parents=True)
    (install / "Preprocessing" / "pdb2rr.py").write_text(
        "# stub pdb2rr\n", encoding="utf-8",
    )
    (install / "input_details" / "supporting_scripts").mkdir(parents=True)
    (install / "input_details" / "supporting_scripts" /
     "esm2_15B_rep_5120.py").write_text("# stub esm2\n", encoding="utf-8")
    return install


class TestGeneratePSSM(unittest.TestCase):
    """``_generate_pssm`` shells out to ``psiblast`` and writes
    ``input/<target>.pssm``. Mock the subprocess and verify command
    shape + the cache / fallback behaviour."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.preprocess = self.tmp / "equipnas_in"
        (self.preprocess / "input").mkdir(parents=True)
        # FASTA must exist; helper raises FileNotFoundError otherwise.
        (self.preprocess / "input" / "1un6_B_F.fasta").write_text(
            ">1un6_B_F\nMYV\n", encoding="utf-8",
        )
        self.adapter = EquiPNASAdapter()

    def _tool_cfg(self, **overrides) -> dict:
        cfg = {
            "pssm_db": "/fake/uniref50",
            "pssm_num_iterations": 3,
            "pssm_evalue": 0.001,
            "pssm_threads": 8,
        }
        cfg.update(overrides)
        return cfg

    def test_writes_pssm_and_uses_psiblast(self):
        observed: dict = {}

        def fake_run(env, cmd, **kwargs):
            observed["env"] = env
            observed["cmd"] = cmd
            # Materialise the output the helper looks for.
            tokens = shlex.split(cmd)
            i = tokens.index("-out_ascii_pssm")
            Path(tokens[i + 1]).write_text("PSSM stub\n", encoding="utf-8")
            return _ok_run()

        with patch(
            "step4_tool_adapters.adapters.equipnas_adapter.run_in_conda_env",
            side_effect=fake_run,
        ):
            self.adapter._generate_pssm(
                env_name="EquiPNAS",
                preprocess_dir=self.preprocess,
                target="1un6_B_F",
                tool_cfg=self._tool_cfg(),
                timeout=600,
                log_dir=None,
            )
        self.assertEqual(observed["env"], "EquiPNAS")
        self.assertIn("psiblast ", observed["cmd"])
        self.assertIn("-num_iterations 3", observed["cmd"])
        self.assertIn("-evalue 0.001", observed["cmd"])
        self.assertIn("-num_threads 8", observed["cmd"])
        self.assertIn("/fake/uniref50", observed["cmd"])
        self.assertTrue(
            (self.preprocess / "input" / "1un6_B_F.pssm").is_file(),
        )

    def test_skips_when_pssm_already_exists(self):
        # Pre-stage the output → helper must not invoke psiblast.
        (self.preprocess / "input" / "1un6_B_F.pssm").write_text(
            "cached\n", encoding="utf-8",
        )
        with patch(
            "step4_tool_adapters.adapters.equipnas_adapter.run_in_conda_env",
        ) as m:
            self.adapter._generate_pssm(
                env_name="EquiPNAS",
                preprocess_dir=self.preprocess,
                target="1un6_B_F",
                tool_cfg=self._tool_cfg(),
                timeout=600,
                log_dir=None,
            )
        self.assertFalse(m.called)

    def test_falls_back_to_secondary_db(self):
        # Primary fails (no output written) → helper retries with fallback.
        seen: list[str] = []

        def fake_run(env, cmd, **kwargs):
            seen.append(cmd)
            tokens = shlex.split(cmd)
            i = tokens.index("-db")
            db = tokens[i + 1]
            if db == "/fake/uniref50":
                return _bad_run("BLAST Database error: no DB", rc=2)
            # Fallback succeeds: write the output file.
            j = tokens.index("-out_ascii_pssm")
            Path(tokens[j + 1]).write_text("PSSM\n", encoding="utf-8")
            return _ok_run()

        with patch(
            "step4_tool_adapters.adapters.equipnas_adapter.run_in_conda_env",
            side_effect=fake_run,
        ):
            self.adapter._generate_pssm(
                env_name="EquiPNAS",
                preprocess_dir=self.preprocess,
                target="1un6_B_F",
                tool_cfg=self._tool_cfg(
                    pssm_fallback_db="/fake/swissprot",
                ),
                timeout=600,
                log_dir=None,
            )
        self.assertEqual(len(seen), 2)
        self.assertIn("/fake/uniref50", seen[0])
        self.assertIn("/fake/swissprot", seen[1])

    def test_raises_when_all_dbs_fail(self):
        with patch(
            "step4_tool_adapters.adapters.equipnas_adapter.run_in_conda_env",
            return_value=_bad_run("DB missing", rc=2),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                self.adapter._generate_pssm(
                    env_name="EquiPNAS",
                    preprocess_dir=self.preprocess,
                    target="1un6_B_F",
                    tool_cfg=self._tool_cfg(
                        pssm_fallback_db="/fake/swissprot",
                    ),
                    timeout=600,
                    log_dir=None,
                )
        msg = str(ctx.exception)
        self.assertIn("PSI-BLAST", msg)
        self.assertIn("1un6_B_F", msg)

    def test_missing_fasta_raises(self):
        # Wipe the FASTA we wrote in setUp.
        (self.preprocess / "input" / "1un6_B_F.fasta").unlink()
        with self.assertRaises(FileNotFoundError):
            self.adapter._generate_pssm(
                env_name="EquiPNAS",
                preprocess_dir=self.preprocess,
                target="1un6_B_F",
                tool_cfg=self._tool_cfg(),
                timeout=600,
                log_dir=None,
            )


class TestGenerateESM2Embedding(unittest.TestCase):
    """``_generate_esm2_embedding`` shells out to the upstream
    ESM-2 helper and produces ``input/<target>.rep_5120.npy``."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.preprocess = self.tmp / "equipnas_in"
        (self.preprocess / "input").mkdir(parents=True)
        (self.preprocess / "input" / "1un6_B_F.fasta").write_text(
            ">1un6_B_F\nMYV\n", encoding="utf-8",
        )
        self.install = _make_install_dir(self.tmp)
        self.adapter = EquiPNASAdapter()

    def _tool_cfg(self, **overrides) -> dict:
        cfg = {"install_dir": str(self.install)}
        cfg.update(overrides)
        return cfg

    def test_invokes_script_and_creates_npy(self):
        observed: dict = {}

        def fake_run(env, cmd, **kwargs):
            observed["cmd"] = cmd
            tokens = shlex.split(cmd)
            i = tokens.index("-o")
            stem = tokens[i + 1]
            # Simulate the upstream script writing <stem>.npy.
            np.save(stem + ".npy", np.zeros((5, 5120), dtype=np.float32))
            return _ok_run()

        with patch(
            "step4_tool_adapters.adapters.equipnas_adapter.run_in_conda_env",
            side_effect=fake_run,
        ):
            self.adapter._generate_esm2_embedding(
                env_name="EquiPNAS",
                preprocess_dir=self.preprocess,
                target="1un6_B_F",
                tool_cfg=self._tool_cfg(),
                timeout=600,
                log_dir=None,
            )
        self.assertIn("esm2_15B_rep_5120.py", observed["cmd"])
        self.assertTrue(
            (self.preprocess / "input" / "1un6_B_F.rep_5120.npy").is_file(),
        )

    def test_skips_when_npy_already_exists(self):
        np.save(
            self.preprocess / "input" / "1un6_B_F.rep_5120.npy",
            np.zeros((5, 5120), dtype=np.float32),
        )
        with patch(
            "step4_tool_adapters.adapters.equipnas_adapter.run_in_conda_env",
        ) as m:
            self.adapter._generate_esm2_embedding(
                env_name="EquiPNAS",
                preprocess_dir=self.preprocess,
                target="1un6_B_F",
                tool_cfg=self._tool_cfg(),
                timeout=600,
                log_dir=None,
            )
        self.assertFalse(m.called)

    def test_failure_raises(self):
        with patch(
            "step4_tool_adapters.adapters.equipnas_adapter.run_in_conda_env",
            return_value=_bad_run("CUDA OOM", rc=1),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                self.adapter._generate_esm2_embedding(
                    env_name="EquiPNAS",
                    preprocess_dir=self.preprocess,
                    target="1un6_B_F",
                    tool_cfg=self._tool_cfg(),
                    timeout=600,
                    log_dir=None,
                )
        self.assertIn("ESM-2", str(ctx.exception))


class TestGenerateDummyMSAFirstRow(unittest.TestCase):
    """``_generate_dummy_msa_first_row`` writes a zero-filled placeholder.
    No subprocess; just numpy."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.preprocess = self.tmp / "equipnas_in"
        (self.preprocess / "input").mkdir(parents=True)
        self.adapter = EquiPNASAdapter()

    def test_writes_zero_array_with_correct_shape(self):
        self.adapter._generate_dummy_msa_first_row(
            preprocess_dir=self.preprocess,
            target="1un6_B_F",
            seq_len=42,
        )
        out = self.preprocess / "input" / "1un6_B_Fmsa_first_row.npy"
        self.assertTrue(out.is_file())
        arr = np.load(out)
        # +2 for BOS/EOS tokens; 256 is the upstream embedding dim.
        self.assertEqual(arr.shape, (44, 256))
        self.assertTrue(np.all(arr == 0))
        self.assertEqual(arr.dtype, np.float32)

    def test_idempotent(self):
        out = self.preprocess / "input" / "1un6_B_Fmsa_first_row.npy"
        # Pre-write a marker we want preserved.
        np.save(out, np.full((10, 256), 7.0, dtype=np.float32))
        mtime_before = out.stat().st_mtime_ns
        self.adapter._generate_dummy_msa_first_row(
            preprocess_dir=self.preprocess,
            target="1un6_B_F",
            seq_len=42,
        )
        # File untouched on idempotent path.
        self.assertEqual(out.stat().st_mtime_ns, mtime_before)
        arr = np.load(out)
        self.assertTrue(np.all(arr == 7.0))

    def test_zero_seq_len_raises(self):
        with self.assertRaises(ValueError):
            self.adapter._generate_dummy_msa_first_row(
                preprocess_dir=self.preprocess,
                target="1un6_B_F",
                seq_len=0,
            )


class TestGenerateDistanceMap(unittest.TestCase):
    """``_generate_distance_map`` runs ``Preprocessing/pdb2rr.py`` and
    writes ``distmaps/<target>.dist``."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.preprocess = self.tmp / "equipnas_in"
        (self.preprocess / "input").mkdir(parents=True)
        (self.preprocess / "distmaps").mkdir(parents=True)
        # PDB stub — helper checks existence only, content irrelevant.
        (self.preprocess / "input" / "1un6_B_F.pdb").write_text(
            "ATOM\n", encoding="utf-8",
        )
        self.install = _make_install_dir(self.tmp)
        self.adapter = EquiPNASAdapter()

    def test_invokes_pdb2rr_with_threshold(self):
        observed: dict = {}

        def fake_run(env, cmd, **kwargs):
            observed["cmd"] = cmd
            tokens = shlex.split(cmd)
            i = tokens.index("-o")
            Path(tokens[i + 1]).write_text("DIST\n", encoding="utf-8")
            return _ok_run()

        with patch(
            "step4_tool_adapters.adapters.equipnas_adapter.run_in_conda_env",
            side_effect=fake_run,
        ):
            self.adapter._generate_distance_map(
                env_name="EquiPNAS",
                preprocess_dir=self.preprocess,
                install_dir=self.install,
                target="1un6_B_F",
                tool_cfg={"distance_map_threshold": 14},
                timeout=600,
                log_dir=None,
            )
        self.assertIn("pdb2rr.py", observed["cmd"])
        self.assertIn("-t 14", observed["cmd"])
        self.assertTrue(
            (self.preprocess / "distmaps" / "1un6_B_F.dist").is_file(),
        )

    def test_skip_when_dist_exists(self):
        (self.preprocess / "distmaps" / "1un6_B_F.dist").write_text(
            "cached\n", encoding="utf-8",
        )
        with patch(
            "step4_tool_adapters.adapters.equipnas_adapter.run_in_conda_env",
        ) as m:
            self.adapter._generate_distance_map(
                env_name="EquiPNAS",
                preprocess_dir=self.preprocess,
                install_dir=self.install,
                target="1un6_B_F",
                tool_cfg={},
                timeout=600,
                log_dir=None,
            )
        self.assertFalse(m.called)

    def test_failure_raises(self):
        with patch(
            "step4_tool_adapters.adapters.equipnas_adapter.run_in_conda_env",
            return_value=_bad_run("malformed PDB", rc=1),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                self.adapter._generate_distance_map(
                    env_name="EquiPNAS",
                    preprocess_dir=self.preprocess,
                    install_dir=self.install,
                    target="1un6_B_F",
                    tool_cfg={},
                    timeout=600,
                    log_dir=None,
                )
        self.assertIn("pdb2rr.py", str(ctx.exception))


class TestEnsurePsiblastAvailable(unittest.TestCase):
    """``_ensure_psiblast_available`` is the v7 analogue of the existing
    DSSP / ESM probes — fail fast if psiblast is missing."""

    def setUp(self):
        self.adapter = EquiPNASAdapter()

    def test_passes_when_path_returned(self):
        with patch(
            "step4_tool_adapters.adapters.equipnas_adapter.run_in_conda_env",
            return_value=_ok_run(stdout="/opt/conda/bin/psiblast"),
        ):
            self.adapter._ensure_psiblast_available("EquiPNAS", None)

    def test_raises_with_install_command_when_missing(self):
        with patch(
            "step4_tool_adapters.adapters.equipnas_adapter.run_in_conda_env",
            return_value=_ok_run(stdout=""),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                self.adapter._ensure_psiblast_available("EquiPNAS", None)
        msg = str(ctx.exception)
        self.assertIn("psiblast", msg)
        self.assertIn("conda install", msg)
        self.assertIn("bioconda", msg)


class TestRunPreprocessingStage7(unittest.TestCase):
    """End-to-end verification that ``_run_preprocessing`` wires every
    v7 generator in, in the documented order."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.raw = self.tmp / "raw"
        self.raw.mkdir()
        self.install = _make_install_dir(self.tmp)
        # Symlink helper looks for *.py in Preprocessing — add a stub
        # sibling so the symlink step has something to mirror.
        (self.install / "Preprocessing" / "extract_dssp_feat.py").write_text(
            "# stub\n", encoding="utf-8",
        )

    @unittest.skipUnless(
        (RNA2P_DIR / "1un6.pdb").is_file(),
        "rna2p_balanced/1un6.pdb not available — skip",
    )
    def test_full_v7_pipeline_call_order(self):
        (self.raw / "1un6.pdb").write_bytes(
            (RNA2P_DIR / "1un6.pdb").read_bytes()
        )
        adapter = EquiPNASAdapter()
        sample = _sample_json()
        seq_len = len(sample["protein"]["sequence"])

        events: list[str] = []

        def fake_run(env, cmd, **kwargs):
            # Tag each call by its conceptual step so we can assert on
            # ordering without coupling to exact command strings.
            if cmd.startswith("bash -c 'command -v mkdssp"):
                events.append("probe_dssp")
                return _ok_run(stdout="/opt/conda/bin/mkdssp")
            if cmd.startswith("bash -c 'command -v psiblast"):
                events.append("probe_psiblast")
                return _ok_run(stdout="/opt/conda/bin/psiblast")
            if "import esm" in cmd:
                events.append("probe_esm")
                return _ok_run()
            if cmd.startswith("mkdssp "):
                events.append("mkdssp")
                tokens = shlex.split(cmd)
                out = Path(tokens[tokens.index("-o") + 1]
                           if "-o" in tokens else tokens[2])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text("HEADER mock\n", encoding="utf-8")
                return _ok_run()
            if "psiblast " in cmd:
                events.append("pssm")
                tokens = shlex.split(cmd)
                Path(tokens[tokens.index("-out_ascii_pssm") + 1]).write_text(
                    "PSSM stub\n", encoding="utf-8",
                )
                return _ok_run()
            if "esm2_15B_rep_5120.py" in cmd:
                events.append("esm2")
                tokens = shlex.split(cmd)
                stem = tokens[tokens.index("-o") + 1]
                np.save(stem + ".npy",
                        np.zeros((seq_len + 2, 5120), dtype=np.float32))
                return _ok_run()
            if "pdb2rr.py" in cmd:
                events.append("distance_map")
                tokens = shlex.split(cmd)
                Path(tokens[tokens.index("-o") + 1]).write_text(
                    "DIST\n", encoding="utf-8",
                )
                return _ok_run()
            if "gen_aa_structural_features.py" in cmd:
                events.append("gen_aa")
                return _ok_run()
            if "genpssmto20feat.py" in cmd:
                events.append("genpssm")
                # Produce a too-short feat file to exercise alignment.
                cwd = Path(kwargs["cwd"])
                tmp_dir = cwd / "tmp"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                np.save(tmp_dir / f"{sample['sample_id']}.feat.npy",
                        np.ones((seq_len - 5, 14), dtype=np.float32))
                return _ok_run()
            if "gen_preprocessed_node_5461features_new.py" in cmd:
                events.append("gen_preprocessed")
                return _ok_run()
            return _ok_run()

        with patch(
            "step4_tool_adapters.adapters.equipnas_adapter.run_in_conda_env",
            side_effect=fake_run,
        ):
            paths = adapter.prepare_input(
                sample, self.tmp / "work",
                _config_v7(self.raw, self.install),
            )

        # The expected order matches the v7 docstring; probes can come in
        # any order before mkdssp, but the artifact-generation steps are
        # strictly sequenced.
        self.assertIn("probe_dssp", events)
        self.assertIn("probe_psiblast", events)
        self.assertIn("probe_esm", events)
        # Probes must precede mkdssp.
        for probe in ("probe_dssp", "probe_psiblast", "probe_esm"):
            self.assertLess(events.index(probe), events.index("mkdssp"))
        # mkdssp → pssm → esm2 → distance_map → gen_aa → genpssm → gen_preprocessed.
        ordered = [
            "mkdssp", "pssm", "esm2", "distance_map",
            "gen_aa", "genpssm", "gen_preprocessed",
        ]
        idxs = [events.index(name) for name in ordered]
        self.assertEqual(idxs, sorted(idxs),
                         f"steps out of order: events={events}")

        # Side effects: PSSM / ESM-2 / MSA / distmap files all materialised.
        pp = paths["preprocessed_dir"]
        self.assertTrue((pp / "input" / "1un6_B_F.pssm").is_file())
        self.assertTrue((pp / "input" / "1un6_B_F.rep_5120.npy").is_file())
        self.assertTrue((pp / "input" / "1un6_B_Fmsa_first_row.npy").is_file())
        self.assertTrue((pp / "distmaps" / "1un6_B_F.dist").is_file())
        # MSA file shape = (seq_len+2, 256).
        msa = np.load(pp / "input" / "1un6_B_Fmsa_first_row.npy")
        self.assertEqual(msa.shape, (seq_len + 2, 256))

    @unittest.skipUnless(
        (RNA2P_DIR / "1un6.pdb").is_file(),
        "rna2p_balanced/1un6.pdb not available — skip",
    )
    def test_skip_pssm_omits_psiblast_probe_and_call(self):
        # When the user pre-stages PSSM, neither the probe nor the
        # generator should run.
        (self.raw / "1un6.pdb").write_bytes(
            (RNA2P_DIR / "1un6.pdb").read_bytes()
        )
        cfg = _config_v7(self.raw, self.install)
        cfg["tools"]["equipnas"]["skip_pssm"] = True
        cfg["tools"]["equipnas"]["skip_esm2"] = True
        cfg["tools"]["equipnas"]["skip_msa_dummy"] = True
        cfg["tools"]["equipnas"]["skip_distance_map"] = True

        seen_cmds: list[str] = []

        def fake_run(env, cmd, **kwargs):
            seen_cmds.append(cmd)
            if "command -v" in cmd:
                return _ok_run(stdout="/opt/conda/bin/x")
            if "import esm" in cmd:
                return _ok_run()
            if cmd.startswith("mkdssp "):
                tokens = shlex.split(cmd)
                out = Path(tokens[tokens.index("-o") + 1]
                           if "-o" in tokens else tokens[2])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text("HEADER\n", encoding="utf-8")
            return _ok_run()

        adapter = EquiPNASAdapter()
        with patch(
            "step4_tool_adapters.adapters.equipnas_adapter.run_in_conda_env",
            side_effect=fake_run,
        ):
            adapter.prepare_input(_sample_json(), self.tmp / "work", cfg)

        self.assertFalse(any("psiblast" in c for c in seen_cmds))
        self.assertFalse(any("esm2_15B_rep_5120.py" in c for c in seen_cmds))
        self.assertFalse(any("pdb2rr.py" in c for c in seen_cmds))


if __name__ == "__main__":
    unittest.main(verbosity=2)
