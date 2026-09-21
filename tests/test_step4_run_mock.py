"""Mock tests for step 4 orchestrator (run.py + run_all.py).

What this covers:
  - ADAPTER_REGISTRY shape (4 deployed adapters keyed by canonical id)
  - get_adapter raises on unknown tool_id
  - order_tools puts Cat A before C / B / D, stable within category
  - load_sample finds samples/<id>.json or <id>.json under processed_dir
  - run_sample collects predictions in deterministic order
  - run_sample writes ToolPredictionSet with consistent metadata
  - one tool failing does not abort the rest of the batch
  - JSONL writer round-trips (read back → equal model)
  - run_all extract_planned_tools / filter_deployed
  - run_batch behaviour: skips failed step3 plans by default, drops
    undeployed tools quietly, returns counters, writes per-sample files
  - argparse main entry points (run.main / run_all.main) execute end-to-end
    against patched adapters
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step3_tool_selection.tool_registry import get_all_tool_ids  # noqa: E402
from step4_tool_adapters.base_adapter import BaseAdapter  # noqa: E402
from step4_tool_adapters.run import (  # noqa: E402
    ADAPTER_REGISTRY, _summarize, get_adapter, load_sample, main as run_main,
    order_tools, run_sample, write_jsonl_record,
)
from step4_tool_adapters.run_all import (  # noqa: E402
    extract_planned_tools, filter_deployed, load_step3_records,
    main as run_all_main, run_batch,
)
from step4_tool_adapters.schemas import (  # noqa: E402
    Pocket, ToolPrediction, ToolPredictionSet,
)


# ---------- fake adapters -------------------------------------------------


class _FakeOK(BaseAdapter):
    """Adapter that always succeeds with a fixed prediction shape."""
    def __init__(self, *, tool_id: str, category: str,
                 binding: list[int] | None = None):
        self.tool_id = tool_id
        self.category = category
        self._binding = binding or [1, 2]

    def prepare_input(self, sample_json, work_dir, config):
        return {}

    def run_tool(self, input_paths, work_dir, config):
        return Path(work_dir)

    def parse_output(self, output_dir, sample_json, config):
        if self.category == "B":
            return ToolPrediction(
                tool_id=self.tool_id, category=self.category,
                sample_id=sample_json["sample_id"], success=True,
                binding_protein_residues=self._binding,
                pockets=[Pocket(rank=1, score=10.0, residues=self._binding)],
            )
        return ToolPrediction(
            tool_id=self.tool_id, category=self.category,
            sample_id=sample_json["sample_id"], success=True,
            binding_protein_residues=self._binding,
            per_residue_confidence={i: 0.7 for i in self._binding},
        )


class _FakeFail(BaseAdapter):
    def __init__(self, *, tool_id: str, category: str,
                 message: str = "boom"):
        self.tool_id = tool_id
        self.category = category
        self._msg = message

    def prepare_input(self, sample_json, work_dir, config):
        raise RuntimeError(self._msg)

    def run_tool(self, input_paths, work_dir, config):  # pragma: no cover
        return Path(work_dir)

    def parse_output(self, output_dir, sample_json, config):  # pragma: no cover
        raise RuntimeError("unreachable")


def _sample(sid: str = "1un6_B_F") -> dict:
    return {
        "sample_id": sid,
        "source_pdb": sid.split("_")[0],
        "protein": {"chain_id": "B", "sequence": "MKTV"},
        "rna": {"chain_id": "F", "sequence": "GCCG"},
    }


def _config() -> dict:
    return {
        "work_dir": "data/step4_workdir",
        "output": {"batch_jsonl_dir": "data/step4_outputs"},
    }


# ---------- registry & ordering -------------------------------------------


class TestRegistry(unittest.TestCase):
    def test_deployed_adapters_registered(self):
        # ADAPTER_REGISTRY carries one adapter per tool of the library
        # (paper Table 1) — the 12 that run locally and the 3 whose
        # adapter ingests results from a manual web submission.
        self.assertEqual(
            set(ADAPTER_REGISTRY.keys()),
            set(get_all_tool_ids()),
        )

    def test_get_adapter_returns_instance(self):
        ad = get_adapter("p2rank")
        self.assertEqual(ad.tool_id, "p2rank")
        self.assertEqual(ad.category, "B")

    def test_get_adapter_unknown_raises(self):
        with self.assertRaises(ValueError) as ctx:
            get_adapter("nope")
        self.assertIn("nope", str(ctx.exception))


class TestOrderTools(unittest.TestCase):
    def test_cat_a_before_b_c(self):
        # Input order is intentionally B, A, C — output should be A, C, B.
        out = order_tools(["p2rank", "boltz2", "equipnas"])
        self.assertEqual(out, ["boltz2", "equipnas", "p2rank"])

    def test_dedupe(self):
        out = order_tools(["p2rank", "p2rank", "boltz2"])
        self.assertEqual(out, ["boltz2", "p2rank"])

    def test_cat_a_two_tools_keep_input_order(self):
        # Both Cat A; tie broken by input position.
        out = order_tools(["rosettafold2na", "boltz2"])
        self.assertEqual(out, ["rosettafold2na", "boltz2"])

    def test_unknown_tool_raises(self):
        with self.assertRaises(ValueError):
            order_tools(["mystery"])


# ---------- IO helpers ----------------------------------------------------


class TestLoadSample(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_finds_in_samples_subdir(self):
        d = self.tmp / "samples"
        d.mkdir()
        (d / "x.json").write_text(json.dumps({"sample_id": "x"}), encoding="utf-8")
        self.assertEqual(load_sample(self.tmp, "x"), {"sample_id": "x"})

    def test_finds_at_processed_root(self):
        (self.tmp / "y.json").write_text(json.dumps({"sample_id": "y"}), encoding="utf-8")
        self.assertEqual(load_sample(self.tmp, "y"), {"sample_id": "y"})

    def test_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_sample(self.tmp, "missing")


class TestWriteJsonl(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_atomic_write_round_trip(self):
        pred = ToolPrediction(
            tool_id="p2rank", category="B", sample_id="x", success=True,
            binding_protein_residues=[1, 2, 3],
            pockets=[Pocket(rank=1, score=5.0, residues=[1, 2])],
        )
        ps = ToolPredictionSet(
            sample_id="x",
            tools_run=["p2rank"],
            predictions=[pred],
            total_runtime_seconds=1.5,
            timestamp="2026-04-29T12:00:00Z",
        )
        out = self.tmp / "x.jsonl"
        write_jsonl_record(ps, out)

        # File should be exactly one line of valid JSON.
        text = out.read_text(encoding="utf-8")
        self.assertEqual(len(text.splitlines()), 1)
        loaded = ToolPredictionSet.model_validate(json.loads(text.strip()))
        self.assertEqual(loaded.sample_id, "x")
        self.assertEqual(loaded.predictions[0].binding_protein_residues, [1, 2, 3])
        # No leftover .tmp file.
        self.assertFalse(out.with_suffix(out.suffix + ".tmp").exists())


# ---------- run_sample ----------------------------------------------------


class TestRunSample(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        # Patch ADAPTER_REGISTRY so get_adapter returns fakes.
        # Use spec-matching factories — caller calls cls(), not cls(tool_id=...).
        self.patches = patch.dict(
            "step4_tool_adapters.run.ADAPTER_REGISTRY",
            {
                "p2rank": lambda: _FakeOK(tool_id="p2rank", category="B"),
                "boltz2": lambda: _FakeOK(tool_id="boltz2", category="A",
                                          binding=[3, 4]),
                "equipnas": lambda: _FakeOK(tool_id="equipnas", category="C",
                                            binding=[5, 6]),
                "broken": lambda: _FakeFail(tool_id="broken", category="C"),
            },
            clear=True,
        )
        self.patches.start()

    def tearDown(self):
        self.patches.stop()

    def test_runs_in_category_order(self):
        ps = run_sample(_sample(), ["p2rank", "equipnas", "boltz2"], _config(), self.tmp)
        # Cat A first, then C, then B
        self.assertEqual([p.tool_id for p in ps.predictions],
                         ["boltz2", "equipnas", "p2rank"])
        self.assertEqual(ps.tools_run, ["boltz2", "equipnas", "p2rank"])
        # all succeeded
        self.assertTrue(all(p.success for p in ps.predictions))

    def test_set_metadata(self):
        ps = run_sample(_sample(), ["p2rank"], _config(), self.tmp)
        self.assertEqual(ps.sample_id, "1un6_B_F")
        self.assertIsNotNone(ps.total_runtime_seconds)
        self.assertGreaterEqual(ps.total_runtime_seconds, 0.0)
        self.assertTrue(ps.timestamp.endswith("Z"))

    def test_failure_isolated_to_failing_tool(self):
        ps = run_sample(
            _sample(), ["broken", "p2rank"], _config(), self.tmp,
        )
        # Both tools were attempted; broken failed gracefully.
        ids = {p.tool_id: p for p in ps.predictions}
        self.assertEqual(set(ids), {"broken", "p2rank"})
        self.assertFalse(ids["broken"].success)
        self.assertIn("boom", ids["broken"].error_message)
        self.assertTrue(ids["p2rank"].success)

    def test_on_prediction_callback_fires_per_tool(self):
        # Serial mode preserves the original "callback order = category
        # order" contract. Parallel mode does not — see
        # TestRunSampleParallel.test_callback_fires_for_every_tool.
        seen = []
        run_sample(
            _sample(), ["p2rank", "boltz2"], _config(), self.tmp,
            on_prediction=lambda p: seen.append(p.tool_id),
            parallel_workers=1,
        )
        self.assertEqual(seen, ["boltz2", "p2rank"])  # category-ordered


# ---------- parallel-mode behaviour ---------------------------------------


class _FakeSlow(BaseAdapter):
    """Adapter that sleeps before returning success — used to assert
    that parallel mode actually overlaps tool execution."""
    def __init__(self, *, tool_id: str, category: str, sleep_s: float = 0.10):
        self.tool_id = tool_id
        self.category = category
        self._sleep = sleep_s

    def prepare_input(self, sample_json, work_dir, config):
        return {}

    def run_tool(self, input_paths, work_dir, config):
        return Path(work_dir)

    def parse_output(self, output_dir, sample_json, config):
        import time as _time
        _time.sleep(self._sleep)
        return ToolPrediction(
            tool_id=self.tool_id, category=self.category,
            sample_id=sample_json["sample_id"], success=True,
            binding_protein_residues=[1, 2],
            per_residue_confidence={1: 0.9, 2: 0.8},
        )


class TestRunSampleParallel(unittest.TestCase):
    """Parallel execution of run_sample.

    Verifies:
      - explicit parallel_workers >= 2 + multiple tools → tools overlap
        (wall-clock < sum of individual sleeps)
      - predictions still come out in category order regardless of
        completion order
      - failures stay isolated (one slow + one failing → other survives)
      - callback fires once per tool (order does NOT matter in parallel)
      - parallel_workers from config is honoured when CLI override is None
      - effective_workers caps at len(tools) (no spurious thread pool)
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        # 3 slow tools, ~100 ms each. Serial → ~300 ms; parallel → ~100 ms.
        self.patches = patch.dict(
            "step4_tool_adapters.run.ADAPTER_REGISTRY",
            {
                "boltz2": lambda: _FakeSlow(tool_id="boltz2", category="A",
                                             sleep_s=0.10),
                "equipnas": lambda: _FakeSlow(tool_id="equipnas", category="C",
                                               sleep_s=0.10),
                "p2rank": lambda: _FakeSlow(tool_id="p2rank", category="B",
                                             sleep_s=0.10),
                "broken": lambda: _FakeFail(tool_id="broken", category="C"),
            },
            clear=True,
        )
        self.patches.start()

    def tearDown(self):
        self.patches.stop()

    def test_parallel_overlaps_tool_execution(self):
        # 3 × 0.10 s = 0.30 s serial; parallel should be much closer to 0.10 s.
        # We give a generous 0.25 s ceiling to absorb thread-pool overhead +
        # CI machine jitter.
        import time as _time
        t0 = _time.monotonic()
        ps = run_sample(
            _sample(), ["boltz2", "equipnas", "p2rank"],
            _config(), self.tmp,
            parallel_workers=3,
        )
        elapsed = _time.monotonic() - t0
        self.assertLess(
            elapsed, 0.25,
            f"parallel run took {elapsed:.3f}s, expected < 0.25s "
            f"(serial baseline ~0.30s for 3 × 0.10s sleeps)",
        )
        self.assertEqual([p.tool_id for p in ps.predictions],
                         ["boltz2", "equipnas", "p2rank"])

    def test_predictions_in_category_order_regardless_of_completion(self):
        # Stagger the sleeps so completion order differs from category order.
        with patch.dict(
            "step4_tool_adapters.run.ADAPTER_REGISTRY",
            {
                "boltz2": lambda: _FakeSlow(tool_id="boltz2", category="A",
                                             sleep_s=0.20),  # finishes LAST
                "equipnas": lambda: _FakeSlow(tool_id="equipnas", category="C",
                                               sleep_s=0.05),  # finishes 1st
                "p2rank": lambda: _FakeSlow(tool_id="p2rank", category="B",
                                             sleep_s=0.10),  # finishes 2nd
            },
            clear=True,
        ):
            ps = run_sample(
                _sample(), ["p2rank", "boltz2", "equipnas"],
                _config(), self.tmp,
                parallel_workers=3,
            )
        # boltz2 finished last in wall-clock terms but must come FIRST in
        # the predictions list (Cat A → C → B).
        self.assertEqual([p.tool_id for p in ps.predictions],
                         ["boltz2", "equipnas", "p2rank"])

    def test_failure_isolated_in_parallel_mode(self):
        ps = run_sample(
            _sample(), ["boltz2", "broken", "p2rank"],
            _config(), self.tmp,
            parallel_workers=3,
        )
        ids = {p.tool_id: p for p in ps.predictions}
        self.assertEqual(set(ids), {"boltz2", "broken", "p2rank"})
        self.assertFalse(ids["broken"].success)
        self.assertIn("boom", ids["broken"].error_message)
        self.assertTrue(ids["boltz2"].success)
        self.assertTrue(ids["p2rank"].success)
        # Order still canonical despite the failure happening at
        # arbitrary times in the pool.
        self.assertEqual([p.tool_id for p in ps.predictions],
                         ["boltz2", "broken", "p2rank"])

    def test_callback_fires_for_every_tool(self):
        # In parallel mode the callback ORDER is unspecified (depends on
        # completion order), but every tool must trigger it exactly once.
        seen = []
        lock = __import__("threading").Lock()

        def _record(p):
            with lock:
                seen.append(p.tool_id)

        run_sample(
            _sample(), ["boltz2", "equipnas", "p2rank"],
            _config(), self.tmp,
            on_prediction=_record,
            parallel_workers=3,
        )
        self.assertEqual(set(seen), {"boltz2", "equipnas", "p2rank"})
        self.assertEqual(len(seen), 3)

    def test_config_parallel_workers_honoured_when_no_override(self):
        # parallel_workers=None on the function call → falls back to
        # config.execution.parallel_workers. Set it to 1 (serial) and
        # assert serial-mode timing.
        cfg = _config()
        cfg["execution"] = {"parallel_workers": 1}
        import time as _time
        t0 = _time.monotonic()
        ps = run_sample(
            _sample(), ["boltz2", "equipnas", "p2rank"],
            cfg, self.tmp,
            parallel_workers=None,  # explicit: defer to config
        )
        elapsed = _time.monotonic() - t0
        # 3 × 0.10 s sleeps must run sequentially → wall clock ≥ 0.27 s.
        self.assertGreaterEqual(
            elapsed, 0.27,
            f"serial run took {elapsed:.3f}s, expected >= 0.27s",
        )
        self.assertEqual(len(ps.predictions), 3)

    def test_cli_override_beats_config(self):
        # config says serial, override says parallel → must run parallel.
        cfg = _config()
        cfg["execution"] = {"parallel_workers": 1}
        import time as _time
        t0 = _time.monotonic()
        run_sample(
            _sample(), ["boltz2", "equipnas", "p2rank"],
            cfg, self.tmp,
            parallel_workers=3,
        )
        elapsed = _time.monotonic() - t0
        self.assertLess(elapsed, 0.25,
                        f"override should force parallel (took {elapsed:.3f}s)")

    def test_single_tool_skips_pool(self):
        # 1 tool requested → effective_workers=1 → use serial path even
        # if config asks for 4. Asserting via behaviour (output correct +
        # no exception); a thread leak would surface as a hung test run.
        ps = run_sample(
            _sample(), ["p2rank"],
            _config(), self.tmp,
            parallel_workers=4,
        )
        self.assertEqual([p.tool_id for p in ps.predictions], ["p2rank"])

    def test_no_parallel_flag_in_cli(self):
        # Cover the CLI mapping: --no-parallel → parallel_workers=1.
        sample_dir = self.tmp / "processed" / "samples"
        sample_dir.mkdir(parents=True)
        sample_path = sample_dir / "1un6_B_F.json"
        sample_path.write_text(json.dumps(_sample()), encoding="utf-8")

        cfg_path = self.tmp / "step4_cfg.yaml"
        cfg_path.write_text(
            "tools:\n  p2rank:\n    install_dir: x\n    timeout: 60\n"
            "work_dir: " + str(self.tmp / "wd") + "\n"
            "output:\n  batch_jsonl_dir: " + str(self.tmp / "out") + "\n"
            "execution:\n  parallel_workers: 4\n",
            encoding="utf-8",
        )
        rc = run_main([
            "--processed-dir", str(self.tmp / "processed"),
            "--sample-id", "1un6_B_F",
            "--tool", "p2rank",
            "--config", str(cfg_path),
            "--no-parallel",
        ])
        self.assertEqual(rc, 0)


class TestSummarize(unittest.TestCase):
    def test_format_includes_per_tool_status(self):
        ps = ToolPredictionSet(
            sample_id="x",
            tools_run=["a", "b"],
            predictions=[
                ToolPrediction(tool_id="a", category="A", sample_id="x",
                               success=True,
                               binding_protein_residues=[1]),
                ToolPrediction(tool_id="b", category="B", sample_id="x",
                               success=False, error_message="nope"),
            ],
            total_runtime_seconds=2.0,
            timestamp="2026-04-29T12:00:00Z",
        )
        text = _summarize(ps)
        self.assertIn("ok=1", text)
        self.assertIn("fail=1", text)
        self.assertIn("a=OK", text)
        self.assertIn("b=FAIL", text)


# ---------- run_all helpers ----------------------------------------------


class TestExtractPlannedTools(unittest.TestCase):
    def test_normal(self):
        rec = {"tool_plan": {"selected_tools": ["p2rank", "boltz2"]}}
        self.assertEqual(extract_planned_tools(rec), ["p2rank", "boltz2"])

    def test_missing_returns_empty(self):
        self.assertEqual(extract_planned_tools({}), [])
        self.assertEqual(extract_planned_tools({"tool_plan": {}}), [])

    def test_non_list_returns_empty(self):
        self.assertEqual(
            extract_planned_tools({"tool_plan": {"selected_tools": "p2rank"}}),
            [],
        )


class TestFilterDeployed(unittest.TestCase):
    def test_splits_correctly(self):
        # Everything in the library has an adapter, so nothing is dropped;
        # an id that is not in the registry at all still lands in `dropped`
        # (that is what protects against stale step-3 records).
        deployed, dropped = filter_deployed(
            ["p2rank", "alphafold3", "boltz2", "fpocket", "chai1"],
        )
        self.assertEqual(
            deployed, ["p2rank", "alphafold3", "boltz2", "fpocket", "chai1"],
        )
        self.assertEqual(dropped, [])

        deployed, dropped = filter_deployed(["p2rank", "no_such_tool"])
        self.assertEqual(deployed, ["p2rank"])
        self.assertEqual(dropped, ["no_such_tool"])


class TestLoadStep3Records(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_skips_blank_and_comment_lines(self):
        path = self.tmp / "x.jsonl"
        path.write_text(
            '{"sample_id":"a"}\n'
            "\n"
            "# comment\n"
            '{"sample_id":"b"}\n',
            encoding="utf-8",
        )
        records = load_step3_records(path)
        self.assertEqual([r["sample_id"] for r in records], ["a", "b"])


# ---------- run_batch ----------------------------------------------------


class TestRunBatch(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.processed = self.tmp / "processed"
        (self.processed / "samples").mkdir(parents=True)
        for sid in ("1un6_B_F", "2bgg_A_P"):
            (self.processed / "samples" / f"{sid}.json").write_text(
                json.dumps(_sample(sid)),
                encoding="utf-8",
            )
        self.out_dir = self.tmp / "step4_out"
        self.work_dir = self.tmp / "step4_work"
        self.patches = patch.dict(
            "step4_tool_adapters.run.ADAPTER_REGISTRY",
            {
                "p2rank": lambda: _FakeOK(tool_id="p2rank", category="B"),
                "boltz2": lambda: _FakeOK(tool_id="boltz2", category="A"),
            },
            clear=True,
        )
        self.patches.start()

    def tearDown(self):
        self.patches.stop()

    def _step3(self, *, sid: str, tools: list[str], success: bool = True) -> dict:
        return {
            "sample_id": sid,
            "tool_plan": {"selected_tools": tools},
            "success": success,
        }

    def test_skips_failed_step3_plans(self):
        records = [
            self._step3(sid="1un6_B_F", tools=["p2rank"], success=False),
            self._step3(sid="2bgg_A_P", tools=["p2rank"]),
        ]
        summary = run_batch(
            records, processed_dir=self.processed, config=_config(),
            work_dir=self.work_dir, out_dir=self.out_dir,
        )
        self.assertEqual(summary["ok"], 1)
        self.assertEqual(summary["skip"], 1)
        # Only the second sample should have an output file.
        files = sorted(p.name for p in self.out_dir.glob("*.jsonl"))
        self.assertEqual(files, ["2bgg_A_P.jsonl"])

    def test_drops_undeployed_tools_quietly(self):
        rec = self._step3(
            sid="1un6_B_F",
            tools=["p2rank", "alphafold3", "fpocket"],
        )
        summary = run_batch(
            [rec], processed_dir=self.processed, config=_config(),
            work_dir=self.work_dir, out_dir=self.out_dir,
        )
        self.assertEqual(summary["ok"], 1)
        # Output should reflect only the deployed subset.
        loaded = json.loads((self.out_dir / "1un6_B_F.jsonl").read_text(encoding="utf-8"))
        self.assertEqual([p["tool_id"] for p in loaded["predictions"]], ["p2rank"])

    def test_strict_plan_fails_when_undeployed_present(self):
        rec = self._step3(sid="1un6_B_F", tools=["p2rank", "alphafold3"])
        summary = run_batch(
            [rec], processed_dir=self.processed, config=_config(),
            work_dir=self.work_dir, out_dir=self.out_dir,
            strict_plan=True,
        )
        self.assertEqual(summary["fail"], 1)
        self.assertEqual(summary["ok"], 0)
        self.assertFalse((self.out_dir / "1un6_B_F.jsonl").exists())

    def test_skip_when_no_deployed_tools_in_plan(self):
        rec = self._step3(sid="1un6_B_F", tools=["alphafold3"])
        summary = run_batch(
            [rec], processed_dir=self.processed, config=_config(),
            work_dir=self.work_dir, out_dir=self.out_dir,
        )
        self.assertEqual(summary["skip"], 1)

    def test_skip_when_sample_missing(self):
        rec = self._step3(sid="missing", tools=["p2rank"])
        summary = run_batch(
            [rec], processed_dir=self.processed, config=_config(),
            work_dir=self.work_dir, out_dir=self.out_dir,
        )
        self.assertEqual(summary["skip"], 1)

    def test_sample_filter(self):
        records = [
            self._step3(sid="1un6_B_F", tools=["p2rank"]),
            self._step3(sid="2bgg_A_P", tools=["p2rank"]),
        ]
        summary = run_batch(
            records, processed_dir=self.processed, config=_config(),
            work_dir=self.work_dir, out_dir=self.out_dir,
            sample_filter={"2bgg_A_P"},
        )
        self.assertEqual(summary["ok"], 1)
        files = sorted(p.name for p in self.out_dir.glob("*.jsonl"))
        self.assertEqual(files, ["2bgg_A_P.jsonl"])


# ---------- argparse main entries ----------------------------------------


class TestRunCli(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.processed = self.tmp / "processed"
        (self.processed / "samples").mkdir(parents=True)
        (self.processed / "samples" / "1un6_B_F.json").write_text(
            json.dumps(_sample("1un6_B_F")), encoding="utf-8",
        )
        # Minimal config file.
        self.cfg = self.tmp / "step4_config.yaml"
        self.cfg.write_text(
            "work_dir: " + str(self.tmp / "_work").replace("\\", "/") + "\n"
            "output:\n"
            "  batch_jsonl_dir: "
            + str(self.tmp / "_out").replace("\\", "/") + "\n",
            encoding="utf-8",
        )
        self.patches = patch.dict(
            "step4_tool_adapters.run.ADAPTER_REGISTRY",
            {
                "p2rank": lambda: _FakeOK(tool_id="p2rank", category="B"),
                "boltz2": lambda: _FakeOK(tool_id="boltz2", category="A"),
            },
            clear=True,
        )
        self.patches.start()

    def tearDown(self):
        self.patches.stop()

    def test_single_tool_writes_jsonl(self):
        out = self.tmp / "out.jsonl"
        rc = run_main([
            "--processed-dir", str(self.processed),
            "--sample-id", "1un6_B_F",
            "--tool", "p2rank",
            "--config", str(self.cfg),
            "--output", str(out),
        ])
        self.assertEqual(rc, 0)
        loaded = json.loads(out.read_text(encoding="utf-8").strip())
        self.assertEqual(loaded["sample_id"], "1un6_B_F")
        self.assertEqual([p["tool_id"] for p in loaded["predictions"]], ["p2rank"])

    def test_csv_tools_runs_in_category_order(self):
        out = self.tmp / "out.jsonl"
        rc = run_main([
            "--processed-dir", str(self.processed),
            "--sample-id", "1un6_B_F",
            "--tools", "p2rank,boltz2",
            "--config", str(self.cfg),
            "--output", str(out),
        ])
        self.assertEqual(rc, 0)
        loaded = json.loads(out.read_text(encoding="utf-8").strip())
        # Cat A (boltz2) before Cat B (p2rank)
        self.assertEqual(
            [p["tool_id"] for p in loaded["predictions"]],
            ["boltz2", "p2rank"],
        )

    def test_default_output_path_uses_config(self):
        rc = run_main([
            "--processed-dir", str(self.processed),
            "--sample-id", "1un6_B_F",
            "--tool", "p2rank",
            "--config", str(self.cfg),
        ])
        self.assertEqual(rc, 0)
        expected = self.tmp / "_out" / "1un6_B_F.jsonl"
        self.assertTrue(expected.is_file())


class TestRunAllCli(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.processed = self.tmp / "processed"
        (self.processed / "samples").mkdir(parents=True)
        for sid in ("1un6_B_F", "2bgg_A_P"):
            (self.processed / "samples" / f"{sid}.json").write_text(
                json.dumps(_sample(sid)), encoding="utf-8",
            )
        self.step3 = self.tmp / "step3.jsonl"
        self.step3.write_text(
            json.dumps({"sample_id": "1un6_B_F",
                        "tool_plan": {"selected_tools": ["p2rank", "boltz2"]},
                        "success": True}) + "\n"
            + json.dumps({"sample_id": "2bgg_A_P",
                          "tool_plan": {"selected_tools": ["p2rank"]},
                          "success": True}) + "\n",
            encoding="utf-8",
        )
        self.cfg = self.tmp / "step4_config.yaml"
        self.cfg.write_text(
            "work_dir: " + str(self.tmp / "_work").replace("\\", "/") + "\n"
            "output:\n"
            "  batch_jsonl_dir: "
            + str(self.tmp / "_out").replace("\\", "/") + "\n",
            encoding="utf-8",
        )
        self.patches = patch.dict(
            "step4_tool_adapters.run.ADAPTER_REGISTRY",
            {
                "p2rank": lambda: _FakeOK(tool_id="p2rank", category="B"),
                "boltz2": lambda: _FakeOK(tool_id="boltz2", category="A"),
            },
            clear=True,
        )
        self.patches.start()

    def tearDown(self):
        self.patches.stop()

    def test_writes_one_file_per_sample(self):
        out_dir = self.tmp / "out"
        rc = run_all_main([
            "--processed-dir", str(self.processed),
            "--step3-output", str(self.step3),
            "--config", str(self.cfg),
            "--output", str(out_dir),
        ])
        self.assertEqual(rc, 0)
        files = sorted(p.name for p in out_dir.glob("*.jsonl"))
        self.assertEqual(files, ["1un6_B_F.jsonl", "2bgg_A_P.jsonl"])
        # First file: both tools, A before B
        loaded = json.loads((out_dir / "1un6_B_F.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(
            [p["tool_id"] for p in loaded["predictions"]],
            ["boltz2", "p2rank"],
        )

    def test_limit(self):
        out_dir = self.tmp / "out"
        rc = run_all_main([
            "--processed-dir", str(self.processed),
            "--step3-output", str(self.step3),
            "--config", str(self.cfg),
            "--output", str(out_dir),
            "--limit", "1",
        ])
        self.assertEqual(rc, 0)
        files = sorted(p.name for p in out_dir.glob("*.jsonl"))
        self.assertEqual(files, ["1un6_B_F.jsonl"])

    def test_sample_id_filter(self):
        out_dir = self.tmp / "out"
        rc = run_all_main([
            "--processed-dir", str(self.processed),
            "--step3-output", str(self.step3),
            "--config", str(self.cfg),
            "--output", str(out_dir),
            "--sample-id", "2bgg_A_P",
        ])
        self.assertEqual(rc, 0)
        files = sorted(p.name for p in out_dir.glob("*.jsonl"))
        self.assertEqual(files, ["2bgg_A_P.jsonl"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
