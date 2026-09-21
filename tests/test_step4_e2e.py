"""End-to-end pipeline smoke for step 2 → 3 → 4 (server-only).

NOT a unittest module — manual smoke that runs the LLM-driven steps
2 and 3 inline, then feeds the produced tool plan into step 4 to
actually invoke the deployed tools (P2Rank, Boltz-2, EquiPNAS).

Expected end-to-end runtime
---------------------------
- Step 2 (target characterisation): 5-15 s per sample (one LLM call)
- Step 3 (tool selection):         5-30 s per sample (one LLM call,
                                  occasional retry)
- Step 4 (real tool invocation):   1 s (P2Rank) + several minutes
                                  (Boltz-2, MSA server) +
                                  several minutes (EquiPNAS, depends
                                  on PSSM/ESM-2 cache state)

So budget 5-15 minutes per sample with the standard plan.

⚠ RoseTTAFold2NA is currently DATABASE-BLOCKED. If step 3 selects it
the step-4 invocation will fail at the MSA step — that's expected and
will appear as a ``success=False`` ToolPrediction inside the JSONL
output, not a script crash.

Usage (on the server)::

    cd ~/riboseer
    conda activate riboseer
    export LLM_API_KEY="<your real key>"
    python tests/test_step4_e2e.py \\
        --processed-dir data/processed \\
        --config-dir configs/ \\
        --sample-id 1un6_B_F \\
        --raw-dir /opt/raw_pdb \\
        --p2rank-install-dir /opt/biotools/p2rank_2.5 \\
        --equipnas-install-dir /opt/biotools/EquiPNAS \\
        --rf2na-install-dir /opt/biotools/RoseTTAFold2NA \\
        --output data/step4_outputs/e2e_test.jsonl

What it prints
--------------
Per stage banner, plus a final summary block:

    === STEP 2 ===
      sample 1un6_B_F → category=protein_zinc_finger ...
    === STEP 3 ===
      sample 1un6_B_F → tools=['p2rank','equipnas']  conf=0.78
    === STEP 4 ===
      sample 1un6_B_F: 2 tools  ok=2 fail=0 runtime=18.3s -> 1un6_B_F.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import yaml  # noqa: E402

from step2_target_char.run import (  # noqa: E402
    build_client, load_config as load_step2_config,
)
from step2_target_char.target_char import characterize_target  # noqa: E402
from step3_tool_selection.tool_selector import select_tools  # noqa: E402
from step3_tool_selection.weight_tensor import WeightTensor  # noqa: E402
from step4_tool_adapters.run import (  # noqa: E402
    ADAPTER_REGISTRY, load_config as load_step4_config, load_sample,
    run_sample, write_jsonl_record,
)


# ---------- helpers --------------------------------------------------------


def _banner(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _build_step4_config(args: argparse.Namespace, base_cfg: dict) -> dict:
    """Override step4_config.yaml's tool install/env paths with CLI flags
    so the user can point at the live server install without editing
    files."""
    cfg = dict(base_cfg)  # shallow copy
    tools = dict(cfg.get("tools") or {})

    if args.p2rank_install_dir:
        p = dict(tools.get("p2rank") or {})
        p["install_dir"] = str(args.p2rank_install_dir)
        tools["p2rank"] = p
    if args.boltz_env:
        b = dict(tools.get("boltz2") or {})
        b["conda_env"] = args.boltz_env
        tools["boltz2"] = b
    if args.equipnas_install_dir:
        e = dict(tools.get("equipnas") or {})
        e["install_dir"] = str(args.equipnas_install_dir)
        if args.skip_equipnas_preprocess:
            e["skip_preprocess"] = True
        tools["equipnas"] = e
    if args.rf2na_install_dir:
        r = dict(tools.get("rosettafold2na") or {})
        r["install_dir"] = str(args.rf2na_install_dir)
        tools["rosettafold2na"] = r

    cfg["tools"] = tools

    if args.raw_dir:
        ss = dict(cfg.get("structure_source") or {})
        ss["raw_dir"] = str(args.raw_dir)
        cfg["structure_source"] = ss

    return cfg


def _run_step2(sample: dict, client, step2_cfg: dict) -> dict:
    """Step 2: LLM target characterisation."""
    return characterize_target(sample, client, step2_cfg)


def _run_step3(s2_record: dict, client, weight_tensor, step3_cfg: dict) -> dict:
    """Step 3: LLM tool selection from step-2 output."""
    return select_tools(
        sample_id=s2_record.get("sample_id"),
        target_char=s2_record.get("output") or {},
        target_features=s2_record.get("input_features") or {},
        client=client,
        weight_tensor=weight_tensor,
        config=step3_cfg,
    )


# ---------- main -----------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--config-dir", type=Path, required=True,
                   help="dir containing step2_config.yaml / step3_config.yaml "
                        "/ step4_config.yaml")
    p.add_argument("--sample-id", type=str, action="append", required=True,
                   help="sample to run (repeat for multiple)")

    # Step-4 install/env overrides (otherwise picks up config defaults)
    p.add_argument("--raw-dir", type=Path, default=None,
                   help="override config.structure_source.raw_dir")
    p.add_argument("--p2rank-install-dir", type=Path, default=None)
    p.add_argument("--boltz-env", type=str, default=None,
                   help="override conda env for Boltz-2 (default 'boltz')")
    p.add_argument("--equipnas-install-dir", type=Path, default=None)
    p.add_argument("--skip-equipnas-preprocess", action="store_true",
                   help="set tools.equipnas.skip_preprocess=True (assumes "
                        "the user has pre-staged PSSM/ESM-2 features)")
    p.add_argument("--rf2na-install-dir", type=Path, default=None)

    p.add_argument("--output", type=Path, default=None,
                   help="JSONL output path (default: <step4 cfg>/<sid>.jsonl). "
                        "When multiple --sample-id given, this is a directory.")

    args = p.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not os.environ.get("LLM_API_KEY"):
        print("ERROR: LLM_API_KEY environment variable not set.",
              file=sys.stderr)
        return 2

    cfg_dir = args.config_dir
    step2_cfg = load_step2_config(cfg_dir / "step2_config.yaml")
    with (cfg_dir / "step3_config.yaml").open("r", encoding="utf-8") as f:
        step3_cfg = yaml.safe_load(f) or {}
    step4_base = load_step4_config(cfg_dir / "step4_config.yaml")
    step4_cfg = _build_step4_config(args, step4_base)

    client = build_client(step2_cfg)

    wt_path = Path(step3_cfg.get(
        "weight_tensor_path", "data/step3_weights/weight_tensor.json",
    ))
    weight_tensor = (WeightTensor.load(wt_path) if wt_path.is_file()
                     else WeightTensor.from_config(step3_cfg))

    work_dir = Path(step4_cfg.get("work_dir", "data/step4_workdir"))
    out_arg = args.output
    if out_arg is None:
        out_dir = Path((step4_cfg.get("output") or {}).get(
            "batch_jsonl_dir", "data/step4_outputs",
        ))
    elif len(args.sample_id) > 1:
        out_dir = Path(out_arg); out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    summary: list[dict] = []
    for sid in args.sample_id:
        _banner(f"SAMPLE {sid}")
        sample = load_sample(args.processed_dir, sid)

        # ---- step 2 ----
        _banner(f"STEP 2 — {sid}")
        s2 = _run_step2(sample, client, step2_cfg)
        if not s2.get("success"):
            print(f"  step2 failed: {s2.get('error_message')}")
            summary.append({"sample_id": sid, "step": 2, "ok": False})
            continue
        s2_out = s2.get("output") or {}
        print(f"  category   : {s2_out.get('category')}")
        print(f"  confidence : {s2_out.get('confidence')}")

        # ---- step 3 ----
        _banner(f"STEP 3 — {sid}")
        s3 = _run_step3(s2, client, weight_tensor, step3_cfg)
        if not s3.get("success"):
            print(f"  step3 failed; using fallback plan: {s3.get('tool_plan')}")
        plan = s3.get("tool_plan") or {}
        selected = plan.get("selected_tools") or []
        print(f"  tools      : {selected}")
        print(f"  strategy   : {plan.get('execution_strategy')}")
        print(f"  confidence : {plan.get('confidence')}")

        # ---- step 4 ----
        _banner(f"STEP 4 — {sid}")
        deployed = [t for t in selected if t in ADAPTER_REGISTRY]
        dropped = [t for t in selected if t not in ADAPTER_REGISTRY]
        if dropped:
            print(f"  note: dropping undeployed tools {dropped}")
        if not deployed:
            print(f"  no deployed tools in plan; skipping step 4 for {sid}")
            summary.append({"sample_id": sid, "step": 4, "ok": False,
                            "reason": "no deployed tools"})
            continue

        pred_set = run_sample(sample, deployed, step4_cfg, work_dir)
        n_ok = sum(1 for p in pred_set.predictions if p.success)
        n_fail = len(pred_set.predictions) - n_ok
        print(f"  tools_run  : {pred_set.tools_run}")
        for p_ in pred_set.predictions:
            tag = "OK" if p_.success else "FAIL"
            extra = ""
            if p_.success:
                if p_.binding_protein_residues is not None:
                    extra += f" prot={len(p_.binding_protein_residues)}"
                if p_.binding_rna_nucleotides is not None:
                    extra += f" rna={len(p_.binding_rna_nucleotides)}"
                if p_.plddt_mean is not None:
                    extra += f" plddt={p_.plddt_mean}"
            else:
                extra = f" err={(p_.error_message or '')[:80]}"
            print(f"    {p_.tool_id:<14} [{tag}]{extra}")

        # Output path (per-sample file)
        if out_arg is not None and len(args.sample_id) == 1:
            out_path = out_arg
        else:
            out_path = out_dir / f"{sid}.jsonl"
        write_jsonl_record(pred_set, out_path)
        print(f"  wrote      : {out_path}")
        summary.append({
            "sample_id": sid, "step": 4,
            "ok": n_fail == 0, "n_ok": n_ok, "n_fail": n_fail,
            "runtime_seconds": pred_set.total_runtime_seconds,
        })

    # ---- final summary ----
    _banner("SUMMARY")
    n_overall_ok = sum(1 for s in summary if s.get("ok"))
    n_overall_fail = len(summary) - n_overall_ok
    for s in summary:
        print(f"  {json.dumps(s, ensure_ascii=False)}")
    print(f"\nTotal: ok={n_overall_ok}  fail={n_overall_fail}")
    return 0 if n_overall_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
