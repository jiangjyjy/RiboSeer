"""CLI entry point for Step 5 — fusion.

Two modes
---------

Mode 1 — consume an existing step 4 ``ToolPredictionSet`` JSONL and an
existing step 2 JSONL (both per-sample one-line records)::

    python -m step5_fusion.run \\
        --processed-dir data/processed \\
        --step2-output data/step2_outputs/e2e_smoke.jsonl \\
        --step4-output data/step4_outputs/1un6_B_F.jsonl \\
        --config configs/step5_config.yaml \\
        --step2-config configs/step2_config.yaml \\
        --step3-config configs/step3_config.yaml \\
        --output data/step5_outputs/

Mode 2 — pipeline mode: re-run step 2 + step 3 + step 4 + step 5 for
one sample id from raw step 1 output. **Heavy**: the user typically
runs step 4 ahead of time and uses Mode 1; this exists for end-to-end
smoke runs::

    python -m step5_fusion.run \\
        --processed-dir data/processed \\
        --sample-id 1un6_B_F \\
        --run-pipeline \\
        --config configs/step5_config.yaml \\
        --output data/step5_outputs/

Environment
-----------
Requires ``LLM_API_KEY`` (read from env by ``LLMClient``). Run on the
server. Output is one JSONL line per fused sample; if ``--output`` is a
directory, each sample lands in ``<dir>/<sample_id>.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

import yaml

from step2_target_char.llm_client import LLMClient
from step2_target_char.run import build_client, load_config as load_step2_config
from step3_tool_selection.weight_tensor import WeightTensor
from step4_tool_adapters.schemas import ToolPrediction, ToolPredictionSet

from .fusion import build_output_record, fuse_predictions


# ---------- IO helpers -----------------------------------------------------


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(json.loads(line))
    return rows


def _load_sample(processed_dir: Path, sample_id: str) -> dict:
    candidates = [
        processed_dir / "samples" / f"{sample_id}.json",
        processed_dir / f"{sample_id}.json",
    ]
    for p in candidates:
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        f"sample {sample_id!r} not found in {[str(p) for p in candidates]}"
    )


def _step4_record_to_predictions(record: dict) -> list[ToolPrediction]:
    """Coerce a raw step 4 ToolPredictionSet record into ToolPrediction list.

    ``ToolPredictionSet.model_validate`` re-runs all the field validators
    so we catch malformed step 4 output before fusion (better error than
    a confusing schema failure deeper in noisy_or).
    """
    pset = ToolPredictionSet.model_validate(record)
    return list(pset.predictions)


def _index_step2_records(rows: Iterable[dict]) -> dict[str, dict]:
    """``{sample_id: target_char_dict}`` from step 2 JSONL records."""
    out: dict[str, dict] = {}
    for r in rows:
        sid = r.get("sample_id")
        if not sid:
            continue
        # step 2's record has output={analysis,...}; fall back to whole
        # record if "output" key isn't present (older / mock format).
        tc = r.get("output") or r
        out[sid] = tc
    return out


def _index_step4_records(rows: Iterable[dict]) -> dict[str, dict]:
    """``{sample_id: ToolPredictionSet record}`` keyed by step 4 record."""
    out: dict[str, dict] = {}
    for r in rows:
        sid = r.get("sample_id")
        if not sid:
            continue
        out[sid] = r
    return out


# ---------- output writer --------------------------------------------------


def _resolve_output_path(out_arg: Optional[Path], sample_id: str) -> Optional[Path]:
    """If ``--output`` is a dir, write ``<dir>/<sid>.jsonl``; if a file,
    write everything to that one file (one line per sample)."""
    if out_arg is None:
        return None
    if out_arg.suffix in (".jsonl", ".json"):
        out_arg.parent.mkdir(parents=True, exist_ok=True)
        return out_arg
    out_arg.mkdir(parents=True, exist_ok=True)
    return out_arg / f"{sample_id}.jsonl"


def _write_record_atomic(path: Path, record: dict, *, append: bool = False) -> None:
    """Write JSONL line atomically (``.tmp`` + ``replace``).

    When ``append=True`` we read the existing file and re-emit it plus
    the new line atomically, so a SIGINT mid-write can't leave a partial
    line at the end.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if append and path.is_file():
        existing = path.read_text(encoding="utf-8")
    else:
        existing = ""
    body = existing + json.dumps(record, ensure_ascii=False) + "\n"
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)


def _summarize(record: dict) -> str:
    tools = record.get("tools_fused") or []
    p = record.get("precision", 0.0)
    r = record.get("recall", 0.0)
    f1 = record.get("f1", 0.0)
    usage = record.get("api_usage") or {}
    status = usage.get("status", "?")
    return (
        f"[{status:<10}] {record.get('sample_id', '?'):<16} "
        f"cat={record.get('step2_category', '?'):<28} "
        f"tools={','.join(tools):<40} "
        f"thresh={record.get('threshold', 0.5):.2f}  "
        f"P={p:.3f} R={r:.3f} F1={f1:.3f}  "
        f"tokens={usage.get('total_tokens', 0)}"
    )


# ---------- pipeline mode (Mode 2) -----------------------------------------


def _run_inline_pipeline(
    sample_id: str,
    processed_dir: Path,
    step2_cfg: dict,
    step3_cfg: dict,
    step4_cfg_path: Path,
    client: LLMClient,
) -> tuple[dict, dict, list[ToolPrediction]]:
    """Run step 2 → 3 → 4 inline so ``fuse_predictions`` has fresh inputs.

    Returns ``(target_char_dict, sample_json, tool_predictions)``. Heavy
    operation — only used when ``--run-pipeline`` is set.
    """
    from step2_target_char.target_char import characterize_target
    from step3_tool_selection.tool_selector import select_tools
    from step4_tool_adapters.run import run_sample as run_step4_sample

    sample = _load_sample(processed_dir, sample_id)

    # Step 2
    s2 = characterize_target(sample, client, step2_cfg)
    target_char = s2.get("output") or {}

    # Step 3 (need WeightTensor)
    wt_path = Path(step3_cfg.get(
        "weight_tensor_path",
        "data/step3_weights/weight_tensor.json",
    ))
    wt = (WeightTensor.load(wt_path) if wt_path.is_file()
          else WeightTensor.from_config(step3_cfg))

    s3 = select_tools(
        sample_id=sample_id,
        target_char=target_char,
        target_features=s2.get("input_features") or {},
        client=client,
        weight_tensor=wt,
        config=step3_cfg,
    )
    plan = s3.get("tool_plan") or {}
    selected = plan.get("selected_tools") or []

    # Step 4 — single-sample run on the selected tools
    step4_config = load_config(step4_cfg_path)
    work_dir = Path(step4_config.get("work_dir") or "data/step4_workdir") / sample_id
    work_dir.mkdir(parents=True, exist_ok=True)
    pset = run_step4_sample(
        sample=sample,
        tool_ids=selected,
        config=step4_config,
        work_dir=work_dir,
    )
    return target_char, sample, list(pset.predictions)


# ---------- main -----------------------------------------------------------


def _process_one(
    sample_id: str,
    sample_json: dict,
    target_char: dict,
    predictions: list[ToolPrediction],
    client: Optional[LLMClient],
    weight_tensor: Optional[WeightTensor],
    config: dict,
    history,
    fusion_model=None,
    step3_record: Optional[dict] = None,
) -> dict:
    if fusion_model is not None:
        # HARMONY: the LightGBM model reported in the paper. Uses this
        # sample's SCOPE profile and MAESTRO selection, both already loaded.
        from .lightgbm_fusion import fuse_sample  # noqa: PLC0415

        composite = fuse_sample(
            fusion_model,
            sample_id=sample_id, sample_json=sample_json,
            step4_record={"predictions":
                          [p.model_dump(mode="json") for p in predictions]},
            scope_profiles={sample_id: target_char or {}},
            selections={sample_id: (step3_record or {}).get("tool_plan") or {}},
            polish_actions=None,
        )
        return build_output_record(composite, sample_json, target_char)

    composite = fuse_predictions(
        sample_json=sample_json,
        target_char=target_char,
        tool_predictions=predictions,
        client=client,
        weight_tensor=weight_tensor,
        config=config,
        history=history,
    )
    return build_output_record(composite, sample_json, target_char)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Step 5 — LLM-weighted noisy-OR fusion of step 4 predictions",
    )
    ap.add_argument("--processed-dir", type=Path, required=True,
                    help="step 1 output dir containing samples/<id>.json")
    ap.add_argument("--config", type=Path, required=True,
                    help="step5_config.yaml")
    ap.add_argument("--step2-config", type=Path, default=None,
                    help="step2_config.yaml (for LLM client + JSON mode)")
    ap.add_argument("--step3-config", type=Path, default=None,
                    help="step3_config.yaml (used for weight tensor + "
                         "fallback during pipeline mode)")
    ap.add_argument("--step4-config", type=Path, default=None,
                    help="step4_config.yaml (only used in --run-pipeline mode)")
    ap.add_argument("--step2-output", type=Path, default=None,
                    help="JSONL from step 2; provides target_char per sample")
    ap.add_argument("--step4-output", type=Path, default=None,
                    help="JSONL from step 4 (one line per sample, "
                         "ToolPredictionSet shape)")
    ap.add_argument("--sample-id", type=str, default=None,
                    help="restrict processing to this sample (or required "
                         "for --run-pipeline)")
    ap.add_argument("--run-pipeline", action="store_true",
                    help="run step 2 → 3 → 4 inline before step 5 (heavy)")
    ap.add_argument("--output", type=Path, default=None,
                    help="output dir or single .jsonl file")
    ap.add_argument("--fusion", default="auto",
                    choices=["auto", "lightgbm", "noisy-or"],
                    help="'auto' uses the LightGBM model when --model-dir "
                         "holds one, else the prototype noisy-OR path")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="saved LightGBM bundle (HARMONY)")
    ap.add_argument("--no-llm", action="store_true",
                    help="force the equal-weights fallback path "
                         "(skip LLM call entirely; useful for offline smoke)")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    config = load_config(args.config)

    # HARMONY model (optional): the LightGBM bundle the paper reports.
    fusion_model = None
    if args.fusion != "noisy-or":
        model_dir = args.model_dir or (args.output or Path("data/step5_outputs"))
        model_dir = Path(model_dir).parent / "harmony_model" if args.model_dir is None else args.model_dir
        try:
            from .prediction_io import has_lightgbm_model, load_lightgbm_model  # noqa: PLC0415

            if has_lightgbm_model(model_dir):
                fusion_model = load_lightgbm_model(model_dir)
                print(f"harmony: using the LightGBM model in {model_dir}")
        except Exception as e:  # noqa: BLE001
            if args.fusion == "lightgbm":
                print(f"ERROR loading the LightGBM model: {e}", file=sys.stderr)
                return 2
            print(f"harmony: LightGBM unavailable ({e}); using noisy-OR")
    if args.fusion == "lightgbm" and fusion_model is None:
        print("ERROR: --fusion lightgbm needs --model-dir pointing at a "
              "LightGBM bundle", file=sys.stderr)
        return 2

    # Resolve sibling configs.
    cfg_dir = args.config.parent
    step2_cfg_path = args.step2_config or (cfg_dir / "step2_config.yaml")
    step2_cfg = load_step2_config(step2_cfg_path)

    # Build LLM client (unless --no-llm).
    client: Optional[LLMClient]
    if args.no_llm:
        client = None
    else:
        client = build_client(step2_cfg)

    # Optional weight tensor for prompt embedding.
    step3_cfg_path = args.step3_config or (cfg_dir / "step3_config.yaml")
    step3_cfg: dict = load_config(step3_cfg_path) if step3_cfg_path.is_file() else {}
    wt_path = Path(step3_cfg.get(
        "weight_tensor_path",
        "data/step3_weights/weight_tensor.json",
    ))
    weight_tensor = (
        WeightTensor.load(wt_path) if wt_path.is_file()
        else WeightTensor.from_config(step3_cfg) if step3_cfg
        else None
    )

    # Optional history (cold-start: file doesn't exist → None).
    history = None
    history_path_cfg = config.get("history_path")
    if history_path_cfg:
        try:
            from step2_target_char.history import PredictionHistory
            hp = Path(history_path_cfg)
            if hp.is_file():
                history = PredictionHistory.load(hp)
        except Exception:
            history = None

    # ---- choose mode -----------------------------------------------------

    if args.run_pipeline:
        if not args.sample_id:
            raise SystemExit("--run-pipeline requires --sample-id")
        if client is None:
            raise SystemExit(
                "--run-pipeline cannot be combined with --no-llm "
                "(steps 2/3 need the LLM client)"
            )
        step4_cfg_path = args.step4_config or (cfg_dir / "step4_config.yaml")
        target_char, sample_json, predictions = _run_inline_pipeline(
            sample_id=args.sample_id,
            processed_dir=args.processed_dir,
            step2_cfg=step2_cfg,
            step3_cfg=step3_cfg,
            step4_cfg_path=step4_cfg_path,
            client=client,
        )
        record = _process_one(
            args.sample_id, sample_json, target_char, predictions,
            client, weight_tensor, config, history,
            fusion_model=fusion_model,
        )
        print(_summarize(record))
        out_path = _resolve_output_path(args.output, args.sample_id)
        if out_path is not None:
            _write_record_atomic(out_path, record, append=False)
            print(f"Wrote: {out_path}")
        return 0

    # Mode 1: consume existing step 2 + step 4 JSONL
    if args.step4_output is None:
        raise SystemExit(
            "specify --step4-output (Mode 1) or --run-pipeline + "
            "--sample-id (Mode 2)"
        )
    if args.step2_output is None:
        raise SystemExit("Mode 1 requires --step2-output")

    s2_index = _index_step2_records(_load_jsonl(args.step2_output))
    s4_index = _index_step4_records(_load_jsonl(args.step4_output))

    sample_ids = [args.sample_id] if args.sample_id else sorted(s4_index.keys())
    if not sample_ids:
        raise SystemExit("no sample_ids resolved from --step4-output")

    # If --output is a single .jsonl file, append all records there;
    # otherwise resolve per-sample paths.
    write_to_one_file = (
        args.output is not None and args.output.suffix in (".jsonl", ".json")
    )
    one_file_path: Optional[Path] = None
    if write_to_one_file:
        one_file_path = args.output
        one_file_path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate if it exists — semantics match step 2/3 batch runs.
        one_file_path.write_text("", encoding="utf-8")

    n_ok = n_fb = n_empty = n_single = 0
    total_tokens = 0
    for sid in sample_ids:
        s4_record = s4_index.get(sid)
        if not s4_record:
            print(f"[SKIP      ] {sid}: no step 4 record")
            continue
        try:
            sample_json = _load_sample(args.processed_dir, sid)
        except FileNotFoundError as e:
            print(f"[SKIP      ] {sid}: {e}")
            continue
        target_char = s2_index.get(sid) or {}
        try:
            predictions = _step4_record_to_predictions(s4_record)
        except Exception as e:
            print(f"[SKIP      ] {sid}: step4 record invalid ({e})")
            continue

        record = _process_one(
            sid, sample_json, target_char, predictions,
            client, weight_tensor, config, history,
        )
        print(_summarize(record))

        out_path = (one_file_path if write_to_one_file
                    else _resolve_output_path(args.output, sid))
        if out_path is not None:
            _write_record_atomic(out_path, record, append=write_to_one_file)

        status = (record.get("api_usage") or {}).get("status", "ok")
        if status == "ok":
            n_ok += 1
        elif status == "fallback":
            n_fb += 1
        elif status == "empty":
            n_empty += 1
        elif status == "single_tool":
            n_single += 1
        total_tokens += (record.get("api_usage") or {}).get("total_tokens", 0) or 0

    print()
    print(
        f"Processed: {len(sample_ids)}  "
        f"ok={n_ok}  fallback={n_fb}  single={n_single}  empty={n_empty}"
    )
    print(f"Total tokens: {total_tokens}")
    if args.output:
        print(f"Wrote: {args.output}")
    return 0 if n_fb == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
