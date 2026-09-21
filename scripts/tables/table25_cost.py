"""Table 26 — computational cost per module.

Reports, per sample: wall-clock seconds, number of LLM API calls, and
input/output token usage for each pipeline module
(GENESIS / SCOPE / MAESTRO / RELAY / HARMONY / VERDICT / POLISH / MEMORY).

What is measured vs read vs unavailable
---------------------------------------
* **GENESIS / HARMONY / MEMORY** — actually profiled here on ``--profile-n``
  samples (load + feature build / LightGBM forward / EMA weight update).
* **RELAY** — tool execution time, read from step4 ``total_runtime_seconds``
  (or the per-tool ``runtime_seconds`` sum). Marked "–" when the step4
  records carry no runtime (e.g. a stripped local copy).
* **SCOPE / MAESTRO / POLISH** — LLM calls. The call COUNT is deterministic
  (SCOPE 1, MAESTRO 1, POLISH = the mean call count read from ``--polish-dir``
  — the iterative ``total_rounds``/``rounds`` length, or 1 per sample for the
  single-round action format). Wall-clock + token usage come from a LLM log
  (``--usage-log``, e.g. ``logs/llm_usage.jsonl``) — but that log is NOT
  module-tagged, so the per-call latency/tokens shown for SCOPE/MAESTRO/
  POLISH are the SAME average across all LLM calls (noted below). Without a
  usage log these are "–".
* **VERDICT** — step6 PocketQA; not profiled here (would need the step6
  pipeline) → "–".

No LLM is called; nothing is written. Tokens/latency are read, never re-run.

Run
---
::

    python scripts/tables/table25_cost.py \\
        --data-dir    data/processed_quality \\
        --step4-dir   data/batch_test_v7/step4 \\
        --model-dir   data/enriched_v7_lgbm \\
        --split-file  data/processed_quality/splits_tmscore_035/test.txt \\
        --polish-dir  data/batch_test_v7/polish_actions_llm_v2 \\
        --usage-log   logs/llm_usage.jsonl \\
        --profile-n   20
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from step5_fusion.data_collector import read_jsonl_record  # noqa: E402
from step5_fusion.enriched_fusion import EnrichedFusion  # noqa: E402
from step3_tool_selection.weight_tensor import METRICS, WeightTensor  # noqa: E402
from scripts.riboseer.ablation_fusion_method import collect_sample_data  # noqa: E402
from scripts.tables.table15_rna_length import riboseer_predict  # noqa: E402


# ---------------------------------------------------------------------------
# Profilers (actually time these locally)
# ---------------------------------------------------------------------------


def profile_genesis(step4_dir: Path, data_dir: Path, ids: list[str],
                    model: EnrichedFusion, n: int):
    """Time GENESIS (load + GT + feature build) over ``n`` samples; returns
    (per_sample_seconds, collected_samples) — samples reused by HARMONY."""
    sub = ids[:max(1, n)]
    t0 = time.perf_counter()
    samples = collect_sample_data(
        step4_dir, data_dir, sub,
        feature_set=model.feature_set, use_context=model.use_context)
    dt = time.perf_counter() - t0
    per = dt / len(samples) if samples else None
    return per, samples


def profile_harmony(samples, model: EnrichedFusion) -> Optional[float]:
    """Time HARMONY (LightGBM forward) per sample."""
    if not samples:
        return None
    t0 = time.perf_counter()
    for s in samples:
        riboseer_predict(model, s.X)
    return (time.perf_counter() - t0) / len(samples)


def profile_memory(n: int) -> float:
    """Time MEMORY (EMA weight-tensor update) per sample — 15 tools × 5
    metrics, the system's update footprint."""
    W = WeightTensor()
    tools = [f"tool{i}" for i in range(15)]
    reps = max(1, n)
    t0 = time.perf_counter()
    for _ in range(reps):
        for t in tools:
            for m in METRICS:
                old = W.get_weight(t, m, "cat")
                W.update(t, "cat", m, 0.9 * old + 0.1 * 0.5)
    return (time.perf_counter() - t0) / reps


# ---------------------------------------------------------------------------
# Read from existing data
# ---------------------------------------------------------------------------


def relay_time(step4_dir: Path, ids: list[str]) -> tuple[Optional[float], int]:
    """Mean per-sample tool runtime from step4 ``total_runtime_seconds`` (or
    the per-tool ``runtime_seconds`` sum). (mean_or_None, n_with_data)."""
    vals: list[float] = []
    for sid in ids:
        rec = read_jsonl_record(step4_dir / f"{sid}.jsonl")
        if rec is None:
            continue
        tot = rec.get("total_runtime_seconds")
        if not isinstance(tot, (int, float)):
            parts = [p.get("runtime_seconds") for p in
                     (rec.get("predictions") or [])]
            parts = [p for p in parts if isinstance(p, (int, float))]
            tot = sum(parts) if parts else None
        if isinstance(tot, (int, float)):
            vals.append(float(tot))
    return (round(statistics.fmean(vals), 2) if vals else None), len(vals)


def mean_polish_rounds(polish_dir: Optional[Path], ids: list[str]
                       ) -> tuple[Optional[float], int]:
    """Mean POLISH API calls per sample, over the ``{sid}.json`` files in the
    polish output dir. Two layouts are supported per file:

    * **Iterative** — a ``total_rounds`` int (or a ``rounds`` list): that
      round count is the number of LLM calls for the sample.
    * **Single-round** — a flat action record (``{"action": ..., "residues":
      ..., "source": "llm", ...}``) with no rounds field: exactly one LLM
      call, so it counts as 1.

    Returns (mean_calls_or_None, n_samples_with_data)."""
    if polish_dir is None or not polish_dir.is_dir():
        return None, 0
    vals: list[int] = []
    for sid in ids:
        f = polish_dir / f"{sid}.json"
        if not f.is_file():
            continue
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(rec, dict):
            continue
        tr = rec.get("total_rounds")
        if not isinstance(tr, int):
            rounds = rec.get("rounds")
            if isinstance(rounds, list) and rounds:
                tr = len(rounds)               # iterative: one call per round
            elif rec.get("action") is not None:
                tr = 1                          # single-round POLISH: 1 call
            else:
                tr = 0
        if tr:
            vals.append(tr)
    return (round(statistics.fmean(vals), 2) if vals else None), len(vals)


def llm_usage(usage_log: Optional[Path]) -> Optional[dict]:
    """Mean per-call latency (s) + prompt/completion tokens from a LLM usage
    log. NOT module-tagged → one average across all LLM calls."""
    if usage_log is None or not usage_log.is_file():
        return None
    lat, pin, pout = [], [], []
    for line in usage_log.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not rec.get("success"):
            continue
        if isinstance(rec.get("latency_ms"), (int, float)):
            lat.append(rec["latency_ms"] / 1000.0)
        if isinstance(rec.get("prompt_tokens"), (int, float)):
            pin.append(rec["prompt_tokens"])
        if isinstance(rec.get("completion_tokens"), (int, float)):
            pout.append(rec["completion_tokens"])
    if not lat and not pin:
        return None
    return {
        "latency_s": round(statistics.fmean(lat), 2) if lat else None,
        "in_tok": int(round(statistics.fmean(pin))) if pin else None,
        "out_tok": int(round(statistics.fmean(pout))) if pout else None,
        "n_calls": len(lat) or len(pin),
    }


# ---------------------------------------------------------------------------
# Assemble + print
# ---------------------------------------------------------------------------


def _f(v, fmt="{:.2f}"):
    return fmt.format(v) if isinstance(v, (int, float)) else "–"


def _tok(i, o):
    if i is None and o is None:
        return "–"
    return f"{i if i is not None else '?'} / {o if o is not None else '?'}"


def build_rows(*, genesis, harmony, memory, relay, polish_rounds,
               llm) -> list[dict]:
    lw = llm["latency_s"] if llm else None
    li = llm["in_tok"] if llm else None
    lo = llm["out_tok"] if llm else None
    pr = polish_rounds
    polish_wall = (lw * pr if (lw is not None and pr is not None) else None)
    polish_in = (li * pr if (li is not None and pr is not None) else None)
    polish_out = (lo * pr if (lo is not None and pr is not None) else None)

    def num(*xs):
        xs = [x for x in xs if isinstance(x, (int, float))]
        return sum(xs) if xs else None

    total_calls = 2 + (pr if isinstance(pr, (int, float)) else 0)
    rows = [
        {"mod": "GENESIS (preprocessing)", "wall": genesis, "calls": 0,
         "in": None, "out": None},
        {"mod": "SCOPE (LLM profiling)", "wall": lw, "calls": 1,
         "in": li, "out": lo},
        {"mod": "MAESTRO (LLM selection)", "wall": lw, "calls": 1,
         "in": li, "out": lo},
        {"mod": "RELAY (tool execution)", "wall": relay, "calls": 0,
         "in": None, "out": None},
        {"mod": "HARMONY (fusion)", "wall": harmony, "calls": 0,
         "in": None, "out": None},
        {"mod": "VERDICT (quality assessment)", "wall": None, "calls": 0,
         "in": None, "out": None},
        {"mod": "POLISH (LLM refinement)", "wall": polish_wall,
         "calls": pr, "in": polish_in, "out": polish_out},
        {"mod": "MEMORY (weight update)", "wall": memory, "calls": 0,
         "in": None, "out": None},
        {"mod": "Total per sample",
         "wall": num(genesis, lw, lw, relay, harmony, memory, polish_wall),
         "calls": total_calls,
         "in": num(li, li, polish_in), "out": num(lo, lo, polish_out)},
    ]
    return rows


def print_table(rows: list[dict]) -> None:
    print("=== Table 26: Computational Cost (per sample) ===")
    hdr = (f"{'Module':<28s} {'Wall clock (s)':>14s} {'# API calls':>11s} "
           f"{'In/Out tokens':>16s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r["mod"].startswith("Total"):
            print("-" * len(hdr))
        calls = (_f(r["calls"], "{:g}") if r["calls"] not in (0, None)
                 else ("0" if r["calls"] == 0 else "–"))
        print(f"{r['mod']:<28s} {_f(r['wall']):>14s} {calls:>11s} "
              f"{_tok(r['in'], r['out']):>16s}")


def _load_split(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"split file not found: {path}")
    return [ln.split()[0] for ln in
            path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--step4-dir", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--split-file", type=Path, required=True)
    p.add_argument("--polish-dir", type=Path, default=None,
                   help="POLISH iterative output dir (for mean round count)")
    p.add_argument("--usage-log", type=Path,
                   default=Path("logs/llm_usage.jsonl"),
                   help="LLM usage JSONL for LLM latency/token stats")
    p.add_argument("--profile-n", type=int, default=20,
                   help="#samples to time for GENESIS/HARMONY/MEMORY")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for d in (args.data_dir, args.step4_dir):
        if not d.is_dir():
            print(f"ERROR: not a directory: {d}", file=sys.stderr)
            return 1
    try:
        ids = _load_split(args.split_file)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    try:
        model = EnrichedFusion.load(args.model_dir)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: failed to load model from {args.model_dir}: {e}",
              file=sys.stderr)
        return 1

    print(f"profiling {args.profile_n} samples for GENESIS/HARMONY/MEMORY ...")
    genesis, samples = profile_genesis(args.step4_dir, args.data_dir, ids,
                                       model, args.profile_n)
    harmony = profile_harmony(samples, model)
    memory = profile_memory(args.profile_n)
    relay, n_relay = relay_time(args.step4_dir, ids)
    polish_rounds, n_polish = mean_polish_rounds(args.polish_dir, ids)
    llm = llm_usage(args.usage_log)

    rows = build_rows(genesis=genesis, harmony=harmony, memory=memory,
                      relay=relay, polish_rounds=polish_rounds, llm=llm)
    print()
    print_table(rows)
    print()
    # Provenance / caveats so the "–" cells are self-explanatory.
    print("notes:")
    print(f"  GENESIS/HARMONY/MEMORY: timed on {len(samples)} samples "
          f"(this machine).")
    if relay is None:
        print("  RELAY: '–' — step4 records carry no total_runtime_seconds / "
              "runtime_seconds here (populated on the full server step4).")
    else:
        print(f"  RELAY: mean over {n_relay} step4 records with runtime.")
    if polish_rounds is None:
        print("  POLISH calls: '–' — pass --polish-dir (reads iterative "
              "total_rounds, or counts each single-round action file as 1).")
    else:
        print(f"  POLISH: mean {polish_rounds} call(s)/sample over "
              f"{n_polish} samples.")
    if llm is None:
        print(f"  SCOPE/MAESTRO/POLISH wall+tokens: '–' — no usable "
              f"--usage-log ({args.usage_log}).")
    else:
        print(f"  LLM latency/tokens: mean over {llm['n_calls']} calls in "
              f"{args.usage_log} — NOT module-tagged, so SCOPE/MAESTRO/POLISH "
              f"per-call figures are the same average.")
    print("  VERDICT: '–' — step6 PocketQA not profiled here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
