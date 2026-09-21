"""Shared per-residue data collection for the learned-fusion path(s).

Both ``LearnedFusion`` (Ridge) and ``XGBFusion`` need the same input:
per-(sample, residue) rows carrying tool scores, gating flags, the
amino-acid letter, and the binary GT label. This module owns that
collection so the two model classes can't drift from each other —
swap the model and you keep the same training set verbatim.

What "score field" means
------------------------
Cat A tools (Boltz-2, Chai-1) emit a per-residue confidence dict on
two fields:

  - ``per_residue_pae_score``  (interface-affinity proxy, paper §4 fix)
  - ``per_residue_confidence`` (pLDDT — historical; structure-quality)

Cat B/C tools (P2Rank, Fpocket, EquiPNAS) only emit the second.
``score_fields`` decides which one to read per tool. The collector
falls back to the alternative field when the configured one is empty,
so a Cat A run that lost PAE still contributes pLDDT.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional


# Default per-tool field map — same as in learned_fusion's defaults.
DEFAULT_SCORE_FIELDS: dict[str, str] = {
    "boltz2":   "per_residue_pae_score",
    "chai1":    "per_residue_pae_score",
    "equipnas": "per_residue_confidence",
    "p2rank":   "per_residue_confidence",
    "fpocket":  "per_residue_confidence",
}


# ---- single-record JSONL + sample JSON readers ---------------------------


def read_jsonl_record(path: Path) -> Optional[dict]:
    """First parseable record in a one-per-file JSONL.

    Returns ``None`` on missing / malformed file (the caller decides
    whether that's a skip or an error). batch_predict.write_step_record
    emits exactly one record per file, so this is sufficient.
    """
    if not path.is_file():
        return None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            return json.loads(line)
    except (OSError, json.JSONDecodeError):
        return None
    return None


def load_sample_json(processed_dir: Path, sample_id: str) -> Optional[dict]:
    """Load step-1 sample JSON with case-insensitive fallback.

    Mirrors batch_predict.load_sample / scripts/evaluate's case scan —
    splits.json sometimes carries a different case than the on-disk
    filename (e.g. ``3j46_y_1`` in splits vs ``3j46_Y_1.json`` on disk).
    """
    samples_dir = processed_dir / "samples"
    for p in (samples_dir / f"{sample_id}.json",
              processed_dir / f"{sample_id}.json"):
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
    if samples_dir.is_dir():
        target = sample_id.lower()
        for f in samples_dir.iterdir():
            if f.suffix == ".json" and f.stem.lower() == target:
                try:
                    return json.loads(f.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    return None
    return None


def per_residue_to_int_dict(per_res: Optional[dict]) -> dict[int, float]:
    """Coerce a per-residue map to ``{int: float}`` (drops bad entries).

    JSON serialisation forces dict keys to strings; some adapters
    round-trip with int keys already. Accept either.
    """
    if not per_res:
        return {}
    out: dict[int, float] = {}
    for k, v in per_res.items():
        try:
            ki = int(k)
            vf = float(v)
        except (TypeError, ValueError):
            continue
        out[ki] = vf
    return out


def aa_for(seq: str, residue_id_1based: int) -> str:
    """1-based index into ``seq``; out-of-range / missing → ''."""
    if not seq or residue_id_1based < 1:
        return ""
    if residue_id_1based > len(seq):
        return ""
    return seq[residue_id_1based - 1]


# ---- score-field selection ----------------------------------------------


def resolve_score_fields(cfg: dict) -> dict[str, str]:
    """Merge per-tool overrides on top of DEFAULT_SCORE_FIELDS.

    Two equivalent shapes are accepted:
      - flat ``score_fields: {tool_id: field_name}``
      - nested ``tools: {tool_id: {score_field: field_name}}``

    The nested form matches the YAML schema used in
    learned_fusion_config.yaml / xgb_fusion_config.yaml, so callers
    can pass the loaded YAML block straight in.
    """
    out = dict(DEFAULT_SCORE_FIELDS)
    flat = (cfg or {}).get("score_fields") or {}
    for k, v in flat.items():
        if isinstance(v, str):
            out[k] = v
    nested = (cfg or {}).get("tools") or {}
    for tool_id, sub in nested.items():
        if isinstance(sub, dict):
            field = sub.get("score_field")
            if isinstance(field, str):
                out[tool_id] = field
    return out


def extract_score(
    prediction: dict, residue_id: int, score_fields: dict[str, str],
) -> Optional[float]:
    """Pull the configured score-field value for ``residue_id`` (1-based).

    Falls back to the alternative field
    (``per_residue_confidence`` ↔ ``per_residue_pae_score``) so a Cat A
    run that lost PAE still contributes pLDDT instead of dropping out
    entirely. ``None`` means "tool didn't score this residue" — both
    Ridge and XGB downstream treat that as 0.
    """
    tool_id = prediction.get("tool_id") or ""
    field = score_fields.get(tool_id, "per_residue_confidence")
    per_res = per_residue_to_int_dict(prediction.get(field))
    if not per_res:
        alt = (
            "per_residue_confidence"
            if field == "per_residue_pae_score"
            else "per_residue_pae_score"
        )
        per_res = per_residue_to_int_dict(prediction.get(alt))
    return per_res.get(residue_id)


# ---- main collector -----------------------------------------------------


def collect_per_residue_data(
    *,
    step4_dir: Path,
    processed_dir: Path,
    score_fields: Optional[dict[str, str]] = None,
    sample_ids: Optional[Iterable[str]] = None,
) -> dict:
    """Walk step4/ JSONLs + sample JSONs → per-residue training rows.

    ``sample_ids`` restricts to a known subset (typical: the train
    split's ``train.txt``); if None, every step4 JSONL under
    ``step4_dir`` is processed.

    Skip semantics
    --------------
    Samples without ground truth in ``interaction.binding_protein_residues``
    are excluded — a no-positive sample contributes only negatives,
    which would skew the bias and starve a discriminative fit.

    Residues that NO tool scored AND that no tool flagged on its
    binding list are skipped — a fully-zero row only adds to the
    bias term, biasing it toward the negative class proportional to
    protein length.

    Returns
    -------
    bundle : dict with keys:
      - ``per_tool_scores`` : ``{tool_id: [score, ...]}`` flat lists for
        the standardiser to fit on.
      - ``rows`` : per-(sample, residue) entries, each with
        ``sample_id``, ``residue_id``, ``tool_scores``,
        ``tool_binding_flags``, ``aa``, ``label``.
      - ``n_samples_used`` / ``n_samples_skipped`` : provenance.
    """
    if score_fields is None:
        score_fields = dict(DEFAULT_SCORE_FIELDS)
    step4_dir = Path(step4_dir)
    processed_dir = Path(processed_dir)
    per_tool_scores: dict[str, list[float]] = defaultdict(list)
    rows: list[dict] = []
    n_used = 0
    n_skip = 0

    if sample_ids is not None:
        sids = list(sample_ids)
    else:
        sids = sorted(p.stem for p in step4_dir.glob("*.jsonl"))

    for sid in sids:
        s4 = read_jsonl_record(step4_dir / f"{sid}.jsonl")
        if s4 is None:
            n_skip += 1
            continue
        sample = load_sample_json(processed_dir, sid)
        if sample is None:
            n_skip += 1
            continue
        seq = (sample.get("protein") or {}).get("sequence") or ""
        length = (sample.get("protein") or {}).get("length") or len(seq)
        gt = set(
            (sample.get("interaction") or {}).get(
                "binding_protein_residues") or []
        )
        if not length or not gt:
            n_skip += 1
            continue

        preds = [p for p in (s4.get("predictions") or []) if p.get("success")]
        if not preds:
            n_skip += 1
            continue

        # Pre-compute each tool's binding-list as a set so the
        # per-residue inner loop is O(1) per (residue, tool).
        binding_sets: dict[str, set[int]] = {}
        for pred in preds:
            tid = pred.get("tool_id") or ""
            bset: set[int] = set()
            for v in (pred.get("binding_protein_residues") or []):
                try:
                    bset.add(int(v))
                except (TypeError, ValueError):
                    continue
            binding_sets[tid] = bset

        for res_id in range(1, int(length) + 1):
            tool_scores: dict[str, float] = {}
            for pred in preds:
                score = extract_score(pred, res_id, score_fields)
                if score is None:
                    continue
                tool_scores[pred.get("tool_id") or ""] = score
            # Gating flag PER TOOL — across every successful tool,
            # not only the ones that scored this residue.
            flags: dict[str, bool] = {
                tid: (res_id in bset)
                for tid, bset in binding_sets.items()
            }
            # Skip rows with no scores AND no flags — pure zero rows
            # only inflate the bias term.
            if not tool_scores and not any(flags.values()):
                continue
            aa = aa_for(seq, res_id)
            label = 1 if res_id in gt else 0
            rows.append({
                "sample_id": sid,
                "residue_id": res_id,
                "tool_scores": tool_scores,
                "tool_binding_flags": flags,
                "aa": aa,
                "label": label,
            })
            for tid, sc in tool_scores.items():
                per_tool_scores[tid].append(sc)
        n_used += 1

    return {
        "per_tool_scores": dict(per_tool_scores),
        "rows": rows,
        "n_samples_used": n_used,
        "n_samples_skipped": n_skip,
    }
