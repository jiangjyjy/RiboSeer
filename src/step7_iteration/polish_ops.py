"""POLISH probability-modification ops for paper Table 9.

Problem (Table 9 plan §3)
-------------------------
The shipped POLISH (step7) only edits the *binary* binding-residue set;
it never touches ``per_residue_probability``. So on the headline metric
(per-residue Pearson R) POLISH is a no-op and can't be ablated.

This module makes POLISH act on the continuous probability vector so the
ablation bites. The edits are **soft** (scale, never hard-zero) so a wrong
call costs less and the LLM's ``confidence`` can modulate the strength:

* ``mask``     → ``p *= 1 - 0.8*confidence`` (default 0.2 when no
                 confidence ⇒ high-confidence behaviour). Suppresses a
                 likely false positive without erasing it.
* ``extend``   → ``p = max(p, 0.8 * local binding-neighbour mean)`` (the
                 neighbour mean uses a 0.5 fallback when no binding
                 neighbour); only ever raises a residue.
* ``relocate`` → suppress the old region (``p *= 1 - 0.8*confidence``,
                 default 0.2) and lift the new region to
                 ``max(p, 0.8 * within-region max)``.
* ``accept``   → no change.

``confidence`` (optional, in ``[0, 1]``) is read from the action dict —
the LLM's self-rated certainty in the edit. Low confidence → gentler
removal (``confidence`` 0 ⇒ factor 1.0, i.e. no-op); high confidence ⇒
factor 0.2. Absent confidence defaults to the strong 0.2 factor so the
deterministic ``auto_polish_action`` control keeps its original strength.

The probability vector is represented as a ``dict[int, float]``
(residue id → fused probability), matching what the fusion layer / the
ablation driver produce.

Two action sources
-------------------
* ``apply_polish_to_probability`` — applies a single action dict (the
  shape a LLM POLISH call returns; see plan §3.4). Used on the server
  with saved LLM actions, and in tests.
* ``auto_polish_action`` — a deterministic, no-LLM control action used in
  Phase 1 so the POLISH=on control arm differs from POLISH=off without
  any API call. It denoises: masks predicted-binding residues whose
  ±2 neighbour-smoothed probability is weak (likely false positives),
  and extends short gaps between strong binding residues.
"""
from __future__ import annotations

from typing import Iterable, Optional

# Soft-edit constants (plan §3, gentler-strategy revision).
_REMOVE_FACTOR_DEFAULT = 0.2   # mask/relocate-away factor when no confidence
_CONF_WEIGHT = 0.8             # confidence scaling: factor = 1 - w*confidence
_LIFT_FACTOR = 0.8             # extend / relocate-into lift fraction


def _removal_factor(action: Optional[dict]) -> float:
    """Multiplicative factor for a masked / relocated-away residue.

    With an LLM ``confidence`` in ``[0, 1]``: ``1 - 0.8*confidence`` — high
    confidence drives the residue down toward 0.2×, low confidence barely
    touches it (confidence 0 ⇒ factor 1.0, a no-op). Without a usable
    ``confidence`` field, defaults to ``0.2`` (≡ confidence 1.0), matching
    the deterministic control's original strength.
    """
    conf = (action or {}).get("confidence")
    if conf is None:
        return _REMOVE_FACTOR_DEFAULT
    try:
        c = float(conf)
    except (TypeError, ValueError):
        return _REMOVE_FACTOR_DEFAULT
    c = max(0.0, min(1.0, c))
    return 1.0 - _CONF_WEIGHT * c


def _as_int_set(values: Optional[Iterable]) -> set[int]:
    out: set[int] = set()
    for v in values or []:
        try:
            out.add(int(v))
        except (TypeError, ValueError):
            continue
    return out


def apply_polish_to_probability(
    probability: dict[int, float],
    action: dict,
    *,
    binding_set: Optional[Iterable[int]] = None,
    extend_fallback: float = 0.5,
    remove_factor: Optional[float] = None,
    lift_factor: float = _LIFT_FACTOR,
) -> dict[int, float]:
    """Return a NEW probability dict with ``action`` applied.

    ``action`` schema (plan §3.4)::

        {"action": "mask"|"extend"|"relocate"|"accept",
         "residues": [...],
         "target_residues": [...],   # relocate only
         "confidence": 0.0-1.0}      # optional; scales mask / relocate-away

    ``binding_set`` is the current predicted binding residues (used by
    ``extend`` to find binding neighbours); defaults to ``{r : p > 0.5}``.
    The input dict is not mutated.

    Strength overrides (Table-12 gentle ablation; defaults preserve the
    headline behaviour exactly):

    * ``remove_factor`` — when given, the multiplicative factor for a
      masked / relocated-away residue, *overriding* the confidence-derived
      ``_removal_factor``. 1.0 ⇒ no suppression, 0.2 ⇒ the current strength.
    * ``lift_factor`` — the fraction applied to extend / relocate-into
      lifts (default ``_LIFT_FACTOR`` = 0.8). 0.0 ⇒ no lift.
    """
    prob = dict(probability)
    if not prob:
        return prob
    act = (action or {}).get("action", "accept")
    residues = _as_int_set(action.get("residues") if action else None)

    if binding_set is None:
        bind = {r for r, p in prob.items() if p > 0.5}
    else:
        bind = _as_int_set(binding_set)

    if act == "accept" or not act:
        return prob

    factor = (_removal_factor(action) if remove_factor is None
              else remove_factor)

    if act == "mask":
        for r in residues:
            if r in prob:
                prob[r] *= factor          # soft suppress (default 0.2×)
        return prob

    if act == "extend":
        for r in residues:
            if r not in prob:
                continue
            nb = _neighbour_binding_mean(prob, bind, r,
                                         fallback=extend_fallback)
            prob[r] = max(prob[r], lift_factor * nb)   # only ever raises
        return prob

    if act == "relocate":
        target = _as_int_set(action.get("target_residues"))
        for r in residues:            # old region → soft suppress
            if r in prob:
                prob[r] *= factor
        if target:
            region_max = max((prob.get(r, 0.0) for r in target), default=0.0)
            if region_max <= 0.0:
                region_max = extend_fallback
            lift = lift_factor * region_max
            for r in target:
                if r in prob:
                    prob[r] = max(prob[r], lift)
        return prob

    # Unknown action → no-op (defensive; never raise on a bad LLM dict).
    return prob


def _neighbour_binding_mean(prob: dict[int, float], bind: set[int],
                            residue: int, *, fallback: float,
                            radius: int = 2) -> float:
    """Mean probability of binding residues within ±radius of ``residue``;
    ``fallback`` when none are present."""
    vals = [prob[r] for d in range(1, radius + 1)
            for r in (residue - d, residue + d)
            if r in bind and r in prob]
    return sum(vals) / len(vals) if vals else fallback


# ---------------------------------------------------------------------------
# Deterministic control action (no LLM) — Phase 1
# ---------------------------------------------------------------------------


def auto_polish_action(
    probability: dict[int, float],
    *,
    binding_threshold: float = 0.5,
    weak_quantile: float = 0.25,
    smooth_radius: int = 2,
) -> dict:
    """Build a deterministic POLISH action from the probability vector.

    Strategy (conservative false-positive removal): among residues
    currently predicted as binding (``p > binding_threshold``), mask the
    weakest by *neighbour-smoothed* probability — those sitting below the
    ``weak_quantile`` of the smoothed binding scores. Smoothing avoids
    masking a strong residue that merely has one weak neighbour.

    Returns a ``mask`` action (possibly with an empty ``residues`` list,
    which ``apply_polish_to_probability`` treats as a no-op). Deterministic
    given the same input — no randomness, no API.
    """
    if not probability:
        return {"action": "accept", "residues": []}
    bind = [r for r, p in probability.items() if p > binding_threshold]
    if len(bind) < 4:
        # Too few binding residues to safely prune.
        return {"action": "accept", "residues": []}

    smoothed = {
        r: _neighbour_smoothed(probability, r, smooth_radius)
        for r in bind
    }
    ordered = sorted(bind, key=lambda r: (smoothed[r], r))
    k = max(1, int(len(bind) * weak_quantile))
    to_mask = sorted(ordered[:k])
    return {"action": "mask", "residues": to_mask,
            "reasoning": "auto: prune weakest neighbour-smoothed binders"}


def _neighbour_smoothed(prob: dict[int, float], residue: int,
                        radius: int) -> float:
    """Mean of ``prob`` over ``[residue-radius, residue+radius]`` for the
    positions that exist in the dict (always includes the centre)."""
    vals = [prob[r] for r in range(residue - radius, residue + radius + 1)
            if r in prob]
    return sum(vals) / len(vals) if vals else prob.get(residue, 0.0)


def apply_actions(
    probability: dict[int, float],
    actions: Iterable[dict],
    *,
    binding_set: Optional[Iterable[int]] = None,
    remove_factor: Optional[float] = None,
    lift_factor: float = _LIFT_FACTOR,
) -> dict[int, float]:
    """Apply a sequence of POLISH actions in order (each sees the prior's
    result). ``binding_set`` is recomputed from the live vector before
    each action unless explicitly pinned. ``remove_factor`` / ``lift_factor``
    are passed through to every action (Table-12 gentle ablation; defaults
    preserve the headline strength)."""
    prob = dict(probability)
    for act in actions or []:
        prob = apply_polish_to_probability(
            prob, act, binding_set=binding_set,
            remove_factor=remove_factor, lift_factor=lift_factor)
    return prob
