"""Stage 1.6b — dataset loading API.

A small class that reads stage-1 outputs and yields filtered sample dicts.
Designed to be callable from step 2+ without any other step1 dependency.

Filter parameters
-----------------
- `quality_tier`: `"strict"` / `"standard"` / `"low"` / `"discard"` / `"all"`
  (or a list/set to include multiple). Default `"all"`.
- `split`: `"train"` / `"val"` / `"test"` / `"all"` / `None`.
  `None` means "return everything, including samples with split=null"
  (i.e. unclusterable samples from stage 1.5). `"all"` means
  "everything that HAS a non-null split"; useful once stage 1.5 has run.
  Default `None`.
- `has_pi` (`bool | None`): if `True`, exclude samples where
  `protein.features.pI` is null (the 122 all-X edge cases from 1.4).
  If `False`, return only those. If `None`, no filter. Default `True`.
- `has_bfactor` (`bool | None`): same semantics for
  `protein.features.mean_bfactor`. Default `None`.
- `has_ss` (`bool | None`): filter on whether `rna.features.ss_status`
  equals `"done"` (i.e. 1.3b has populated this sample). Default `None`.

Fast path
---------
If `data/processed/_feature_flags.csv` exists (written by `validate.py`),
the loader filters at the index level — zero sample JSON reads to build
the filtered id list. If the sidecar is missing, the loader falls back to
opening each sample JSON to check `has_pi` / `has_bfactor` / `has_ss`,
which is ~10x slower. Just run `validate.py` once after changing anything
in the pipeline to keep the sidecar fresh.

Usage
-----
```python
from dataset_loader import SampleLoader
from pathlib import Path

# strict tier, with pI, no split filter (pre-1.5)
loader = SampleLoader(
    processed_dir=Path("data/processed"),
    quality_tier="strict",
    has_pi=True,
)
print(f"{len(loader)} samples")
for sample in loader:
    ...

# after stage 1.5 finishes, this picks up the split filter for free
train = SampleLoader(
    processed_dir=Path("data/processed"),
    quality_tier=("strict", "standard"),
    split="train",
)
```
"""

import csv
import json
import warnings
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from _common import sid_to_filename  # noqa: E402 — Windows DLL path fix


_ALL_TIERS = ("strict", "standard", "low", "discard")


def _norm_tier_filter(value: Any) -> set[str] | None:
    if value is None or value == "all":
        return None
    if isinstance(value, str):
        return {value}
    return set(value)


class SampleLoader:
    """Lightweight, filter-aware iterator over stage-1 sample JSONs."""

    def __init__(
        self,
        processed_dir: Path,
        quality_tier: Any = "all",
        split: str | None = None,
        has_pi: bool | None = True,
        has_bfactor: bool | None = None,
        has_ss: bool | None = None,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.samples_dir = self.processed_dir / "samples"
        self.quality_tier = quality_tier
        self.split = split
        self.has_pi = has_pi
        self.has_bfactor = has_bfactor
        self.has_ss = has_ss

        self._sidecar_rows = self._load_sidecar()
        self._index_rows = self._load_index()
        self._ids = self._build_filtered_ids()

    # ---------- loading helpers -----------------------------------------

    def _load_index(self) -> dict[str, dict]:
        path = self.processed_dir / "index.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run stage 1.2 first"
            )
        with path.open("r", encoding="utf-8") as f:
            return {r["sample_id"]: r for r in csv.DictReader(f)}

    def _load_sidecar(self) -> dict[str, dict] | None:
        path = self.processed_dir / "_feature_flags.csv"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            rows = {}
            for r in csv.DictReader(f):
                # csv roundtrips bools as strings — normalize.
                r["has_pi"] = r["has_pi"] == "True"
                r["has_bfactor"] = r["has_bfactor"] == "True"
                r["has_gc"] = r["has_gc"] == "True"
                rows[r["sample_id"]] = r
            return rows

    def _sample_path(self, sid: str) -> Path:
        return self.samples_dir / (sid_to_filename(sid) + ".json")

    def _load_sample(self, sid: str) -> dict:
        with self._sample_path(sid).open("r", encoding="utf-8") as f:
            return json.load(f)

    # ---------- filtering -----------------------------------------------

    def _build_filtered_ids(self) -> list[str]:
        tier_filter = _norm_tier_filter(self.quality_tier)
        ids = list(self._index_rows.keys())

        # Cheap filter 1: quality_tier (from index.csv)
        if tier_filter is not None:
            ids = [sid for sid in ids if self._index_rows[sid]["quality_tier"] in tier_filter]

        need_json = (
            self._sidecar_rows is None
            and (
                self.split not in (None, "all")
                or self.has_pi is not None
                or self.has_bfactor is not None
                or self.has_ss is not None
            )
        )

        if need_json:
            warnings.warn(
                "_feature_flags.csv not found — falling back to opening sample JSONs "
                "for has_pi / has_bfactor / has_ss / split filters. "
                "Run `validate.py` once to regenerate the sidecar."
            )

        out: list[str] = []
        for sid in ids:
            # Get flags for this sample — from sidecar if present, else
            # from the sample JSON (slow path).
            if self._sidecar_rows is not None and sid in self._sidecar_rows:
                flags = self._sidecar_rows[sid]
                sample_split = flags["split"] or None
                flag_pi = flags["has_pi"]
                flag_bf = flags["has_bfactor"]
                flag_ss_status = flags["ss_status"]
            elif need_json:
                try:
                    sample = self._load_sample(sid)
                except FileNotFoundError:
                    continue
                pf = sample["protein"]["features"]
                rf = sample["rna"]["features"]
                sample_split = sample["split_info"].get("split")
                flag_pi = pf.get("pI") is not None
                flag_bf = pf.get("mean_bfactor") is not None
                flag_ss_status = rf.get("ss_status") or "pending"
            else:
                sample_split = None
                flag_pi = flag_bf = True
                flag_ss_status = "pending"

            # split filter
            if self.split is None:
                pass  # include everything
            elif self.split == "all":
                if sample_split is None:
                    continue
            else:
                if sample_split != self.split:
                    continue

            if self.has_pi is True and not flag_pi:
                continue
            if self.has_pi is False and flag_pi:
                continue
            if self.has_bfactor is True and not flag_bf:
                continue
            if self.has_bfactor is False and flag_bf:
                continue
            if self.has_ss is True and flag_ss_status != "done":
                continue
            if self.has_ss is False and flag_ss_status == "done":
                continue

            out.append(sid)

        return out

    # ---------- public API -----------------------------------------------

    def __len__(self) -> int:
        return len(self._ids)

    def __iter__(self) -> Iterator[dict]:
        for sid in self._ids:
            yield self._load_sample(sid)

    def sample_ids(self) -> list[str]:
        return list(self._ids)

    def get(self, sample_id: str) -> dict:
        if sample_id not in self._index_rows:
            raise KeyError(f"sample_id not in index.csv: {sample_id!r}")
        return self._load_sample(sample_id)

    def head(self, n: int = 3) -> list[dict]:
        """Convenience: load and return the first n samples as full dicts."""
        out = []
        for i, sid in enumerate(self._ids):
            if i >= n:
                break
            out.append(self._load_sample(sid))
        return out


# ---------- demo ------------------------------------------------------------


def _demo():
    import argparse

    parser = argparse.ArgumentParser(
        description="Demo: load strict-tier samples with pI and print a few."
    )
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--tier", default="strict")
    parser.add_argument("--split", default=None)
    args = parser.parse_args()

    print(f"Loading quality_tier={args.tier!r} split={args.split!r} has_pi=True …")
    loader = SampleLoader(
        processed_dir=args.processed_dir,
        quality_tier=args.tier,
        split=args.split,
        has_pi=True,
    )
    print(f"  → {len(loader)} samples matched")
    print()
    print("First 3 sample summaries:")
    for sample in loader.head(3):
        pf = sample["protein"]["features"]
        n_contact = len(sample["interaction"]["contact_pairs"])
        print(
            f"  {sample['sample_id']:20s} "
            f"prot_len={sample['protein']['length']:>5d} "
            f"rna_len={sample['rna']['length']:>5d} "
            f"contacts={n_contact:>4d} "
            f"pI={pf['pI']:.2f} "
            f"tier={sample['data_availability']['quality_tier']}"
        )


if __name__ == "__main__":
    _demo()
