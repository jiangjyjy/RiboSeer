"""BindUP adapter — Category C, web submission only (paper Table 1).

BindUP predicts RNA-binding residues without homology, from sequence alone.
The authors' server exposes two forms — a batch mode that takes a list of
PDB IDs and a single mode that takes an uploaded structure — and returns
per-chain "patch" lists. Nothing here can be automated end to end:

  * ``prepare_input`` writes the PDB-ID list to submit;
  * ``run_tool`` only locates the downloaded patch files;
  * ``parse_output`` maps each patch's author-numbered residues onto the
    dataset's 1-based polymer index and emits the ToolPrediction.

Drop the downloaded ``*patch_list*.txt`` files under
``data/external/bindup/`` (or ``tools.bindup.results_dir``).
"""
from __future__ import annotations

from pathlib import Path

from ..external import bindup_parse
from ..tool_io import find_raw_structure
from ..base_adapter import BaseAdapter


class BindUPAdapter(BaseAdapter):
    tool_id = "bindup"
    category = "C"

    # ------------------------------------------------------------------ input

    def prepare_input(
        self,
        sample_json: dict,
        work_dir: Path,
        config: dict,
    ) -> dict:
        sample_id = sample_json["sample_id"]
        source_pdb = sample_json["source_pdb"]
        chain_id = (sample_json.get("protein") or {}).get("chain_id")
        if not chain_id:
            raise ValueError(f"sample {sample_id!r}: missing protein.chain_id")

        ids_path = Path(work_dir) / "bindup_pdb_ids.txt"
        if not ids_path.is_file():
            ids_path.write_text(f"{source_pdb}\n", encoding="utf-8")
        else:  # batch runs append, so one submission covers the whole split
            existing = ids_path.read_text(encoding="utf-8").split()
            if source_pdb not in existing:
                with ids_path.open("a", encoding="utf-8") as f:
                    f.write(f"{source_pdb}\n")
        return {"ids_file": ids_path, "sample_id": sample_id,
                "source_pdb": source_pdb, "chain_id": chain_id}

    # -------------------------------------------------------------------- run

    def run_tool(
        self,
        input_paths: dict,
        work_dir: Path,
        config: dict,
    ) -> Path:
        tool_cfg = (config.get("tools") or {}).get("bindup") or {}
        results_dir = Path(tool_cfg.get("results_dir") or "data/external/bindup")

        if not bindup_parse.build_bindup_index(results_dir):
            raise FileNotFoundError(
                f"BindUP runs on its web server: submit the PDB-ID list at "
                f"{input_paths['ids_file']} through "
                f"https://bindup.technion.ac.il/ (batch form) and save the "
                f"returned patch files under {results_dir}/ (see the README, "
                f"'manual web submission')"
            )
        return results_dir

    # ------------------------------------------------------------------ parse

    def parse_output(
        self,
        output_dir: Path,
        sample_json: dict,
        config: dict,
    ) -> bindup_parse.ToolPrediction:
        tool_cfg = (config.get("tools") or {}).get("bindup") or {}
        sample_id = sample_json["sample_id"]
        source_pdb = sample_json["source_pdb"]
        chain_id = (sample_json.get("protein") or {}).get("chain_id")

        index = bindup_parse.build_bindup_index(Path(output_dir))
        patches = bindup_parse.lookup_patches(index, source_pdb, chain_id)
        if patches is None:
            return self.fail(
                sample_id,
                f"no BindUP result for ({source_pdb}, chain {chain_id})",
                raw_output_dir=str(output_dir),
            )

        ss_cfg = config.get("structure_source") or {}
        raw_dir = Path(ss_cfg.get("raw_dir") or "data/raw")
        raw_path = find_raw_structure(raw_dir, source_pdb)
        if raw_path is None:
            return self.fail(
                sample_id,
                f"no raw structure for {source_pdb!r} under {raw_dir} "
                f"(needed to map BindUP's author numbering)",
                raw_output_dir=str(output_dir),
            )

        amap = bindup_parse.build_auth_to_label(raw_path, chain_id)
        patch_label_lists: list[list[int]] = []
        n_tok = n_unmapped = 0
        for patch in patches:
            labels: list[int] = []
            for num, icode in patch:
                n_tok += 1
                label = bindup_parse.map_token(amap, num, icode)
                if label is None:
                    n_unmapped += 1
                    continue
                labels.append(label)
            patch_label_lists.append(labels)
        if n_tok and n_unmapped == n_tok:
            return self.fail(
                sample_id,
                f"none of {n_tok} BindUP residues mapped to the polymer index",
                raw_output_dir=str(output_dir),
            )

        patch_scores = tool_cfg.get("patch_scores") or [1.0, 0.67, 0.33]
        return bindup_parse.build_prediction(
            sample_id, patch_label_lists,
            patch_scores=[float(s) for s in patch_scores],
            raw_output_dir=str(output_dir),
        )
