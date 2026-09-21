"""RNABindRPlus adapter — Category C, web submission only (paper Table 1).

RNABindRPlus is a sequence + homology predictor hosted by the authors; the
form takes a FASTA batch and **returns the predictions by email**, so there
is nothing to drive programmatically:

  * ``prepare_input`` writes the FASTA to submit (plus the index map that
    translates submitted positions back to the sample's own residue
    numbering, which matters because the FASTA drops gaps and
    non-standard residues);
  * ``run_tool`` only locates the downloaded predictions;
  * ``parse_output`` applies the index map and emits the ToolPrediction.

Drop the ``*finalpredictions*.txt`` files under
``data/external/rnabindrplus/`` (or ``tools.rnabindrplus.results_dir``).
"""
from __future__ import annotations

from pathlib import Path

from ..external import rnabindrplus_inputs as rbp_in
from ..external import rnabindrplus_parse
from ..base_adapter import BaseAdapter


class RNABindRPlusAdapter(BaseAdapter):
    tool_id = "rnabindrplus"
    category = "C"

    # ------------------------------------------------------------------ input

    def prepare_input(
        self,
        sample_json: dict,
        work_dir: Path,
        config: dict,
    ) -> dict:
        sample_id = sample_json["sample_id"]
        sequence = (sample_json.get("protein") or {}).get("sequence") or ""
        if not sequence:
            raise ValueError(f"sample {sample_id!r}: missing protein.sequence")

        clean_seq, kept_positions = rbp_in.clean_sequence(str(sequence))
        if not clean_seq:
            raise ValueError(
                f"sample {sample_id!r}: protein sequence is empty after cleaning"
            )

        fasta_path = Path(work_dir) / f"{sample_id}.fasta"
        rbp_in.write_fasta(fasta_path, [(sample_id, clean_seq)])
        return {"fasta": fasta_path, "sample_id": sample_id,
                "kept_positions": kept_positions}

    # -------------------------------------------------------------------- run

    def run_tool(
        self,
        input_paths: dict,
        work_dir: Path,
        config: dict,
    ) -> Path:
        tool_cfg = (config.get("tools") or {}).get("rnabindrplus") or {}
        results_dir = Path(
            tool_cfg.get("results_dir") or "data/external/rnabindrplus")

        if not results_dir.is_dir():
            raise FileNotFoundError(
                f"RNABindRPlus results arrive by email: submit "
                f"{input_paths['fasta']} through the form at "
                f"http://ailab-projects2.ist.psu.edu/RNABindRPlus/ and save the "
                f"returned predictions under {results_dir}/ (see the README, "
                f"'manual web submission')"
            )
        return results_dir

    # ------------------------------------------------------------------ parse

    def parse_output(
        self,
        output_dir: Path,
        sample_json: dict,
        config: dict,
    ) -> rnabindrplus_parse.ToolPrediction:
        tool_cfg = (config.get("tools") or {}).get("rnabindrplus") or {}
        sample_id = sample_json["sample_id"]

        index_map_path = tool_cfg.get("index_map")
        index_map = rnabindrplus_parse.load_index_map(
            Path(index_map_path) if index_map_path else None)
        combined = rnabindrplus_parse.load_all_results(
            Path(output_dir), str(tool_cfg.get("results_glob", "*finalpredictions*.txt")))

        rows = combined.get(sample_id)
        if rows is None:
            return self.fail(
                sample_id, f"no RNABindRPlus predictions for {sample_id}",
                raw_output_dir=str(output_dir),
            )

        kept_positions = rnabindrplus_parse.kept_positions_for(
            sample_id, index_map, None)
        if kept_positions is None:
            # No index map on disk: re-derive the submitted positions from the
            # sample's own sequence (non-gap residues are what was sent).
            seq = (sample_json.get("protein") or {}).get("sequence") or ""
            kept_positions = [
                i for i, c in enumerate(seq.upper(), start=1)
                if c in rnabindrplus_parse.STANDARD_AA
            ] or None

        return rnabindrplus_parse.build_prediction(
            sample_id, rows, kept_positions,
            threshold=float(tool_cfg.get("threshold", 0.5)),
            raw_output_dir=str(output_dir),
        )
