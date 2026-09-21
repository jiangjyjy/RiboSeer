"""AlphaFold 3 adapter — Category A, web submission only (paper Table 1).

AlphaFold 3 is distributed as the AlphaFold Server, which takes jobs
through a web form (30 jobs/day) and returns a result folder per job. There
is no local API to drive, so this adapter does not execute anything:

  * ``prepare_input`` writes the ready-to-upload job payload
    (``<work_dir>/<sample_id>.json``, alphafoldserver dialect) and a
    manifest of the whole submission when several samples share a run;
  * ``run_tool`` only locates the downloaded results — it raises a message
    telling you where to put them;
  * ``parse_output`` reads ``fold_<sample_id>_model_0.cif`` +
    ``fold_<sample_id>_summary_confidences_0.json`` exactly like the local
    Cat-A tools (same helpers as Boltz-2/Chai-1).

Drop downloaded jobs under ``data/external/af3/<sample_id>/`` (or whatever
``tools.alphafold3.results_dir`` points at). The README documents the
submission flow.
"""
from __future__ import annotations

from pathlib import Path

from ..external import af3_inputs, af3_parse
from ..base_adapter import BaseAdapter


def _sequences_from_sample(sample_json: dict) -> tuple[str, str]:
    """Protein and RNA sequence from a step-1 sample JSON."""
    protein = (sample_json.get("protein") or {}).get("sequence") or ""
    rna = (sample_json.get("rna") or {}).get("sequence") or ""
    return str(protein), str(rna)


class AlphaFold3Adapter(BaseAdapter):
    tool_id = "alphafold3"
    category = "A"

    # ------------------------------------------------------------------ input

    def prepare_input(
        self,
        sample_json: dict,
        work_dir: Path,
        config: dict,
    ) -> dict:
        sample_id = sample_json["sample_id"]
        protein_seq, rna_seq = _sequences_from_sample(sample_json)
        if not protein_seq or not rna_seq:
            raise ValueError(
                f"sample {sample_id!r} needs both protein and RNA sequences "
                f"for an AlphaFold Server job"
            )

        row = af3_inputs.clean_row(sample_id, protein_seq, rna_seq)
        job_path = Path(work_dir) / f"{sample_id}.json"
        af3_inputs.write_job_jsons(Path(work_dir), [row])
        if not job_path.is_file():
            raise RuntimeError(
                f"AlphaFold Server payload was not written for {sample_id}"
            )
        return {"job_json": job_path, "sample_id": sample_id}

    # -------------------------------------------------------------------- run

    def run_tool(
        self,
        input_paths: dict,
        work_dir: Path,
        config: dict,
    ) -> Path:
        tool_cfg = (config.get("tools") or {}).get("alphafold3") or {}
        results_dir = Path(tool_cfg.get("results_dir") or "data/external/af3")
        sample_id = input_paths["sample_id"]

        if af3_parse._resolve_sample_dir(results_dir, sample_id) is None:
            raise FileNotFoundError(
                f"AlphaFold 3 runs on its web server: submit "
                f"{input_paths['job_json']} at https://alphafoldserver.com and "
                f"unpack the downloaded job folder under {results_dir}/"
                f"{sample_id}/ (see the README, 'manual web submission')"
            )
        return results_dir

    # ------------------------------------------------------------------ parse

    def parse_output(
        self,
        output_dir: Path,
        sample_json: dict,
        config: dict,
    ) -> af3_parse.ToolPrediction:
        tool_cfg = (config.get("tools") or {}).get("alphafold3") or {}
        sample_id = sample_json["sample_id"]

        sample_dir = af3_parse._resolve_sample_dir(Path(output_dir), sample_id)
        if sample_dir is None:
            return self.fail(
                sample_id, f"no AlphaFold 3 job folder for {sample_id}",
                raw_output_dir=str(output_dir),
            )
        cif = af3_parse._find_structure(sample_dir, sample_id)
        if cif is None:
            return self.fail(
                sample_id, f"no predicted model CIF under {sample_dir}",
                raw_output_dir=str(sample_dir),
            )
        return af3_parse.build_prediction(
            cif, sample_json,
            contact_cutoff=float(config.get("contact_threshold", 4.5)),
            distance_scale=float(tool_cfg.get("distance_scale", 8.0)),
        )
