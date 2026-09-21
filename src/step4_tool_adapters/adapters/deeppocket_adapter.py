"""DeepPocket adapter — Category B pocket detection (paper Table 1).

DeepPocket runs fpocket to enumerate candidate pockets on the protein
surface, ranks them with a classification CNN and segments the top ones.
Like P2Rank it consumes a single-chain protein PDB and emits per-residue
pocket scores, so the parsing side reuses the same Cat-B recipe (pockets →
per-residue max → threshold) via
:mod:`step4_tool_adapters.external.deeppocket_parse`.

The tool is driven exactly as verified on the server::

    CUDA_VISIBLE_DEVICES=<dev> python predict.py \\
        -p <protein.pdb> \\
        -c first_model_fold1_best_test_auc_85001.pth.tar \\
        -s seg0_best_test_IOU_91.pth.tar \\
        -r 3

``predict.py`` writes its results next to the input PDB (``<stem>_nowat_out/``),
which is why the adapter stages the protein inside ``work_dir``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ..external import deeppocket_parse
from ..tool_io import extract_protein_chain_pdb, find_raw_structure
from ..base_adapter import BaseAdapter


class DeepPocketAdapter(BaseAdapter):
    tool_id = "deeppocket"
    category = "B"

    # ------------------------------------------------------------------ input

    def prepare_input(
        self,
        sample_json: dict,
        work_dir: Path,
        config: dict,
    ) -> dict:
        sample_id = sample_json["sample_id"]
        source_pdb = sample_json["source_pdb"]
        chain_id = sample_json["protein"]["chain_id"]

        ss_cfg = config.get("structure_source") or {}
        raw_dir = Path(ss_cfg.get("raw_dir") or "data/raw")
        raw_path = find_raw_structure(raw_dir, source_pdb)
        if raw_path is None:
            raise FileNotFoundError(
                f"no raw structure for {source_pdb!r} under {raw_dir}"
            )

        out_pdb = work_dir / f"{sample_id}.pdb"
        extract_protein_chain_pdb(raw_path, chain_id, out_pdb)
        return {"protein_pdb": out_pdb, "sample_id": sample_id}

    # -------------------------------------------------------------------- run

    def run_tool(
        self,
        input_paths: dict,
        work_dir: Path,
        config: dict,
    ) -> Path:
        tool_cfg = (config.get("tools") or {}).get("deeppocket") or {}
        install_dir = tool_cfg.get("install_dir")
        if not install_dir:
            raise ValueError("config['tools']['deeppocket']['install_dir'] not set")
        timeout = int(tool_cfg.get("timeout", 300))

        pdb_path = Path(input_paths["protein_pdb"]).resolve()
        if not pdb_path.is_file():
            raise FileNotFoundError(f"prepared PDB missing: {pdb_path}")

        cmd = [
            sys.executable, "predict.py",
            "-p", str(pdb_path),
            "-c", str(tool_cfg.get(
                "class_checkpoint", "first_model_fold1_best_test_auc_85001.pth.tar")),
            "-s", str(tool_cfg.get(
                "seg_checkpoint", "seg0_best_test_IOU_91.pth.tar")),
            "-r", str(int(tool_cfg.get("rank", 3))),
        ]

        env = os.environ.copy()
        if tool_cfg.get("device") is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(tool_cfg["device"])

        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            cmd, cwd=str(install_dir), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=timeout if timeout > 0 else None,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"DeepPocket exited {proc.returncode}: "
                f"{(proc.stdout or '')[-800:]}"
            )
        return work_dir

    # ------------------------------------------------------------------ parse

    def parse_output(
        self,
        output_dir: Path,
        sample_json: dict,
        config: dict,
    ) -> deeppocket_parse.ToolPrediction:
        tool_cfg = (config.get("tools") or {}).get("deeppocket") or {}
        sample_id = sample_json["sample_id"]

        pockets_dir = deeppocket_parse.find_pockets_dir(Path(output_dir) / sample_id)
        if pockets_dir is None:
            return self.fail(
                sample_id,
                f"no DeepPocket pockets dir under {output_dir}/{sample_id}",
                raw_output_dir=str(output_dir),
            )

        protein = sample_json.get("protein") or {}
        length = protein.get("length") or len(protein.get("sequence") or "")
        return deeppocket_parse.build_prediction(
            sample_id,
            pockets_dir,
            threshold=float(tool_cfg.get("residue_score_threshold", 0.5)),
            max_residue=int(length) if length else None,
        )
