"""GraphBind adapter — Category C RNA-binding residue prediction.

GraphBind is a hierarchical graph network over the protein surface. The
local CLI is used here::

    cd <install_dir>/scripts
    python prediction.py --querypath <dir> --filename <stem> \\
        --chainid A --ligands RNA --cpu 4

which writes ``RNA-binding_result.csv`` into the query directory. The
authors also expose the same model through the BindWeb web server; that
route is described in the README (results downloaded by hand and dropped
under ``data/external/graphbind/``) and is not driven from here.
"""
from __future__ import annotations

import shlex
from pathlib import Path

from ..external import graphbind_parse
from ..tool_io import extract_protein_chain_pdb, find_raw_structure
from ..tool_runner import run_in_conda_env
from ..base_adapter import BaseAdapter


class GraphBindAdapter(BaseAdapter):
    tool_id = "graphbind"
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
        chain_id = sample_json["protein"]["chain_id"]

        ss_cfg = config.get("structure_source") or {}
        raw_dir = Path(ss_cfg.get("raw_dir") or "data/raw")
        raw_path = find_raw_structure(raw_dir, source_pdb)
        if raw_path is None:
            raise FileNotFoundError(
                f"no raw structure for {source_pdb!r} under {raw_dir}"
            )

        query_dir = work_dir / sample_id
        query_dir.mkdir(parents=True, exist_ok=True)
        out_pdb = query_dir / f"{sample_id}.pdb"
        extract_protein_chain_pdb(raw_path, chain_id, out_pdb)
        return {"protein_pdb": out_pdb, "query_dir": query_dir,
                "sample_id": sample_id}

    # -------------------------------------------------------------------- run

    def run_tool(
        self,
        input_paths: dict,
        work_dir: Path,
        config: dict,
    ) -> Path:
        tool_cfg = (config.get("tools") or {}).get("graphbind") or {}
        install_dir = tool_cfg.get("install_dir")
        if not install_dir:
            raise ValueError("config['tools']['graphbind']['install_dir'] not set")
        env_name = tool_cfg.get("conda_env", "GraphBind")
        timeout = int(tool_cfg.get("timeout", 600))

        query_dir = Path(input_paths["query_dir"]).resolve()
        sample_id = input_paths["sample_id"]

        cmd = " ".join([
            "python", "prediction.py",
            "--querypath", shlex.quote(str(query_dir)),
            "--filename", shlex.quote(f"{sample_id}.pdb"),
            "--chainid", shlex.quote(str(tool_cfg.get("chainid", "A"))),
            "--ligands", shlex.quote(str(tool_cfg.get("ligands", "RNA"))),
            "--cpu", str(int(tool_cfg.get("cpu", 4))),
        ])
        scripts_dir = Path(install_dir) / "scripts"

        log_dir_cfg = config.get("log_dir")
        run_in_conda_env(
            env_name, cmd,
            cwd=scripts_dir if scripts_dir.is_dir() else Path(install_dir),
            timeout=timeout,
            log_dir=Path(log_dir_cfg) if log_dir_cfg else None,
            log_tag=f"graphbind_{sample_id}",
        )
        return work_dir

    # ------------------------------------------------------------------ parse

    def parse_output(
        self,
        output_dir: Path,
        sample_json: dict,
        config: dict,
    ) -> graphbind_parse.ToolPrediction:
        tool_cfg = (config.get("tools") or {}).get("graphbind") or {}
        sample_id = sample_json["sample_id"]
        query_dir = Path(output_dir) / sample_id

        csv_path = graphbind_parse.find_result_csv(query_dir)
        if csv_path is None:
            return self.fail(
                sample_id,
                f"no GraphBind result CSV under {query_dir}",
                raw_output_dir=str(query_dir),
            )

        per_res, binding = graphbind_parse.parse_graphbind_csv(
            csv_path,
            prob_threshold=tool_cfg.get("prob_threshold"),
        )
        if not per_res:
            return self.fail(
                sample_id,
                f"GraphBind result CSV parsed to 0 rows: {csv_path}",
                raw_output_dir=str(csv_path.parent),
            )
        return graphbind_parse.build_prediction(
            sample_id, per_res, binding,
            raw_output_dir=str(csv_path.parent),
        )
