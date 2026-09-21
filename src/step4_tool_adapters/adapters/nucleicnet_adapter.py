"""NucleicNet adapter — Category C RNA-binding residue prediction.

NucleicNet voxelises a halo shell around the protein surface and classifies
every grid point into ``{Base / Phosphate / Ribose / Nonsite}`` (the "SXPR"
head). Only SXPR is run: the finer AUCG base-identity head is not needed for
a per-residue binding score and roughly halves the GPU time.

Unlike the other tools, NucleicNet is driven **in process** — its published
code is a Python library, not a CLI — through the verified sequence::

    Server = Server(SaveCleansed=True, SaveDf=True, Select_HeavyAtoms=True,
                    DIR_ServerFolder=<out>)
    ServerC.SimpleSanitise(DIR_InputPdbFile=<single-chain protein pdb>)
    ServerC.MakeHalo(); ServerC.MakeDssp(); ServerC.MakeFeature()
    ServerC.MakeDummyTypi(); ServerC.MakeSXPR()
    # -> <out>/sxpr/Result_EnsembleAvDf.pkl

NucleicNet resolves its weights and helper binaries through paths relative to
its ``Notebooks/`` directory, so the library insists on being imported from
there. That makes this adapter change the process working directory for the
duration of ``run_tool`` (restored afterwards), which in turn means it is not
thread-safe: run RELAY with ``execution.parallel_workers: 1`` when NucleicNet
is part of the plan.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from ..external import nucleicnet_parse
from ..tool_io import extract_protein_chain_pdb, find_raw_structure
from ..base_adapter import BaseAdapter


class NucleicNetAdapter(BaseAdapter):
    tool_id = "nucleicnet"
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

        sample_dir = Path(work_dir) / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        out_pdb = sample_dir / f"{sample_id}.pdb"
        extract_protein_chain_pdb(raw_path, chain_id, out_pdb)
        return {"protein_pdb": out_pdb, "sample_dir": sample_dir,
                "sample_id": sample_id}

    # -------------------------------------------------------------------- run

    def _import_server(self, nucleicnet_dir: Path):
        """Make the local NucleicNet checkout importable and return its
        ``Server`` class (plus the chdir target the caller must enter)."""
        nucleicnet_dir = Path(nucleicnet_dir).expanduser().resolve()
        notebooks_dir = nucleicnet_dir / "Notebooks"
        if not notebooks_dir.is_dir():
            raise FileNotFoundError(
                f"expected Notebooks/ under the NucleicNet checkout: {notebooks_dir}"
            )
        if str(nucleicnet_dir) not in sys.path:
            sys.path.insert(0, str(nucleicnet_dir))
        from NucleicNet.DatasetBuilding.commandServer import Server  # noqa: PLC0415
        return Server, notebooks_dir

    def run_tool(
        self,
        input_paths: dict,
        work_dir: Path,
        config: dict,
    ) -> Path:
        tool_cfg = (config.get("tools") or {}).get("nucleicnet") or {}
        install_dir = tool_cfg.get("install_dir")
        if not install_dir:
            raise ValueError("config['tools']['nucleicnet']['install_dir'] not set")

        sample_id = input_paths["sample_id"]
        pdb_path = Path(input_paths["protein_pdb"]).resolve()
        sample_dir = Path(input_paths["sample_dir"]).resolve()
        server_out = sample_dir / "nucleicnet_outputs" / sample_id
        server_out.mkdir(parents=True, exist_ok=True)

        Server, notebooks_dir = self._import_server(Path(install_dir))

        # The library's relative paths are anchored on Notebooks/: enter it for
        # the flow and restore the previous cwd no matter how it exits.
        previous_cwd = Path.cwd()
        try:
            os.chdir(notebooks_dir)
            server = Server(
                SaveCleansed=True, SaveDf=True, Select_HeavyAtoms=True,
                DIR_ServerFolder=str(server_out),
            )
            server.SimpleSanitise(DIR_InputPdbFile=str(pdb_path))
            server.MakeHalo()
            server.MakeDssp()
            server.MakeFeature()
            server.MakeDummyTypi()
            server.MakeSXPR()
        finally:
            os.chdir(previous_cwd)
        return sample_dir

    # ------------------------------------------------------------------ parse

    def parse_output(
        self,
        output_dir: Path,
        sample_json: dict,
        config: dict,
    ) -> nucleicnet_parse.ToolPrediction:
        tool_cfg = (config.get("tools") or {}).get("nucleicnet") or {}
        sample_id = sample_json["sample_id"]
        sample_dir = Path(output_dir) / sample_id

        try:
            return nucleicnet_parse.parse_one_sample(
                sample_id,
                output_dir=sample_dir,
                inputs_dir=sample_dir,
                prediction_type=str(tool_cfg.get("prediction_type", "Smoothened")),
                nonsite_name=str(tool_cfg.get("nonsite_name", "Nonsite")),
                nonsite_index_override=tool_cfg.get("nonsite_index"),
                radius=float(tool_cfg.get("radius", 6.0)),
                agg=str(tool_cfg.get("agg", "max")),
                threshold=float(tool_cfg.get("threshold", 0.5)),
            )
        except FileNotFoundError as exc:
            return self.fail(
                sample_id, str(exc),
                raw_output_dir=str(nucleicnet_parse.sxpr_dir(sample_dir, sample_id)),
            )
