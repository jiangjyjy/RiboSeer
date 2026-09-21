"""HDOCK adapter — Category D rigid-body docking.

HDOCK (HDOCKlite) takes a receptor (protein) and a ligand (RNA) PDB and
produces a sampled set of docked complexes ranked by its shape +
statistical score::

    hdock receptor.pdb ligand.pdb          # -> Hdock.out
    createpl Hdock.out model.pdb -nmax 10 -complex -models   # -> model_1.pdb ..

Two HDOCKlite quirks shape this adapter: both binaries always write into
the current working directory (they ignore any output-path argument), and
the Fortran runtime is happier with short paths — so every sample gets its
own directory and is run from inside it. Both binaries come from the
HDOCKlite distribution and need its bundled FFTW on ``LD_LIBRARY_PATH``.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ..external import hdock_parse
from ..tool_io import extract_protein_chain_pdb, find_raw_structure
from ..base_adapter import BaseAdapter
from .structure_utils import extract_rna_chain_pdb


class HdockAdapter(BaseAdapter):
    tool_id = "hdock"
    category = "D"

    # ------------------------------------------------------------------ input

    def prepare_input(
        self,
        sample_json: dict,
        work_dir: Path,
        config: dict,
    ) -> dict:
        sample_id = sample_json["sample_id"]
        source_pdb = sample_json["source_pdb"]
        protein_chain = (sample_json.get("protein") or {}).get("chain_id")
        rna_chain = (sample_json.get("rna") or {}).get("chain_id")
        if not protein_chain:
            raise ValueError(f"sample {sample_id!r}: missing protein.chain_id")
        if not rna_chain:
            raise ValueError(f"sample {sample_id!r}: missing rna.chain_id")

        ss_cfg = config.get("structure_source") or {}
        raw_dir = Path(ss_cfg.get("raw_dir") or "data/raw")
        raw_path = find_raw_structure(raw_dir, source_pdb)
        if raw_path is None:
            raise FileNotFoundError(
                f"no raw structure for {source_pdb!r} under {raw_dir}"
            )

        run_dir = Path(work_dir) / sample_id
        run_dir.mkdir(parents=True, exist_ok=True)
        receptor = run_dir / "receptor.pdb"
        ligand = run_dir / "ligand.pdb"
        extract_protein_chain_pdb(raw_path, protein_chain, receptor)
        extract_rna_chain_pdb(raw_path, rna_chain, ligand)
        return {"receptor": receptor, "ligand": ligand,
                "run_dir": run_dir, "sample_id": sample_id}

    # -------------------------------------------------------------------- run

    def run_tool(
        self,
        input_paths: dict,
        work_dir: Path,
        config: dict,
    ) -> Path:
        tool_cfg = (config.get("tools") or {}).get("hdock") or {}
        install_dir = tool_cfg.get("install_dir")
        if not install_dir:
            raise ValueError("config['tools']['hdock']['install_dir'] not set")
        timeout = int(tool_cfg.get("timeout", 600))
        nmodels = int(tool_cfg.get("nmodels", 10))

        run_dir = Path(input_paths["run_dir"]).resolve()
        receptor = Path(input_paths["receptor"]).resolve()
        ligand = Path(input_paths["ligand"]).resolve()

        env = os.environ.copy()
        # The bundled FFTW ships next to the binaries; a caller-provided
        # conda prefix takes priority when the tool was installed there.
        lib_dirs = [str(Path(install_dir))]
        conda_prefix = tool_cfg.get("conda_prefix") or os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            lib_dirs.insert(0, str(Path(conda_prefix) / "lib"))
        prev = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            lib_dirs + ([prev] if prev else []))

        def _run(cmd: list[str], step: str, budget: int) -> None:
            proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
                cmd, cwd=str(run_dir), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=budget if budget > 0 else None,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"HDOCK {step} exited {proc.returncode}: "
                    f"{(proc.stdout or '')[-800:]}"
                )

        hdock_bin = str(Path(install_dir) / "hdock")
        createpl_bin = str(Path(install_dir) / "createpl")
        _run([hdock_bin, receptor.name, ligand.name], "docking", timeout)
        _run([createpl_bin, "Hdock.out", "model.pdb",
              "-nmax", str(nmodels), "-complex", "-models"],
             "createpl", timeout)
        return run_dir

    # ------------------------------------------------------------------ parse

    def parse_output(
        self,
        output_dir: Path,
        sample_json: dict,
        config: dict,
    ) -> hdock_parse.ToolPrediction:
        tool_cfg = (config.get("tools") or {}).get("hdock") or {}
        sample_id = sample_json["sample_id"]

        model = hdock_parse.find_best_model(
            Path(output_dir), str(tool_cfg.get("model_name", "model_1.pdb")))
        if model is None:
            return self.fail(
                sample_id,
                f"no docked model under {output_dir}",
                raw_output_dir=str(output_dir),
            )

        return hdock_parse.build_prediction(
            model, sample_id,
            contact_cutoff=float(config.get("contact_threshold", 4.5)),
            distance_scale=float(tool_cfg.get("distance_scale", 8.0)),
        )
