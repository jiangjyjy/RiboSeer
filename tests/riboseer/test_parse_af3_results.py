#!/usr/bin/env python3
"""Tests for af3_parse.py against a synthetic AF3-Server job folder."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

gemmi = pytest.importorskip("gemmi")

from step4_tool_adapters.external import af3_parse as paf  # noqa: E402
from step4_tool_adapters.schemas import (  # noqa: E402
    ToolPrediction,
    ToolPredictionSet,
)


def _write_mini_cif(path: Path) -> None:
    """Small protein(3 res) + RNA(2 res) complex; pLDDT in B-factor column.

    Both chains need >=2 residues so gemmi's ``setup_entities`` /
    ``assign_label_seq_id`` classify them as polymers and assign the
    ``label_seq`` indices that ``extract_contacts`` relies on. Residue 2's CA
    sits within 4.5 A of the RNA, giving exactly one binding protein residue.
    """
    st = gemmi.Structure()
    model = gemmi.Model("1")
    ch_a = gemmi.Chain("A")
    for i, (resn, x, b) in enumerate(
        [("ALA", 0.0, 90.0), ("GLY", 3.0, 80.0), ("SER", 20.0, 70.0)], start=1
    ):
        res = gemmi.Residue()
        res.name = resn
        res.seqid = gemmi.SeqId(i, " ")
        at = gemmi.Atom()
        at.name = "CA"
        at.element = gemmi.Element("C")
        at.pos = gemmi.Position(x, 0.0, 0.0)
        at.b_iso = b
        at.occ = 1.0  # gemmi defaults occ to 0.0, which makes B-factor round-trip flaky
        res.add_atom(at)
        ch_a.add_residue(res)
    ch_b = gemmi.Chain("B")
    for j, x in enumerate([3.5, 23.5], start=1):  # nt 1 contacts GLY (res 2)
        res = gemmi.Residue()
        res.name = "A"  # adenine -> RNA
        res.seqid = gemmi.SeqId(j, " ")
        for an, el in (("P", "P"), ("O5'", "O"), ("C1'", "C")):
            at = gemmi.Atom()
            at.name = an
            at.element = gemmi.Element(el)
            at.pos = gemmi.Position(x, 0.0, 0.0)
            at.b_iso = 70.0
            at.occ = 1.0
            res.add_atom(at)
        ch_b.add_residue(res)
    model.add_chain(ch_a)
    model.add_chain(ch_b)
    st.add_model(model)
    st.setup_entities()
    st.make_mmcif_document().write_file(str(path))


def _make_job_folder(root: Path, folder_name: str, sample_id: str) -> Path:
    """Create an AF3-Server-style job folder; files use ``sample_id`` casing."""
    job = root / folder_name
    job.mkdir(parents=True)
    _write_mini_cif(job / f"fold_{sample_id}_model_0.cif")
    summary = {
        "chain_iptm": [0.44, 0.44],
        "chain_pair_iptm": [[0.79, 0.44], [0.44, 0.25]],
        "iptm": 0.44,
        "ptm": 0.73,
        "ranking_score": 0.5,
        "has_clash": 0.0,
    }
    (job / f"fold_{sample_id}_summary_confidences_0.json").write_text(
        json.dumps(summary)
    )
    return job


def test_discover_sample_ids_skips_non_job_entries(tmp_path: Path) -> None:
    af3 = tmp_path / "af3"
    _make_job_folder(af3, "8z4l_c_m", "8z4l_c_m")
    _make_job_folder(af3, "4wkr_b_d", "4wkr_b_d")
    (af3 / "terms_of_use.md").write_text("terms")  # a file, not a job folder
    assert paf._discover_sample_ids(af3) == ["4wkr_b_d", "8z4l_c_m"]


def test_resolve_sample_dir_case_insensitive(tmp_path: Path) -> None:
    af3 = tmp_path / "af3"
    _make_job_folder(af3, "8z4l_c_m", "8z4l_c_m")  # lower-cased folder
    resolved = paf._resolve_sample_dir(af3, "8z4l_C_M")  # canonical casing
    # Windows is case-insensitive, so the reported name may keep the requested
    # casing; only the (case-folded) identity matters.
    assert resolved is not None and resolved.name.lower() == "8z4l_c_m"
    assert paf._resolve_sample_dir(af3, "nope_a_b") is None


def test_read_summary(tmp_path: Path) -> None:
    job = _make_job_folder(tmp_path, "8z4l_c_m", "8z4l_c_m")
    summary = paf._read_summary(job, "8z4l_c_m")
    assert summary["iptm"] == 0.44 and summary["ptm"] == 0.73


def test_build_prediction(tmp_path: Path) -> None:
    job = _make_job_folder(tmp_path, "8z4l_c_m", "8z4l_c_m")
    cif = job / "fold_8z4l_c_m_model_0.cif"
    pred = paf.build_prediction(
        cif, {"sample_id": "8z4l_c_m"}, contact_cutoff=4.5, distance_scale=8.0
    )
    assert isinstance(pred, ToolPrediction)
    assert pred.tool_id == "alphafold3" and pred.category == "A"
    assert pred.sample_id == "8z4l_c_m" and pred.success
    assert pred.binding_protein_residues  # contacting residues populated
    assert pred.binding_rna_nucleotides   # RNA side populated too
    assert pred.per_residue_pae_score  # CA->RNA distance binding probabilities
    assert pred.per_residue_confidence == {1: 90.0, 2: 80.0, 3: 70.0}  # pLDDT
    assert pred.plddt_mean == 80.0
    assert pred.iptm_score == 0.44
    assert "fold_8z4l_c_m_model_0.cif" in pred.predicted_structure_path


def test_build_prediction_remaps_with_index_map(tmp_path: Path) -> None:
    job = _make_job_folder(tmp_path, "8z4l_c_m", "8z4l_c_m")
    cif = job / "fold_8z4l_c_m_model_0.cif"
    # cleaned residue 2 -> original residue 42
    sample = {"sample_id": "8z4l_c_m", "protein_index_map": {1: 10, 2: 42, 3: 99}}
    pred = paf.build_prediction(cif, sample, contact_cutoff=4.5, distance_scale=8.0)
    assert pred.per_residue_confidence == {10: 90.0, 42: 80.0, 99: 70.0}
    # binding residues are remapped through the same map (all >= 10 now)
    assert pred.binding_protein_residues
    assert all(r in (10, 42, 99) for r in pred.binding_protein_residues)


def test_main_writes_per_sample_set(tmp_path: Path) -> None:
    af3 = tmp_path / "af3"
    _make_job_folder(af3, "8z4l_c_m", "8z4l_c_m")
    _make_job_folder(af3, "4wkr_b_d", "4wkr_b_d")
    step4 = tmp_path / "step4"
    processed = tmp_path / "processed"
    processed.mkdir()
    sample_list = tmp_path / "test.txt"
    sample_list.write_text("# samples\n8z4l_C_M\n4wkr_b_d\n")  # mixed case + comment

    rc = paf.main(
        [
            "--af3-dir", str(af3),
            "--step4-dir", str(step4),
            "--processed-dir", str(processed),
            "--sample-list", str(sample_list),
            "--log-level", "WARNING",
        ]
    )
    assert rc == 0

    out = step4 / "8z4l_C_M.jsonl"  # canonical casing preserved in filename
    assert out.exists()
    pset = ToolPredictionSet.model_validate_json(out.read_text().strip())
    assert pset.sample_id == "8z4l_C_M"
    assert pset.tools_run == ["alphafold3"]
    assert len(pset.predictions) == 1
    assert pset.predictions[0].tool_id == "alphafold3"
    assert pset.predictions[0].sample_id == "8z4l_C_M"
    assert (step4 / "4wkr_b_d.jsonl").exists()


def test_main_merges_without_clobbering_other_tools(tmp_path: Path) -> None:
    af3 = tmp_path / "af3"
    _make_job_folder(af3, "8z4l_c_m", "8z4l_c_m")
    step4 = tmp_path / "step4"
    step4.mkdir()
    processed = tmp_path / "processed"
    processed.mkdir()
    # Pre-existing set: a boltz2 prediction + a stale (failed) alphafold3 one.
    pre = ToolPredictionSet(
        sample_id="8z4l_C_M",
        tools_run=["boltz2", "alphafold3"],
        predictions=[
            ToolPrediction(
                tool_id="boltz2", category="A", sample_id="8z4l_C_M",
                success=True, binding_protein_residues=[5],
            ),
            ToolPrediction(
                tool_id="alphafold3", category="A", sample_id="8z4l_C_M",
                success=False, error_message="old run",
            ),
        ],
    )
    (step4 / "8z4l_C_M.jsonl").write_text(pre.model_dump_json() + "\n")
    sample_list = tmp_path / "test.txt"
    sample_list.write_text("8z4l_C_M\n")

    rc = paf.main(
        [
            "--af3-dir", str(af3),
            "--step4-dir", str(step4),
            "--processed-dir", str(processed),
            "--sample-list", str(sample_list),
            "--log-level", "WARNING",
        ]
    )
    assert rc == 0

    pset = ToolPredictionSet.model_validate_json(
        (step4 / "8z4l_C_M.jsonl").read_text().strip()
    )
    by_tool = {p.tool_id: p for p in pset.predictions}
    assert set(by_tool) == {"boltz2", "alphafold3"}  # boltz2 preserved
    assert len(pset.predictions) == 2  # stale alphafold3 replaced, not duplicated
    assert by_tool["alphafold3"].success  # the fresh, successful entry
    assert by_tool["boltz2"].binding_protein_residues == [5]  # untouched


def test_main_reports_missing_without_crashing(tmp_path: Path) -> None:
    af3 = tmp_path / "af3"
    _make_job_folder(af3, "8z4l_c_m", "8z4l_c_m")
    step4 = tmp_path / "step4"
    processed = tmp_path / "processed"
    processed.mkdir()
    sample_list = tmp_path / "test.txt"
    sample_list.write_text("8z4l_C_M\nghost_x_y\n")  # second has no folder

    rc = paf.main(
        [
            "--af3-dir", str(af3),
            "--step4-dir", str(step4),
            "--processed-dir", str(processed),
            "--sample-list", str(sample_list),
            "--log-level", "ERROR",
        ]
    )
    assert rc == 0
    assert (step4 / "8z4l_C_M.jsonl").exists()
    assert not (step4 / "ghost_x_y.jsonl").exists()
