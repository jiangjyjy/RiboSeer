"""Mock tests for src/step5_fusion/features_15tool.py."""
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from step5_fusion import features_15tool as t9  # noqa: E402


def _pred(tool_id, *, success=True, pae=None, conf=None, binding=None):
    return {
        "tool_id": tool_id,
        "success": success,
        "per_residue_pae_score": pae or {},
        "per_residue_confidence": conf or {},
        "binding_protein_residues": binding or [],
    }


class TestScopeEncoding(unittest.TestCase):
    def test_dim_and_names(self):
        self.assertEqual(t9.SCOPE_FEATURE_DIM, 16)
        self.assertEqual(len(t9.SCOPE_FEATURE_NAMES), 16)

    def test_none_profile_is_zero_block(self):
        v = t9.encode_scope_profile(None, 300, 80)
        self.assertEqual(v.shape, (16,))
        self.assertTrue(np.all(v == 0.0))

    def test_one_hot_family_and_rna(self):
        v = t9.encode_scope_profile(
            {"protein_family": "KH", "rna_context": "stem-loop",
             "difficulty": "Hard", "confidence": 0.9}, 250, 50)
        kh_idx = t9.SCOPE_FAMILIES.index("KH")
        self.assertEqual(v[kh_idx], 1.0)
        self.assertEqual(v[:6].sum(), 1.0)  # exactly one family hot
        sl_idx = 6 + t9.SCOPE_RNA_CONTEXTS.index("stem-loop")
        self.assertEqual(v[sl_idx], 1.0)
        self.assertEqual(v[6:12].sum(), 1.0)
        # difficulty Hard=1.0, conf=0.9
        self.assertEqual(v[12], 1.0)
        self.assertAlmostEqual(v[15], 0.9)

    def test_len_bins_clamped(self):
        v = t9.encode_scope_profile({"protein_family": "Novel"}, 100000, 1)
        self.assertEqual(v[13], 1.0)  # protein bin clamps at 1
        self.assertAlmostEqual(v[14], 1.0 / t9.RNA_LEN_NORM)

    def test_unknown_labels_fall_back(self):
        v = t9.encode_scope_profile(
            {"protein_family": "ZZZ", "rna_context": "weird",
             "difficulty": "???"}, 200, 40)
        novel = t9.SCOPE_FAMILIES.index("Novel")
        unstr = 6 + t9.SCOPE_RNA_CONTEXTS.index("unstructured")
        self.assertEqual(v[novel], 1.0)
        self.assertEqual(v[unstr], 1.0)
        self.assertEqual(v[12], 0.5)  # unknown difficulty → Medium

    def test_auto_profile_difficulty_rule(self):
        easy = t9.auto_scope_profile(
            {"protein": {"length": 80}, "rna": {"length": 20}})
        self.assertEqual(easy["difficulty"], "easy")
        hard = t9.auto_scope_profile(
            {"protein": {"length": 400}, "rna": {"length": 120}})
        self.assertEqual(hard["difficulty"], "hard")
        self.assertEqual(hard["source"], "cauto")


class TestFeatureLayout(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(len(t9.ALL_KNOWN_TOOLS), 15)
        self.assertEqual(t9.N_BASE_T9, 15 * 6 + 3)        # 93
        self.assertEqual(t9.N_CROSS_T9, 30)
        self.assertEqual(t9.N_CONTEXT_T9, 15)

    def test_default_width_154(self):
        names = t9.table9_feature_names(use_context=True, use_scope=True)
        self.assertEqual(len(names), 93 + 30 + 15 + 16)
        self.assertEqual(len(names), 154)

    def test_full_matrix_shape(self):
        step4 = {"predictions": [
            _pred("boltz2", pae={1: 0.9, 2: 0.1}, conf={1: 80, 2: 70},
                  binding=[1])]}
        X = t9.build_15tool_features(step4, [1, 2])
        self.assertEqual(X.shape, (2, 154))


class TestActiveInactiveTools(unittest.TestCase):
    def setUp(self):
        self.step4 = {"predictions": [
            _pred("boltz2", pae={1: 0.8, 2: 0.2, 3: 0.5},
                  conf={1: 90, 2: 60, 3: 70}, binding=[1]),
            _pred("equipnas", conf={1: 0.7, 2: 0.3, 3: 0.6}, binding=[1, 3]),
        ]}

    def test_present_selected_tool_has_values(self):
        X = t9.build_15tool_features(self.step4, [1, 2, 3],
                                     selected_tools=["boltz2", "equipnas"])
        b_col = t9.ALL_KNOWN_TOOLS.index("boltz2") * 6
        # boltz2 main score for residue 1 = 0.8
        self.assertAlmostEqual(X[0, b_col], 0.8)
        self.assertFalse(np.isnan(X[0, b_col]))

    def test_unselected_tool_is_nan(self):
        X = t9.build_15tool_features(self.step4, [1, 2, 3],
                                     selected_tools=["boltz2"])
        e_col = t9.ALL_KNOWN_TOOLS.index("equipnas") * 6
        self.assertTrue(np.all(np.isnan(X[:, e_col:e_col + 6])))

    def test_absent_tool_is_nan(self):
        # chai1 not in step4 at all → NaN even if "selected".
        X = t9.build_15tool_features(self.step4, [1, 2, 3],
                                     selected_tools=["boltz2", "chai1"])
        c_col = t9.ALL_KNOWN_TOOLS.index("chai1") * 6
        self.assertTrue(np.all(np.isnan(X[:, c_col:c_col + 6])))

    def test_failed_tool_is_nan(self):
        step4 = {"predictions": [
            _pred("boltz2", pae={1: 0.8}, binding=[1]),
            _pred("chai1", success=False)]}
        X = t9.build_15tool_features(step4, [1],
                                     selected_tools=["boltz2", "chai1"])
        c_col = t9.ALL_KNOWN_TOOLS.index("chai1") * 6
        self.assertTrue(np.all(np.isnan(X[:, c_col:c_col + 6])))

    def test_summary_counts_active_only(self):
        X = t9.build_15tool_features(self.step4, [1, 2, 3],
                                     selected_tools=["boltz2"])
        # n_tools_active is the 3rd summary col; only boltz2 active → 1.0
        n_active_col = 15 * 6 + 2
        self.assertEqual(X[0, n_active_col], 1.0)


class TestCrossAndContext(unittest.TestCase):
    def test_cross_term_product_when_both_active(self):
        step4 = {"predictions": [
            _pred("boltz2", pae={1: 0.5}, binding=[1]),
            _pred("chai1", pae={1: 0.4}, binding=[1])]}
        X = t9.build_15tool_features(step4, [1],
                                     selected_tools=["boltz2", "chai1"])
        names = t9.table9_feature_names()
        ci = names.index("cross_boltz2_chai1")
        self.assertAlmostEqual(X[0, ci], 0.5 * 0.4)
        gi = names.index("gate_boltz2_chai1")
        self.assertAlmostEqual(X[0, gi], 1.0)  # both gate residue 1

    def test_cross_term_nan_when_one_inactive(self):
        step4 = {"predictions": [_pred("boltz2", pae={1: 0.5}, binding=[1])]}
        X = t9.build_15tool_features(step4, [1],
                                     selected_tools=["boltz2", "chai1"])
        names = t9.table9_feature_names()
        ci = names.index("cross_boltz2_chai1")
        self.assertTrue(np.isnan(X[0, ci]))

    def test_scope_block_broadcast(self):
        step4 = {"predictions": [_pred("boltz2", pae={1: 0.5, 2: 0.6},
                                       binding=[1])]}
        sv = t9.encode_scope_profile({"protein_family": "RRM"}, 200, 40)
        X = t9.build_15tool_features(step4, [1, 2], selected_tools=["boltz2"],
                                     scope_vector=sv)
        scope_block = X[:, -16:]
        # every residue row shares the same scope block
        self.assertTrue(np.allclose(scope_block[0], scope_block[1]))
        self.assertTrue(np.allclose(scope_block[0], sv))

    def test_scope_vector_wrong_dim_raises(self):
        step4 = {"predictions": [_pred("boltz2", pae={1: 0.5}, binding=[1])]}
        with self.assertRaises(ValueError):
            t9.build_15tool_features(step4, [1], selected_tools=["boltz2"],
                                     scope_vector=np.zeros(5))

    def test_no_scope_no_context_width(self):
        step4 = {"predictions": [_pred("boltz2", pae={1: 0.5}, binding=[1])]}
        X = t9.build_15tool_features(step4, [1], selected_tools=["boltz2"],
                                     use_context=False, use_scope=False)
        self.assertEqual(X.shape[1], 93 + 30)

    def test_alias_canonicalisation(self):
        # rf2na alias should map to rosettafold2na.
        step4 = {"predictions": [_pred("rf2na", pae={1: 0.7}, binding=[1])]}
        X = t9.build_15tool_features(step4, [1],
                                     selected_tools=["rosettafold2na"])
        rf_col = t9.ALL_KNOWN_TOOLS.index("rosettafold2na") * 6
        self.assertAlmostEqual(X[0, rf_col], 0.7)


if __name__ == "__main__":
    unittest.main()
