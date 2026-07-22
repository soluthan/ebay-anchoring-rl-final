import ast
import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULES = [
    "project_constants.py",
    "policy_utils.py",
    "data_preprocess.py",
    "phase1_supervised.py",
    "phase2_cql.py",
    "phase3_ppo.py",
    "ope.py",
    "recommend.py",
    "recommend_one.py",
    "results.py",
    "run_pipeline.py",
]


class StaticRepoTests(unittest.TestCase):
    def source(self, name):
        return (ROOT / name).read_text(encoding="utf-8")

    def test_modules_parse(self):
        for module in MODULES:
            with self.subTest(module=module):
                ast.parse(self.source(module), filename=module)

    def test_acceptance_semantics_are_explicit(self):
        namespace = {}
        exec(self.source("project_constants.py"), namespace)
        self.assertEqual(namespace["ACCEPTED_STATUSES"], (1, 9))
        self.assertEqual(namespace["FIRST_BUYER_OFFER_TYPE"], 0)
        self.assertEqual(namespace["ANCHOR_MAX"], 1.0)
        self.assertNotIn("DEAL_STATUS", namespace)

    def test_preprocess_extracts_first_event_and_immediate_label(self):
        source = self.source("data_preprocess.py")
        for needle in [
            "BUYER_COL",
            "offr_type_id",
            "extract_opening_offers",
            "group_by(group_cols",
            "FIRST_BUYER_OFFER_TYPE",
            ".is_in(ACCEPTED_STATUSES)",
            "thread_eventual_accepted",
            "opening_status_counts.csv",
            "acceptance_by_anchor_bin.csv",
            "status_price_diagnostics.csv",
        ]:
            self.assertIn(needle, source)

    def test_preprocess_uses_listing_group_splits(self):
        source = self.source("data_preprocess.py")
        self.assertIn("split_random", source)
        self.assertIn("split_temporal", source)
        self.assertIn("assert_disjoint_items", source)
        self.assertIn("item_split_overlap.json", source)
        self.assertNotIn("rng.permutation(n)\n", source)

    def test_phase1_has_calibration_and_real_greedy(self):
        source = self.source("phase1_supervised.py")
        self.assertIn("brier_score_loss", source)
        self.assertIn("fixed_anchor_metrics.json", source)
        self.assertIn("supervised_greedy_benchmark", source)
        self.assertIn("interior_optimum_fraction", source)
        self.assertIn("anchor_response_curve.csv", source)

    def test_one_step_cql_has_no_heuristic_ope(self):
        source = self.source("phase2_cql.py")
        self.assertIn("CQL_GAMMA = 0.0", source)
        self.assertIn("CQL_ALPHA", source)
        self.assertIn('"cql_alpha"', source)
        self.assertIn("One-step terminal target", source)
        self.assertNotIn("heuristic_ope_ips", source)

    def test_ppo_uses_bounded_beta_and_runs_both_variants(self):
        source = self.source("phase3_ppo.py")
        self.assertIn("from torch.distributions import Beta", source)
        self.assertIn("PPO_RUN_BOTH", source)
        self.assertIn("gamma=0.0", source)
        self.assertIn("ACTION_EPS", source)
        self.assertNotIn("from torch.distributions import Normal", source)

    def test_ope_has_real_on_policy_sanity_check(self):
        source = self.source("ope.py")
        self.assertIn("on_policy_sanity", source)
        self.assertIn("unit_on_policy", source)
        self.assertNotIn('"behavioral_logged":', source)

    def test_recommender_uses_training_preprocess_stats(self):
        source = self.source("recommend_one.py")
        self.assertIn("preprocess_stats.json", source)
        self.assertIn("--leaf_category", source)
        self.assertIn("PPO is omitted", source)
        self.assertNotIn("clean_master_dataset.parquet", source)

    def test_pipeline_order_and_flat_imports(self):
        source = self.source("run_pipeline.py")
        for needle in [
            "from data_preprocess import main as prep",
            "from phase1_supervised import main as phase1",
            "from phase2_cql import main as phase2",
            "from phase3_ppo import main as phase3",
            "from ope import main as ope",
            "from recommend import main as recommend",
            "from results import main as results",
        ]:
            self.assertIn(needle, source)
        self.assertLess(source.index('if "ope" in phases'), source.index('if "results" in phases'))

    def test_config_is_valid_and_lists_corrected_outputs(self):
        config = json.loads((ROOT / "configs/default_experiment.json").read_text())
        self.assertIn(
            "models/opening_acceptance_classifier.ubj",
            config["phase1_supervised"]["outputs"],
        )
        self.assertIn("models/fixed_anchor_metrics.json", config["phase1_supervised"]["outputs"])
        self.assertEqual(config["phase3_ppo"]["PPO_RUN_BOTH"], 1)
        self.assertIn("data/item_split_overlap.json", config["data_preprocess"]["outputs"])
        self.assertNotIn("models/price_regressor.ubj", str(config))

    def test_gitignore_keeps_private_artifacts_out(self):
        ignore = self.source(".gitignore")
        for pattern in ["*.parquet", "*.pt", "*.pkl", "*.ubj", "data/", "models/", "outputs/"]:
            self.assertIn(pattern, ignore)


if __name__ == "__main__":
    unittest.main()
