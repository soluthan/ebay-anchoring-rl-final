import ast
import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]

MODULES = [
    "project_constants.py",
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
    def test_required_modules_parse(self):
        for module in MODULES:
            path = ROOT / module
            with self.subTest(module=module):
                ast.parse(path.read_text(encoding="utf-8"), filename=module)

    def test_run_pipeline_uses_flat_imports(self):
        source = (ROOT / "run_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("from data_preprocess import main as prep", source)
        self.assertIn("from phase1_supervised import main as phase1", source)
        self.assertIn("from phase2_cql import main as phase2", source)
        self.assertIn("from phase3_ppo import main as phase3", source)
        self.assertIn("from results import main as results", source)
        self.assertNotIn("from models.", source)
        self.assertNotIn("from data.", source)

    def test_shared_constants_exist(self):
        source = (ROOT / "project_constants.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        names = {node.targets[0].id for node in tree.body if isinstance(node, ast.Assign)}
        for name in ["SEED", "STATE_COLS", "ACTION_COL", "ANCHOR_MIN", "ANCHOR_MAX"]:
            with self.subTest(name=name):
                self.assertIn(name, names)

    def test_results_prefers_faithful_ppo_outputs(self):
        source = (ROOT / "results.py").read_text(encoding="utf-8")
        self.assertIn("ppo_metrics_faithful.json", source)
        self.assertIn("ppo_history_faithful.json", source)
        self.assertIn("artifact(", source)
        self.assertIn("Evidence Type", source)
        self.assertIn("simulator-only estimate", source)

    def test_config_is_valid_json(self):
        config_path = ROOT / "configs" / "default_experiment.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["experiment_name"], "ebay_best_offer_anchoring_rl")
        self.assertIn("run_order", config)
        self.assertIn("data_preprocess", config)
        self.assertIn("phase2_cql", config)
        self.assertIn("phase3_ppo", config)
        self.assertIn("ope", config)
        phase1_outputs = config["phase1_supervised"]["outputs"]
        self.assertIn("models/deal_classifier.ubj", phase1_outputs)
        self.assertNotIn("models/price_regressor.ubj", phase1_outputs)

    def test_price_regressor_removed_from_reward_path(self):
        for module in ["phase1_supervised.py", "phase2_cql.py", "phase3_ppo.py"]:
            source = (ROOT / module).read_text(encoding="utf-8")
            with self.subTest(module=module):
                self.assertNotIn("price_regressor", source)
                self.assertNotIn("XGBRegressor", source)
        phase1 = (ROOT / "phase1_supervised.py").read_text(encoding="utf-8")
        self.assertIn("1 - anchor_ratio", phase1)
        self.assertIn("1.0 - anchors", phase1)

    def test_preprocess_has_robust_split_modes_and_train_only_fit(self):
        source = (ROOT / "data_preprocess.py").read_text(encoding="utf-8")
        self.assertIn("SPLIT_MODE", source)
        self.assertIn("PREPROCESS_MAX_ROWS", source)
        self.assertIn("split_temporal", source)
        self.assertIn("split_leaf_holdout", source)
        self.assertIn("fit_preprocess_stats(train", source)
        self.assertIn("apply_preprocess_stats(val_raw", source)
        self.assertIn("leaf_holdout_overlap.json", source)

    def test_smoke_and_ppo_sensitivity_knobs_exist(self):
        phase1 = (ROOT / "phase1_supervised.py").read_text(encoding="utf-8")
        phase3 = (ROOT / "phase3_ppo.py").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("PHASE1_MAX_ROWS", phase1)
        self.assertIn("PPO_FAITHFUL_P5", phase3)
        self.assertIn("faithful_support_threshold", phase3)
        self.assertIn("PPO_FAITHFUL_P5", readme)

    def test_ope_reports_support_diagnostics(self):
        source = (ROOT / "ope.py").read_text(encoding="utf-8")
        for needle in [
            "snips",
            "effective_sample_size",
            "bootstrap_ci",
            "cql_offline_rl",
            "OPE_INCLUDE_CQL",
            "OPE_BANDWIDTHS",
            "ope_weight_diagnostics.csv",
        ]:
            with self.subTest(needle=needle):
                self.assertIn(needle, source)

    def test_data_schema_doc_exists(self):
        doc = ROOT / "docs" / "data_schema.md"
        text = doc.read_text(encoding="utf-8")
        self.assertIn("anchor_ratio", text)
        self.assertIn("savings_pct", text)
        self.assertIn("contextual bandit", text)
        self.assertIn("SPLIT_MODE=leaf_holdout", text)
        self.assertIn("Off-Policy Evaluation", text)

    def test_gitignore_keeps_private_artifacts_out(self):
        ignore_text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        required_patterns = [
            "*.parquet",
            "*.csv",
            "*.pt",
            "*.pkl",
            "*.ubj",
            "data/",
            "data_*/",
            "models/",
            "models_*/",
            "outputs/",
            "outputs_*/",
            "__pycache__/",
            ".venv/",
        ]
        for pattern in required_patterns:
            with self.subTest(pattern=pattern):
                self.assertIn(pattern, ignore_text)

    def test_report_allows_final_deliverables(self):
        ignore_text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("report/*", ignore_text)
        self.assertIn("!report/.gitkeep", ignore_text)
        self.assertIn("!report/technical_report.md", ignore_text)
        self.assertIn("!report/technical_report.pdf", ignore_text)
        self.assertIn("!report/technical_report_results_dashboard.png", ignore_text)
        self.assertNotIn("!report/*.pdf", ignore_text)
        self.assertNotIn("!report/*.md", ignore_text)
        self.assertNotIn("!report/*.png", ignore_text)


if __name__ == "__main__":
    unittest.main()
