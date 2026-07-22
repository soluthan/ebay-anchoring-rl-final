"""Runtime semantic tests; skipped automatically when ML dependencies are absent."""

import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HAS_RUNTIME = all(importlib.util.find_spec(name) for name in ("numpy", "polars"))


@unittest.skipUnless(HAS_RUNTIME, "numpy/polars not installed in lightweight CI")
class PreprocessRuntimeTests(unittest.TestCase):
    def test_opening_event_label_and_reward(self):
        import polars as pl
        from data_preprocess import build_features
        from project_constants import BUYER_COL, ITEM_COL, LABEL_COL, REWARD_COL

        rows = [
            # Countered opening, later accepted: immediate label must remain zero.
            (1, 11, 101, "01jan2013 10:00:00", 0, 7, 60.0),
            (1, 11, 101, "02jan2013 10:00:00", 2, 1, 80.0),
            (2, 22, 202, "01jan2013 10:00:00", 0, 1, 75.0),
            (3, 33, 303, "01jan2013 10:00:00", 0, 9, 70.0),
            (4, 44, 404, "01jan2013 10:00:00", 0, 2, 50.0),
            # First event is a seller counter, so this is not a valid PB1 thread.
            (5, 55, 505, "01jan2013 10:00:00", 2, 7, 80.0),
            (5, 55, 505, "02jan2013 10:00:00", 0, 1, 90.0),
        ]
        frame = pl.DataFrame(
            rows,
            schema=[
                ITEM_COL,
                BUYER_COL,
                "anon_thread_id",
                "src_cre_date",
                "offr_type_id",
                "status_id",
                "offr_price",
            ],
            orient="row",
        ).with_columns(
            pl.lit(100.0).alias("start_price_usd"),
            pl.when(pl.col("status_id").is_in([1, 9]))
            .then(pl.col("offr_price"))
            .otherwise(100.0)
            .alias("item_price"),
            pl.lit(11450).alias("meta_categ_id"),
            pl.lit(9001).alias("anon_leaf_categ_id"),
            pl.lit(1000).alias("fdbk_score_src"),
            pl.lit(99.0).alias("fdbk_pstv_src"),
        )
        result = build_features(frame.lazy()).collect().sort(ITEM_COL)
        self.assertEqual(result[ITEM_COL].to_list(), [1, 2, 3, 4])
        self.assertEqual(result[LABEL_COL].to_list(), [0, 1, 1, 0])
        self.assertAlmostEqual(result[REWARD_COL][0], 0.0)
        self.assertAlmostEqual(result[REWARD_COL][1], 0.25)
        self.assertTrue(result.filter(pl.col(ITEM_COL) == 1)["thread_eventual_accepted"][0])
        accepted_prices = result.filter(pl.col(LABEL_COL) == 1)
        self.assertTrue(
            (accepted_prices["item_price"] == accepted_prices["offr_price"]).all()
        )

    def test_listing_split_has_zero_overlap(self):
        import polars as pl
        from data_preprocess import assert_disjoint_items, split_random
        from project_constants import ITEM_COL

        frame = pl.DataFrame(
            {
                ITEM_COL: [item for item in range(20) for _ in range(2)],
                "x": list(range(40)),
            }
        )
        train, val, test = split_random(frame)
        assert_disjoint_items(train, val, test)
        self.assertEqual(len(train) + len(val) + len(test), len(frame))


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy not installed")
class PolicyRuntimeTests(unittest.TestCase):
    def test_greedy_policy_finds_interior_optimum(self):
        import numpy as np
        from policy_utils import greedy_policy

        class FakeClassifier:
            def predict_proba(self, features):
                action = features[:, -1]
                p = np.clip(action ** 2, 0.0, 1.0)
                return np.column_stack([1.0 - p, p])

        anchors, _, _, indices = greedy_policy(FakeClassifier(), np.zeros((4, 4), np.float32))
        self.assertTrue(np.all((anchors > 0.5) & (anchors < 0.8)))
        self.assertTrue(np.all(indices > 0))

    def test_greedy_grid_contains_fixed_baseline_anchor(self):
        """A tree-model jump at 0.70 must not make greedy lose to fixed 0.70."""
        import numpy as np
        from policy_utils import ACTION_GRID, greedy_policy, score_actions

        self.assertTrue(np.isclose(ACTION_GRID, 0.70).any())

        class ThresholdClassifier:
            def predict_proba(self, features):
                action = features[:, -1]
                p = np.where(np.isclose(action, 0.70), 0.9, 0.1)
                return np.column_stack([1.0 - p, p])

        states = np.zeros((3, 4), np.float32)
        _, _, greedy_values, _ = greedy_policy(ThresholdClassifier(), states)
        _, fixed_values = score_actions(
            ThresholdClassifier(), states, np.full(len(states), 0.70, np.float32)
        )
        self.assertTrue(np.all(greedy_values >= fixed_values - 1e-8))


HAS_PPO = all(importlib.util.find_spec(name) for name in ("numpy", "torch", "gymnasium"))


@unittest.skipUnless(HAS_PPO, "torch/gymnasium not installed in lightweight CI")
class PPORuntimeTests(unittest.TestCase):
    def test_beta_policy_samples_strictly_bounded_unit_actions(self):
        import torch
        from phase3_ppo import ActorCritic

        network = ActorCritic(4, hidden=16)
        distribution, values = network.get_dist(torch.zeros(128, 4))
        actions = distribution.sample()
        self.assertTrue(torch.all(actions > 0.0))
        self.assertTrue(torch.all(actions < 1.0))
        self.assertTrue(torch.isfinite(distribution.log_prob(actions)).all())
        self.assertEqual(tuple(values.shape), (128,))


if __name__ == "__main__":
    unittest.main()
