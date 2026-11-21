import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
from compare_models import (
    bootstrap_mean_ci,
    compare_paired_results,
    resolve_checkpoint,
    validate_model_spaces,
    write_csv,
    write_report,
)


class ModelComparisonTests(unittest.TestCase):
    def test_checkpoint_resolver_uses_separate_model_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ppo_path = root / "saved_models" / "best_model.zip"
            a2c_path = root / "saved_models" / "a2c" / "best_model.zip"
            ppo_path.parent.mkdir(parents=True)
            a2c_path.parent.mkdir(parents=True)
            ppo_path.touch()
            a2c_path.touch()

            self.assertEqual(
                resolve_checkpoint("MaskablePPO", "best", root), ppo_path
            )
            self.assertEqual(
                resolve_checkpoint("MaskedA2C", "best", root), a2c_path
            )

    def test_checkpoint_resolver_reports_missing_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                resolve_checkpoint("MaskedA2C", "final", temp_dir)

    def test_incompatible_observation_shape_is_rejected(self):
        model = SimpleNamespace(
            observation_space=gym.spaces.Box(
                low=0, high=1, shape=(140,), dtype=float
            ),
            action_space=gym.spaces.Discrete(181),
        )
        current_observation_space = gym.spaces.Box(
            low=0, high=1, shape=(234,), dtype=float
        )

        with self.assertRaisesRegex(ValueError, "Retrain the checkpoint"):
            validate_model_spaces(
                "MaskedA2C",
                model,
                current_observation_space,
                gym.spaces.Discrete(181),
            )

    def test_bootstrap_interval_is_reproducible(self):
        first = bootstrap_mean_ci(
            [1, 2, 3, 4], seed=7, n_resamples=500
        )
        second = bootstrap_mean_ci(
            [1, 2, 3, 4], seed=7, n_resamples=500
        )

        self.assertEqual(first, second)
        self.assertLessEqual(first[0], 2.5)
        self.assertGreaterEqual(first[1], 2.5)

    def test_paired_comparison_uses_matching_seeds(self):
        ppo_results = [
            {"seed": 1, "episode_return": 10, "survived": True},
            {"seed": 2, "episode_return": 5, "survived": False},
            {"seed": 3, "episode_return": 7, "survived": True},
        ]
        a2c_results = [
            {"seed": 3, "episode_return": 7, "survived": False},
            {"seed": 1, "episode_return": 15, "survived": True},
            {"seed": 2, "episode_return": 3, "survived": True},
        ]

        comparison = compare_paired_results(
            ppo_results, a2c_results, seed=10, n_resamples=500
        )

        self.assertEqual(
            comparison["reward_difference_a2c_minus_ppo"]["mean"], 1.0
        )
        self.assertEqual(comparison["a2c_reward_wins"], 1)
        self.assertEqual(comparison["ppo_reward_wins"], 1)
        self.assertEqual(comparison["reward_ties"], 1)
        self.assertEqual(
            comparison["survival_difference_a2c_minus_ppo"]["mean"], 0.0
        )

    def test_paired_comparison_rejects_different_seed_sets(self):
        with self.assertRaises(ValueError):
            compare_paired_results(
                [{"seed": 1, "episode_return": 0, "survived": False}],
                [{"seed": 2, "episode_return": 0, "survived": False}],
                seed=1,
                n_resamples=100,
            )

    def test_writers_create_csv_and_comparison_report(self):
        summary = {
            "model": "MaskedA2C",
            "mean_return": 2.0,
            "median_return": 2.0,
            "return_ci_95_low": 1.0,
            "return_ci_95_high": 3.0,
            "survival_rate": 0.5,
            "survival_ci_95_low": 0.0,
            "survival_ci_95_high": 1.0,
            "mean_decision_steps": 12.0,
            "invalid_actions": 0,
        }
        metadata = {
            "episodes": 2,
            "seed_start": 4,
            "seed_end": 5,
            "device": "cpu",
            "bootstrap_samples": 100,
            "checkpoints": {
                "MaskablePPO": "ppo.zip",
                "MaskedA2C": "a2c.zip",
            },
        }
        paired = {
            "reward_difference_a2c_minus_ppo": {
                "mean": 1.0,
                "ci_95_low": 0.0,
                "ci_95_high": 2.0,
            },
            "survival_difference_a2c_minus_ppo": {
                "mean": 0.5,
                "ci_95_low": 0.0,
                "ci_95_high": 1.0,
            },
            "a2c_reward_wins": 1,
            "ppo_reward_wins": 0,
            "reward_ties": 1,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            write_csv(output_dir / "summary.csv", [summary])
            write_report(output_dir, metadata, [summary], paired)

            self.assertIn("mean_return", (output_dir / "summary.csv").read_text())
            report = (output_dir / "report.md").read_text()
            self.assertIn("Paired comparison", report)
            self.assertIn("Survival means", report)


if __name__ == "__main__":
    unittest.main()
