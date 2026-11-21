import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from watch_a2c import resolve_model_path


class A2CModelSelectionTests(unittest.TestCase):
    def test_best_checkpoint_is_preferred_and_final_can_be_selected(self):
        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            best_model = model_dir / "best_model.zip"
            final_model = model_dir / "a2c_pvz_final.zip"
            best_model.touch()
            final_model.touch()

            self.assertEqual(resolve_model_path(model_dir), best_model)
            self.assertEqual(
                resolve_model_path(model_dir, "final"), final_model
            )

    def test_missing_best_checkpoint_falls_back_to_final(self):
        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            final_model = model_dir / "a2c_pvz_final.zip"
            final_model.touch()

            self.assertEqual(resolve_model_path(model_dir), final_model)

    def test_missing_requested_model_raises_an_error(self):
        with TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                resolve_model_path(temp_dir, "final")


if __name__ == "__main__":
    unittest.main()
