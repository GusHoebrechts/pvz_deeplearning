import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gym_pvz.envs.pvz_env_v2 import PVZEnv_V2
from pvz import Scene, config
from pvz.entities.plants.potatomine import Potatomine
from pvz.entities.zombies.zombie import Zombie
from pvz.entities.zombies.zombie_cone import Zombie_cone
from train import (
    EntropyCoefficientScheduleCallback,
    RenderTrainingMilestonesCallback,
    TrainingControlCallback,
    entropy_coefficient,
    _training_game_label,
)
from watch import resolve_model_path


class NoopSpawner:
    def spawn(self, scene):
        pass


class DamageProjectile:
    def __init__(self, zombie, damage):
        self.zombie = zombie
        self.damage = damage
        self.hp = 1

    def step(self, scene):
        scene.damage_zombie(self.zombie, self.damage)
        self.hp = 0

    def __bool__(self):
        return self.hp > 0


class DamageRewardTests(unittest.TestCase):
    def setUp(self):
        self.scene = Scene({}, NoopSpawner())

    def test_partial_and_overkill_damage_award_at_most_configured_score(self):
        zombie = Zombie(lane=1, pos=5)
        self.scene.add_zombie(zombie)
        partial_damage = 50
        self.scene.projectiles.append(DamageProjectile(zombie, partial_damage))

        self.scene.step()

        expected_partial_reward = round(
            zombie.SCORE * partial_damage / zombie.MAX_HP
        )
        self.assertEqual(self.scene.score, expected_partial_reward)
        self.assertEqual(int(self.scene.score), self.scene.score)

        self.scene.projectiles.append(DamageProjectile(zombie, zombie.MAX_HP * 10))
        self.scene.step()

        self.assertEqual(self.scene.total_score, zombie.SCORE)

    def test_many_small_hits_remain_integer_and_sum_to_configured_score(self):
        zombie = Zombie(lane=1, pos=5)

        for _ in range(zombie.MAX_HP):
            self.scene.damage_zombie(zombie, 1)
            self.assertIsInstance(self.scene._damage_reward, int)

        self.assertEqual(self.scene._damage_reward, zombie.SCORE)

    def test_reaching_house_does_not_award_damage_score(self):
        zombie = Zombie(lane=0, pos=0)
        zombie._offset = 0
        self.scene.add_zombie(zombie)

        self.scene.step()

        self.assertEqual(self.scene.score, -config.DEATH_PENALTY)

    def test_armor_conversion_does_not_award_a_kill_score(self):
        zombie = Zombie_cone(lane=0, pos=5)
        zombie.hp = 189
        self.scene.add_zombie(zombie)

        self.scene.step()

        self.assertEqual(self.scene.score, 0)
        self.assertEqual(len(self.scene.zombies), 1)
        self.assertIsInstance(self.scene.zombies[0], Zombie)
        self.assertEqual(self.scene.zombies[0]._reward_max_hp, 189)

    def test_reaching_frame_limit_awards_survival_once(self):
        self.scene._chrono = config.MAX_FRAMES

        self.scene.step()
        self.scene.step()

        self.assertEqual(self.scene.score, 0)
        self.assertEqual(self.scene.total_score, config.SURVIVAL)

    def test_dying_on_frame_limit_does_not_award_survival(self):
        zombie = Zombie(lane=0, pos=0)
        zombie._offset = 0
        self.scene.add_zombie(zombie)
        self.scene._chrono = config.MAX_FRAMES

        self.scene.step()

        self.assertEqual(self.scene.score, -config.DEATH_PENALTY)


class CooldownObservationTests(unittest.TestCase):
    def test_observation_exposes_continuous_cooldown_fraction(self):
        env = PVZEnv_V2()
        try:
            plant_name = "sunflower"
            max_cooldown = (
                env.plant_deck[plant_name].COOLDOWN * config.FPS - 1
            )
            remaining = max_cooldown // 2
            env._scene.plant_cooldowns[plant_name] = remaining
            observation = env._get_obs()

            cooldown_start = 5 * config.N_LANES * config.LANE_LENGTH + 1
            cooldown_index = cooldown_start + env._plant_names.index(plant_name)

            self.assertEqual(observation.shape, env.observation_space.shape)
            self.assertEqual(len(observation), env.num_observations())
            self.assertAlmostEqual(
                observation[cooldown_index], remaining / max_cooldown
            )
            self.assertGreater(observation[cooldown_index], 0)
            self.assertLess(observation[cooldown_index], 1)
        finally:
            env.close()

    def test_observation_exposes_potato_mine_arming_progress(self):
        env = PVZEnv_V2()
        try:
            mine = Potatomine(lane=1, pos=3)
            env._scene.plants.append(mine)
            cell_index = mine.lane * config.LANE_LENGTH + mine.pos
            arming_grid_start = 4 * config.N_LANES * config.LANE_LENGTH
            max_arming_cooldown = (
                Potatomine.ATTACK_COOLDOWN * config.FPS - 1
            )

            observation = env._get_obs()
            arming_index = arming_grid_start + cell_index
            self.assertEqual(observation[arming_index], 0)

            mine.attack_cooldown = max_arming_cooldown // 2
            observation = env._get_obs()
            self.assertAlmostEqual(
                observation[arming_index],
                1 - mine.attack_cooldown / max_arming_cooldown,
            )

            mine.attack_cooldown = 0
            observation = env._get_obs()
            self.assertEqual(observation[arming_index], 1)
            self.assertEqual(observation.shape, env.observation_space.shape)
            self.assertEqual(len(observation), env.num_observations())
            self.assertTrue(env.observation_space.contains(observation))
        finally:
            env.close()


class TrainingControlTests(unittest.TestCase):
    def test_stop_requires_q_then_y(self):
        callback = TrainingControlCallback()
        callback._interactive = True
        keys = iter(["q", "y"])
        callback._read_key = lambda wait=False: next(keys, None)

        self.assertFalse(callback._on_step())
        self.assertTrue(callback.stop_requested)

    def test_ctrl_q_does_not_stop_training(self):
        callback = TrainingControlCallback()
        callback._interactive = True
        keys = iter(["\x11", None])
        callback._read_key = lambda wait=False: next(keys, None)

        self.assertTrue(callback._on_step())
        self.assertFalse(callback.stop_requested)


class TrainingScheduleTests(unittest.TestCase):
    def test_entropy_coefficient_decays_linearly_and_clamps(self):
        self.assertAlmostEqual(entropy_coefficient(1.0), 0.05)
        self.assertAlmostEqual(entropy_coefficient(0.5), 0.03)
        self.assertAlmostEqual(entropy_coefficient(0.0), 0.01)
        self.assertAlmostEqual(entropy_coefficient(-1.0), 0.01)

    def test_entropy_callback_updates_from_training_progress(self):
        from types import SimpleNamespace

        callback = EntropyCoefficientScheduleCallback(total_timesteps=1_000_000)
        callback.model = SimpleNamespace(num_timesteps=500_000, ent_coef=0.05)
        callback.num_timesteps = callback.model.num_timesteps

        self.assertTrue(callback._on_step())

        self.assertAlmostEqual(callback.model.ent_coef, 0.03)

    def test_render_callback_keeps_best_episode_in_each_window(self):
        callback = RenderTrainingMilestonesCallback(interval=10_000)
        lower_reward_frames = [{"frame": 1}]
        higher_reward_frames = [{"frame": 2}]
        callback._record_episode(10, 5, lower_reward_frames)
        callback._record_episode(11, 12, higher_reward_frames)
        callback._finish_window(10_000)

        best_game = callback._milestone_games[-1]
        self.assertEqual(best_game["episode"], 11)
        self.assertEqual(best_game["reward"], 12)
        self.assertEqual(best_game["window_start"], 0)
        self.assertEqual(best_game["window_end"], 10_000)
        self.assertEqual(best_game["frames"], higher_reward_frames)

    def test_renderer_supports_legacy_sample_labels(self):
        label = _training_game_label(
            {"milestone": 10_000, "episode": 42, "frames": []}
        )
        current_label = _training_game_label(
            {
                "window_start": 10_000,
                "window_end": 20_000,
                "episode": 43,
                "reward": 25,
                "frames": [],
            }
        )

        self.assertEqual(label, "Checkpoint 10,000 | Training game #42")
        self.assertEqual(
            current_label,
            "Checkpoint 20,000 | Window 10,000-20,000 "
            "| Training game #43 | Reward 25",
        )

    def test_renderer_failure_does_not_stop_training_callback(self):
        class FailedRenderProcess:
            exitcode = 1

            def is_alive(self):
                return False

            def join(self):
                pass

        callback = RenderTrainingMilestonesCallback()
        callback._render_process = FailedRenderProcess()

        callback._collect_finished_render()

        self.assertIsNone(callback._render_process)


class WatchModelSelectionTests(unittest.TestCase):
    def test_best_checkpoint_is_preferred_and_final_can_be_selected(self):
        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            best_model = model_dir / "best_model.zip"
            final_model = model_dir / "maskable_ppo_pvz_final.zip"
            best_model.touch()
            final_model.touch()

            self.assertEqual(resolve_model_path(model_dir), best_model)
            self.assertEqual(
                resolve_model_path(model_dir, "final"), final_model
            )

    def test_missing_best_checkpoint_falls_back_to_final(self):
        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            final_model = model_dir / "maskable_ppo_pvz_final.zip"
            final_model.touch()

            self.assertEqual(resolve_model_path(model_dir), final_model)


if __name__ == "__main__":
    unittest.main()
