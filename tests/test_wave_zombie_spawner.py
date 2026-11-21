import unittest

from pvz import config
from pvz.entities.zombies.zombie import Zombie
from pvz.entities.zombies.zombie_flag import Zombie_flag
from pvz.entities.zombies.wave_zombie_spawner import WaveZombieSpawner


class RecordingScene:
    def __init__(self):
        self.zombies = []
        self.survival_rewards = 0

    def add_zombie(self, zombie):
        self.zombies.append(zombie)

    def wave_survived(self):
        self.survival_rewards += 1


class WaveZombieSpawnerTests(unittest.TestCase):
    def test_regular_spawns_use_configured_rate_and_only_basic_zombies(self):
        spawner = WaveZombieSpawner(seed=1)
        scene = RecordingScene()

        for _ in range(config.BASIC_ZOMBIE_INITIAL_OFFSET * config.FPS):
            spawner.spawn(scene)

        self.assertEqual(len(scene.zombies), 1)
        self.assertIs(type(scene.zombies[0]), Zombie)

        for _ in range(config.BASIC_ZOMBIE_SPAWN_INTERVAL * config.FPS):
            spawner.spawn(scene)

        regular_zombies = [
            zombie for zombie in scene.zombies if type(zombie) is Zombie
        ]
        self.assertEqual(len(regular_zombies), 2)

    def test_wave_timers_keep_the_configured_intervals(self):
        spawner = WaveZombieSpawner(seed=1)
        scene = RecordingScene()

        for _ in range(config.FIRST_WAVE_DELAY_SECONDS * config.FPS):
            spawner.spawn(scene)

        wave_zombies = [
            zombie
            for zombie in scene.zombies
            if getattr(zombie, "_wave_id", None) == 0
        ]
        self.assertEqual(len(wave_zombies), 1)
        self.assertIsInstance(wave_zombies[0], Zombie_flag)
        self.assertEqual(
            spawner._minimum_wave_timer,
            config.MIN_WAVE_INTERVAL_SECONDS * config.FPS,
        )
        self.assertEqual(
            spawner._wave_timer,
            config.WAVE_TIMEOUT_SECONDS * config.FPS,
        )

    def test_wave_survival_reward_requires_all_wave_zombies_to_be_dead(self):
        spawner = WaveZombieSpawner(seed=1)
        scene = RecordingScene()
        zombie = Zombie(lane=0)
        zombie._wave_id = 0
        zombie.hp = 50
        scene.zombies.append(zombie)
        spawner._active_wave_id = 0
        spawner._active_wave_initial_hp = 100
        spawner._minimum_wave_timer = 1
        spawner._wave_timer = 100

        spawner.spawn(scene)

        self.assertEqual(scene.survival_rewards, 0)

    def test_wave_survival_reward_is_awarded_once_when_cleared(self):
        spawner = WaveZombieSpawner(seed=1)
        scene = RecordingScene()
        zombie = Zombie(lane=0)
        zombie._wave_id = 0
        zombie.hp = 0
        scene.zombies.append(zombie)
        spawner._active_wave_id = 0
        spawner._active_wave_initial_hp = 100
        spawner._minimum_wave_timer = 100
        spawner._wave_timer = 100

        spawner.spawn(scene)
        spawner.spawn(scene)

        self.assertEqual(scene.survival_rewards, 1)

    def test_wave_timeout_does_not_award_survival_with_zombies_remaining(self):
        spawner = WaveZombieSpawner(seed=1)
        scene = RecordingScene()
        zombie = Zombie(lane=0)
        zombie._wave_id = 0
        zombie.hp = 100
        scene.zombies.append(zombie)
        spawner._active_wave_id = 0
        spawner._active_wave_initial_hp = 100
        spawner._minimum_wave_timer = 1
        spawner._wave_timer = 1

        spawner.spawn(scene)

        self.assertEqual(scene.survival_rewards, 0)


if __name__ == "__main__":
    unittest.main()
