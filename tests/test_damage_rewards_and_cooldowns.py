import unittest

from gym_pvz.envs.pvz_env_v2 import PVZEnv_V2
from pvz import Scene, config
from pvz.entities.zombies.zombie import Zombie
from pvz.entities.zombies.zombie_cone import Zombie_cone


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

        expected_partial_reward = zombie.SCORE * partial_damage / zombie.MAX_HP
        self.assertAlmostEqual(self.scene.score, expected_partial_reward)

        self.scene.projectiles.append(DamageProjectile(zombie, zombie.MAX_HP * 10))
        self.scene.step()

        self.assertAlmostEqual(self.scene.total_score, zombie.SCORE)

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

            cooldown_start = 3 * config.N_LANES * config.LANE_LENGTH + 1
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


if __name__ == "__main__":
    unittest.main()
