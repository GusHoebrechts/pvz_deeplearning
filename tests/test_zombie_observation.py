import unittest

from gym_pvz.envs.pvz_env_v2 import PVZEnv_V2
from pvz import config
from pvz.entities.zombies.zombie import Zombie
from pvz.entities.zombies.zombie_cone import Zombie_cone


class ZombieObservationTests(unittest.TestCase):
    def test_observation_exposes_hp_total_and_count_per_tile(self):
        env = PVZEnv_V2()
        try:
            lane = 2
            pos = 4
            basic_zombie = Zombie(lane=lane, pos=pos)
            cone_zombie = Zombie_cone(lane=lane, pos=pos)
            env._scene.add_zombie(basic_zombie)
            env._scene.add_zombie(cone_zombie)

            observation = env._get_obs()
            grid_size = config.N_LANES * config.LANE_LENGTH
            cell_index = lane * config.LANE_LENGTH + pos
            zombie_hp_index = 2 * grid_size + cell_index
            zombie_count_index = 3 * grid_size + cell_index

            self.assertEqual(
                observation[zombie_hp_index],
                basic_zombie.hp + cone_zombie.hp,
            )
            self.assertEqual(observation[zombie_count_index], 2)
            self.assertEqual(
                observation[3 * grid_size + cell_index + 1], 0
            )
            self.assertEqual(observation.shape, env.observation_space.shape)
            self.assertEqual(len(observation), env.num_observations())
            self.assertTrue(env.observation_space.contains(observation))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
