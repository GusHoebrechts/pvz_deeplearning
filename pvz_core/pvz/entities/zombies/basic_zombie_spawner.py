from .zombie_spawner import ZombieSpawner
from .zombie import Zombie
from ... import config
import random


class BasicZombieSpawner(ZombieSpawner):

    def __init__(self, rng=None):
        self._rng = rng or random
        self._timer = config.BASIC_ZOMBIE_INITIAL_OFFSET * config.FPS - 1
        
    def spawn(self, scene):
        if self._timer <= 0:
            lane = self._rng.choice(range(config.N_LANES))
            scene.add_zombie(Zombie(lane))
            self._timer = config.BASIC_ZOMBIE_SPAWN_INTERVAL * config.FPS - 1
        else:
            self._timer -= 1