from .zombie_spawner import ZombieSpawner
from .zombie import Zombie
from .zombie_cone import Zombie_cone
from .zombie_bucket import Zombie_bucket
from .zombie_flag import Zombie_flag
from .basic_zombie_spawner import BasicZombieSpawner
from ... import config
import random
from math import floor

WAVE_ZOMBIE_TYPES = {
    "basic": Zombie,
    "cone": Zombie_cone,
    "bucket": Zombie_bucket,
}


class WaveZombieSpawner(ZombieSpawner):

    def __init__(self, seed=None):
        self._rng = random.Random(seed)
        self._basic_zombie_spawner = BasicZombieSpawner(self._rng)
        self._next_wave_index = 0
        self._planned_wave = self._plan_wave(self._next_wave_index)
        self._next_wave_index += 1
        self._active_wave_id = None
        self._active_wave_initial_hp = 0
        self._active_wave_rewarded = False
        self._wave_timer = 0
        self._minimum_wave_timer = 0
        self._first_wave_timer = config.FIRST_WAVE_DELAY_SECONDS * config.FPS

    def _plan_wave(self, wave_index):
        remaining_points = floor(
            config.POINTS_PER_WAVE
            + config.POINTS_ADDED_PER_WAVE * wave_index
        )
        zombie_points = config.WAVE_ZOMBIE_POINT_COSTS
        zombies = []

        while remaining_points:
            affordable_types = [
                zombie_name
                for zombie_name, point_cost in zombie_points.items()
                if point_cost <= remaining_points
            ]
            zombie_name = self._rng.choice(affordable_types)
            lane = self._rng.randrange(config.N_LANES)
            zombies.append((WAVE_ZOMBIE_TYPES[zombie_name], lane))
            remaining_points -= zombie_points[zombie_name]

        return wave_index, zombies

    def _spawn_planned_wave(self, scene):
        wave_id, planned_zombies = self._planned_wave
        flag_lane = self._rng.randrange(config.N_LANES)
        wave_zombies = [Zombie_flag(flag_lane)]
        wave_zombies.extend(
            zombie_type(lane) for zombie_type, lane in planned_zombies
        )

        for zombie in wave_zombies:
            zombie._wave_id = wave_id
            scene.add_zombie(zombie)

        self._active_wave_id = wave_id
        self._active_wave_initial_hp = sum(zombie.hp for zombie in wave_zombies)
        self._active_wave_rewarded = False
        self._wave_timer = config.WAVE_TIMEOUT_SECONDS * config.FPS
        self._minimum_wave_timer = (
            config.MIN_WAVE_INTERVAL_SECONDS * config.FPS
        )

        self._planned_wave = self._plan_wave(self._next_wave_index)
        self._next_wave_index += 1

    def spawn(self, scene):
        self._basic_zombie_spawner.spawn(scene)

        if self._active_wave_id is None:
            self._first_wave_timer -= 1
            if self._first_wave_timer <= 0:
                self._spawn_planned_wave(scene)
            return

        remaining_hp = sum(
            max(0, zombie.hp)
            for zombie in scene.zombies
            if getattr(zombie, "_wave_id", None) == self._active_wave_id
        )
        if remaining_hp == 0 and not self._active_wave_rewarded:
            scene.wave_survived()
            self._active_wave_rewarded = True

        self._wave_timer -= 1
        self._minimum_wave_timer -= 1
        if (
            self._minimum_wave_timer <= 0
            and (
                remaining_hp <= self._active_wave_initial_hp / 2
                or self._wave_timer <= 0
            )
        ):
            self._spawn_planned_wave(scene)
