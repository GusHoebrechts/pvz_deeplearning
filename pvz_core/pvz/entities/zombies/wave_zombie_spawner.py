from .zombie_spawner import ZombieSpawner
from .zombie import Zombie
from .zombie_cone import Zombie_cone
from .zombie_bucket import Zombie_bucket
from .zombie_flag import Zombie_flag
from ... import config
import random

WAVE_TIMEOUT_SECONDS = 75
FIRST_WAVE_DELAY_SECONDS = 15
MIN_WAVE_INTERVAL_SECONDS = 20
POINTS_PER_WAVE = 0
POINTS_ADDED_PER_WAVE = 2
ZOMBIE_POINTS = {
    Zombie: 1,
    Zombie_cone: 3,
    Zombie_bucket: 5,
}


class WaveZombieSpawner(ZombieSpawner):

    def __init__(self, seed=None):
        self._rng = random.Random(seed)
        self._next_wave_index = 0
        self._planned_wave = self._plan_wave(self._next_wave_index)
        self._next_wave_index += 1
        self._active_wave_id = None
        self._active_wave_initial_hp = 0
        self._wave_timer = 0
        self._minimum_wave_timer = 0
        self._first_wave_timer = FIRST_WAVE_DELAY_SECONDS * config.FPS

    def _plan_wave(self, wave_index):
        remaining_points = POINTS_PER_WAVE + POINTS_ADDED_PER_WAVE * wave_index
        zombies = []

        while remaining_points:
            affordable_types = [
                zombie_type
                for zombie_type, point_cost in ZOMBIE_POINTS.items()
                if point_cost <= remaining_points
            ]
            zombie_type = self._rng.choice(affordable_types)
            lane = self._rng.randrange(config.N_LANES)
            zombies.append((zombie_type, lane))
            remaining_points -= ZOMBIE_POINTS[zombie_type]

        return wave_index, zombies

    def _spawn_planned_wave(self, scene):
        if self._active_wave_id is not None:
            scene.wave_survived()

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
        self._wave_timer = WAVE_TIMEOUT_SECONDS * config.FPS
        self._minimum_wave_timer = MIN_WAVE_INTERVAL_SECONDS * config.FPS

        self._planned_wave = self._plan_wave(self._next_wave_index)
        self._next_wave_index += 1

    def spawn(self, scene):
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
