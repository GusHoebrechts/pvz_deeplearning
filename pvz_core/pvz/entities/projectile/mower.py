from ... import config
from .projectile import Projectile

MOWER_SPEED = 5 # Cells per second

class Mower(Projectile):

    def __init__(self, lane):
        super().__init__(MOWER_SPEED, lane, 0)

    def _attack_zombies(self, zombies, scene): # Kills all zombies
        for zombie in zombies:
            scene.damage_zombie(zombie, zombie.hp)
    
    def _render(self):
        return True