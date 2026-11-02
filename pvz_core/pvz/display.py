from pathlib import Path

import pygame

from . import config


BOARD_X = 24
BOARD_Y = 88
CELL_WIDTH = 92
CELL_HEIGHT = 90
HEADER_HEIGHT = 64
FOOTER_HEIGHT = 48
WINDOW_WIDTH = BOARD_X * 2 + config.LANE_LENGTH * CELL_WIDTH
WINDOW_HEIGHT = BOARD_Y + config.N_LANES * CELL_HEIGHT + FOOTER_HEIGHT
ASSET_DIR = Path(__file__).resolve().parents[2] / "assets"

ASSET_FILES = {
    "Sunflower": "sunflower_scaled.png",
    "Peashooter": "peashooter_scaled.png",
    "Wallnut": "wallnut_scaled.png",
    "Potatomine": "potatomine_scaled.png",
    "Zombie": "zombie_scaled.png",
    "ZombieBucket": "zombie_bucket_scaled.png",
    "ZombieCone": "zombie_cone_scaled.png",
    "ZombieFlag": "zombie_flag_scaled.png",
}


def display_game(render_info):
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Plants vs. Zombies - Trained model")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 26)
    small_font = pygame.font.Font(None, 21)
    images = {}

    try:
        for frame in render_info:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    return False

            screen.fill((34, 49, 39))
            pygame.draw.rect(screen, (27, 38, 31), (0, 0, WINDOW_WIDTH, HEADER_HEIGHT))
            pygame.draw.rect(
                screen,
                (27, 38, 31),
                (0, WINDOW_HEIGHT - FOOTER_HEIGHT, WINDOW_WIDTH, FOOTER_HEIGHT),
            )
            header = (
                f"Sun: {frame['sun']}     Score: {frame['score']}     "
                f"Time: {frame['time']}s"
            )
            screen.blit(font.render(header, True, (245, 245, 220)), (BOARD_X, 22))

            for lane in range(config.N_LANES):
                for pos in range(config.LANE_LENGTH):
                    color = (104, 161, 75) if (lane + pos) % 2 == 0 else (94, 149, 68)
                    cell = pygame.Rect(
                        BOARD_X + pos * CELL_WIDTH,
                        BOARD_Y + lane * CELL_HEIGHT,
                        CELL_WIDTH,
                        CELL_HEIGHT,
                    )
                    pygame.draw.rect(screen, color, cell)
                    pygame.draw.rect(screen, (68, 112, 55), cell, 1)

            for lane, plants in enumerate(frame["plants"]):
                for plant_name, pos in plants:
                    image = _load_image(images, plant_name)
                    if image is not None:
                        screen.blit(
                            image,
                            image.get_rect(
                                center=(
                                    BOARD_X + (pos + 0.5) * CELL_WIDTH,
                                    BOARD_Y + (lane + 0.5) * CELL_HEIGHT,
                                )
                            ),
                        )

            for lane, zombies in enumerate(frame["zombies"]):
                for zombie_name, pos, offset in zombies:
                    image = _load_image(images, zombie_name)
                    if image is not None:
                        screen.blit(
                            image,
                            image.get_rect(
                                center=(
                                    BOARD_X + (pos + offset + 0.5) * CELL_WIDTH,
                                    BOARD_Y + (lane + 0.5) * CELL_HEIGHT,
                                )
                            ),
                        )

            for lane, projectiles in enumerate(frame["projectiles"]):
                for projectile_name, pos, offset in projectiles:
                    x = int(BOARD_X + (pos + offset + 0.5) * CELL_WIDTH)
                    y = int(BOARD_Y + (lane + 0.5) * CELL_HEIGHT)
                    if projectile_name == "Mower":
                        pygame.draw.rect(screen, (190, 190, 190), (x - 26, y - 10, 52, 20))
                        pygame.draw.circle(screen, (35, 35, 35), (x - 14, y + 10), 6)
                        pygame.draw.circle(screen, (35, 35, 35), (x + 14, y + 10), 6)
                    else:
                        pygame.draw.circle(screen, (65, 220, 60), (x, y), 9)

            cooldowns = "Plant cooldowns: " + "   ".join(
                f"{name}: {seconds}s" for name, seconds in frame["cooldowns"].items()
            )
            screen.blit(
                small_font.render(cooldowns, True, (245, 245, 220)),
                (BOARD_X, WINDOW_HEIGHT - FOOTER_HEIGHT + 14),
            )
            pygame.display.flip()
            clock.tick(12)

        pygame.time.wait(1000)
        return True
    finally:
        pygame.quit()


def _load_image(images, name):
    if name not in ASSET_FILES:
        return None
    if name not in images:
        image_path = ASSET_DIR / ASSET_FILES[name]
        image = pygame.image.load(str(image_path)).convert_alpha()
        images[name] = pygame.transform.smoothscale(image, (72, 72))
    return images[name]
