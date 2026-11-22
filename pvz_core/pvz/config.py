#simulations
FPS = 2
MAX_FRAMES = 2000

#map layout
N_LANES = 5 # Height
LANE_LENGTH = 9 # Width

#currency
INITIAL_SUN_AMOUNT = 50
NATURAL_SUN_PRODUCTION = 25 # Sun that falls from the sky
NATURAL_SUN_PRODUCTION_COOLDOWN = 8

# Regular zombie spawning (seconds)
BASIC_ZOMBIE_INITIAL_OFFSET = 20
BASIC_ZOMBIE_SPAWN_INTERVAL = 40

# Wave zombie spawning
WAVE_TIMEOUT_SECONDS = 85
FIRST_WAVE_DELAY_SECONDS = 60
MIN_WAVE_INTERVAL_SECONDS = 60
POINTS_PER_WAVE = 0
POINTS_ADDED_PER_WAVE = 1.4
WAVE_ZOMBIE_POINT_COSTS = {
    "basic": 1,
    "cone": 3,
    "bucket": 5,
}

# Mowers
MOWERS = False

# Score values; zombie scores are awarded proportionally as damage is dealt.
#Positive scores by killing & surviving
SURVIVAL = 60
SCORE_ZOMBIE = 120
SCORE_ZOMBIE_CONE = 350
SCORE_ZOMBIE_BUCKET = 695
SCORE_ZOMBIE_FLAG = 120
SCORE_ALIVE_PLANT = 0
SCORE_ALIVE_MOWER = 0
PLANT_OFFENSIVE_ZOMBIE_BONUS = 50

#Negative scores by losing plants & zombies advancing on the map
DEATH_PENALTY = 1200
PLANT_DEATH_PENALTY = 10
PLANT_REPLACEMENT_PENALTY = 30
ZOMBIE_DANGER_COLUMN_PENALTIES = {2: 1, 1: 3, 0: 10}

# Score 10/10
# Scorebis : 10/20