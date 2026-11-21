#simulations
FPS = 2
MAX_FRAMES = 1000

#map layout
N_LANES = 5 # Height
LANE_LENGTH = 9 # Width

#currency
INITIAL_SUN_AMOUNT = 75
NATURAL_SUN_PRODUCTION = 25 # Sun that falls from the sky
NATURAL_SUN_PRODUCTION_COOLDOWN = 10

# Mowers
MOWERS = False

# Score values; zombie scores are awarded proportionally as damage is dealt.
#Positive scores by killing & surviving
SURVIVAL = 30
SCORE_ZOMBIE = 120
SCORE_ZOMBIE_CONE = 40
SCORE_ZOMBIE_BUCKET = 120
SCORE_ZOMBIE_FLAG = 120
SCORE_ALIVE_PLANT = 0
SCORE_ALIVE_MOWER = 0

#Negative scores by losing plants & zombies advancing on the map
DEATH_PENALTY = 1200
PLANT_DEATH_PENALTY = 15
PLANT_REPLACEMENT_PENALTY = 5
PLANT_OFFENSIVE_ZOMBIE_BONUS = 1
ZOMBIE_DANGER_COLUMN_PENALTIES = {2: 5, 1: 25, 0: 100}

# Score 10/10
# Scorebis : 10/20