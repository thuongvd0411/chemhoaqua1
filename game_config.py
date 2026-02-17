import os

# Resolution Config
GAME_WIDTH = 640
GAME_HEIGHT = 480

# Paths
ASSETS_DIR = "assets"
if not os.path.exists(ASSETS_DIR):
    os.makedirs(ASSETS_DIR)

# Game Settings
DEFAULT_TARGET_SCORE = 100
DEFAULT_MAX_LIVES = 10

# Leaderboard Ranks
RANKS = [
    {"name": "Top1 Server", "emoji": "🏆", "color": "#FFD700", "min_stars": 0}, # Special
    {"name": "Chiến thần", "emoji": "⚔️", "color": "#F44336", "min_stars": 76},
    {"name": "Cao thủ", "emoji": "🔥", "color": "#FF9800", "min_stars": 51},
    {"name": "Cũng biết chơi", "emoji": "🐣", "color": "#4CAF50", "min_stars": 21},
    {"name": "Gà mờ", "emoji": "🥚", "color": "#808080", "min_stars": 0},
]

# Vietnamese Asset Labels
# Vietnamese Asset Labels - Mapped to AVAILABLE images
ASSET_LABELS = {
    'apple': 'Táo (Apple)',
    'orange': 'Cam (Orange)',
    'mango': 'Xoài (Mango)',
    'banana': 'Chuối (Banana)',
    'cherry': 'Cherry',
    'bomb': 'Bom (Bomb)',
    'watermelon': 'Dưa hấu (Watermelon)',
    # Fallbacks/Mappings for missing assets
    'mole': 'Chuột (Mole)', # Will map to orange/mango if missing
    'spike': 'Gai (Spike)', # Will map to bomb
    'first_aid': 'Túi cứu thương' # Will likely disable if image missing
}

# Particle Colors (BGR)
PARTICLE_COLORS = {
    'apple': (0, 0, 255),       # Red
    'orange': (0, 165, 255),    # Orange
    'mango': (0, 255, 255),     # Yellow
    'banana': (0, 255, 255),    # Yellow
    'cherry': (0, 0, 139),      # Dark Red
    'watermelon': (0, 0, 200),  # Red-ish
    'bomb': (50, 50, 50),       # Dark Grey
    'default': (200, 200, 200)  # White/Grey
}
