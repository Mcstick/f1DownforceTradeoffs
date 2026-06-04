# ─────────────────────────────────────────────
#  config.py  –  single source of truth
# ─────────────────────────────────────────────

# ── Session ───────────────────────────────────
YEAR        = 2025
GRAND_PRIX  = "British"          # Silverstone
SESSIONS    = ["R", "Q"]         # Race, Qualifying

# ── Drivers & teams ───────────────────────────
DRIVERS = {
    "VER": {"full_name": "Max Verstappen",  "team": "Red Bull Racing", "color": "#3671C6"},
    "NOR": {"full_name": "Lando Norris",    "team": "McLaren",         "color": "#FF8000"},
}

DRIVER_CODES = list(DRIVERS.keys())   # ["VER", "NOR"]

# ── Corner speed thresholds (km/h) ────────────
CORNER_SPEED = {
    "slow":   (0,   160),
    "medium": (160, 220),
    "fast":   (220, 999),
}

# ── Silverstone named corners (approximate distances in metres from start/finish)
# Used for annotation; not hard-required for classification.
SILVERSTONE_CORNERS = {
    "Copse":             580,
    "Maggotts":          1200,
    "Becketts":          1400,
    "Chapel":            1650,
    "Hangar Straight":   2400,   # not a corner – top-speed reference point
    "Stowe":             2900,
    "Vale":              3300,
    "Club":              3600,
    "Abbey":             4200,
    "Farm":              4500,
    "Village":           4900,
    "The Loop":          5100,
    "Aintree":           5400,
    "Wellington":        5700,
    "Brooklands":        5950,
    "Luffield":          6250,
    "Woodcote":          6500,
}

# ── Telemetry channels we care about ──────────
TELEMETRY_CHANNELS = [
    "Speed",        # km/h
    "Throttle",     # 0-100 %
    "Brake",        # bool / 0-100
    "nGear",        # gear number
    "RPM",
    "DRS",          # 0/8/10/12/14 – >8 means open
    "Distance",     # metres into lap
    "Time",         # timedelta from lap start
    "X", "Y", "Z",  # track position (used for corner mapping)
]

# ── Feature engineering ───────────────────────
MIN_VALID_LAP_TIME_S  = 80      # discard outliers / safety-car laps
MAX_VALID_LAP_TIME_S  = 140

# ── ML model ──────────────────────────────────
MODEL_RANDOM_SEED   = 42
XGBOOST_PARAMS = {
    "n_estimators":     400,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "random_state":     MODEL_RANDOM_SEED,
}
TEST_SIZE = 0.2

# ── Output ────────────────────────────────────
CACHE_DIR   = "data/cache"
OUTPUT_DIR  = "outputs"
FIG_DPI     = 150
FIG_STYLE   = "dark_background"   # matplotlib style
