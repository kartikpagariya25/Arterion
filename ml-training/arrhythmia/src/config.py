import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MITDB_DIR = os.path.join(DATA_DIR, "mitdb")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

SEED = 42

SAMPLING_RATE = 360
BEAT_WINDOW = 256
PRE_R = 90
POST_R = 166
MIN_PEAK_DISTANCE_MS = 200

CLASSES = ["N", "S", "V", "F", "Q"]

AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}

EXCLUDED_RECORDS = [102, 104, 107, 217]

DS1_RECORDS = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
               201, 203, 205, 207, 208, 209, 215, 220, 223, 230]

DS2_RECORDS = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
               213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

VAL_FRACTION = 0.10

BANDPASS_LOW = 0.5
BANDPASS_HIGH = 40.0
BANDPASS_ORDER = 4

# training
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 30
EARLY_STOP_PATIENCE = 5
FOCAL_GAMMA = 2.0

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
