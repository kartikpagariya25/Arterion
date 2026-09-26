import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_TRAINING_DIR = os.path.dirname(BASE_DIR)

ECHONET_RAW_DIR = os.path.join(ML_TRAINING_DIR, "echonet_data", "raw")
ECHONET_EXTRACTED_DIR = os.path.join(ML_TRAINING_DIR, "echonet_data", "extracted")

DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

SEED = 42
IMAGE_SIZE = 112

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
