import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
EXTRACTED_DIR = os.path.join(DATA_DIR, "extracted")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

SEED = 42
IMAGE_SIZE = 512

HF_REPO = "BearSubj13/CoronaryDominance"
HF_BASE_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"

FILES = {
    "main_normal.7z": {
        "sha256": "9cf9ff25f1393080eae9f23345f281c44368fb4a073bb1246faf2bac4675c285",
        "size": 40722004587,
    },
    "main_others.7z": {
        "sha256": "fcf46fb5feee41bab9467ca13c367581375062f87ba38277c5cd806d6c0f26fa",
        "size": 14176216862,
    },
}

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
