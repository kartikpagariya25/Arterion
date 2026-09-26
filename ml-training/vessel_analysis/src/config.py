import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
EXTRACTED_DIR = os.path.join(DATA_DIR, "extracted")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

SEED = 42
IMAGE_SIZE = 512

ROOT = os.path.join(EXTRACTED_DIR, "arcade_challenge_datasets")

SEGMENTATION_SPLITS = {
    "train": {
        "images": os.path.join(ROOT, "dataset_phase_1", "segmentation_dataset", "seg_train", "images"),
        "annotations": os.path.join(ROOT, "dataset_phase_1", "segmentation_dataset", "seg_train", "annotations", "seg_train.json"),
    },
    "val": {
        "images": os.path.join(ROOT, "dataset_phase_1", "segmentation_dataset", "seg_val", "images"),
        "annotations": os.path.join(ROOT, "dataset_phase_1", "segmentation_dataset", "seg_val", "annotations", "seg_val.json"),
    },
    "test": {
        "images": os.path.join(ROOT, "dataset_final_phase", "test_case_segmentation", "images"),
        "annotations": os.path.join(ROOT, "dataset_final_phase", "test_case_segmentation", "annotations", "instances_default.json"),
    },
}

STENOSIS_SPLITS = {
    "train": {
        "images": os.path.join(ROOT, "dataset_phase_1", "stenosis_dataset", "sten_train", "images"),
        "annotations": os.path.join(ROOT, "dataset_phase_1", "stenosis_dataset", "sten_train", "annotations", "sten_train.json"),
    },
    "val": {
        "images": os.path.join(ROOT, "dataset_phase_1", "stenosis_dataset", "sten_val", "images"),
        "annotations": os.path.join(ROOT, "dataset_phase_1", "stenosis_dataset", "sten_val", "annotations", "sten_val.json"),
    },
    "test": {
        "images": os.path.join(ROOT, "dataset_final_phase", "test_cases_stenosis", "images"),
        "annotations": os.path.join(ROOT, "dataset_final_phase", "test_cases_stenosis", "annotations", "instances_default.json"),
    },
}

# category id 26 is "stenosis" in every annotation file's category list;
# ids 1-25 are the SYNTAX-style anatomical vessel segments
VESSEL_CATEGORY_IDS = list(range(1, 26))
STENOSIS_CATEGORY_ID = 26

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
