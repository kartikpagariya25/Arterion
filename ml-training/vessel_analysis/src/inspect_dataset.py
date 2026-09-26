import os
import json
import zipfile

from config import RAW_DIR, EXTRACTED_DIR

ZIP_PATH = os.path.join(RAW_DIR, "arcade_challenge_datasets.zip")


def extract():
    if os.path.exists(EXTRACTED_DIR) and os.listdir(EXTRACTED_DIR):
        print("already extracted, skipping")
        return
    os.makedirs(EXTRACTED_DIR, exist_ok=True)
    print("extracting (a 450MB archive can take a few minutes)...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACTED_DIR)
    print("extraction complete")


def print_tree(root, max_depth=4, max_entries_per_dir=15):
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        if depth > max_depth:
            dirnames[:] = []
            continue
        indent = "  " * depth
        print(f"{indent}{os.path.basename(dirpath) or dirpath}/")
        sub_indent = "  " * (depth + 1)
        for f in filenames[:max_entries_per_dir]:
            print(f"{sub_indent}{f}")
        if len(filenames) > max_entries_per_dir:
            print(f"{sub_indent}... and {len(filenames) - max_entries_per_dir} more files")


def inspect_json_files(root):
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if not f.endswith(".json"):
                continue
            path = os.path.join(dirpath, f)
            print(f"\n--- {path} ---")
            try:
                with open(path) as jf:
                    data = json.load(jf)
            except Exception as e:
                print(f"failed to parse: {e}")
                continue

            if isinstance(data, dict):
                print("top-level keys:", list(data.keys()))
                if "categories" in data:
                    print("categories:", data["categories"][:30])
                if "images" in data and data["images"]:
                    print("sample image entry:", data["images"][0])
                if "annotations" in data and data["annotations"]:
                    print("sample annotation entry:", data["annotations"][0])
            else:
                print("type:", type(data))


if __name__ == "__main__":
    extract()
    print("\n=== DIRECTORY STRUCTURE ===")
    print_tree(EXTRACTED_DIR)
    print("\n=== COCO JSON INSPECTION ===")
    inspect_json_files(EXTRACTED_DIR)
