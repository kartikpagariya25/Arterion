import os
import glob
import numpy as np
import py7zr

from config import RAW_DIR, EXTRACTED_DIR


def extract():
    for fname in ["main_normal.7z", "main_others.7z"]:
        path = os.path.join(RAW_DIR, fname)
        marker = os.path.join(EXTRACTED_DIR, f".{fname}.done")
        if os.path.exists(marker):
            print(f"{fname}: already extracted, skipping")
            continue
        print(f"extracting {fname} (large archive, can take a long time)...")
        with py7zr.SevenZipFile(path, mode="r") as z:
            z.extractall(path=EXTRACTED_DIR)
        open(marker, "w").close()
        print(f"{fname}: extraction complete")


def print_tree(root, max_depth=3, max_entries=15):
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        if depth > max_depth:
            dirnames[:] = []
            continue
        indent = "  " * depth
        print(f"{indent}{os.path.basename(dirpath) or dirpath}/")
        for f in filenames[:max_entries]:
            print(f"{indent}  {f}")
        if len(filenames) > max_entries:
            print(f"{indent}  ... and {len(filenames) - max_entries} more files")


def inspect_npz_files(root, limit=3):
    found = 0
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if not f.endswith(".npz"):
                continue
            if found >= limit:
                return
            path = os.path.join(dirpath, f)
            print(f"\n--- {path} ---")
            try:
                data = np.load(path, allow_pickle=True)
                print("keys:", data.files)
                for key in data.files:
                    arr = data[key]
                    print(f"  {key}: shape={arr.shape if hasattr(arr, 'shape') else 'N/A'}, dtype={arr.dtype if hasattr(arr, 'dtype') else type(arr)}")
            except Exception as e:
                print(f"failed to load: {e}")
            found += 1


def inspect_label_files(root):
    for ext in ("*.csv", "*.json"):
        for path in glob.glob(os.path.join(root, "**", ext), recursive=True):
            print(f"\n--- {path} ---")
            if path.endswith(".csv"):
                import pandas as pd
                df = pd.read_csv(path)
                print("shape:", df.shape)
                print("columns:", list(df.columns))
                print(df.head(3).to_string())
            else:
                import json
                with open(path) as f:
                    data = json.load(f)
                print("type:", type(data))
                if isinstance(data, dict):
                    print("keys:", list(data.keys())[:20])
                elif isinstance(data, list) and data:
                    print("first item:", data[0])


if __name__ == "__main__":
    extract()
    print("\n=== DIRECTORY STRUCTURE ===")
    print_tree(EXTRACTED_DIR)
    print("\n=== NPZ FILE INSPECTION ===")
    inspect_npz_files(EXTRACTED_DIR)
    print("\n=== LABEL FILE INSPECTION ===")
    inspect_label_files(EXTRACTED_DIR)
