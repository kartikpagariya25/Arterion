"""One-off: index studies, make splits, and cache 8 evenly spaced 192px frames per video."""
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np

from config import PROCESSED_DIR, SEED
from dataset import (CACHE_DIR, CACHE_SIZE, CACHED_FRAMES, INDEX_PATH, SPLITS_PATH,
                     build_splits, cache_name, scan_index)


def cache_one(path: str) -> str:
    out = os.path.join(CACHE_DIR, cache_name(path))
    if os.path.exists(out):
        return "skip"
    try:
        arr = np.load(path)["pixel_array"]
        if arr.ndim == 4:
            arr = arr[..., 0]
        idx = np.linspace(0, arr.shape[0] - 1, CACHED_FRAMES).round().astype(int)
        small = np.stack([cv2.resize(arr[i], (CACHE_SIZE, CACHE_SIZE), interpolation=cv2.INTER_AREA) for i in idx])
        np.save(out, small.astype(np.uint8))
        return "ok"
    except Exception as e:  # corrupt npz: report and continue
        return f"fail:{path}:{e}"


if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)
    index = scan_index()
    with open(INDEX_PATH, "w") as f:
        json.dump(index, f)
    splits = build_splits(index, seed=SEED)
    with open(SPLITS_PATH, "w") as f:
        json.dump(splits, f)
    by = {r["study"]: r["label"] for r in index}
    for k, ids in splits.items():
        print(k, len(ids), "left=", sum(by[i] for i in ids))
    paths = [v["path"] for r in index for v in r["videos"]]
    print("videos:", len(paths))
    with Pool(4) as p:
        res = []
        for i, r in enumerate(p.imap_unordered(cache_one, paths, chunksize=8), 1):
            res.append(r)
            if i % 250 == 0:
                print(i, "/", len(paths), flush=True)
    fails = [r for r in res if r.startswith("fail")]
    print("done; failures:", len(fails))
    for r in fails[:20]:
        print(r)
