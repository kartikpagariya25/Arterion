import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataset  # noqa: E402


def _fake_index():
    rng = np.random.default_rng(0)
    index = []
    for i in range(300):
        label = 1 if i % 3 == 0 else 0  # ~1:2 left:right
        index.append({"study": f"S{i:04d}", "label": label, "videos": [{"view": "RCA", "path": f"p{i}"}]})
    rng.shuffle(index)
    return index


def test_study_split_has_no_overlap_and_keeps_class_ratio():
    splits = dataset.build_splits(_fake_index(), seed=42)
    tr, va, te = (set(splits[k]) for k in ("train", "val", "test"))
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(tr | va | te) == 300
    ratio = {}
    for k in ("train", "val", "test"):
        labels = [int(s[1:]) % 3 == 0 for s in splits[k]]
        ratio[k] = sum(labels) / len(labels)
    for k in ratio:
        assert abs(ratio[k] - 1 / 3) < 0.05


def test_single_view_study_makes_tensor():
    cache = {"p1": np.zeros((8, 64, 64), dtype=np.uint8), "p2": np.zeros((8, 64, 64), dtype=np.uint8)}
    study = {"study": "S1", "label": 0, "videos": [{"view": "RCA", "path": "p1"}, {"view": "RCA", "path": "p2"}]}
    frames, views = dataset.study_to_tensors(study, cache, max_videos=4, frames_per_video=4, size=64, train=False)
    assert frames.shape == (2, 4, 3, 64, 64)
    assert views.tolist() == [0, 0]
