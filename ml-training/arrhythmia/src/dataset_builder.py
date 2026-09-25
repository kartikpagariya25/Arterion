import os
import numpy as np
import wfdb

from config import (
    MITDB_DIR, PROCESSED_DIR, AAMI_MAP, CLASSES,
    EXCLUDED_RECORDS, DS1_RECORDS, DS2_RECORDS, SEED,
)
from preprocess import bandpass_filter, extract_beat_window, normalize_beat, compute_rr_features


def process_record(record_id: int):
    path = os.path.join(MITDB_DIR, str(record_id))
    record = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")

    sig_names = record.sig_name
    ch = sig_names.index("MLII") if "MLII" in sig_names else 0
    signal = record.p_signal[:, ch]
    fs = record.fs

    cleaned = bandpass_filter(signal, fs)

    beats, rrs, labels, records = [], [], [], []
    for i, sym in enumerate(ann.symbol):
        if sym not in AAMI_MAP:
            continue
        r_index = ann.sample[i]
        beat = extract_beat_window(cleaned, r_index)
        if beat is None:
            continue
        beats.append(normalize_beat(beat))
        rrs.append(compute_rr_features(ann.sample, i, fs))
        labels.append(CLASSES.index(AAMI_MAP[sym]))
        records.append(record_id)

    return np.array(beats, dtype=np.float32), np.array(rrs, dtype=np.float32), \
        np.array(labels, dtype=np.int64), np.array(records, dtype=np.int64)


def build_set(record_ids):
    all_beats, all_rrs, all_labels, all_records = [], [], [], []
    for rid in record_ids:
        beats, rrs, labels, records = process_record(rid)
        all_beats.append(beats)
        all_rrs.append(rrs)
        all_labels.append(labels)
        all_records.append(records)
        print(f"record {rid}: {len(labels)} beats")

    return (
        np.concatenate(all_beats),
        np.concatenate(all_rrs),
        np.concatenate(all_labels),
        np.concatenate(all_records),
    )


def save_npz(path, beats, rrs, labels, records):
    np.savez(path, beats=beats, rr=rrs, labels=labels, record=records)
    print(f"saved {path}: beats={beats.shape}, rr={rrs.shape}")
    dist = {CLASSES[i]: int((labels == i).sum()) for i in range(len(CLASSES))}
    print(f"class distribution: {dist}")


def compute_rr_stats(rr: np.ndarray):
    mean = rr.mean(axis=0)
    std = rr.std(axis=0)
    return mean, std


def main():
    print("Building DS1 (train + val)...")
    ds1_beats, ds1_rr, ds1_labels, ds1_records = build_set(DS1_RECORDS)
    save_npz(os.path.join(PROCESSED_DIR, "ds1.npz"), ds1_beats, ds1_rr, ds1_labels, ds1_records)

    print("\nBuilding DS2 (test, held out)...")
    ds2_beats, ds2_rr, ds2_labels, ds2_records = build_set(DS2_RECORDS)
    save_npz(os.path.join(PROCESSED_DIR, "ds2.npz"), ds2_beats, ds2_rr, ds2_labels, ds2_records)

    print("\nBuilding intra-patient comparison set (DS1+DS2 combined)...")
    combined_beats = np.concatenate([ds1_beats, ds2_beats])
    combined_rr = np.concatenate([ds1_rr, ds2_rr])
    combined_labels = np.concatenate([ds1_labels, ds2_labels])
    combined_records = np.concatenate([ds1_records, ds2_records])
    save_npz(os.path.join(PROCESSED_DIR, "intra.npz"), combined_beats, combined_rr, combined_labels, combined_records)

    print("\nDone. RR feature stats will be computed and saved by train.py "
          "from the actual training split (not the full DS1 set).")


if __name__ == "__main__":
    main()
