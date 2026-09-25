import os
import wfdb
import numpy as np

from config import MITDB_DIR

OUTPUT_DIR = os.path.join(os.path.dirname(MITDB_DIR), "test_signals")


def export(record_id: int, n_samples: int = 20000):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    record = wfdb.rdrecord(os.path.join(MITDB_DIR, str(record_id)))

    sig_names = record.sig_name
    ch = sig_names.index("MLII") if "MLII" in sig_names else 0
    signal = record.p_signal[:n_samples, ch]

    out_path = os.path.join(OUTPUT_DIR, f"record{record_id}_sample.csv")
    np.savetxt(out_path, signal, delimiter=",")
    print(f"exported {len(signal)} samples to {out_path}")


if __name__ == "__main__":
    export(record_id=100)
