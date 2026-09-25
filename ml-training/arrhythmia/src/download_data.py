import os
import time
import wfdb

from config import MITDB_DIR, EXCLUDED_RECORDS, DS1_RECORDS, DS2_RECORDS

ALL_RECORDS = sorted(set(EXCLUDED_RECORDS) | set(DS1_RECORDS) | set(DS2_RECORDS))


def already_downloaded(record_id):
    for ext in (".dat", ".hea", ".atr"):
        if not os.path.exists(os.path.join(MITDB_DIR, f"{record_id}{ext}")):
            return False
    return True


def main():
    os.makedirs(MITDB_DIR, exist_ok=True)
    total = len(ALL_RECORDS)

    for i, record_id in enumerate(ALL_RECORDS, 1):
        if already_downloaded(record_id):
            print(f"[{i}/{total}] record {record_id}: already present, skipping")
            continue

        print(f"[{i}/{total}] downloading record {record_id}...", flush=True)
        for attempt in range(1, 4):
            try:
                wfdb.dl_database("mitdb", dl_dir=MITDB_DIR, records=[str(record_id)])
                break
            except Exception as e:
                print(f"  attempt {attempt} failed: {e}")
                time.sleep(2)
        else:
            print(f"  FAILED to download record {record_id} after 3 attempts")

    missing = [r for r in ALL_RECORDS if not already_downloaded(r)]
    print(f"\nDone: {total - len(missing)}/{total} records present.")
    if missing:
        print(f"Missing: {missing}")
        print("Re-run this script — it will skip completed records and retry only these.")


if __name__ == "__main__":
    main()
