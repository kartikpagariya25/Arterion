import os
import hashlib
import time
import requests

from config import RAW_DIR, HF_BASE_URL, FILES


def _sha256(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(filename: str, max_retries: int = 10):
    info = FILES[filename]
    url = f"{HF_BASE_URL}/{filename}"
    out_path = os.path.join(RAW_DIR, filename)
    total_size = info["size"]

    for attempt in range(1, max_retries + 1):
        resume_pos = os.path.getsize(out_path) if os.path.exists(out_path) else 0

        if resume_pos >= total_size:
            break

        headers = {"Range": f"bytes={resume_pos}-"} if resume_pos else {}
        mode = "ab" if resume_pos else "wb"
        if resume_pos:
            print(f"{filename}: resuming from {resume_pos / 1e9:.2f}GB (attempt {attempt}/{max_retries})")

        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                downloaded = resume_pos
                with open(out_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                        f.write(chunk)
                        downloaded += len(chunk)
                        pct = downloaded / total_size * 100
                        print(f"\r{filename}: {downloaded / 1e9:.2f}GB / {total_size / 1e9:.2f}GB ({pct:.1f}%)",
                              end="", flush=True)
            print()
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            print(f"\n{filename}: connection dropped ({e.__class__.__name__}), retrying...")
            time.sleep(5)
            continue

    print(f"{filename}: verifying checksum (this can take a few minutes for a large file)...")
    actual = _sha256(out_path)
    if actual != info["sha256"]:
        print(f"{filename}: CHECKSUM MISMATCH — expected {info['sha256']}, got {actual}")
        print("the downloaded file is likely corrupt or incomplete — delete it and re-run this script")
        return False

    print(f"{filename}: checksum verified.")
    return True


def main():
    results = {}
    for filename in FILES:
        results[filename] = download_file(filename)

    if all(results.values()):
        print("\nAll files downloaded and verified successfully.")
    else:
        failed = [f for f, ok in results.items() if not ok]
        print(f"\nFailed: {failed} — re-run this script to retry.")


if __name__ == "__main__":
    main()
