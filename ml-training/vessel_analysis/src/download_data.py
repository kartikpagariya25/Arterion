import os
import hashlib
import requests

from config import RAW_DIR

ZENODO_RECORD = "8386059"
FILENAME = "arcade_challenge_datasets.zip"
DOWNLOAD_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{FILENAME}/content"
EXPECTED_MD5 = "50a57df6882469468a0b980a059ed3c8"


def _md5(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download():
    out_path = os.path.join(RAW_DIR, FILENAME)
    resume_pos = os.path.getsize(out_path) if os.path.exists(out_path) else 0

    head = requests.head(DOWNLOAD_URL, allow_redirects=True)
    total_size = int(head.headers.get("content-length", 0))

    if resume_pos and resume_pos >= total_size:
        print("file already fully downloaded, verifying checksum...")
    else:
        headers = {"Range": f"bytes={resume_pos}-"} if resume_pos else {}
        mode = "ab" if resume_pos else "wb"
        if resume_pos:
            print(f"resuming download from {resume_pos / 1e6:.1f}MB")

        with requests.get(DOWNLOAD_URL, headers=headers, stream=True, timeout=60) as r:
            r.raise_for_status()
            downloaded = resume_pos
            with open(out_path, mode) as f:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    pct = downloaded / total_size * 100 if total_size else 0
                    print(f"\r{downloaded / 1e6:.1f}MB / {total_size / 1e6:.1f}MB ({pct:.1f}%)", end="", flush=True)
        print()

    print("verifying checksum...")
    actual_md5 = _md5(out_path)
    if actual_md5 != EXPECTED_MD5:
        print(f"CHECKSUM MISMATCH: expected {EXPECTED_MD5}, got {actual_md5}")
        print("the downloaded file is likely corrupt — delete it and re-run this script")
        return False

    print("checksum verified, download complete.")
    return True


if __name__ == "__main__":
    download()
