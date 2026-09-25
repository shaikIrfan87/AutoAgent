"""
Download and extract official ARC-AGI evaluation split into benchmarks/data/evaluation/.
"""
import io
import os
import sys
import zipfile
import urllib.request

ARC_ZIP_URL = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"
TARGET_DIR = os.path.join(os.path.dirname(__file__), "data", "evaluation")

def download_and_extract_arc(target_dir: str = TARGET_DIR, max_files: int = 50):
    os.makedirs(target_dir, exist_ok=True)
    existing = [f for f in os.listdir(target_dir) if f.endswith(".json")]
    if len(existing) >= max_files:
        print(f"ARC evaluation dataset already present: {len(existing)} tasks found in {target_dir}")
        return len(existing)

    print(f"Downloading official ARC-AGI dataset from {ARC_ZIP_URL}...")
    req = urllib.request.Request(ARC_ZIP_URL, headers={"User-Agent": "AutoAgent-Benchmark/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read()

    print(f"Downloaded {len(content)} bytes. Extracting evaluation tasks...")
    extracted_count = 0
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        for member in zf.namelist():
            if "data/evaluation/" in member and member.endswith(".json"):
                fname = os.path.basename(member)
                if not fname:
                    continue
                out_path = os.path.join(target_dir, fname)
                with open(out_path, "wb") as out_f:
                    out_f.write(zf.read(member))
                extracted_count += 1

    print(f"Successfully extracted {extracted_count} ARC evaluation tasks to {target_dir}")
    return extracted_count

if __name__ == "__main__":
    download_and_extract_arc()
