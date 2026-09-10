"""
download_dataset.py
Trace-Oceans Dataset Downloader for Zenodo Record 8346860 (Part I)

Automates downloading, extracting, and cleaning up the 40+ GB Sentinel-1 SAR Oil Spill dataset:
- 01_Train_Val_Oil_Spill_mask.7z (~6.2 MB) -> dataset/masks/
- 01_Train_Val_Oil_Spill_images.7z (~40.7 GB) -> dataset/images/

Optimized for Google Colab and cloud instances with automatic disk checks,
multi-threaded download acceleration (aria2c), robust fallback, and immediate
archive deletion to prevent running out of disk space.
"""

import os
import sys
import shutil
import subprocess
import requests
from pathlib import Path

ZENODO_RECORD_ID = "8346860"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

# Default paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
MASKS_DIR = DATASET_DIR / "masks"


def check_disk_space(required_gb=50):
    """Checks if there is enough free disk space."""
    stat = shutil.disk_usage(BASE_DIR)
    free_gb = stat.free / (1024 ** 3)
    total_gb = stat.total / (1024 ** 3)
    print(f"📊 Disk Status: {free_gb:.1f} GB free out of {total_gb:.1f} GB total.")
    if free_gb < required_gb:
        print(f"⚠️  WARNING: Less than {required_gb} GB free! Extracting 40 GB requires headroom.")
        print("💡 In Google Colab, free up space by running:")
        print("   !rm -rf /usr/local/cuda-11* /usr/local/cuda-12.0 /root/.cache")
    return free_gb


def ensure_tools():
    """Ensures 7z is available for extraction."""
    has_7z = shutil.which("7z") is not None
    if not has_7z:
        print("🔧 '7z' not found. Attempting to install p7zip-full...")
        try:
            subprocess.run(["apt-get", "update", "-qq"], check=False)
            subprocess.run(["apt-get", "install", "-y", "-qq", "p7zip-full"], check=True)
            print("✅ '7z' successfully installed.")
        except Exception as e:
            print(f"❌ Could not install 7z via apt-get: {e}")
            print("Please install p7zip-full manually: sudo apt-get install p7zip-full")
            sys.exit(1)


def get_zenodo_file_urls():
    """Queries Zenodo API to obtain exact download links and sizes."""
    print(f"🌐 Querying Zenodo API for record {ZENODO_RECORD_ID}...")
    resp = requests.get(ZENODO_API_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    files = {}
    for f in data.get("files", []):
        filename = f.get("key")
        download_url = f.get("links", {}).get("self")
        size_bytes = f.get("size", 0)
        files[filename] = {
            "url": download_url,
            "size": size_bytes,
            "size_gb": size_bytes / (1024 ** 3)
        }
    return files


def download_file(url, target_path):
    """
    Downloads file using aria2c (fast multi-connection) or falls back to curl/wget.
    """
    target_path = Path(target_path)
    filename = target_path.name
    dest_dir = target_path.parent

    # Prefer wget (pre-installed on Colab/Linux, rock-solid with Zenodo, supports resume)
    if shutil.which("wget"):
        print(f"📥 Downloading {filename} with wget (resumable, retry enabled)...")
        cmd = [
            "wget",
            "-c",
            "--retry-connrefused",
            "--tries=10",
            "--timeout=30",
            "-O", str(target_path),
            url
        ]
        ret = subprocess.run(cmd)
        if ret.returncode == 0:
            return True
        print("⚠️ wget failed, attempting aria2c/curl fallback...")

    # Fallback to aria2c with safe 4 connections (avoids Zenodo rate limit HTTP 429)
    if shutil.which("aria2c"):
        print(f"🚀 Downloading {filename} with aria2c (4 connections)...")
        cmd = [
            "aria2c",
            "-x", "4",
            "-s", "4",
            "-k", "1M",
            "--continue=true",
            "--dir", str(dest_dir),
            "--out", filename,
            url
        ]
        ret = subprocess.run(cmd)
        if ret.returncode == 0:
            return True

    # Fallback to curl
    if shutil.which("curl"):
        print(f"📥 Downloading {filename} with curl...")
        cmd = ["curl", "-L", "-C", "-", "-o", str(target_path), url]
        ret = subprocess.run(cmd)
        if ret.returncode == 0:
            return True

    # Python stream fallback
    print(f"📥 Downloading {filename} with Python requests (fallback)...")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(target_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
    return True


def extract_and_organize(archive_path, target_dest_dir):
    """
    Extracts 7z archive to a temporary staging folder, moves all .tif files into
    target_dest_dir, and immediately deletes archive and temp folder.
    """
    archive_path = Path(archive_path)
    target_dest_dir = Path(target_dest_dir)
    target_dest_dir.mkdir(parents=True, exist_ok=True)

    staging_dir = target_dest_dir.parent / f"_temp_extract_{archive_path.stem}"
    staging_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 Extracting {archive_path.name} to {staging_dir} using 7z...")
    cmd = ["7z", "x", "-y", f"-o{staging_dir}", str(archive_path)]
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        raise RuntimeError(f"7z extraction failed for {archive_path}")

    # Immediately delete the 7z archive to reclaim disk space!
    print(f"🗑️ Reclaiming disk space: deleting {archive_path.name} ({archive_path.stat().st_size / (1024**3):.2f} GB)...")
    try:
        archive_path.unlink()
        print(f"✅ Deleted {archive_path.name}")
    except Exception as e:
        print(f"⚠️ Could not delete {archive_path}: {e}")

    # Move extracted files into target directory
    print(f"📂 Flattening and organizing .tif files into {target_dest_dir}...")
    tif_count = 0
    for root, _, files in os.walk(staging_dir):
        for file in files:
            if file.lower().endswith((".tif", ".tiff")):
                src_file = Path(root) / file
                dst_file = target_dest_dir / file
                if dst_file.exists():
                    dst_file.unlink()
                shutil.move(str(src_file), str(dst_file))
                tif_count += 1

    # Remove temporary staging directory
    shutil.rmtree(staging_dir, ignore_errors=True)
    print(f"✨ Successfully placed {tif_count} files into {target_dest_dir}")
    return tif_count


def run_pipeline():
    print("=" * 70)
    print("🌊 Trace-Oceans Dataset Setup (Zenodo Record 8346860)")
    print("=" * 70)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    check_disk_space(required_gb=50)
    ensure_tools()

    files = get_zenodo_file_urls()

    # Step 1: Masks first (~6 MB, minimal disk space)
    mask_file = "01_Train_Val_Oil_Spill_mask.7z"
    if mask_file in files:
        mask_archive = DATASET_DIR / mask_file
        print(f"\n--- [1/2] Processing Ground Truth Masks ({files[mask_file]['size_gb']*1024:.1f} MB) ---")
        download_file(files[mask_file]["url"], mask_archive)
        extract_and_organize(mask_archive, MASKS_DIR)
    else:
        print(f"⚠️ {mask_file} not found in Zenodo record!")

    # Step 2: Images (~40.7 GB, requires sequential extract + delete)
    img_file = "01_Train_Val_Oil_Spill_images.7z"
    if img_file in files:
        img_archive = DATASET_DIR / img_file
        print(f"\n--- [2/2] Processing Sentinel-1 Images ({files[img_file]['size_gb']:.2f} GB) ---")
        check_disk_space(required_gb=45)
        download_file(files[img_file]["url"], img_archive)
        extract_and_organize(img_archive, IMAGES_DIR)
    else:
        print(f"⚠️ {img_file} not found in Zenodo record!")

    # Final summary
    num_images = len(list(IMAGES_DIR.glob("*.tif*")))
    num_masks = len(list(MASKS_DIR.glob("*.tif*")))
    print("\n" + "=" * 70)
    print("🎉 DATASET SETUP COMPLETE!")
    print(f"   Images: {num_images} in {IMAGES_DIR}")
    print(f"   Masks:  {num_masks} in {MASKS_DIR}")
    check_disk_space(required_gb=10)
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
