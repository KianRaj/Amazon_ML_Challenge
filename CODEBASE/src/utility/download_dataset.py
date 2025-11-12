import os
import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import time as timer

# ===============================
# CONFIG
# ===============================
CSV_PATH = "../dataset/train.csv"
SAVE_DIR = "../images/train"
MAX_THREADS = 32  # Tune: 16–64 depending on network

# ===============================
# DOWNLOAD FUNCTION
# ===============================
def download_image(image_link, savefolder):
    """Download a single image using requests (more reliable than urllib)."""
    if not isinstance(image_link, str) or not image_link.startswith("http"):
        return "skip"

    # Use filename from URL to match existing naming
    filename = Path(image_link).name
    image_save_path = os.path.join(savefolder, filename)

    # Resumability: Skip if image already exists
    if os.path.exists(image_save_path):
        return "exists"

    try:
        response = requests.get(image_link, timeout=10, stream=True)
        if response.status_code == 200:
            with open(image_save_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return "ok"
        else:
            return f"fail_{response.status_code}"
    except Exception as e:
        return f"error: {e}"

# ===============================
# MAIN DOWNLOADER
# ===============================
def download_images(image_links, download_folder, max_threads=MAX_THREADS):
    os.makedirs(download_folder, exist_ok=True)
    print(f"\n🧩 Starting download of {len(image_links)} images using {max_threads} threads...\n")

    start = timer()
    success, fail, skipped = 0, 0, 0

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(download_image, link, download_folder): link for link in image_links}
        for i, future in enumerate(tqdm(futures, total=len(futures), dynamic_ncols=True)):
            result = future.result()
            if result == "ok":
                success += 1
            elif result == "exists":
                skipped += 1
            else:
                fail += 1
            if i % 500 == 0 and i > 0:
                print(f"🧾 Progress: {i}/{len(image_links)} | Success: {success} | Skipped: {skipped} | Fail: {fail}")

    end = timer()
    print(f"\n✅ Finished! {success} succeeded, {skipped} skipped, {fail} failed in {end - start:.2f} seconds.\n")

# ===============================
# EXECUTION
# ===============================
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    if "image_link" not in df.columns:
        raise KeyError(f"❌ Required columns not found. Found: {df.columns.tolist()}")

    # Filter to only download missing images
    image_links = []
    for _, row in df.iterrows():
        image_link = row['image_link']
        filename = Path(image_link).name
        if not os.path.exists(os.path.join(SAVE_DIR, filename)):
            image_links.append(image_link)

    print(f"📦 Total images to download (missing): {len(image_links)}")
    print(f"📁 Save directory: {SAVE_DIR}")

    if image_links:
        download_images(image_links, SAVE_DIR, MAX_THREADS)
    else:  
        print("All images already downloaded!")
