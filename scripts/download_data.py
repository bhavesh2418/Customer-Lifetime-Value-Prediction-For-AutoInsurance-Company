"""
Download the vehicle insurance customer dataset from Kaggle.
Requires KAGGLE_USERNAME and KAGGLE_KEY in the parent project .env.
"""
import os
import sys
import zipfile
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the auto_md_file project (where credentials live)
ENV_PATH = Path(__file__).resolve().parents[2] / "auto_md_file" / ".env"
if not ENV_PATH.exists():
    # fallback: try same directory
    ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)

KAGGLE_DATASET = "ranja7/vehicle-insurance-customer-data"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def download():
    import kaggle  # imported after env is loaded so auth picks up credentials
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {KAGGLE_DATASET} ...")
    kaggle.api.dataset_download_files(KAGGLE_DATASET, path=str(RAW_DIR), unzip=True)
    print(f"Done. Files in {RAW_DIR}:")
    for f in RAW_DIR.iterdir():
        print(f"  {f.name}  ({f.stat().st_size:,} bytes)")


if __name__ == "__main__":
    download()
