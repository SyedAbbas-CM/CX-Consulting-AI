#!/usr/bin/env python
"""Utility for downloading and activating vector database bundles (Chroma collections).
Currently supports Google-Drive direct links and ZIP/TAR archives.

This mirrors the structure used in `model_manager.py` but is greatly simplified.
The script can be imported from FastAPI routes or executed standalone:

    python -m app.scripts.vector_db_manager list
    python -m app.scripts.vector_db_manager download prod-v1
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Optional

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

logger = logging.getLogger("vector_db_manager")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Directory where the live Chroma db lives
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # <workspace>/
VECTOR_DB_DIR = BASE_DIR / "app" / "data" / "vectorstore"
ARCHIVE_DIR = BASE_DIR / "app" / "data" / "vectorstore_archives"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

# Name of small marker file stored inside VECTOR_DB_DIR to indicate which bundle is active
META_FILENAME = ".bundle_id"

# ---------------------------------------------------------------------------
# Register available bundles here. The user can update the gdrive_id later.
# ---------------------------------------------------------------------------
AVAILABLE_VECTOR_DBS: Dict[str, Dict] = {
    "prod-v1": {
        "gdrive_id": "1YtF-r8HV8YRAQvTDC0HH-yzSmXXw5q3H",  # Provided by admin
        "filename": "vectorstore-prod-v1.tar.gz",
        "size_gb": 0.3,  # approx uncompressed size 297 MiB
        "description": "Production vector DB (Chroma) built 2024-06-16",
    },
    # Add more vector DB bundles here as needed
}

# ---------------------------------------------------------------------------

CHUNK_SIZE = 1024 * 1024  # 1 MiB


def _gdrive_url(file_id: str) -> str:
    """Return a direct download url for a public Google-Drive file."""
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def _download(url: str, dest_path: Path) -> None:
    """Download a file from Google-Drive or direct URL.

    If `gdown` is available, prefer it for Google-Drive links (handles the
    anti-virus confirmation step so we don't accidentally save the 2 KB html
    page). Fallback to streaming via requests.
    """

    # Use gdown when possible and the URL points to Google Drive
    if "drive.google.com" in url:
        try:
            import gdown  # type: ignore

            logger.info("Using gdown to fetch Google-Drive file…")

            # gdown expects either the file-id or the full URL
            try:
                gdown.download(url, str(dest_path), quiet=False, resume=True)
            except Exception as ge:
                # gdown sometimes throws at the very end when renaming temp files; capture but don't bail yet
                logger.warning(
                    "gdown raised %s – will validate file on disk anyway.", ge
                )

            # ------------------------------------------------------------------
            # SUCCESS CRITERIA – bail out early if the final file is plausibly OK
            # ------------------------------------------------------------------
            if dest_path.exists():
                size_mb = dest_path.stat().st_size / (1024 * 1024)
                if size_mb > 5:  # Anything smaller is likely an HTML interstitial
                    logger.info(
                        "gdown finished – archive size %.1f MB (treated as success).",
                        size_mb,
                    )
                    return  # ✅  We're done.

            # If we reach here the file is missing or too small – fall back to requests
            logger.warning(
                "gdown did not yield a valid archive (<5 MB); falling back to raw requests download…"
            )
        except ImportError:
            logger.warning(
                "gdown not installed – falling back to raw requests download (may fail for large Drive files). Run `pip install gdown` for robust behaviour."
            )
        except Exception as e:  # Any gdown problem – fallback to requests
            logger.warning("gdown failed (%s). Falling back to requests download.", e)

    # Fallback: basic streaming download
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with tqdm(
            total=total if total else None,
            unit="B",
            unit_scale=True,
            desc=dest_path.name,
        ) as pbar:
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    # sanity-check for html warning page
    if dest_path.stat().st_size < 1024 * 1024:  # <1 MiB → suspicious
        logger.error(
            "Downloaded file appears too small (%d bytes). The request probably fetched an interstitial HTML page instead of the archive. Re-download with gdown or ensure the link is public.",
            dest_path.stat().st_size,
        )
        raise RuntimeError(
            "Vector DB download failed – received unexpected small file."
        )


def _extract_archive(archive_path: Path, extract_to: Path) -> None:
    """Extract ZIP or TAR archive to the specified directory."""
    logger.info(f"Extracting {archive_path} to {extract_to}")

    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.suffix.lower() in [".tar", ".gz"] or archive_path.name.endswith(
        ".tar.gz"
    ):
        with tarfile.open(archive_path, "r:*") as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")

    logger.info(f"Successfully extracted {archive_path}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dir_has_files(path: Path) -> bool:
    """Return True if directory exists *and* contains at least one file."""
    return path.exists() and any(path.iterdir())


def _write_active_marker(db_id: str):
    """Write a tiny text file into VECTOR_DB_DIR to mark which bundle is active."""
    try:
        marker_path = VECTOR_DB_DIR / META_FILENAME
        marker_path.write_text(db_id, encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to write %s marker: %s", META_FILENAME, e)


def _read_active_marker() -> Optional[str]:
    marker_path = VECTOR_DB_DIR / META_FILENAME
    if marker_path.exists():
        try:
            return marker_path.read_text(encoding="utf-8").strip()
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Public helpers used by API routes
# ---------------------------------------------------------------------------


def get_available_vector_dbs() -> Dict[str, Dict]:
    return AVAILABLE_VECTOR_DBS


def _archive_path(filename: str) -> Path:
    return ARCHIVE_DIR / filename


def get_vector_db_status(db_id: str) -> Dict[str, str]:
    info = AVAILABLE_VECTOR_DBS.get(db_id)
    if not info:
        return {"status": "not_found", "message": f"Unknown db id {db_id}"}

    archive = _archive_path(info["filename"])
    live_dir = VECTOR_DB_DIR

    marker_db_id = _read_active_marker()
    live_has_files = _dir_has_files(live_dir)

    if marker_db_id == db_id and live_has_files:
        status = "active"
    elif archive.exists() and not live_has_files:
        status = "downloaded_not_active"
    else:
        status = "not_downloaded"

    return {
        "status": status,
        "archive": str(archive),
        "live_dir": str(live_dir),
        "marker_id": marker_db_id,
    }


def download_vector_db(db_id: str, force: bool = False) -> bool:
    info = AVAILABLE_VECTOR_DBS.get(db_id)
    if not info:
        logger.error("Unknown vector db id %s", db_id)
        return False

    archive = _archive_path(info["filename"])
    if archive.exists() and not force:
        logger.info(
            "Archive already downloaded – skipping. Use force=True to redownload."
        )
    else:
        url = _gdrive_url(info["gdrive_id"])
        logger.info("Downloading vector DB from %s", url)
        _download(url, archive)

    # Remove existing vectorstore directory if it exists
    if VECTOR_DB_DIR.exists():
        logger.info("Removing existing vectorstore directory")
        shutil.rmtree(VECTOR_DB_DIR)

    # Extract / activate - extract to the parent directory so vectorstore/ is created
    logger.info("Extracting archive %s", archive)
    _extract_archive(archive, VECTOR_DB_DIR.parent)

    # Write marker file to record which bundle is active
    _write_active_marker(db_id)

    logger.info("Vector DB '%s' activated at %s", db_id, VECTOR_DB_DIR)
    return True


# ---------------------------------------------------------------------------
# CLI utility (optional)
# ---------------------------------------------------------------------------


def _cli():
    parser = argparse.ArgumentParser("vector_db_manager")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List available bundles")
    dl = sub.add_parser("download", help="Download and activate a bundle")
    dl.add_argument("db_id")
    dl.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.cmd == "list":
        for k, v in AVAILABLE_VECTOR_DBS.items():
            st = get_vector_db_status(k)["status"]
            print(f"{k:<10} {st:<20} {v['description']}")
    elif args.cmd == "download":
        download_vector_db(args.db_id, force=args.force)
    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
