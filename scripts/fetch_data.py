"""Fetch the Heart Disease Risk 2026 dataset and verify its integrity.

The copy of ``heart_disease_risk_2026.csv`` originally committed to this
repository was corrupted (it arrived mangled through a file-sharing step). It is
not a CSV at all: only 6.8% of its bytes are printable, it contains 782,680 NUL
bytes, and it holds fragments of unrelated BMP data.

The file is recoverable because its byte length matches the genuine Kaggle
release exactly, so this script downloads that release, verifies it by SHA-256,
and writes it where the pipeline expects to find it.

Usage:
    python scripts/fetch_data.py
    python scripts/fetch_data.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

DATASET_SLUG = "uditjain13/heart-disease-risk-2026"
DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_SLUG}"
ARCHIVE_MEMBER = "heart_disease_risk_2026.csv"

# Pinned identity of version 3, the release this project was built against.
EXPECTED_CSV_SHA256 = "f30b84b747c1d26382cb57af138d3422a3f0a10d21a70efefba9282b55fe77fa"
EXPECTED_CSV_BYTES = 1_137_455

REPO_ROOT = Path(__file__).resolve().parents[1]
DESTINATION = REPO_ROOT / "data" / "raw" / ARCHIVE_MEMBER


class IntegrityError(RuntimeError):
    """Raised when downloaded bytes do not match the pinned release."""


def download_archive(url: str = DOWNLOAD_URL, timeout: int = 120) -> bytes:
    """Download the Kaggle dataset archive into memory."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def extract_member(archive: bytes, member: str = ARCHIVE_MEMBER) -> bytes:
    """Return one member of a ZIP archive held in memory."""
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        try:
            return bundle.read(member)
        except KeyError as exc:
            raise IntegrityError(
                f"archive does not contain {member!r}; found {bundle.namelist()}"
            ) from exc


def verify(payload: bytes, expected_sha256: str, expected_bytes: int) -> str:
    """Fail loudly unless the payload matches the pinned release."""
    digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != expected_bytes:
        raise IntegrityError(
            f"expected {expected_bytes} bytes, received {len(payload)}"
        )
    if digest != expected_sha256:
        raise IntegrityError(
            f"sha256 mismatch: expected {expected_sha256}, received {digest}"
        )
    return digest


def fetch(destination: Path = DESTINATION, force: bool = False) -> Path:
    """Download, verify and write the dataset. Returns the written path."""
    if destination.exists() and not force:
        existing = destination.read_bytes()
        try:
            verify(existing, EXPECTED_CSV_SHA256, EXPECTED_CSV_BYTES)
        except IntegrityError as exc:
            raise IntegrityError(
                f"{destination} exists but is not the pinned release ({exc}). "
                "Re-run with --force to replace it."
            ) from exc
        print(f"already verified: {destination}")
        return destination

    print(f"downloading {DOWNLOAD_URL}")
    payload = extract_member(download_archive())
    digest = verify(payload, EXPECTED_CSV_SHA256, EXPECTED_CSV_BYTES)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    print(f"wrote {destination} ({len(payload)} bytes, sha256 {digest})")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing file even if it fails verification",
    )
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    args = parser.parse_args(argv)

    try:
        fetch(destination=args.destination, force=args.force)
    except (IntegrityError, urllib.error.URLError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
