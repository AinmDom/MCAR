"""Acquire and inventory the frozen ARI in-ear `hrtf b_nh` cohort.

The official ARI directory mixes HRTFs/DTFs and several measurement variants.
This script intentionally admits only ``hrtf b_nh*.sofa``: human, in-ear,
semi-anechoic-room HRTFs with the common 50 Hz--18 kHz equalization variant.
It never reads ``Data.IR`` values; SOFA dataset shapes and attributes are
inspected so that the split can be frozen before any test waveform access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import ssl
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import h5py
import certifi


ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://sofacoustics.org/data/database/ari/"
LICENSE_NOTE = (
    "No machine-readable licence was published in the official directory or "
    "ARI database page when acquired. Retain OEAW/ARI attribution; this "
    "experiment uses the publicly downloadable research data only."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str) -> tuple[bytes, dict[str, str]]:
    request = urllib.request.Request(url, headers={"User-Agent": "MCAR-ARI-replication/1.0"})
    # The host machine's Windows certificate store can be unavailable in a
    # headless worker.  Certifi still verifies the server certificate, unlike
    # disabling TLS verification.
    context = ssl.create_default_context(cafile=certifi.where())
    error: Exception | None = None
    for attempt in range(1, 6):
        try:
            with urllib.request.urlopen(request, timeout=120, context=context) as response:
                return response.read(), {key: value for key, value in response.headers.items()}
        except Exception as exc:  # network transfers can be truncated by the host
            error = exc
            if attempt == 5:
                break
            time.sleep(attempt * 2)
    raise RuntimeError(f"Failed to fetch after five verified-TLS attempts: {url}") from error


def remote_files() -> list[str]:
    listing, _ = fetch(SOURCE_URL)
    # The official Apache index URL-encodes the embedded space. The second
    # form makes this robust to a future server-side HTML formatting change.
    names = re.findall(r'href="(hrtf(?:%20| )b_nh\d+\.sofa)"', listing.decode("utf-8"))
    normalized = sorted({name.replace(" ", "%20") for name in names}, key=lambda x: int(re.search(r"nh(\d+)", x).group(1)))
    if len(normalized) < 80:
        raise RuntimeError(f"Unexpectedly small ARI hrtf-b cohort: {len(normalized)} files")
    return normalized


def sofa_metadata(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        if "Data.IR" not in handle or "SourcePosition" not in handle:
            raise ValueError(f"Missing required SOFA datasets: {path}")
        ir = handle["Data.IR"]
        source = handle["SourcePosition"]
        convention = handle.attrs.get("SOFAConventions", "")
        if isinstance(convention, bytes):
            convention = convention.decode("utf-8")
        shape = tuple(int(v) for v in ir.shape)
        source_shape = tuple(int(v) for v in source.shape)
        sampling_rate = handle["Data.SamplingRate"]
        sampling_shape = tuple(int(v) for v in sampling_rate.shape)
    eligible = shape == (1550, 2, 256) and source_shape == (1550, 3)
    return {
        "data_ir_shape": list(shape),
        "source_position_shape": list(source_shape),
        "sampling_rate_shape": list(sampling_shape),
        "sofa_conventions": str(convention),
        "eligible_metadata": eligible,
        "exclusion_reason": "" if eligible else "unexpected SOFA dimensions",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=ROOT / "data" / "raw" / "ari_hrtf_b_nh_v20260911")
    parser.add_argument("--inventory", type=Path, default=ROOT / "configs" / "data" / "ari_hrtf_b_nh_source_inventory_v1.json")
    args = parser.parse_args()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    acquired_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    rows: list[dict[str, object]] = []
    for number, remote_name in enumerate(remote_files(), start=1):
        filename = remote_name.replace("%20", " ")
        local = destination / filename
        url = SOURCE_URL + remote_name
        headers: dict[str, str] = {}
        if not local.is_file():
            payload, headers = fetch(url)
            temporary = local.with_suffix(".partial")
            temporary.write_bytes(payload)
            temporary.replace(local)
        metadata = sofa_metadata(local)
        subject_match = re.search(r"nh(\d+)", filename)
        if subject_match is None:
            raise ValueError(f"Unable to parse ARI subject ID: {filename}")
        subject = f"nh{subject_match.group(1)}"
        rows.append({
            "subject_id": subject,
            "filename": filename,
            "url": url,
            "bytes": local.stat().st_size,
            "sha256": sha256(local),
            "http_last_modified": headers.get("Last-Modified", "not-requested-existing-file"),
            **metadata,
        })
        print(f"inventory [{number}/{len(rows) if False else '?'}] {filename}", flush=True)
    if len({row["subject_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate subject IDs")
    payload = {
        "schema_version": "1.0",
        "acquired_at": acquired_at,
        "source_url": SOURCE_URL,
        "source_variant": "hrtf b_nh*.sofa",
        "source_description": "ARI measured in-ear semi-anechoic-room HRTFs; b variant only",
        "selection_excludes": ["dtf*", "hrtf_nh*", "hrtf c*", "simulated", "behind-the-ear", "LAS"],
        "licence_information": LICENSE_NOTE,
        "destination": str(destination),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "all_metadata_eligible": all(bool(row["eligible_metadata"]) for row in rows),
        "files": rows,
    }
    args.inventory.parent.mkdir(parents=True, exist_ok=True)
    args.inventory.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("file_count", "total_bytes", "all_metadata_eligible")}, indent=2))


if __name__ == "__main__":
    main()
