"""Download the clean measured SONICOM cohort used by MCAR.

The downloader intentionally fetches one HRTF variant only:

``PXXXX_FreeFieldCompMinPhase_44kHz.sofa``

This variant retains ITD, uses the same 44.1 kHz sampling rate as the
existing HUTUBS experiments, and has already been windowed and free-field
compensated.  Subject selection is derived from the official metadata and
the dated outlier snapshot instead of from directory names alone.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
import urllib.response
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from mcar.paths import project_root


DATASET_BASE_URL = (
    "https://transfer.ic.ac.uk:9090/2022_SONICOM-HRTF-DATASET"
)
VARIANT_NAME = "FreeFieldCompMinPhase_44kHz"
OFFICIAL_PATH_TEMPLATE = (
    "{subject}/HRTF/HRTF/44kHz/"
    "{subject}_FreeFieldCompMinPhase_44kHz.sofa"
)
LOCAL_PATH_TEMPLATE = (
    "subjects/{subject}_FreeFieldCompMinPhase_44kHz.sofa"
)
OUTLIER_SNAPSHOT = "Outliers_2026-05-06.csv"
EXPECTED_METADATA_ROWS = 372
EXPECTED_CLEAN_SUBJECTS = 350
USER_AGENT = "MCAR-SONICOM-downloader/1.0"

METADATA_PATHS = (
    "metadata_and_readme/README.txt",
    "metadata_and_readme/metadata.csv",
    (
        "metadata_and_readme/important_information/"
        "README_SONICOM_OUTLIERS.txt"
    ),
    (
        "metadata_and_readme/important_information/"
        f"{OUTLIER_SNAPSHOT}"
    ),
)


@dataclass(frozen=True)
class SubjectDownload:
    """One selected measured HRTF and its provenance."""

    subject_id: str
    free_field_eq_file: str
    official_relative_path: str
    local_relative_path: str
    source_url: str


@dataclass(frozen=True)
class DownloadResult:
    """Result of checking or downloading one remote file."""

    subject_id: str
    status: str
    bytes_remote: int | None
    local_path: str


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the official clean SONICOM measured-HRTF cohort "
            "for MCA residual learning."
        )
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=(
            project_root()
            / "data"
            / "HRTF"
            / "sonicom_measured_ffcmp_minphase_44k1"
        ),
        help="Dataset output root.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent HTTP requests (default: 8).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=8,
        help="Maximum attempts for each HTTP request (default: 8).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds (default: 120).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Fetch metadata and issue HEAD checks, but do not write any "
            "files."
        ),
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Write metadata and manifests without downloading SOFA files.",
    )
    parser.add_argument(
        "--allow-metadata-drift",
        action="store_true",
        help=(
            "Allow official metadata row/selection counts to differ from "
            f"{EXPECTED_METADATA_ROWS}/{EXPECTED_CLEAN_SUBJECTS}."
        ),
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        metavar="PXXXX",
        help=(
            "Optional clean-subject subset, useful for a pilot download. "
            "Every ID must pass the official selection rules."
        ),
    )
    arguments = parser.parse_args()
    if arguments.workers < 1:
        parser.error("--workers must be at least 1")
    if arguments.retries < 1:
        parser.error("--retries must be at least 1")
    if arguments.timeout <= 0:
        parser.error("--timeout must be positive")
    return arguments


def _request(
    url: str,
    *,
    method: str,
    timeout: float,
    headers: dict[str, str] | None = None,
) -> urllib.response.addinfourl:
    request_headers = {"User-Agent": USER_AGENT}
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(
        url, headers=request_headers, method=method
    )
    context = ssl.create_default_context()
    return urllib.request.urlopen(
        request, timeout=timeout, context=context
    )


def _retry(
    operation,
    *,
    attempts: int,
    description: str,
):
    last_exception: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except (
            OSError,
            TimeoutError,
            urllib.error.HTTPError,
            urllib.error.URLError,
        ) as exception:
            last_exception = exception
            if attempt == attempts:
                break
            wait_seconds = min(30.0, 1.5 ** (attempt - 1))
            print(
                f"Retry {attempt}/{attempts - 1} for {description} "
                f"after {type(exception).__name__}: {exception}",
                file=sys.stderr,
            )
            time.sleep(wait_seconds)
    assert last_exception is not None
    raise RuntimeError(
        f"Failed after {attempts} attempts: {description}"
    ) from last_exception


def fetch_bytes(url: str, *, timeout: float, retries: int) -> bytes:
    def operation() -> bytes:
        with _request(url, method="GET", timeout=timeout) as response:
            return response.read()

    return _retry(
        operation,
        attempts=retries,
        description=f"GET {url}",
    )


def remote_size(
    url: str, *, timeout: float, retries: int
) -> int | None:
    def operation() -> int | None:
        with _request(url, method="HEAD", timeout=timeout) as response:
            value = response.headers.get("Content-Length")
            return int(value) if value is not None else None

    return _retry(
        operation,
        attempts=retries,
        description=f"HEAD {url}",
    )


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(content)
    os.replace(partial, path)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def decode_csv(content: bytes) -> list[dict[str, str]]:
    text = content.decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def build_clean_cohort(
    metadata_content: bytes,
    outlier_content: bytes,
) -> tuple[list[SubjectDownload], list[dict[str, str]]]:
    metadata_rows = decode_csv(metadata_content)
    outlier_rows = decode_csv(outlier_content)
    outliers = {
        row["SubjectID"].strip().upper()
        for row in outlier_rows
        if row.get("HRTF_Outlier", "").strip() == "1"
    }

    selected: list[SubjectDownload] = []
    excluded: list[dict[str, str]] = []
    for row in metadata_rows:
        subject = row.get("Subject", "").strip().upper()
        if not re.fullmatch(r"P\d{4}", subject):
            raise ValueError(f"Unexpected SONICOM subject ID: {subject!r}")

        reasons: list[str] = []
        if row.get("HRTF", "").strip().upper() != "TRUE":
            reasons.append("metadata_hrtf_not_true")
        if subject in outliers:
            reasons.append("official_outlier")
        free_field_eq = row.get("Free Field EQ File", "").strip()
        if free_field_eq == "EQ File Corrupted":
            reasons.append("free_field_eq_corrupted")
        if reasons:
            excluded.append(
                {
                    "subject_id": subject,
                    "reasons": ";".join(reasons),
                }
            )
            continue

        official_path = OFFICIAL_PATH_TEMPLATE.format(subject=subject)
        local_path = LOCAL_PATH_TEMPLATE.format(subject=subject)
        selected.append(
            SubjectDownload(
                subject_id=subject,
                free_field_eq_file=free_field_eq,
                official_relative_path=official_path,
                local_relative_path=local_path,
                source_url=f"{DATASET_BASE_URL}/{official_path}",
            )
        )

    selected.sort(key=lambda item: item.subject_id)
    excluded.sort(key=lambda item: item["subject_id"])
    return selected, excluded


def restrict_subjects(
    cohort: Sequence[SubjectDownload],
    requested_subjects: Sequence[str] | None,
) -> list[SubjectDownload]:
    if not requested_subjects:
        return list(cohort)
    requested = {
        subject.strip().upper() for subject in requested_subjects
    }
    invalid_format = sorted(
        subject
        for subject in requested
        if not re.fullmatch(r"P\d{4}", subject)
    )
    if invalid_format:
        raise ValueError(
            f"Invalid --subjects values: {', '.join(invalid_format)}"
        )
    available = {item.subject_id: item for item in cohort}
    rejected = sorted(requested - available.keys())
    if rejected:
        raise ValueError(
            "Requested subjects do not pass the clean-cohort rules: "
            + ", ".join(rejected)
        )
    return [available[subject] for subject in sorted(requested)]


def validate_snapshot(
    *,
    metadata_row_count: int,
    clean_subject_count: int,
    allow_metadata_drift: bool,
) -> None:
    expected = (EXPECTED_METADATA_ROWS, EXPECTED_CLEAN_SUBJECTS)
    actual = (metadata_row_count, clean_subject_count)
    if actual == expected:
        return
    message = (
        "Official SONICOM metadata changed: expected "
        f"{expected[0]} rows/{expected[1]} clean subjects, found "
        f"{actual[0]}/{actual[1]}. Review the new metadata and outlier "
        "snapshot before changing the frozen experiment cohort."
    )
    if allow_metadata_drift:
        print(f"WARNING: {message}", file=sys.stderr)
    else:
        raise RuntimeError(
            message + " Re-run with --allow-metadata-drift only after review."
        )


def download_one(
    item: SubjectDownload,
    destination: Path,
    *,
    timeout: float,
    retries: int,
    dry_run: bool,
) -> DownloadResult:
    output = destination / item.local_relative_path
    size = remote_size(
        item.source_url, timeout=timeout, retries=retries
    )
    if dry_run:
        return DownloadResult(
            item.subject_id, "available", size, str(output)
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and (
        size is None or output.stat().st_size == size
    ):
        return DownloadResult(
            item.subject_id, "already_complete", size, str(output)
        )

    partial = output.with_name(output.name + ".part")
    partial_size = partial.stat().st_size if partial.is_file() else 0
    if size is not None and partial_size > size:
        partial.unlink()
        partial_size = 0
    if size is not None and partial_size == size and partial_size > 0:
        os.replace(partial, output)
        return DownloadResult(
            item.subject_id, "resumed_complete", size, str(output)
        )

    def operation() -> None:
        start = partial.stat().st_size if partial.is_file() else 0
        headers: dict[str, str] = {}
        if start:
            headers["Range"] = f"bytes={start}-"
        with _request(
            item.source_url,
            method="GET",
            timeout=timeout,
            headers=headers,
        ) as response:
            append = start > 0 and response.status == 206
            mode = "ab" if append else "wb"
            with partial.open(mode) as handle:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    handle.write(block)

    _retry(
        operation,
        attempts=retries,
        description=f"download {item.subject_id}",
    )
    downloaded_size = partial.stat().st_size
    if size is not None and downloaded_size != size:
        raise RuntimeError(
            f"{item.subject_id} size mismatch: downloaded "
            f"{downloaded_size}, expected {size}"
        )
    os.replace(partial, output)
    return DownloadResult(
        item.subject_id, "downloaded", size, str(output)
    )


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)


def write_manifests(
    destination: Path,
    cohort: Sequence[SubjectDownload],
    excluded: Sequence[dict[str, str]],
    metadata_contents: dict[str, bytes],
) -> None:
    manifests = destination / "manifests"
    write_csv(
        manifests / "clean_subjects.csv",
        (
            "subject_id",
            "free_field_eq_file",
            "official_relative_path",
            "local_relative_path",
            "source_url",
        ),
        (asdict(item) for item in cohort),
    )
    write_csv(
        manifests / "excluded_subjects.csv",
        ("subject_id", "reasons"),
        excluded,
    )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_base_url": DATASET_BASE_URL,
        "variant": VARIANT_NAME,
        "selection_rules": {
            "metadata_hrtf": "TRUE",
            "official_outlier": False,
            "free_field_eq_file_not_equal": "EQ File Corrupted",
        },
        "clean_subject_count": len(cohort),
        "excluded_subject_count": len(excluded),
        "metadata_sha256": {
            path: sha256_bytes(content)
            for path, content in sorted(metadata_contents.items())
        },
    }
    atomic_write_bytes(
        manifests / "selection_report.json",
        json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8"),
    )


def format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    return f"{value / (1024 ** 2):.2f} MiB"


def main() -> None:
    arguments = parse_arguments()
    destination = arguments.destination.resolve()

    metadata_contents: dict[str, bytes] = {}
    print("Reading official SONICOM metadata snapshot...")
    for relative_path in METADATA_PATHS:
        url = f"{DATASET_BASE_URL}/{relative_path}"
        content = fetch_bytes(
            url, timeout=arguments.timeout, retries=arguments.retries
        )
        metadata_contents[relative_path] = content
        if not arguments.dry_run:
            atomic_write_bytes(destination / relative_path, content)

    metadata_rows = decode_csv(
        metadata_contents["metadata_and_readme/metadata.csv"]
    )
    clean_cohort, excluded = build_clean_cohort(
        metadata_contents["metadata_and_readme/metadata.csv"],
        metadata_contents[
            "metadata_and_readme/important_information/"
            f"{OUTLIER_SNAPSHOT}"
        ],
    )
    validate_snapshot(
        metadata_row_count=len(metadata_rows),
        clean_subject_count=len(clean_cohort),
        allow_metadata_drift=arguments.allow_metadata_drift,
    )
    selected = restrict_subjects(clean_cohort, arguments.subjects)

    print(
        f"Metadata rows: {len(metadata_rows)}; clean cohort: "
        f"{len(clean_cohort)}; selected for this run: {len(selected)}."
    )
    if not arguments.dry_run:
        write_manifests(
            destination,
            clean_cohort,
            excluded,
            metadata_contents,
        )

    if arguments.metadata_only:
        if arguments.dry_run:
            print("Metadata-only dry run completed; no files were written.")
        else:
            print(f"Metadata and manifests written to {destination}")
        return

    print(
        (
            "Checking remote SOFA files..."
            if arguments.dry_run
            else "Downloading measured SOFA files..."
        )
    )
    results: list[DownloadResult] = []
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        futures = {
            executor.submit(
                download_one,
                item,
                destination,
                timeout=arguments.timeout,
                retries=arguments.retries,
                dry_run=arguments.dry_run,
            ): item
            for item in selected
        }
        completed = 0
        for future in as_completed(futures):
            item = futures[future]
            completed += 1
            try:
                result = future.result()
                results.append(result)
                print(
                    f"[{completed:03d}/{len(selected):03d}] "
                    f"{result.subject_id}: {result.status} "
                    f"({format_bytes(result.bytes_remote)})"
                )
            except Exception as exception:
                failures.append((item.subject_id, str(exception)))
                print(
                    f"[{completed:03d}/{len(selected):03d}] "
                    f"{item.subject_id}: FAILED: {exception}",
                    file=sys.stderr,
                )

    total_remote = sum(
        result.bytes_remote or 0 for result in results
    )
    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[result.status] = (
            status_counts.get(result.status, 0) + 1
        )
    print(
        f"Completed {len(results)}/{len(selected)} files; "
        f"remote total {format_bytes(total_remote)}; "
        f"statuses={status_counts}."
    )
    if failures:
        details = "; ".join(
            f"{subject}: {message}" for subject, message in failures
        )
        raise RuntimeError(
            f"{len(failures)} SONICOM downloads/checks failed: {details}"
        )
    if arguments.dry_run:
        print("Dry run completed; no files were written.")
    else:
        print(f"SONICOM training data are ready under {destination}")


if __name__ == "__main__":
    main()
