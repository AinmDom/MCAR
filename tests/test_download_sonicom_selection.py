"""Offline selection test for the SONICOM downloader."""

from __future__ import annotations

from mcar.data_tools.download_sonicom import (
    build_clean_cohort,
    restrict_subjects,
)


METADATA = b"""\
Subject,HRTF,Free Field EQ File
P0001,TRUE,reference_eq_001
P0002,FALSE,reference_eq_001
P0003,TRUE,reference_eq_001
P0004,TRUE,EQ File Corrupted
P0005,TRUE,reference_eq_002
"""

OUTLIERS = b"""\
SubjectID,HRTF_Outlier
P0003,1
"""


def main() -> None:
    clean, excluded = build_clean_cohort(METADATA, OUTLIERS)
    assert [item.subject_id for item in clean] == ["P0001", "P0005"]
    reasons = {
        item["subject_id"]: item["reasons"] for item in excluded
    }
    assert reasons == {
        "P0002": "metadata_hrtf_not_true",
        "P0003": "official_outlier",
        "P0004": "free_field_eq_corrupted",
    }
    assert [
        item.subject_id
        for item in restrict_subjects(clean, ["p0005"])
    ] == ["P0005"]
    try:
        restrict_subjects(clean, ["P0003"])
    except ValueError as exception:
        assert "do not pass" in str(exception)
    else:
        raise AssertionError("Excluded subject was unexpectedly accepted")
    print(
        {
            "status": "passed",
            "clean_subjects": len(clean),
            "excluded_subjects": len(excluded),
        }
    )


if __name__ == "__main__":
    main()
