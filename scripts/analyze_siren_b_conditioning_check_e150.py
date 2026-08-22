"""Aggregate the symmetric E150 Stage B conditioning check."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from mcar.paths import project_root


SEEDS = (20260821, 20260822, 20260823)
KINDS = ("conditioned", "unconditional")
PROTOCOL = "experiments/film_siren/STAGE_B_CONDITIONING_CHECK_E150_AMENDMENT.md"
EARLY_CYCLES = (100, 105, 110, 115, 120, 125)
LATE_CYCLES = (130, 135, 140, 145, 150)


def run_name(kind: str, seed: int) -> str:
    if kind == "conditioned":
        return f"sonicom_film_siren_b_conditioning_check_all_seed{seed}_e150"
    if kind == "unconditional":
        return f"sonicom_shared_siren_b_unconditional_seed{seed}_e150"
    raise ValueError(f"Unknown kind {kind}")


def plateau_decision(
    best_cycle: int, validation_by_cycle: dict[int, float]
) -> tuple[str, float | None]:
    if best_cycle < 150:
        return "KEEP", None
    if best_cycle != 150:
        raise ValueError(f"Invalid E150 best cycle {best_cycle}")
    required = set(EARLY_CYCLES + LATE_CYCLES)
    if not required.issubset(validation_by_cycle):
        raise ValueError("Missing E150 plateau-window validation points")
    early = float(np.mean([validation_by_cycle[cycle] for cycle in EARLY_CYCLES]))
    late = float(np.mean([validation_by_cycle[cycle] for cycle in LATE_CYCLES]))
    improvement = float(100.0 * (early - late) / early)
    return ("KEEP_PLATEAU" if improvement < 0.1 else "RETEST"), improvement


def read_run(root: Path, kind: str, seed: int) -> dict[str, object] | None:
    name = run_name(kind, seed)
    directory = root / "artifacts" / "training" / name
    report_path = directory / "training_report.json"
    subject_path = directory / "best_validation_per_subject.csv"
    ledger_path = directory / "validation_ledger.json"
    if not report_path.is_file():
        return None
    if not subject_path.is_file() or not ledger_path.is_file():
        raise ValueError(f"Missing E150 result files for {name}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "completed" or report.get("cycles") != 150:
        raise ValueError(f"Incomplete or wrong-budget report for {name}")
    if report.get("run_name") != name or report.get("seed") != seed:
        raise ValueError(f"Run metadata mismatch for {name}")
    if report.get("test_subjects_read") != 0 or bool(report.get("git", {}).get("dirty")):
        raise ValueError(f"Unsafe data/Git state for {name}")
    if kind == "unconditional" and report.get("condition_inputs_read") != 0:
        raise ValueError(f"Unconditional run read condition input: {name}")
    with subject_path.open("r", encoding="utf-8", newline="") as handle:
        subjects = list(csv.DictReader(handle))
    if len(subjects) != 44:
        raise ValueError(f"Expected 44 validation rows for {name}")
    mae = np.asarray([float(row["weighted_mae_db"]) for row in subjects])
    if not np.isfinite(mae).all():
        raise ValueError(f"Non-finite validation result for {name}")
    aggregate = float(np.mean(mae))
    if not np.isclose(aggregate, report["best_validation_weighted_mae_db"], atol=1e-10):
        raise ValueError(f"Aggregate mismatch for {name}")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    validation_by_cycle = {
        int(item["cycle"]): float(item["aggregate"]["aggregate_mae_db"])
        for item in ledger
    }
    if len(validation_by_cycle) != 30 or not np.isfinite(list(validation_by_cycle.values())).all():
        raise ValueError(f"Expected 30 finite validation points for {name}")
    budget_decision, window_improvement = plateau_decision(
        int(report["best_cycle"]), validation_by_cycle
    )
    return {
        "kind": kind,
        "seed": seed,
        "run_name": name,
        "best_cycle": int(report["best_cycle"]),
        "weighted_mae_db": aggregate,
        "budget_decision": budget_decision,
        "plateau_window_improvement_percent": window_improvement,
        "condition_inputs_read": 0 if kind == "unconditional" else "not_applicable",
        "test_subjects_read": 0,
        "git_commit": str(report["git"]["commit"]),
        "git_dirty": False,
    }


def select_status(
    conditioned_mean: float,
    unconditional_mean: float,
    *,
    complete: bool,
    budget_retest: bool,
) -> tuple[bool | None, str]:
    if not complete or budget_retest:
        return None, "PENDING"
    passed = bool(conditioned_mean < unconditional_mean)
    return passed, "CONDITIONING_CHECK_PASSED" if passed else "DO_NOT_FREEZE"


def analyze(root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = [
        row
        for kind in KINDS
        for seed in SEEDS
        if (row := read_run(root, kind, seed)) is not None
    ]
    complete = len(rows) == 6 and {
        (str(row["kind"]), int(row["seed"])) for row in rows
    } == {(kind, seed) for kind in KINDS for seed in SEEDS}
    budget_retest = any(row["budget_decision"] == "RETEST" for row in rows)
    grouped = {
        kind: np.asarray(
            [float(row["weighted_mae_db"]) for row in rows if row["kind"] == kind]
        )
        for kind in KINDS
    }
    conditioned_mean = (
        float(np.mean(grouped["conditioned"])) if grouped["conditioned"].size == 3 else None
    )
    unconditional_mean = (
        float(np.mean(grouped["unconditional"]))
        if grouped["unconditional"].size == 3
        else None
    )
    if conditioned_mean is not None and unconditional_mean is not None:
        passed, status = select_status(
            conditioned_mean,
            unconditional_mean,
            complete=complete,
            budget_retest=budget_retest,
        )
    else:
        passed, status = None, "PENDING"
    paired = []
    if complete:
        lookup = {
            (str(row["kind"]), int(row["seed"])): float(row["weighted_mae_db"])
            for row in rows
        }
        paired = [
            {
                "seed": seed,
                "conditioned_minus_unconditional_mae_db": lookup[("conditioned", seed)]
                - lookup[("unconditional", seed)],
            }
            for seed in SEEDS
        ]
    decision: dict[str, object] = {
        "protocol": PROTOCOL,
        "seeds": list(SEEDS),
        "completed_runs": len(rows),
        "expected_runs": 6,
        "comparison_complete": complete,
        "budget_retest_required": budget_retest,
        "conditioned_mean_weighted_mae_db": conditioned_mean,
        "conditioned_std_seed_mae_db": (
            float(np.std(grouped["conditioned"], ddof=0)) if complete else None
        ),
        "unconditional_mean_weighted_mae_db": unconditional_mean,
        "unconditional_std_seed_mae_db": (
            float(np.std(grouped["unconditional"], ddof=0)) if complete else None
        ),
        "conditioned_relative_improvement_percent": (
            float(100.0 * (1.0 - conditioned_mean / unconditional_mean))
            if complete and unconditional_mean != 0.0
            else None
        ),
        "paired_deltas": paired,
        "conditioned_seed_wins": (
            sum(float(item["conditioned_minus_unconditional_mae_db"]) < 0.0 for item in paired)
            if complete
            else None
        ),
        "conditioning_check_passed": passed,
        "stage_b_status": status,
        "test_subjects_read": 0,
    }
    rows.sort(key=lambda row: (str(row["kind"]), int(row["seed"])))
    return rows, decision


def main() -> None:
    root = project_root()
    rows, decision = analyze(root)
    output_dir = root / "results" / "sonicom_siren_b_conditioning_check_e150"
    output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
