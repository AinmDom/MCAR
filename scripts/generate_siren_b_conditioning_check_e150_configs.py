"""Generate symmetric E150 configs for the Stage B conditioning check."""

from __future__ import annotations

import copy
import json

from mcar.paths import project_root


SEEDS = (20260821, 20260822, 20260823)
AMENDMENT = "experiments/film_siren/STAGE_B_CONDITIONING_CHECK_E150_AMENDMENT.md"


def main() -> None:
    root = project_root()
    decision = json.loads(
        (
            root / "results" / "sonicom_shared_siren_b_unconditional" / "decision.json"
        ).read_text(encoding="utf-8")
    )
    if not (
        decision.get("comparison_complete") is True
        and decision.get("budget_retest_required") is True
        and decision.get("conditioning_check_passed") is None
    ):
        raise ValueError(f"Symmetric E150 conditioning check is not authorized: {decision}")

    outputs: list[str] = []
    for seed in SEEDS:
        conditioned_base = json.loads(
            (
                root
                / "configs"
                / "experiments"
                / f"sonicom_film_siren_b_placement_all_seed{seed}.json"
            ).read_text(encoding="utf-8")
        )
        conditioned = copy.deepcopy(conditioned_base)
        conditioned["experiment_id"] = f"SIREN-B-CONDITIONING-CHECK-ALL-E150-SEED-{seed}"
        conditioned["model_version"] = f"film-siren-b-full-all-e150-seed-{seed}-v1"
        conditioned["search_stage"] = "conditioning_check_e150"
        conditioned["protocol_amendment"] = AMENDMENT
        conditioned["cycles"] = 150
        conditioned["run_name"] = (
            f"sonicom_film_siren_b_conditioning_check_all_seed{seed}_e150"
        )
        conditioned_output = (
            root
            / "configs"
            / "experiments"
            / f"sonicom_film_siren_b_conditioning_check_all_seed{seed}_e150.json"
        )
        conditioned_output.write_text(
            json.dumps(conditioned, indent=2) + "\n", encoding="utf-8"
        )
        outputs.append(str(conditioned_output))

        baseline_base = json.loads(
            (
                root
                / "configs"
                / "experiments"
                / f"sonicom_shared_siren_b_unconditional_seed{seed}.json"
            ).read_text(encoding="utf-8")
        )
        baseline = copy.deepcopy(baseline_base)
        baseline["experiment_id"] = f"SIREN-B-UNCONDITIONAL-E150-SEED-{seed}"
        baseline["model_version"] = f"shared-siren-b-unconditional-e150-seed-{seed}-v1"
        baseline["search_stage"] = "conditioning_check_e150"
        baseline["protocol_amendment"] = AMENDMENT
        baseline["cycles"] = 150
        baseline["run_name"] = f"sonicom_shared_siren_b_unconditional_seed{seed}_e150"
        baseline_output = (
            root
            / "configs"
            / "experiments"
            / f"sonicom_shared_siren_b_unconditional_seed{seed}_e150.json"
        )
        baseline_output.write_text(
            json.dumps(baseline, indent=2) + "\n", encoding="utf-8"
        )
        outputs.append(str(baseline_output))
    print(json.dumps({"generated": outputs}, indent=2))


if __name__ == "__main__":
    main()
