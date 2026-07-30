"""Numerical and gradient checks for strict HRIR-energy ILD loss."""

from __future__ import annotations

import torch

from mcar.losses import strict_hrir_ild_mae


def main() -> None:
    ear_count = 2
    direction_count = 3
    selected_count = 3
    single_sided_count = 5
    corrected_db = torch.tensor(
        [
            [[0.0, -3.0, -6.0]] * direction_count,
            [[-2.0, -4.0, -8.0]] * direction_count,
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    phase = torch.zeros(
        ear_count, direction_count, selected_count, dtype=torch.float32
    )
    outside_real = torch.tensor(
        [
            [[0.5, 0.25]] * direction_count,
            [[0.4, 0.20]] * direction_count,
        ],
        dtype=torch.float32,
    )
    outside_imag = torch.zeros_like(outside_real)
    selected_indices = torch.tensor([1, 2, 3], dtype=torch.int64)
    outside_indices = torch.tensor([0, 4], dtype=torch.int64)

    magnitude = torch.pow(10.0, corrected_db.detach() / 20.0)
    selected_complex = torch.polar(magnitude, phase)
    single_sided = torch.zeros(
        ear_count,
        direction_count,
        single_sided_count,
        dtype=torch.complex64,
    )
    single_sided = single_sided.index_copy(
        -1,
        outside_indices,
        torch.complex(outside_real, outside_imag),
    )
    single_sided = single_sided.index_copy(
        -1, selected_indices, selected_complex
    )
    both_sided = torch.cat(
        (
            single_sided,
            torch.conj(torch.flip(single_sided[..., 1:-1], dims=(-1,))),
        ),
        dim=-1,
    )
    hrir = torch.fft.ifft(both_sided, dim=-1).real
    energy = torch.sum(hrir.square(), dim=-1)
    reference_ild = 10.0 * torch.log10(energy[0] / energy[1])
    metadata: dict[str, torch.Tensor | int] = {
        "mca_selected_phase_rad": phase,
        "mca_outside_real": outside_real,
        "mca_outside_imag": outside_imag,
        "selected_bin_indices_zero_based": selected_indices,
        "outside_bin_indices_zero_based": outside_indices,
        "reference_ild_db": reference_ild,
        "single_sided_frequency_count": single_sided_count,
        "hrir_length": both_sided.shape[-1],
    }
    direction_weights = torch.tensor([0.2, 0.3, 0.5])
    zero_loss = strict_hrir_ild_mae(
        corrected_db, direction_weights, metadata
    )
    if float(zero_loss.detach()) > 1e-6:
        raise AssertionError(f"Expected zero strict ILD loss, got {zero_loss}")

    perturbed = corrected_db.detach().clone()
    perturbed[0, :, 1] += 1.0
    perturbed.requires_grad_(True)
    loss = strict_hrir_ild_mae(perturbed, direction_weights, metadata)
    if not bool(loss > 0):
        raise AssertionError("Perturbation did not increase strict ILD loss")
    loss.backward()
    if perturbed.grad is None or not bool(torch.all(torch.isfinite(perturbed.grad))):
        raise AssertionError("Strict ILD gradient is missing or non-finite")
    print(
        {
            "status": "passed",
            "zero_loss_db": float(zero_loss.detach()),
            "perturbed_loss_db": float(loss.detach()),
            "gradient_abs_max": float(torch.max(torch.abs(perturbed.grad))),
        }
    )


if __name__ == "__main__":
    main()
