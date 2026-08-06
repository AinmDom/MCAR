"""Signal processing used by the SONICOM FSP-AE baseline."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def _require_torchaudio():
    try:
        import torchaudio
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "FSP-AE ITD processing requires a torchaudio build matching torch; "
            "install the project with the 'fsp-ae' optional dependency"
        ) from error
    return torchaudio


def magnitude_db_from_hrir(
    hrir: torch.Tensor,
    nfft: int = 1024,
    top_db: float = 80.0,
) -> torch.Tensor:
    """Return non-DC positive-frequency magnitudes in dB.

    Args:
        hrir: Tensor shaped ``[direction, ear=2, sample]``.
        nfft: Even FFT size. For SONICOM, 1024 preserves the frozen metric grid.
        top_db: Published FSP-AE dynamic-range clipping value.
    """
    if hrir.ndim != 3 or hrir.shape[1] != 2:
        raise ValueError("hrir must have shape [direction, 2, sample]")
    if nfft % 2 != 0 or nfft < hrir.shape[-1]:
        raise ValueError("nfft must be even and no shorter than the HRIR")
    if top_db <= 0:
        raise ValueError("top_db must be positive")
    spectrum = torch.fft.rfft(hrir.to(torch.float32), n=nfft, dim=-1)
    magnitude = torch.abs(spectrum[..., 1:])
    magnitude_db = 20.0 * torch.log10(torch.clamp(magnitude, min=1e-10))
    magnitude_db = torch.clamp(magnitude_db, min=magnitude_db.max() - top_db)
    return magnitude_db


def positive_frequency_hz(
    sampling_rate_hz: float, nfft: int = 1024
) -> torch.Tensor:
    if sampling_rate_hz <= 0 or nfft <= 0 or nfft % 2 != 0:
        raise ValueError("sampling rate and even nfft must be positive")
    return (
        torch.arange(1, nfft // 2 + 1, dtype=torch.float32)
        * float(sampling_rate_hz)
        / nfft
    )


def estimate_itd_seconds(
    hrir: torch.Tensor,
    sampling_rate_hz: float,
    upsampled_rate_hz: float = 384_000.0,
    lowpass_hz: float = 1_600.0,
    maximum_itd_seconds: float = 1e-3,
    direction_batch_size: int = 64,
) -> torch.Tensor:
    """Estimate ITD using the published low-pass/cross-correlation procedure."""
    if hrir.ndim != 3 or hrir.shape[1] != 2:
        raise ValueError("hrir must have shape [direction, 2, sample]")
    if direction_batch_size <= 0:
        raise ValueError("direction_batch_size must be positive")
    torchaudio = _require_torchaudio()
    estimates: list[torch.Tensor] = []
    threshold_index = round(upsampled_rate_hz * maximum_itd_seconds)
    for start in range(0, hrir.shape[0], direction_batch_size):
        batch = hrir[start : start + direction_batch_size].to(torch.float32)
        batch = torchaudio.functional.lowpass_biquad(
            batch, sample_rate=sampling_rate_hz, cutoff_freq=lowpass_hz
        )
        batch = torchaudio.functional.resample(
            batch, orig_freq=sampling_rate_hz, new_freq=upsampled_rate_hz
        )
        direction_count, _, sample_count = batch.shape
        left = batch[:, 0]
        right = batch[:, 1]
        padded_left = functional.pad(left, (sample_count, sample_count))
        correlation = functional.conv1d(
            padded_left.reshape(1, direction_count, -1),
            right.reshape(direction_count, 1, -1),
            groups=direction_count,
        ).reshape(direction_count, -1)
        begin = sample_count - threshold_index
        end = sample_count + threshold_index + 1
        maximum = torch.argmax(correlation[:, begin:end], dim=-1)
        estimates.append(
            (maximum - threshold_index).to(torch.float32)
            / upsampled_rate_hz
        )
    return torch.cat(estimates, dim=0)


def _minimum_phase_hrir(magnitude_db: torch.Tensor) -> torch.Tensor:
    """Reconstruct an oversized minimum-phase HRIR from bins 1..Nyquist."""
    if magnitude_db.ndim != 4 or magnitude_db.shape[2] != 2:
        raise ValueError("magnitude_db must have shape [S, B, 2, L]")
    magnitude = torch.pow(10.0, magnitude_db / 20.0)
    negative = torch.flip(magnitude[..., :-1], dims=(-1,))
    full_magnitude = torch.cat(
        (torch.ones_like(magnitude[..., 0:1]), magnitude, negative), dim=-1
    )
    log_magnitude = torch.log(torch.clamp(full_magnitude, min=1e-10))
    transform = torch.fft.fft(log_magnitude, dim=-1)
    sample_count = transform.shape[-1]
    transform[..., 1 : sample_count // 2] *= -1j
    transform[..., (sample_count + 2) // 2 + 1 :] *= 1j
    transform[..., 0] = 0
    if sample_count % 2 == 0:
        transform[..., sample_count // 2] = 0
    phase = -torch.fft.ifft(transform, dim=-1)
    return torch.real(
        torch.fft.ifft(
            full_magnitude * torch.exp(1j * phase), dim=-1
        )
    )


def assign_itd_seconds(
    hrir: torch.Tensor,
    original_itd_seconds: torch.Tensor,
    desired_itd_seconds: torch.Tensor,
    sampling_rate_hz: float,
    common_delay_seconds: float = 1e-3,
) -> torch.Tensor:
    """Apply the FSP-AE symmetric integer-sample ITD adjustment."""
    subject_count, direction_count = original_itd_seconds.shape
    if desired_itd_seconds.shape != (subject_count, direction_count):
        raise ValueError("desired ITD shape mismatch")
    sample_count = hrir.shape[-1]
    common_delay_samples = common_delay_seconds * sampling_rate_hz
    half_difference = (
        (desired_itd_seconds - original_itd_seconds) * sampling_rate_hz / 2.0
    )
    offsets = torch.full(
        (subject_count, direction_count, 2),
        common_delay_samples,
        device=hrir.device,
    )
    offsets[:, :, 0] += half_difference
    offsets[:, :, 1] -= half_difference
    offsets = torch.round(offsets).to(torch.int64)
    indices = torch.arange(sample_count, device=hrir.device).reshape(
        1, 1, 1, sample_count
    )
    indices = (indices - offsets[:, :, :, None]) % sample_count
    window_length = int(sample_count - common_delay_samples)
    window = torch.cat(
        (
            torch.ones(window_length, device=hrir.device),
            torch.zeros(sample_count - window_length, device=hrir.device),
        )
    )
    return torch.gather(hrir * window, -1, indices)


def reconstruct_hrir_with_itd(
    magnitude_db: torch.Tensor,
    itd_seconds: torch.Tensor,
    sampling_rate_hz: float = 44_100.0,
    upsampled_rate_hz: float = 384_000.0,
) -> torch.Tensor:
    """Reconstruct minimum-phase HRIRs and impose predicted ITDs."""
    minimum_phase = _minimum_phase_hrir(magnitude_db)
    subject_count, direction_count, _, sample_count = minimum_phase.shape
    intrinsic = estimate_itd_seconds(
        minimum_phase.reshape(-1, 2, sample_count),
        sampling_rate_hz,
        upsampled_rate_hz=upsampled_rate_hz,
    ).reshape(subject_count, direction_count)
    return assign_itd_seconds(
        minimum_phase,
        intrinsic.to(minimum_phase.device),
        itd_seconds,
        sampling_rate_hz,
    )


def lsd_loss(
    predicted_magnitude_db: torch.Tensor,
    target_magnitude_db: torch.Tensor,
) -> torch.Tensor:
    if predicted_magnitude_db.shape != target_magnitude_db.shape:
        raise ValueError("predicted and target magnitudes must have equal shapes")
    return torch.sqrt(
        torch.mean(
            torch.square(predicted_magnitude_db - target_magnitude_db), dim=-1
        )
    ).mean()
