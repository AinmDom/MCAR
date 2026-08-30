# FiLM Secondary Metrics — RANF/FSP-AE Baseline Extension Amendment v1

**Status:** implementation prepared; RANF/FSP-AE outputs for the added endpoints have not been computed or inspected  
**Freeze date:** 2026-08-30 (Asia/Hong_Kong)  
**Split:** SONICOM validation only, 44 subjects; test is permanently excluded

## 1. Scope and result-awareness statement

This amendment answers the paper-comparison question left open by the first two
secondary-metric tranches: it adds RANF and FSP-AE to every endpoint for which their
frozen validation predictions provide a definitionally comparable input.

The endpoint formulas, the four-method BOUNDED/HYBRID/FILMENS/MCAR secondary results,
and previously committed four-primary-metric RANF/FSP-AE results were already known
when this amendment was written. Therefore this is not a globally result-blind new
study. It is a **comparator-specific result-blind extension**: before this freeze, no
RANF/FSP-AE result had been computed, aggregated, or inspected for FullSphereLSD,
HF first/second differences, multi-scale notch depth, ERB-band ILD profile/mean,
distance-to-Q26 LSD, ITD, or dominant-notch location. Existing primary rows are used
only to extend the already registered subject-level tail-risk analysis.

The extension cannot alter the frozen candidate, reopen training, authorize test, or
upgrade any secondary/exploratory endpoint to confirmatory evidence.

## 2. Frozen comparator inputs

- **RANF:** `artifacts/reconstruction/sonicom_ranf_q26_validation_frozen/`, exactly
  44 `prediction.sofa` files. Each file must contain `Data.IR [793,2,256]` at
  44.1 kHz. Its `SourcePosition` azimuth/elevation must agree with the processed
  validation direction grid within `1e-8` degrees after azimuth wrapping.
- **FSP-AE:** `artifacts/reconstruction/sonicom_fsp_ae_q26_formal_validation/`, exactly
  44 `prediction.h5` files. Each file must declare `split=val` and the correct subject,
  and contain `predicted_magnitude_db [793,2,512]`, `predicted_hrir [793,2,256]`,
  and a 512-bin non-DC frequency grid.
- **Reference:** the same processed Q26 validation HDF5 and measured minimum-phase
  SOFA inventory used by the frozen v1 secondary/deferred tranches.

All files, provenance records, code, runtime binaries, pre-existing BOUNDED rows and
manifest identity are SHA-256 guarded. The evaluator must abort rather than substitute
or reorder any subject or prediction.

## 3. Frozen spectral mappings

- RANF magnitudes are `20 log10(max(abs(FFT(Data.IR, n=1024)), 1e-10))` sampled at
  the exact zero-based selected-bin indices in each processed subject HDF5. No
  dynamic-range clipping or MCA phase reconstruction is added.
- FSP-AE uses its stored magnitude directly. For a processed selected bin index `i`,
  the FSP-AE magnitude index is `i-1`, because the FSP-AE grid starts at the first
  positive-frequency bin whereas the processed strict index is indexed in the full
  single-sided spectrum. The mapped frequencies must agree exactly within `1e-4 Hz`.
- Direct stored HRIRs, not magnitude-based reconstructions, are used for both methods'
  ITD endpoints.

## 4. Comparable endpoints and statistics

The formulas and constants are inherited without modification from the frozen v1
secondary/deferred protocols:

1. FullSphereLSD;
2. first- and second-order HF spectral-difference MAE above 4 kHz;
3. multi-scale notch-depth MAE, 4–18 kHz;
4. 35-band horizontal ERB-band ILD absolute-error profile and arithmetic mean;
5. LSD in `[0,10)`, `[10,20)`, `[20,30)`, `[30,180]` degree distance-to-Q26 bins;
6. ITD weighted MAE and maximum absolute error using the same official-compatible
   low-pass/resample/cross-correlation estimator; and
7. the five dominant-notch endpoints using the already frozen Savitzky–Golay,
   prominence, separation, matching and penalty rules. These remain exploratory.

Every aggregate uses 44 subject-level values. Means, sample SD and 95% percentile
bootstrap intervals use 10,000 PCG64 resamples, seed `20260829`, linear quantiles.
Paired rows are `BOUNDED - RANF` and `BOUNDED - FSPAE`; negative favors BOUNDED.
Median, P90, P95, worst-five mean, maximum degradation, subject, wins/ties/losses and
win rate follow the original `1e-12` tie tolerance.

The paired table covers the five new scalars, four spatial bins, and four already
committed primary endpoints. Deferred paired rows cover ITD and the four error/rate
dominant-notch endpoints; `ReferenceNotchFraction` is descriptive and is not paired.

## 5. Explicitly non-comparable or unavailable rows

- **Gate/correction mechanism:** `NOT APPLICABLE`. RANF and FSP-AE do not expose the
  BOUNDED model's gate or applied-correction tensors under a common definition.
- **Efficiency:** `NOT AVAILABLE`. Neither comparator has been benchmarked under the
  frozen RTX 5060, batch, precision, warm-up and timing protocol. Literature values or
  measurements from another host must not be mixed into this table.
- **Model-based localization:** `NOT RUN`. The immutable official AMT/SAM dependency
  remains unavailable; no proxy may be substituted.

These status rows are part of the paper table and must not be interpreted as zeros.

## 6. Required integrity and output checks

The evaluator must abort unless:

- the split is exactly validation with 44 unique subjects;
- input inventories and all frozen hashes match;
- shapes are exactly as registered and every numeric input/output is finite;
- Q26/interpolation counts are 26/767, horizontal interpolation count is 72, and the
  common grid is identical across subjects;
- RANF coordinates and FSP-AE frequencies pass their mapping tolerances;
- the ERB profile mean reproduces the registered repository band-ILD loss;
- all candidate/comparator paired tables contain identical 44-subject sets; and
- every summary records `test_subject_count_read=0`.

The sole result directory is
`results/sonicom_film_secondary_baseline_extension_ranf_fsp_v1_validation/`.
The exact command is frozen in the generated manifest. Existing result directories
are read-only; the evaluator refuses to overwrite either final or partial output.
