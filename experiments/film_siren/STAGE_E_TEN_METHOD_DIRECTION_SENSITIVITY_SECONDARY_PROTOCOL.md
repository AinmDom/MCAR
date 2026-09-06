# Stage E ten-method direction-sensitivity supplementary endpoints

Status: frozen before evaluation of these supplementary endpoints.

This validation-only extension supplements, but does not alter, the four primary
endpoints in `STAGE_E_TEN_METHOD_DIRECTION_SENSITIVITY_PROTOCOL.md`.  It applies
the already frozen ten-method registry to Q14/Q26/Q50, with the same 44 validation
listeners and the same fixed 743-direction evaluation mask (all Q50 observations
excluded for every condition).  It must not read test data, choose a model, grid,
checkpoint, or training budget, or change the paper main model: Bounded E25/Q26.

## Endpoints

The following 16 lower-is-better or descriptive endpoints use the frozen
implementations in `secondary_metrics.py` and `deferred_secondary_metrics.py`:

1. FullSphereLSD; HFFirstDifferenceMAE; HFSecondDifferenceMAE;
   MultiScaleNotchDepthMAE; ERBBandILDMean.
2. SpatialLSD in [0,10), [10,20), [20,30), and [30,180] degrees from the
   current Q14/Q26/Q50 input grid.  Bins use separately renormalized solid-angle
   weights and partition the fixed mask.
3. DominantNotchPenalizedMAE_Hz, DominantNotchMatchedMAE_Hz,
   DominantNotchMissRate, DominantNotchSpuriousRate, and
   ReferenceNotchFraction.  These retain their prior exploratory designation.
4. ITDWeightedMAE_us and ITDMaximumAbsoluteError_us, reconstructed with the
   inherited MCA phase/outside-bin spectrum.  They are sanity endpoints, not
   learned-delay claims.

Full-spectrum, shape, notch, and ITD endpoints use 743 directions. ERB-band ILD
uses the intersection of the fixed mask and horizontal directions. Each scalar is
first reduced within listener; listener is the only statistical unit. The ERB
profile is retained as supporting data.

## Statistics and integrity

For every method/endpoint/Q count report mean, sample SD, and listener bootstrap
95% interval (10,000 PCG64 resamples; base seed 20260906). Report Q14-Q26 and
Q50-Q26 paired effects, and each comparator's difference-in-sensitivity against
Bounded. `ReferenceNotchFraction` is descriptive, but remains in auditable tables.
No multiplicity-adjusted or confirmatory inference is claimed.

Expected scalar rows: 44 x 10 x 3 x 16 = 21,120; aggregate rows: 480;
within-method effects: 320; Bounded interactions: 288. Abort on missing cells,
non-finite arrays, a non-44 validation split, mask/grid mismatch, failed source
prediction checks, or any test read. The output must record
`test_subject_count_read=0`.
