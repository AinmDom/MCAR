# Stage E ten-method input-direction sensitivity protocol

Status: preregistered before any missing comparator Q14/Q50 output is computed.

## Question and fixed data boundary

The paper main model remains the frozen equal-weight Bounded E25 ensemble with
Q26 input. This experiment asks whether its nine horizontal comparators show the
same response to one sparser (Q14) and one denser (Q50) observation grid.

Only the 44 frozen SONICOM validation listeners may be read. Q14 is an exact
subset of Q26 and Q26 an exact subset of Q50. Every method and every input count
is evaluated on the same 743 directions obtained by excluding all 50 Q50 input
directions. Test access, model selection, checkpoint selection, tuning, early
stopping changes, and result-dependent protocol changes are forbidden.

## Frozen registry and method-specific handling

The registry order is SH only, SUpDEq+SH, SUpDEq+Natural Neighbor,
SUpDEq+Barycentric, MCA, MCAR v3.5.1, FSP-AE, RANF, Hybrid E190, and Bounded E25.

- The four classical reconstructions and MCA are recomputed from the current-Q
  observations with their already frozen definitions.
- MCAR v3.5.1 is the frozen 0.3 previous + 0.7 candidate residual ensemble and
  receives the current-Q MCA/correction tensors.
- FSP-AE uses its frozen epoch-40 checkpoint and encodes exactly the current-Q
  measured directions; there is no additional training.
- Hybrid E190 uses its three frozen cycle-190 checkpoints, equal residual-dB
  weights, current-Q condition, and current-Q MCA/correction tensors.
- Bounded E25 reuses the already completed, hash-locked sensitivity predictions.
- RANF follows its native protocol: each Q14 and Q50 validation run restarts
  from the same frozen Q26-pretrained checkpoint, recomputes retrieval distances
  using only the 262 training listeners, and performs exactly 1000 adaptation
  epochs with batch size 3. The completed Q26 validation adaptation is reused.

For MCAR v3.5.1, FSP-AE, Hybrid E190, and Bounded E25, Q26 output must reproduce
the existing formal validation artifact within `1e-5 dB` (or `1e-5` absolute
for non-dB FSP-AE arrays). RANF Q26 is the existing frozen native artifact.

## Endpoints and statistics

The four primary endpoints are FullSphereERB, Contralateral25ERB,
ContralateralHighFrequency, and HorizontalILDMAE. Lower is better. The
independent unit is the listener. For each method and endpoint, report the mean,
standard deviation, and listener bootstrap 95% interval at Q14/Q26/Q50, plus
paired Q14-Q26 and Q50-Q26 effects. Also report paired difference-in-sensitivity
against Bounded E25 for each comparator. Bootstrap uses 10,000 replicates and
seed 20260902. No multiplicity-adjusted confirmatory claim is made; the study is
a preregistered sensitivity characterization.

Completeness requires 44 listeners x 10 methods x 3 counts x 4 endpoints = 5280
finite metric rows, 120 aggregate rows, 80 within-method effect rows, and 72
comparator-versus-Bounded interaction rows. Reports must record test reads as 0.
