# ARI external FSC replication

This locked external-database experiment compares MCA with the single-seed FSC E190 endpoint on 22 held-out ARI subjects.

All metrics are lower-is-better. Paired differences are FSC minus MCA; negative values favor FSC.

## Result

The strict replication criterion was met: all five paired confidence intervals exclude zero in favor of FSC.

| Metric | MCA mean ± sample SD | FSC mean ± sample SD | FSC−MCA mean [paired bootstrap 95% CI] |
|---|---:|---:|---:|
| FullSphereERB | 1.125048 ± 0.063907 dB | 0.907512 ± 0.063156 dB | -0.217536 [-0.246811, -0.187570] dB |
| Contralateral25ERB | 1.286699 ± 0.071801 dB | 1.060550 ± 0.072659 dB | -0.226148 [-0.257421, -0.193517] dB |
| ERBBandILDMean | 2.090091 ± 0.191805 dB | 1.628104 ± 0.115208 dB | -0.461988 [-0.531943, -0.393485] dB |
| ITDWeightedMAE | 15.574588 ± 2.598053 us | 15.088884 ± 2.563497 us | -0.485704 [-0.580253, -0.396697] us |
| FullSphereLSD | 4.662866 ± 0.230700 dB | 3.708464 ± 0.223160 dB | -0.954401 [-1.018331, -0.888210] dB |

## Integrity

- Test subjects read: 22/22; all finite: true; failed subjects: 0.
- Split leakage: false; E130 and E190 completed; checkpoint: E190 `last.pt`.
- Test access occurred once after protocol freeze; no post-test tuning or protocol deviations occurred.

ARI and SONICOM use different measured direction grids, weights, and Q26 definitions. Their absolute metric values must not be pooled as one experiment.
