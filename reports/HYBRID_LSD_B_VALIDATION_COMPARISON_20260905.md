# Hybrid LSD B validation comparison

Status: completed on 44 validation subjects; test reads: 0.

FullSphereLSD changed from **3.548925619** to **3.522658135 dB**, a **0.740%** reduction. The paired mean difference was **-0.026267484 dB** (95% bootstrap [-0.029413179, -0.023110672]; wins/ties/losses 43/0/1).

## Candidate minus original Hybrid

| Endpoint | New mean | Old mean | Difference | 95% bootstrap | W/T/L |
|---|---:|---:|---:|---:|---:|
| FullSphereLSD | 3.522658135 | 3.548925619 | -0.026267484 | [-0.029413179, -0.023110672] | 43/0/1 |
| FullSphereERB | 0.810682616 | 0.804420311 | +0.006262305 | [+0.005347438, +0.007189830] | 1/0/43 |
| Contralateral25ERB | 1.230951179 | 1.214593084 | +0.016358095 | [+0.012448234, +0.020235736] | 4/0/40 |
| ContralateralHighFrequency | 3.502457360 | 3.501993960 | +0.000463400 | [-0.003250787, +0.004171865] | 23/0/21 |
| HorizontalILDMAE | 0.628032698 | 0.629811407 | -0.001778710 | [-0.003612555, -0.000016538] | 26/0/18 |
| HFFirstDifferenceMAE | 0.404764339 | 0.402702654 | +0.002061685 | [+0.001834536, +0.002284832] | 0/0/44 |
| HFSecondDifferenceMAE | 0.223154688 | 0.216797570 | +0.006357118 | [+0.005941656, +0.006794687] | 0/0/44 |
| MultiScaleNotchDepthMAE | 0.488821774 | 0.488411853 | +0.000409921 | [+0.000156335, +0.000650022] | 14/0/30 |
| ERBBandILDMean | 1.501545418 | 1.499205253 | +0.002340165 | [-0.000041170, +0.004692116] | 15/0/29 |

LSD gains were clear in the 0–10°, 10–20°, and 20–30° distance bins. The 30–180° mean difference was -0.003499 dB with an interval crossing zero; its maximum subject degradation was +0.101600 dB (P0166).

The intervention slightly worsened full-sphere and contralateral-25 ERB and both high-frequency difference metrics. Contralateral high-frequency was unchanged within uncertainty; strict horizontal ILD improved by 0.001779 dB.

This supports a small LSD/objective tradeoff. It does not isolate MCA's internal ERB correction because that path was unchanged. The result is exploratory validation evidence and must not be described as independent test confirmation.

## Integrity

Three runs: 3/3 exit 0, E190, 49,780 steps each; all history/ledger values and E190 weights finite; checkpoint hashes matched; clean training Git; test reads 0. New inference: 44/44 subjects, shape [2,793,463], all finite. Bootstrap: 10,000 paired subject resamples, seed 20260904.
