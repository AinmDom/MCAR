# Stage E matched-seed E40 expansion

Status: **PRE-REGISTERED AFTER D1 STRICT VALIDATION, BEFORE NEW RUNS**.

The seed-20260821 bounded correction passed the frozen structure gate against
MCAR v3.5.1: all four mean differences were negative; Full ERB and Contra25 ERB
had 95% bootstrap intervals below zero; no metric regressed.  It did not
dominate Hybrid E190, trading slightly worse contralateral spectral metrics for
materially better horizontal ILD.

E200 is explicitly rejected because D1 reached its best validation objective at
cycle 25 and cycles 30, 35, and 40 did not improve it.  The next uncertainty is
matched-seed stability, not optimization horizon.

Two additional runs use seeds `20260822` and `20260823`, their corresponding
frozen corrected FiLM-SIREN E130 `last.pt` checkpoints, and the identical MCAR
v3.5.1 components, architecture, objective, optimizer, scheduler, sampling, and
40-cycle budget.  They may run concurrently because each allocates less than
300 MiB CUDA model/training memory in the D1 report and the RTX 5060 has enough
headroom.  No configuration or metric is selected separately per seed.

After both finish, verify all three histories, ledgers, hashes, tensors, and
best-cycle trajectories.  Then freeze a common cycle as the median of the three
best cycles, rounded to the existing five-cycle validation grid.  If any run is
best at cycle 40, label the expansion endpoint-limited and do not recursively
increase the budget without a new decision.  Otherwise prepare three
from-scratch fixed-cycle formal runs and a 1/3 residual-dB ensemble before the
next strict validation comparison.

Only train and validation subjects may be resolved.  Every run must record
`test_subjects_read=0`; test access remains prohibited.
