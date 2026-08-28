# Stage E formal fixed-E25 three-member ensemble freeze

Status: **PRE-REGISTERED FORMAL TRAINING**, frozen before any formal optimizer
step.

The matched-seed E40 search completed for seeds `20260821`, `20260822`, and
`20260823`.  Their best cycles were `25`, `25`, and `20`; no run was best at
cycle 40.  Applying the pre-registered median rule freezes `E_final=25`.

Three formal runs start again from the zero-initialized bounded gate and the
same frozen seed-matched FiLM-SIREN E130 and MCAR v3.5.1 components.  Every
scientific and optimization setting remains unchanged.  Training stops after
25 cycles while retaining the search scheduler horizon of 40 cycles so the
learning-rate sequence through cycle 25 exactly matches the search runs.

The authoritative checkpoint for every member is cycle-25 `last.pt`, regardless
of its validation-best cycle.  The formal run fields are
`formal_fixed_cycle=true` and `checkpoint_policy=fixed_stop_cycle_last`.
After all three runs pass integrity checks, their cycle-25 residual predictions
are combined with fixed weights `1/3,1/3,1/3`.  No member selection, exclusion,
or post-result weight tuning is allowed.

The formal ensemble is then evaluated on the same 44 validation subjects against
Hybrid E190, FiLM E130 ensemble, and MCAR v3.5.1 with the already frozen strict
four-metric evaluator and 10,000 paired bootstrap resamples.  This validation
remains development evidence; test access is prohibited and every run must
record `test_subjects_read=0`.
