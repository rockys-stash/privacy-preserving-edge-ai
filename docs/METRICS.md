# Metrics

Every metric below is computed by code in `src/edgebench/`; the function is named for each.

## Cells

A **cell** is one architecture (centralized, edge, fully local) at one privacy level (L0–L4): 15 cells. The levels are defined in `docs/PLAN.md` §2 and `edgebench.privacy`.

A **payload** is what one transmission carries across the device boundary. At L0–L3 there is one payload per window; at L4 one payload per block of up to 8 consecutive windows of one segment, and every window in the block receives the block's decision.

## Utility

- **Accuracy** (`bench.evaluate_fold`): share of windows whose decision equals the true activity, computed per test subject, then averaged over subjects. Subject-grouped 5-fold cross-validation: each fold's test people are never seen by any model trained in that fold (cloud model, device model, calibration constants). Repeated for 3 seeds (fold assignment, model seeds, privacy noise).
- **95% interval** (`e1_frontier.bootstrap_ci`): percentile bootstrap over subjects (2,000 resamples) of the per-subject accuracy averaged over seeds.
- **Macro-F1**: per fold, averaged.
- **Official split accuracy**: the same pipeline trained on the release's 21 training people, tested on its 9 test people, seed 0.
- **Utility loss**: signed difference in accuracy, `accuracy(cell) − accuracy(same architecture, L0)` and `accuracy(cell) − accuracy(centralized, L0)`. Negative means less accurate.
- **Floors**: majority-class accuracy (predict the most frequent training activity), averaged over folds; and "transmit nothing", which has that accuracy and chance exposure.

## Exposure

The adversary sits at the cloud and sees exactly the payloads the cloud sees. It has **enrolled every person** with part of their own data and tries to name the person behind each other payload.

- **Enrolment** (`bench.enrolment_split`): for each person and activity, a random share (default 50%) of their recording segments, at least one segment held out. Whole segments are assigned, so no enrolment payload shares samples with a test payload.
- **Model**: the same family as the cloud model for that payload ("strong": random kernels and ridge for raw signals, gradient-boosted trees for features, a logistic regression on the one-hot decision plus any probabilities sent), or a multinomial logistic regression ("linear"; E2).
- **Re-identification accuracy** (`exposure`): share of held-out payloads attributed to the right person among all 30. Chance is 1/30.
- **Advantage**: `(accuracy − 1/n) / (1 − 1/n)`, 0 at chance and 1 at certainty.
- **Per-segment re-identification**: majority vote of the adversary's guesses over each held-out segment, the attacker who links payloads from one session.
- On the Pareto comparison, measured exposure below chance is treated as chance (an attacker can always guess).

Exposure is an empirical lower bound against this attacker. Only L3 and L4 of the edge and fully local architectures carry a formal guarantee (ε-local differential privacy per payload).

## Cost

- **Bytes per payload / per window** (`measure._serialise`): the payload serialised as it would be sent (float32, 8-bit or packed 4-bit integers, float16 for noised features, one byte per label), divided by the windows it covers. **zlib bytes**: the same stream compressed (level 9) over 1,600 consecutive test windows.
- **Uplink bit/s**: bytes per window × 8 ÷ 1.28 s (a window closes every 1.28 s).
- **Operations** (`measure.device_ops`, `measure.cloud_ops`; the cloud count for gradient-boosted trees is trees × mean depth): analytic counts of arithmetic operations and comparisons per window for each stage (feature extraction, quantisation, noise, the int8 device model, the random-kernel transform and the cloud classifier). They do not depend on the machine.
- **Measured wall time**: median of single-window runs of each stage on this machine, one thread. For scale only.
- **Latency** (`measure.latency_s`), modelled: device ops ÷ device ops/s + payload bits ÷ link rate + one-way delay + measured cloud time + L4 block wait ((k − 1)/2 × 1.28 s on average).
- **Energy proxy** (`measure.energy_j_per_window`), modelled: device ops × J/op + bytes per window × radio J/byte.
- **Profiles**: stated assumptions in `configs/e4_cost.yaml` (2 devices × 3 links); the radio's J/byte is swept from 0.01× to 10× in E4.

## Streaming consistency (E5)

For each window of the 9 official test people, the decision produced by the sample-by-sample replay (`stream.replay`) is compared with the batch decision for the same window. L0–L2 are deterministic and must match exactly; L3 and L4 draw fresh noise in the stream, so only their accuracy is compared.
