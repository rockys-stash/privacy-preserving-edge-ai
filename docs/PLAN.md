# Plan: Privacy-Preserving Edge AI Benchmark

> Updated after implementation where the build differs from the first draft; each change is recorded in `docs/DECISIONS.md` (D4, D13).

## 1. Scope

The spec asks for one thing: quantify the **accuracy ↔ privacy ↔ compute-cost** trade-off across three architectures, at graded privacy levels, on a sensor pipeline replayed from a documented public dataset.

| Architecture | Data path | What leaves the device |
|---|---|---|
| **Centralized** | sensor → raw data → cloud → AI | raw sensor windows |
| **Edge** | sensor → local processing → features → cloud | a feature vector per window |
| **Fully local** | sensor → edge AI → decision | the decision (activity label) |

Deliverables:

1. **Sensor replay** (`edgebench.data`, `edgebench.stream`). Continuous per-subject inertial streams (3-axis total acceleration, body acceleration and angular velocity at 50 Hz) reconstructed from the UCI HAR windows, replayed sample by sample on a simulated clock.
2. **Device runtime and cloud service** (`edgebench.stream`, `edgebench.bench`). A device that buffers 2.56 s windows with 50% overlap and runs its architecture's on-device stage, a modelled uplink, and a cloud that receives payloads and produces decisions. The three architectures are configurations of this one pipeline, selected by name.
3. **Privacy transformations** (`edgebench.privacy`). Five cumulative levels per architecture, applied to whatever leaves the device.
4. **Models.** Cloud and on-device classifiers, plus an **adversary** that tries to re-identify the person from the payloads the cloud receives.
5. **Measurement harness** (`edgebench.measure`). For every architecture × level: accuracy, latency, bandwidth, compute and an energy proxy, data exposure, and utility loss. Every number is written with its config and commit.
6. **Console** (`web/`). An edge-device laboratory: replayed telemetry, what each architecture transmits, and the privacy–accuracy–cost frontier charts drawn from measured results.

## 2. Privacy levels

Levels are cumulative: each one keeps the previous transformations and adds one. The same five steps apply to every architecture, acting on that architecture's payload.

| Level | Step | Centralized (raw windows) | Edge (features) | Fully local (decisions) |
|---|---|---|---|---|
| L0 | Full | 9 channels, 50 Hz, float32 | full feature vector, float32 | label + class probabilities |
| L1 | Minimise | drop total acceleration (gravity, so device orientation and posture) | drop gravity- and orientation-derived features | label only, no probabilities |
| L2 | Coarsen | downsample to 25 Hz, 8-bit quantisation | 4-bit quantisation of each feature | label smoothed over the last 4 windows |
| L3 | Perturb | additive Gaussian noise (σ as a share of each channel's spread; **no formal guarantee**) | clipping + Laplace noise: ε-local differential privacy per window | randomised response: ε-local differential privacy per label |
| L4 | Aggregate | one payload per block of 8 windows: the block's middle window | one feature vector per block: the mean of the clean features, then L2–L3 once | one label per block: the majority label, then randomised response |

At L4 the cloud assigns one decision to every window of a block, so per-window accuracy stays defined for every cell of the grid. Noise parameters for L3 are fixed in the config and swept separately in E3.

## 3. Research questions and hypotheses

| | Question | Hypothesis | Experiment |
|---|---|---|---|
| RQ1 | Where does each architecture × level sit on the accuracy–exposure–cost surface, and which cells are Pareto-optimal? | Neither extreme is optimal. Centralized L0 is the most accurate and the most exposing; some edge and local cells give up little accuracy for much less exposure. | E1 |
| RQ2 | Are on-device features less identifying than raw signals? | Only partly. Gait and posture identify people, and summary features keep that, so edge L0 still lets an enrolled adversary re-identify people far above chance. | E1, E2 |
| RQ3 | What does formal local privacy cost in accuracy? | Accuracy falls smoothly with ε for labels (randomised response), and steeply for high-dimensional features (Laplace), because the budget is split across dimensions. | E3 |
| RQ4 | When does fully local win on cost? | Local always wins on bandwidth and on latency to a decision. On energy it wins when the radio costs more per byte than on-device computation, which depends on the link; we report the break-even. | E4 |
| RQ5 | Does the streaming pipeline reproduce the batch results? | Yes: replaying every subject's stream through the device, link and cloud gives the same decisions as batch evaluation, apart from windows lost at stream boundaries, which are counted. | E5 |

**Baselines.** Centralized L0 (the status quo); a majority-class predictor (accuracy floor); and "transmit nothing" (no utility, no exposure). Exposure is compared with chance re-identification (1/number of enrolled subjects).

## 4. Metrics

Defined formally in `docs/METRICS.md`.

| Metric | Measurement |
|---|---|
| Accuracy | Per-window activity accuracy and macro-F1 on subjects never seen in training (subject-grouped cross-validation, 5 folds × 3 seeds, plus the dataset's official 21/9 subject split). |
| Utility loss | Accuracy at a level minus accuracy at L0 of the same architecture, and minus centralized L0. |
| Latency | Time from a window closing on the device to its decision being available at the cloud: measured on-device compute, modelled uplink (payload bits / link rate + one-way delay), measured cloud compute, and, at L4, waiting for the block to fill. Link profiles are stated assumptions. |
| Bandwidth | Bytes per window actually serialised (and after zlib), and bits per second at the 1.28 s window step. |
| Compute | Analytic operation counts (multiply-accumulates and comparisons) per window on device and in the cloud, measured wall time, and model size in bytes. |
| Energy proxy | Device operations × energy per operation + transmitted bytes × radio energy per byte, for stated link profiles. It is a proxy with stated constants, not a measurement, and is reported with a sensitivity range. |
| Data exposure | Re-identification by an adversary at the cloud who has enrolled every subject with part of their own data: top-1 accuracy, and advantage over chance, (acc − 1/n) / (1 − 1/n). Also a categorical inventory of the data types that leave the device. |

## 5. Experiments

| | Design | Output |
|---|---|---|
| E1 | Frontier grid: 3 architectures × 5 levels, subject-grouped 5-fold CV × 3 seeds. Accuracy, exposure, bandwidth, compute and latency for every cell; Pareto set. | frontier table and charts |
| E2 | Adversary strength: enrollment share 10–70% of each subject's data, two attacker model families. Is the exposure ranking stable? | ranking stability |
| E3 | Local differential privacy sweep: ε from 0.1 to 64 for edge features and local labels (L3), subject-grouped 5-fold CV. | accuracy–ε and exposure–ε curves |
| E4 | Cost model: per-stage compute measured, bandwidth, latency and energy proxy for two device and three link profiles, with the radio-energy constant swept from 0.01× to 10×. | break-even analysis |
| E5 | Streaming replay of the 9 official test subjects (models trained on the other 21) through each architecture at L0, L2 and L4; decisions compared with batch evaluation. | consistency report |

One command (`make benchmark`) runs E1–E5 and regenerates the report tables and figures. Configs are YAML with seeds; results go to `results/<exp>/<UTC time>-<commit>/`.

## 6. Risks and dependencies

- **Dataset access.** The UCI repository is blocked by this environment's network policy. A byte-identical copy of the original release archive is fetched from a public GitHub mirror and checked against a pinned SHA-256; its structure and counts are checked against the dataset's own README (`data/DATASET.md`). The data is not committed: the README forbids commercial use and asks for attribution.
- **Hardware.** There is no physical edge device. On-device compute is measured on this machine's CPU, single-threaded, and reported as operation counts as well, which do not depend on the machine. Absolute device latency and energy are therefore proxies; ratios between architectures are the result.
- **Network.** The uplink is modelled, not measured. Three link profiles with stated rates, delays and energy per byte are used, and conclusions are checked across all three.
- **Privacy claims.** Only L3 for edge features and local labels carries a formal guarantee (ε-local DP per window). Everything else is measured empirically against one adversary, which is a lower bound on exposure: a stronger attacker could do better.
- **No language model or API key is needed.**

## 7. Stack

Python 3.13 (numpy, scipy, scikit-learn), FastAPI for the console API, React + TypeScript + Vite for the console. No deep-learning framework: PyTorch's CPU wheels are not reachable here, and the models below are standard and fast enough on CPU.

| Role | Model |
|---|---|
| Cloud model on raw windows (centralized) | Random convolutional kernels (ROCKET-style: random 1-D kernels, proportion-of-positive-values and max pooling) with a ridge classifier |
| Cloud model on features (edge) | Gradient-boosted trees (scikit-learn `HistGradientBoostingClassifier`) |
| On-device model (fully local) | Multinomial logistic regression on the device features, with int8 weights |
| Adversary | The same model family as the cloud model for that payload, trained to predict the subject |
