# Dataset

## Source

**Human Activity Recognition Using Smartphones**, version 1.0 (Reyes-Ortiz, Anguita, Ghio, Oneto and Parra, November 2013), the dataset known as "UCI HAR".

| | |
|---|---|
| People | 30 volunteers, aged 19–48 |
| Device | Samsung Galaxy S II worn on the waist |
| Sensors | 3-axis accelerometer and 3-axis gyroscope at 50 Hz |
| Activities | walking, walking upstairs, walking downstairs, sitting, standing, laying |
| Windows | 10,299 windows of 2.56 s (128 samples) with 50% overlap: 7,352 from 21 training people, 2,947 from 9 test people |
| Signals used | `Inertial Signals/`: total acceleration (g), body acceleration (g) and angular velocity (rad/s), 3 axes each, 9 channels |
| Not used | the 561 precomputed features (the edge architecture computes its own 165 on the device, so their cost can be counted) |

The release's own split assigns people, not windows, to train and test, which matches this benchmark's need to evaluate on people the models have never seen.

## How it is obtained

The UCI repository is blocked by this build environment's network policy. `edgebench fetch` (or `make data`) downloads a copy of the original release archive from a public GitHub mirror and refuses it unless its SHA-256 matches:

| | |
|---|---|
| URL | `https://raw.githubusercontent.com/jbrownlee/Datasets/master/HAR_Smartphones.zip` |
| Size | 60,999,314 bytes |
| SHA-256 | `2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0` |

The loader then checks the structure against the dataset's README: 7,352 training and 2,947 test windows, 21 and 9 people, 9 channels of 128 samples. Every run records the archive's SHA-256 in its `provenance.json`.

## Licence and attribution

From the dataset's README: "Use of this dataset in publications must be acknowledged by referencing the following publication [1] ... This dataset is distributed AS-IS ... Any commercial use is prohibited."

> [1] Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21st European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium, 24–26 April 2013.

For that reason the data is **not** committed (`data/raw/` is ignored by git); it is downloaded and verified on each machine. This repository's MIT licence covers the code, not the dataset. The results and figures in `results/` and `docs/` are derived statistics, attributed above.

## Rebuilt recordings

The release ships windows, not recordings. Because consecutive windows of one recording overlap by exactly 64 samples, `data.segments()` joins each window to the next when the second half of one equals the first half of the next (and the person is the same), rebuilding continuous stretches of signal:

| | Measured on the release |
|---|---|
| Segments | 857 |
| Windows per segment | median 12, range 1–29 |
| Segments mixing two activities | 0 |
| Windows reconstructed exactly from their segment | all (checked in `tests/test_stream.py` on a subset, and by construction) |

Segments are used three ways: the streaming replay plays them sample by sample; L2 smoothing and L4 blocks never cross a segment boundary; and the re-identification adversary is enrolled on whole segments, so a test payload never shares samples with an enrolment payload.

## Class balance

| Activity | Windows |
|---|---:|
| walking | 1,722 |
| walking upstairs | 1,544 |
| walking downstairs | 1,406 |
| sitting | 1,777 |
| standing | 1,906 |
| laying | 1,944 |

## Known limitations

- Lab conditions, one phone model, one wearing position, 30 young-to-middle-aged adults. Re-identification among 30 people is easier than among thousands; exposure numbers are specific to this cohort size and are reported with chance (1/30) beside them.
- The signals were already noise-filtered and the body/gravity split was done by the dataset authors with a 0.3 Hz Butterworth filter; the device stage here starts from those signals, so the filter's cost is not counted.
- Activities were scripted, so segment lengths and transitions are not those of free living.
