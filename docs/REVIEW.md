# Internal review board

Reviewed 2026-10-10 against [REPORT.md](REPORT.md) and the runs at commit `325068e`.
**Overall: pass.**

| Reviewer | Verdict | Required fixes |
|---|---|---|
| Research | Pass | None open |
| Engineering | Pass | None open |
| QA | Pass | None open |
| Security | Pass | None open |
| UX | Pass, one minor note | Frontier labels can touch when cells sit close together (noted, not blocking) |
| Recruiter | Pass | None open |
| PhD supervisor | Pass | None open |

## Research reviewer

*Is this scientifically defensible?*

- Accuracy is on people never seen by any model or calibration constant (subject-grouped CV, 5
  folds × 3 seeds, plus the official 21/9 split), with 95% bootstrap intervals over subjects.
- Baselines are in every comparison: centralized L0, majority class (0.187), chance
  re-identification (1/30) and "transmit nothing", which competes on the Pareto set (D8).
- Exposure is operational (an enrolled adversary) and its robustness is tested rather than assumed
  (E2: 8 adversaries, τ 0.83–0.96). The report says it is a lower bound for 30 people.
- Negative results are reported plainly: features are nearly as identifying as raw signals;
  Laplace LDP on features is useless at every tested ε up to 16; edge L3/L4 fall below the
  majority-class floor.
- Randomised-response accuracy matches its analytic expectation within 0.003, a correctness check
  of the mechanism.
- Scope changes after implementation (L4 definitions, ε range, device profiles, E5 subjects) are
  recorded in D4 and D13 with reasons; none was made to improve a result.

## Engineering reviewer

*Is the code maintainable and the pipeline reproducible?*

- One command (`make benchmark`) produces every result; runs are versioned with config, commit,
  dirty flag, package versions and the dataset hash. A run that raises is marked `FAILED` and never
  becomes `LATEST`.
- The first E1 run was marked dirty; all five experiments were rerun from a clean checkout and E1
  reproduced identically (D10). Only clean runs are kept.
- Architectures are configurations of one pipeline, so a difference between cells is a difference
  in data path, not code path. Streaming and batch share the privacy transforms (E5 checks they
  agree).
- Typed (mypy strict over `src`, `tests`, `scripts`), linted (ruff, ESLint), CI runs tests, smoke
  benchmark, build and browser tests.

## QA reviewer

*Does it work, and is it tested?*

- 35 unit, streaming-consistency, report and API tests and 20 browser tests pass (`make test`,
  `make test-ui`), including the randomised-response and Laplace mechanisms, operation counts, the
  report's pending state and the console's keyboard transport.
- axe-core finds no WCAG 2.1 A/AA violations on any view in light or dark mode.
- 15 real screenshots captured from the running console with the measured results.
- Fresh-clone check (2026-10-10): a new clone of `c1624a2` ran `make setup data test lint` and the
  full `make benchmark`. Every generated table and figure matched the committed one except the two
  wall-clock columns of E4 (*Device ms (here)*, *Cloud ms (here)*), which are timings of this
  machine and differ run to run by design; byte, operation and accuracy values were identical.

## Security reviewer

*Could this leak, be abused, or mislead?*

- The API is read-only, refuses to bind to a non-loopback host without a token, rate-limits per
  client, and sets CSP and other security headers. Run IDs are validated before touching disk.
- No secrets in the repository; `.env` is ignored and `.env.example` holds no values.
- The dataset is not redistributed (non-commercial licence); it is fetched and checksummed.

## UX reviewer

*Is the console clear and honest?*

- The Lab makes the central idea visible: what crosses the perimeter for each architecture, in
  bytes and content, beside the decision it produced.
- Every chart has a table view and every page names the run and commit its numbers come from.
- Minor: on the Frontier chart, point labels for nearby cells (the L0 cells top right, the local
  cells top left) can touch. The table view and selection panel give exact values; not blocking.

## Recruiter reviewer

*Does the README show the work in two minutes?*

- The README leads with the measured table and five findings, then how to run it, then
  screenshots. The negative findings are presented as findings, which reads as research maturity.

## PhD supervisor reviewer

*Would I sign off on this as a study?*

- The research questions are answered with the right evidence and the hypotheses are judged
  against it, including where they held more starkly than expected (RQ3).
- Threats to validity are specific: population size for re-identification, one adversary family,
  modelled costs, light models, one noise draw for randomised streaming cells.
- Future work follows from the limits rather than from ambition: hardware energy measurement, a
  stronger adversary, session-level privacy budgets.
