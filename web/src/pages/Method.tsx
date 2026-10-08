import { PageHead } from "../components/ui";
import { api, type Frontier, useApi } from "../lib/api";
import { ARCH_LABEL, ARCH_PATH, ARCHS } from "../lib/format";

const LEVELS: [string, string, string, string][] = [
  [
    "L0 full",
    "9 channels, 50 Hz, float32",
    "165 features, float32",
    "label + 6 class probabilities",
  ],
  [
    "L1 minimise",
    "drop total acceleration (gravity, posture)",
    "drop the 55 gravity-derived features",
    "label only",
  ],
  [
    "L2 coarsen",
    "25 Hz, 8-bit",
    "4-bit features",
    "majority of the last 4 labels",
  ],
  [
    "L3 perturb",
    "Gaussian noise, 0.5 σ (no guarantee)",
    "clip + Laplace: ε-local DP, ε = 4",
    "randomised response: ε-local DP, ε = 4",
  ],
  [
    "L4 aggregate",
    "one window per block of 8",
    "block mean of 8 windows",
    "block majority label",
  ],
];

export function Method() {
  const { data } = useApi<Frontier>(api.frontier);
  const prof = data?.costs?.profiles;
  return (
    <>
      <PageHead title="Method">
        How the benchmark is built, what it measures, and what it assumes.
      </PageHead>
      <div className="prose stack">
        <section className="panel">
          <div className="panel-head">
            <h2>Data</h2>
          </div>
          <div className="panel-body">
            <p>
              UCI Human Activity Recognition Using Smartphones (Anguita et al.,
              ESANN 2013): 30 people, a waist-worn phone, 3-axis accelerometer
              and gyroscope at 50 Hz, six activities, 10,299 windows of 2.56 s
              with 50% overlap. Consecutive windows that share their 64
              overlapping samples exactly are joined back into 857 continuous
              recording segments, which the streaming replay plays sample by
              sample. The archive is fetched from a public mirror and checked
              against a pinned SHA-256; it is not redistributed here (its
              licence forbids commercial use).
            </p>
          </div>
        </section>
        <section className="panel">
          <div className="panel-head">
            <h2>Three architectures</h2>
          </div>
          <div className="panel-body">
            <ul>
              {ARCHS.map((a) => (
                <li key={a}>
                  <strong>{ARCH_LABEL[a]}</strong>: {ARCH_PATH[a]}.{" "}
                  {a === "centralized"
                    ? "The cloud classifies raw windows with 500 random convolutional kernels and a ridge classifier."
                    : a === "edge"
                      ? "The device computes 165 summary features; the cloud classifies them with gradient-boosted trees."
                      : "The device classifies with a logistic regression with int8 weights; only the decision leaves."}
                </li>
              ))}
            </ul>
          </div>
        </section>
        <section className="panel">
          <div className="panel-head">
            <h2>Five cumulative privacy levels</h2>
          </div>
          <div
            className="table-wrap"
            tabIndex={0}
            role="region"
            aria-label="Privacy levels by architecture"
          >
            <table>
              <thead>
                <tr>
                  <th scope="col">Level</th>
                  <th scope="col">Centralized (raw)</th>
                  <th scope="col">Edge (features)</th>
                  <th scope="col">Fully local (decisions)</th>
                </tr>
              </thead>
              <tbody>
                {LEVELS.map((r) => (
                  <tr key={r[0]}>
                    {r.map((c, i) =>
                      i === 0 ? (
                        <th key={i} scope="row">
                          {c}
                        </th>
                      ) : (
                        <td key={i}>{c}</td>
                      ),
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="panel-body">
            <p className="secondary">
              Each level keeps the transformations before it. At L4 every window
              of a block receives the block's decision, so per-window accuracy
              stays defined. Only the edge and local L3 (and L4) carry a formal
              guarantee, and ε is per payload: k payloads compose to kε.
            </p>
          </div>
        </section>
        <section className="panel">
          <div className="panel-head">
            <h2>Measures</h2>
          </div>
          <div className="panel-body">
            <ul>
              <li>
                <strong>Accuracy</strong>: on people never seen in training.
                Subject-grouped 5-fold cross-validation over 3 seeds;
                per-subject accuracy averaged, with a 95% bootstrap interval
                over subjects. The dataset's own 21/9 split is reported
                alongside.
              </li>
              <li>
                <strong>Exposure</strong>: an adversary at the cloud enrols
                every person with half of their recording segments (split by
                segment, so overlapping windows never straddle enrolment and
                test) and names the person behind each remaining payload.
                Reported as accuracy, advantage over chance and per-segment
                majority vote. It is one attacker, so it is a lower bound on
                what a stronger one could do.
              </li>
              <li>
                <strong>Bandwidth</strong>: bytes per window as serialised, and
                after zlib.
              </li>
              <li>
                <strong>Compute</strong>: analytic operation counts on device
                and in the cloud, and single-window wall times measured on this
                machine (for scale only).
              </li>
              <li>
                <strong>Latency and energy proxy</strong>: modelled. Latency =
                device ops ÷ device speed + payload bits ÷ link rate + one-way
                delay + cloud time + waiting for an L4 block. Energy = ops ×
                J/op + bytes × J/byte.
              </li>
            </ul>
          </div>
        </section>
        <section className="panel">
          <div className="panel-head">
            <h2>Assumed device and link profiles</h2>
          </div>
          <div className="panel-body">
            {prof ? (
              <div
                className="table-wrap"
                tabIndex={0}
                role="region"
                aria-label="Device and link assumptions"
              >
                <table>
                  <thead>
                    <tr>
                      <th scope="col">Profile</th>
                      <th scope="col" className="num">
                        Rate
                      </th>
                      <th scope="col" className="num">
                        Delay or energy
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {prof.devices.map((d) => (
                      <tr key={d.name}>
                        <th scope="row">device: {d.name}</th>
                        <td className="num">
                          {d.ops_per_s.toExponential(0)} ops/s
                        </td>
                        <td className="num">
                          {d.joule_per_op.toExponential(0)} J/op
                        </td>
                      </tr>
                    ))}
                    {prof.links.map((l) => (
                      <tr key={l.name}>
                        <th scope="row">link: {l.name}</th>
                        <td className="num">
                          {l.rate_bps.toExponential(0)} bit/s
                        </td>
                        <td className="num">
                          {l.one_way_s * 1000} ms,{" "}
                          {l.joule_per_byte.toExponential(0)} J/byte
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="secondary">
                Shown once E4 has run (configs/e4_cost.yaml).
              </p>
            )}
            <p className="secondary">
              These are round orders of magnitude, not measurements of any
              product. E4 sweeps the radio's energy per byte from 0.01× to 10×
              to show where the conclusions change.
            </p>
          </div>
        </section>
        <section className="panel">
          <div className="panel-head">
            <h2>Limits</h2>
          </div>
          <div className="panel-body">
            <ul>
              <li>
                No physical edge device or radio: device time and energy are
                proxies; ratios between cells are the result.
              </li>
              <li>
                One dataset of 30 people in a lab setting; re-identification
                gets easier with fewer people.
              </li>
              <li>
                Exposure is measured against one attacker per payload type; it
                is not a privacy guarantee.
              </li>
            </ul>
          </div>
        </section>
      </div>
    </>
  );
}
