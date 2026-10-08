import { useMemo, useState } from "react";
import { DataTable } from "../components/DataTable";
import { type Point, Scatter } from "../components/Scatter";
import {
  ErrorOrPending,
  Loading,
  PageHead,
  ProvenanceLine,
  Segmented,
} from "../components/ui";
import {
  api,
  type Cell,
  type Frontier as FrontierData,
  useApi,
} from "../lib/api";
import { ARCH_LABEL, bytes, num, pct, seconds, signedPct } from "../lib/format";

type CostDim = "bandwidth" | "energy" | "latency";

function cellName(c: Cell) {
  return `${ARCH_LABEL[c.arch]} L${c.level}`;
}

export function Frontier() {
  const { data, error, loading, retry } = useApi<FrontierData>(api.frontier);
  const [selected, setSelected] = useState<string | null>(null);
  const [dim, setDim] = useState<CostDim>("bandwidth");
  const [view, setView] = useState<"chart" | "table">("chart");
  const [device, setDevice] = useState<string>("microcontroller");
  const [link, setLink] = useState<string>("ble");

  const exposurePoints = useMemo<Point[]>(
    () =>
      (data?.cells ?? []).map((c) => ({
        id: c.cell,
        series: c.arch,
        x: c.exposure,
        xlo: c.exposure_lo,
        xhi: c.exposure_hi,
        y: c.accuracy,
        ylo: c.accuracy_lo,
        yhi: c.accuracy_hi,
        label: `L${c.level}`,
        tooltip: [
          `${cellName(c)} (${c.level_name})`,
          `accuracy ${pct(c.accuracy)} [${pct(c.accuracy_lo)}, ${pct(c.accuracy_hi)}]`,
          `re-identification ${pct(c.exposure)}`,
          `${bytes(c.bytes_per_window)} per window`,
        ],
      })),
    [data],
  );

  const costPoints = useMemo<Point[]>(() => {
    if (!data) return [];
    return data.cells.flatMap((c) => {
      let x: number | undefined;
      if (dim === "bandwidth") x = c.bytes_per_window;
      else {
        const m = data.costs?.modelled.find(
          (r) => r.cell === c.cell && r.device === device && r.link === link,
        );
        x = m ? (dim === "energy" ? m.energy_uj : m.latency_s) : undefined;
      }
      if (x === undefined || x <= 0) return [];
      const xs =
        dim === "bandwidth"
          ? bytes(x)
          : dim === "energy"
            ? `${num(x, 2)} µJ`
            : seconds(x);
      return [
        {
          id: c.cell,
          series: c.arch,
          x,
          y: c.accuracy,
          ylo: c.accuracy_lo,
          yhi: c.accuracy_hi,
          label: `L${c.level}`,
          tooltip: [
            `${cellName(c)} (${c.level_name})`,
            `accuracy ${pct(c.accuracy)}`,
            `${dim} ${xs}`,
          ],
        },
      ];
    });
  }, [data, dim, device, link]);

  if (loading && !data) return <Loading label="Loading the frontier" />;
  if (error || !data)
    return error ? (
      <>
        <PageHead title="Frontier" />
        <ErrorOrPending
          error={error}
          retry={retry}
          what="the frontier"
          command="make benchmark"
        />
      </>
    ) : null;

  const b = data.baselines;
  const pareto = data.cells
    .filter((c) => c.pareto)
    .sort((x, y) => x.exposure - y.exposure);
  const labelled = new Set(
    data.cells.filter((c) => c.pareto || c.level === 0).map((c) => c.cell),
  );
  const sel = data.cells.find((c) => c.cell === selected);
  const devices = data.costs?.profiles.devices.map((d) => d.name) ?? [];
  const links = data.costs?.profiles.links.map((l) => l.name) ?? [];

  return (
    <>
      <PageHead
        title="Frontier"
        actions={
          <Segmented
            label="Show as"
            options={[
              { id: "chart", label: "Charts" },
              { id: "table", label: "Table" },
            ]}
            value={view}
            onChange={setView}
          />
        }
      >
        Accuracy on people the models never saw, against how well an adversary
        at the cloud re-identifies the person from what the device sent.{" "}
        {data.cells[0]?.subjects} subjects; majority class{" "}
        {pct(b.majority_accuracy)}; chance re-identification{" "}
        {pct(b.chance_reidentification)}.
      </PageHead>

      {view === "chart" ? (
        <div className="grid-2">
          <section className="panel" aria-labelledby="fr-exp">
            <div className="panel-head">
              <h2 id="fr-exp">Accuracy against exposure</h2>
            </div>
            <div className="panel-body">
              <Scatter
                title="Activity accuracy against re-identification accuracy for 15 cells"
                points={exposurePoints}
                xLabel="Re-identification accuracy per payload (lower is more private)"
                yLabel="Activity accuracy"
                refs={[
                  {
                    axis: "x",
                    value: b.chance_reidentification,
                    label: "chance",
                  },
                  {
                    axis: "y",
                    value: b.majority_accuracy,
                    label: "majority class",
                  },
                ]}
                frontier={pareto.map((c) => c.cell)}
                labelled={labelled}
                selected={selected}
                onSelect={setSelected}
                xFormat={(v) => `${Math.round(v * 100)}%`}
              />
              <p className="muted small">
                Lines through each dot: 95% bootstrap interval over subjects
                (vertical) and range over seeds (horizontal). Labelled: L0 of
                each architecture and the Pareto set.
              </p>
            </div>
          </section>
          <section className="panel" aria-labelledby="fr-cost">
            <div className="panel-head wrap">
              <h2 id="fr-cost">Accuracy against cost</h2>
              <Segmented
                label="Cost dimension"
                options={[
                  { id: "bandwidth", label: "Bandwidth" },
                  {
                    id: "energy",
                    label: "Energy proxy",
                    title: data.costs ? undefined : "needs E4",
                  },
                  {
                    id: "latency",
                    label: "Latency",
                    title: data.costs ? undefined : "needs E4",
                  },
                ]}
                value={dim}
                onChange={setDim}
              />
            </div>
            <div className="panel-body">
              {dim !== "bandwidth" && data.costs && (
                <div className="controls">
                  <label>
                    Device{" "}
                    <select
                      value={device}
                      onChange={(e) => setDevice(e.target.value)}
                    >
                      {devices.map((d) => (
                        <option key={d}>{d}</option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Link{" "}
                    <select
                      value={link}
                      onChange={(e) => setLink(e.target.value)}
                    >
                      {links.map((d) => (
                        <option key={d}>{d}</option>
                      ))}
                    </select>
                  </label>
                </div>
              )}
              {costPoints.length ? (
                <Scatter
                  title={`Activity accuracy against ${dim} for 15 cells`}
                  points={costPoints}
                  xLog
                  xLabel={
                    dim === "bandwidth"
                      ? "Bytes leaving the device per window (log)"
                      : dim === "energy"
                        ? "Device energy proxy per window, µJ (log; modelled)"
                        : "Window closed → decision at the cloud, s (log; modelled)"
                  }
                  yLabel="Activity accuracy"
                  labelled={labelled}
                  selected={selected}
                  onSelect={setSelected}
                />
              ) : (
                <p className="secondary">
                  E4 (cost) has not been run yet: Status: pending.
                </p>
              )}
              {dim !== "bandwidth" && (
                <p className="muted small">
                  Energy and latency are modelled from stated device and link
                  assumptions (see Method), not measured on hardware. Bytes per
                  window are measured.
                </p>
              )}
            </div>
          </section>
        </div>
      ) : (
        <section className="panel">
          <DataTable
            caption="All cells, frontier data"
            rows={data.cells}
            rowKey={(c) => c.cell}
            selected={selected}
            onSelect={setSelected}
            columns={[
              { key: "cell", label: "Cell", render: cellName },
              {
                key: "acc",
                label: "Accuracy",
                align: "r",
                render: (c) => pct(c.accuracy),
              },
              {
                key: "ci",
                label: "95% CI",
                align: "r",
                render: (c) => `${pct(c.accuracy_lo)} – ${pct(c.accuracy_hi)}`,
              },
              {
                key: "reid",
                label: "Re-identification",
                align: "r",
                render: (c) => pct(c.exposure),
              },
              {
                key: "bytes",
                label: "Bytes / window",
                align: "r",
                render: (c) => bytes(c.bytes_per_window),
              },
              {
                key: "pareto",
                label: "Pareto",
                render: (c) => (c.pareto ? "yes" : ""),
              },
            ]}
          />
        </section>
      )}

      <section className="panel" aria-labelledby="fr-sel" aria-live="polite">
        <div className="panel-head">
          <h2 id="fr-sel">
            {sel ? `${cellName(sel)}: ${sel.level_name}` : "Selected cell"}
          </h2>
        </div>
        <div className="panel-body">
          {sel ? (
            <dl className="stat-row">
              <div>
                <dt>Accuracy</dt>
                <dd>{pct(sel.accuracy)}</dd>
              </div>
              <div>
                <dt>vs centralized L0</dt>
                <dd>{signedPct(sel.delta_vs_centralized_L0)}</dd>
              </div>
              <div>
                <dt>Macro-F1</dt>
                <dd>{num(sel.macro_f1)}</dd>
              </div>
              <div>
                <dt>Official 21/9 split</dt>
                <dd>{pct(sel.official_split_accuracy)}</dd>
              </div>
              <div>
                <dt>Re-identification</dt>
                <dd>{pct(sel.exposure)}</dd>
              </div>
              <div>
                <dt>Advantage over chance</dt>
                <dd>{num(sel.advantage)}</dd>
              </div>
              <div>
                <dt>Per-segment re-id</dt>
                <dd>{pct(sel.segment_reid)}</dd>
              </div>
              <div>
                <dt>Bytes / window</dt>
                <dd>{bytes(sel.bytes_per_window)}</dd>
              </div>
            </dl>
          ) : (
            <p className="secondary">
              Select a point or a table row to see every measure for that cell.
            </p>
          )}
        </div>
      </section>
      {data.provenance.e1_frontier && (
        <ProvenanceLine p={data.provenance.e1_frontier} />
      )}
      {data.provenance.e4_cost && (
        <ProvenanceLine p={data.provenance.e4_cost} />
      )}
    </>
  );
}
