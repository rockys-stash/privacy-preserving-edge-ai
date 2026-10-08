import { useState } from "react";
import { DataTable } from "../components/DataTable";
import {
  ErrorOrPending,
  Loading,
  PageHead,
  ProvenanceLine,
} from "../components/ui";
import {
  api,
  type Cell,
  type CostRow,
  type Frontier,
  type Modelled,
  useApi,
} from "../lib/api";
import {
  ARCH_LABEL,
  bytes,
  compact,
  num,
  pct,
  seconds,
  signedPct,
} from "../lib/format";

interface Row {
  c: Cell;
  cost?: CostRow;
  m?: Modelled;
}

export function Grid() {
  const { data, error, loading, retry } = useApi<Frontier>(api.frontier);
  const [device, setDevice] = useState("microcontroller");
  const [link, setLink] = useState("ble");
  if (loading && !data) return <Loading label="Loading the grid" rows={8} />;
  if (error || !data)
    return error ? (
      <>
        <PageHead title="Grid" />
        <ErrorOrPending
          error={error}
          retry={retry}
          what="the grid"
          command="make benchmark"
        />
      </>
    ) : null;
  const rows: Row[] = data.cells.map((c) => ({
    c,
    cost: data.costs?.measured.find(
      (r) => r.arch === c.arch && r.level === c.level,
    ),
    m: data.costs?.modelled.find(
      (r) => r.cell === c.cell && r.device === device && r.link === link,
    ),
  }));
  return (
    <>
      <PageHead title="Grid">
        Every architecture × privacy level with every measure. Accuracy and its
        interval are over held-out subjects; latency and energy are modelled for
        the device and link chosen here.
      </PageHead>
      {data.costs && (
        <div className="controls">
          <label>
            Device{" "}
            <select value={device} onChange={(e) => setDevice(e.target.value)}>
              {data.costs.profiles.devices.map((d) => (
                <option key={d.name}>{d.name}</option>
              ))}
            </select>
          </label>
          <label>
            Link{" "}
            <select value={link} onChange={(e) => setLink(e.target.value)}>
              {data.costs.profiles.links.map((d) => (
                <option key={d.name}>{d.name}</option>
              ))}
            </select>
          </label>
        </div>
      )}
      <section className="panel">
        <DataTable
          caption="Architecture by privacy level grid"
          rows={rows}
          rowKey={(r) => r.c.cell}
          columns={[
            {
              key: "cell",
              label: "Cell",
              render: (r) => `${ARCH_LABEL[r.c.arch]} L${r.c.level}`,
            },
            { key: "level", label: "Level", render: (r) => r.c.level_name },
            {
              key: "acc",
              label: "Accuracy [95% CI]",
              align: "r",
              render: (r) =>
                `${pct(r.c.accuracy)} [${pct(r.c.accuracy_lo, 0)}–${pct(r.c.accuracy_hi, 0)}]`,
            },
            {
              key: "f1",
              label: "Macro-F1",
              align: "r",
              render: (r) => num(r.c.macro_f1),
            },
            {
              key: "loss",
              label: "vs own L0",
              align: "r",
              render: (r) => signedPct(r.c.delta_vs_arch_L0),
              title: "Utility loss: accuracy minus the same architecture at L0",
            },
            {
              key: "lossc",
              label: "vs centralized L0",
              align: "r",
              render: (r) => signedPct(r.c.delta_vs_centralized_L0),
            },
            {
              key: "reid",
              label: "Re-identification",
              align: "r",
              render: (r) => pct(r.c.exposure),
            },
            {
              key: "bytes",
              label: "Bytes / window",
              align: "r",
              render: (r) => bytes(r.c.bytes_per_window),
            },
            {
              key: "zlib",
              label: "zlib B / window",
              align: "r",
              render: (r) => bytes(r.cost?.zlib_bytes_per_window),
            },
            {
              key: "dops",
              label: "Device ops",
              align: "r",
              render: (r) => compact(r.cost?.device_ops),
            },
            {
              key: "cops",
              label: "Cloud ops",
              align: "r",
              render: (r) => compact(r.cost?.cloud_ops),
            },
            {
              key: "lat",
              label: "Latency (model)",
              align: "r",
              render: (r) => seconds(r.m?.latency_s),
            },
            {
              key: "energy",
              label: "Energy µJ (model)",
              align: "r",
              render: (r) => (r.m ? num(r.m.energy_uj, 1) : "–"),
            },
          ]}
        />
      </section>
      {!data.costs && (
        <p className="secondary">
          Cost columns are Status: pending until E4 has run.
        </p>
      )}
      {Object.values(data.provenance).map((p) => (
        <ProvenanceLine key={p.experiment} p={p} />
      ))}
    </>
  );
}
