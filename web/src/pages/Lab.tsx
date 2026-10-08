import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ErrorOrPending,
  Loading,
  PageHead,
  ProvenanceLine,
  Segmented,
} from "../components/ui";
import {
  api,
  type Arch,
  type Lab as LabData,
  type LabSegment,
  type WindowEvent,
  useApi,
} from "../lib/api";
import {
  ARCH_LABEL,
  ARCH_PATH,
  ARCHS,
  bytes,
  LEVEL_NAMES,
  seconds,
} from "../lib/format";

const GROUPS = [
  { id: "body", label: "Body acceleration", channels: [3, 4, 5], unit: "g" },
  { id: "gyro", label: "Angular velocity", channels: [6, 7, 8], unit: "rad/s" },
  { id: "total", label: "Total acceleration", channels: [0, 1, 2], unit: "g" },
] as const;
type GroupId = (typeof GROUPS)[number]["id"];
const AXES = ["x", "y", "z"];
const HALF = 32; // samples per half window at the 25 Hz the Lab trace is stored at
const STEP_MS = 640; // playback: one window per 0.64 s (twice real time)

function label(a: string) {
  return a.replace("_", " ");
}

function SensorLane({
  seg,
  w,
  group,
}: {
  seg: LabSegment;
  w: number;
  group: GroupId;
}) {
  const g = GROUPS.find((x) => x.id === group) ?? GROUPS[0];
  const n = seg.samples[0]?.length ?? 0;
  const W = 600;
  const H = 150;
  const rows = g.channels.map((c) => seg.samples[c] ?? []);
  const all = rows.flat();
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const sx = (i: number) => (i / Math.max(n - 1, 1)) * W;
  const sy = (v: number) => H - 6 - ((v - lo) / (hi - lo || 1)) * (H - 12);
  const x0 = sx(w * HALF);
  const x1 = sx(Math.min(w * HALF + 2 * HALF, n - 1));
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="trace"
      role="img"
      aria-label={`${g.label} (${g.unit}), three axes, for the whole segment; window ${w + 1} outlined`}
      preserveAspectRatio="none"
    >
      {rows.map((r, k) => (
        <polyline
          key={k}
          className={`trace-line t${k}`}
          points={r
            .map((v, i) => `${sx(i).toFixed(1)},${sy(v).toFixed(1)}`)
            .join(" ")}
          vectorEffect="non-scaling-stroke"
        />
      ))}
      <rect
        x={x0}
        y={1}
        width={Math.max(x1 - x0, 2)}
        height={H - 2}
        className="window-box"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}

function PayloadLane({
  evs,
  w,
  arch,
}: {
  evs: WindowEvent[];
  w: number;
  arch: Arch;
}) {
  const e = evs[w];
  if (!e) return null;
  const max = Math.max(...evs.map((x) => x.payload_bytes), 1);
  const carries = evs.filter((x) => x.arrived_at_s === e.arrived_at_s).length;
  const kind =
    e.payload_kind === "raw"
      ? "raw signal"
      : e.payload_kind === "features"
        ? "feature vector"
        : "decision";
  return (
    <div className="stack-sm">
      <dl className="kv">
        <div>
          <dt>Sends</dt>
          <dd>{kind}</dd>
        </div>
        <div>
          <dt>Payload</dt>
          <dd className="mono">{bytes(e.payload_bytes)}</dd>
        </div>
        <div>
          <dt>Covers</dt>
          <dd>
            {carries} window{carries === 1 ? "" : "s"}
          </dd>
        </div>
      </dl>
      <div>
        <p className="muted small lane-sub">
          First values that crossed the perimeter
        </p>
        <p className="preview mono" aria-label="Payload preview">
          {e.preview.length
            ? e.preview
                .map((v) => (Number.isInteger(v) ? v : v.toFixed(3)))
                .join(" ")
            : "–"}
          {arch !== "local" && e.preview.length >= 8 ? " …" : ""}
        </p>
      </div>
      <div
        className="bars"
        role="img"
        aria-label={`Payload bytes for each window of this segment; the largest is ${bytes(max)}`}
      >
        {evs.map((x, i) => (
          <span
            key={i}
            className={i === w ? "bar current" : "bar"}
            style={{ height: `${Math.max(4, (x.payload_bytes / max) * 100)}%` }}
          />
        ))}
      </div>
    </div>
  );
}

function CloudLane({
  evs,
  w,
  activities,
}: {
  evs: WindowEvent[];
  w: number;
  activities: string[];
}) {
  const e = evs[w];
  if (!e) return null;
  const ok = e.decision === e.truth;
  const correct = evs.filter((x) => x.decision === x.truth).length;
  return (
    <div className="stack-sm">
      <dl className="kv">
        <div>
          <dt>Decision</dt>
          <dd>
            {e.decision === null ? "–" : label(activities[e.decision] ?? "?")}
          </dd>
        </div>
        <div>
          <dt>True activity</dt>
          <dd>{label(activities[e.truth] ?? "?")}</dd>
        </div>
        <div>
          <dt>Latency</dt>
          <dd className="mono">
            {e.arrived_at_s === null
              ? "–"
              : seconds(e.arrived_at_s - e.closed_at_s)}
          </dd>
        </div>
      </dl>
      <p className={`mark ${ok ? "ok" : "bad"}`}>
        <span aria-hidden="true">{ok ? "✓" : "✗"}</span>{" "}
        {ok ? "Correct" : "Wrong"} for this window
      </p>
      <div
        className="strip"
        role="img"
        aria-label={`${correct} of ${evs.length} windows in this segment decided correctly`}
      >
        {evs.map((x, i) => (
          <span
            key={i}
            className={`cell-tick ${x.decision === x.truth ? "ok" : "bad"} ${i === w ? "current" : ""}`}
          />
        ))}
      </div>
      <p className="muted small">
        {correct} of {evs.length} windows correct in this segment
      </p>
    </div>
  );
}

export function Lab() {
  const { data, error, loading, retry } = useApi<LabData>(api.lab);
  const [segId, setSegId] = useState<number | null>(null);
  const [arch, setArch] = useState<Arch>("edge");
  const [level, setLevel] = useState<number>(0);
  const [group, setGroup] = useState<GroupId>("body");
  const [w, setW] = useState(0);
  const [playing, setPlaying] = useState(false);

  const levels = useMemo(
    () =>
      [
        ...new Set(
          Object.keys(data?.cells ?? {}).map((k) => Number(k.split("/L")[1])),
        ),
      ].sort((a, b) => a - b),
    [data],
  );
  const seg =
    data?.segments.find((s) => s.segment === segId) ?? data?.segments[0];
  const evs = useMemo(
    () =>
      data && seg
        ? (data.cells[`${arch}/L${level}`] ?? []).filter(
            (e) => e.segment === seg.segment,
          )
        : [],
    [data, seg, arch, level],
  );
  const last = Math.max(evs.length - 1, 0);

  useEffect(() => setW(0), [seg?.segment]);
  useEffect(() => {
    if (w > last) setW(last);
  }, [w, last]);
  useEffect(() => {
    if (!playing) return;
    const t = window.setInterval(
      () => setW((x) => (x >= last ? 0 : x + 1)),
      STEP_MS,
    );
    return () => window.clearInterval(t);
  }, [playing, last]);

  const onKey = useCallback(
    (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (
        t &&
        (t.tagName === "INPUT" ||
          t.tagName === "SELECT" ||
          t.tagName === "TEXTAREA" ||
          t.isContentEditable)
      )
        return;
      if (e.key === " " && t?.tagName !== "BUTTON") {
        e.preventDefault();
        setPlaying((p) => !p);
      } else if (e.key === "ArrowRight") {
        setPlaying(false);
        setW((x) => Math.min(x + 1, last));
      } else if (e.key === "ArrowLeft") {
        setPlaying(false);
        setW((x) => Math.max(x - 1, 0));
      }
    },
    [last],
  );
  useEffect(() => {
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onKey]);

  if (loading && !data) return <Loading label="Loading the lab replay" />;
  if (error || !data || !seg)
    return error ? (
      <>
        <PageHead title="Lab" />
        <ErrorOrPending
          error={error}
          retry={retry}
          what="the lab replay"
          command="make e5"
        />
      </>
    ) : null;

  return (
    <>
      <PageHead title="Lab">
        Test subject {data.subject}, replayed sample by sample through device,
        link and cloud. The models never saw this person. Choose a recording
        segment, an architecture and a privacy level.
      </PageHead>
      <div className="controls lab-controls">
        <label>
          Segment{" "}
          <select
            value={seg.segment}
            onChange={(e) => setSegId(Number(e.target.value))}
          >
            {data.segments.map((s) => (
              <option key={s.segment} value={s.segment}>
                {`#${s.segment} · ${label(data.activities[s.activity] ?? "?")} · ${s.n_windows} windows`}
              </option>
            ))}
          </select>
        </label>
        <Segmented
          label="Architecture"
          options={ARCHS.map((a) => ({
            id: a,
            label: ARCH_LABEL[a],
            title: ARCH_PATH[a],
          }))}
          value={arch}
          onChange={setArch}
        />
        <Segmented
          label="Privacy level"
          options={levels.map((l) => ({
            id: l,
            label: `L${l}`,
            title: LEVEL_NAMES[l],
          }))}
          value={level}
          onChange={setLevel}
        />
      </div>
      <p className="muted small">
        {ARCH_LABEL[arch]}: {ARCH_PATH[arch]}. Level L{level},{" "}
        {LEVEL_NAMES[level]}. The streamed replay covers the levels E5 ran;
        every level is in the Frontier and Grid.
      </p>

      <div className="lanes">
        <section className="panel lane" aria-labelledby="lane-sensor">
          <div className="panel-head wrap">
            <h2 id="lane-sensor">Sensor</h2>
            <Segmented
              label="Channels"
              options={GROUPS.map((g) => ({
                id: g.id,
                label: g.label.split(" ")[0] ?? g.label,
                title: `${g.label} (${g.unit})`,
              }))}
              value={group}
              onChange={setGroup}
            />
          </div>
          <div className="panel-body">
            <SensorLane seg={seg} w={w} group={group} />
            <p className="legend small">
              {AXES.map((a, k) => (
                <span key={a} className="arch-key">
                  <span className={`swatch line t${k}`} aria-hidden="true" />
                  {a}
                </span>
              ))}
              <span className="arch-key">
                <span className="swatch box" aria-hidden="true" />
                current 2.56 s window
              </span>
            </p>
          </div>
        </section>
        <section
          className="panel lane perimeter"
          aria-labelledby="lane-perimeter"
        >
          <div className="panel-head">
            <h2 id="lane-perimeter">Perimeter</h2>
          </div>
          <div className="panel-body">
            <PayloadLane evs={evs} w={w} arch={arch} />
          </div>
        </section>
        <section className="panel lane" aria-labelledby="lane-cloud">
          <div className="panel-head">
            <h2 id="lane-cloud">
              {arch === "local" ? "Decision (sent from device)" : "Cloud"}
            </h2>
          </div>
          <div className="panel-body">
            <CloudLane evs={evs} w={w} activities={data.activities} />
          </div>
        </section>
      </div>

      <div className="transport" role="group" aria-label="Replay transport">
        <button
          className="btn"
          type="button"
          onClick={() => setW((x) => Math.max(x - 1, 0))}
          aria-label="Previous window"
          aria-keyshortcuts="ArrowLeft"
        >
          ←
        </button>
        <button
          className="btn primary"
          type="button"
          onClick={() => setPlaying((p) => !p)}
          aria-keyshortcuts="Space"
        >
          {playing ? "Pause" : "Play"}
        </button>
        <button
          className="btn"
          type="button"
          onClick={() => setW((x) => Math.min(x + 1, last))}
          aria-label="Next window"
          aria-keyshortcuts="ArrowRight"
        >
          →
        </button>
        <label className="grow">
          <span className="sr-only">Window</span>
          <input
            type="range"
            min={0}
            max={last}
            value={w}
            onChange={(e) => setW(Number(e.target.value))}
            aria-valuetext={`window ${w + 1} of ${evs.length}`}
          />
        </label>
        <span className="mono" aria-live="polite">
          window {w + 1}/{evs.length}
        </span>
        <span className="muted small hide-narrow">
          Space plays or pauses; ← → step.
        </span>
      </div>
      <ProvenanceLine p={data.provenance} />
    </>
  );
}
