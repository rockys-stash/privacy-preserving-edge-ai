import { useId, useMemo, useState } from "react";
import type { Arch } from "../lib/api";
import { ARCH_LABEL, ARCHS, archVar } from "../lib/format";

export interface Point {
  id: string;
  series: Arch;
  x: number;
  y: number;
  xlo?: number;
  xhi?: number;
  ylo?: number;
  yhi?: number;
  label: string; // direct label, shown for labelled points only
  tooltip: string[];
}

export interface RefLine {
  axis: "x" | "y";
  value: number;
  label: string;
}

const W = 640;
const H = 380;
const M = { top: 16, right: 24, bottom: 46, left: 56 };

function ticks(lo: number, hi: number, log: boolean): number[] {
  if (log) {
    const out: number[] = [];
    for (
      let e = Math.floor(Math.log10(lo));
      e <= Math.ceil(Math.log10(hi));
      e++
    )
      out.push(10 ** e);
    return out.filter((t) => t >= lo * 0.999 && t <= hi * 1.001);
  }
  const span = hi - lo;
  const raw = span / 5;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step =
    [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= 6) ?? raw;
  const out: number[] = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step)
    out.push(+t.toFixed(10));
  return out;
}

function tickLabel(
  t: number,
  log: boolean,
  fmt?: (v: number) => string,
): string {
  if (fmt) return fmt(t);
  if (log) return t >= 1000 ? `${t / 1000}k` : String(t);
  return t.toFixed(Math.abs(t) < 1 && t !== 0 ? 1 : 0);
}

/**
 * Dot-and-interval scatter for the 15 cells. One y axis; architectures are the only colours;
 * L0 and the Pareto set are direct-labelled; every point has a hover/focus tooltip; the
 * surrounding page always offers the same data as a table.
 */
export function Scatter({
  points,
  xLabel,
  yLabel,
  xLog = false,
  refs = [],
  frontier = [],
  labelled,
  selected,
  onSelect,
  title,
  xFormat,
}: {
  points: Point[];
  xLabel: string;
  yLabel: string;
  xLog?: boolean;
  refs?: RefLine[];
  frontier?: string[];
  labelled: Set<string>;
  selected?: string | null;
  onSelect?: (id: string) => void;
  title: string;
  xFormat?: (v: number) => string;
}) {
  const uid = useId();
  const [hover, setHover] = useState<string | null>(null);

  const { sx, sy, xt, yt } = useMemo(() => {
    const xs = points
      .flatMap((p) => [p.xlo ?? p.x, p.xhi ?? p.x])
      .concat(refs.filter((r) => r.axis === "x").map((r) => r.value));
    const ys = points
      .flatMap((p) => [p.ylo ?? p.y, p.yhi ?? p.y])
      .concat(refs.filter((r) => r.axis === "y").map((r) => r.value));
    let x0 = Math.min(...xs);
    let x1 = Math.max(...xs);
    if (xLog) {
      x0 = 10 ** Math.floor(Math.log10(x0));
      x1 = 10 ** Math.ceil(Math.log10(x1));
    } else {
      const pad = (x1 - x0) * 0.06 || 0.05;
      x0 = Math.max(0, x0 - pad);
      x1 = x1 + pad;
    }
    const y0 = Math.max(0, Math.floor((Math.min(...ys) - 0.03) * 10) / 10);
    const y1 = Math.min(1, Math.ceil((Math.max(...ys) + 0.02) * 10) / 10);
    const fx = (v: number) =>
      M.left +
      ((xLog ? Math.log10(v) - Math.log10(x0) : v - x0) /
        (xLog ? Math.log10(x1) - Math.log10(x0) : x1 - x0)) *
        (W - M.left - M.right);
    const fy = (v: number) =>
      H - M.bottom - ((v - y0) / (y1 - y0)) * (H - M.top - M.bottom);
    return {
      sx: fx,
      sy: fy,
      xt: ticks(x0, x1, xLog),
      yt: ticks(y0, y1, false),
    };
  }, [points, refs, xLog]);

  const byId = new Map(points.map((p) => [p.id, p]));
  const front = frontier
    .map((id) => byId.get(id))
    .filter((p): p is Point => p !== undefined);
  const active = hover ?? selected ?? null;
  const tip = active ? byId.get(active) : undefined;

  return (
    <figure className="chart">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="group"
        aria-labelledby={`${uid}-t`}
        className="chart-svg"
        onMouseLeave={() => setHover(null)}
      >
        <title id={`${uid}-t`}>{title}</title>
        {yt.map((t) => (
          <g key={`y${t}`}>
            <line
              x1={M.left}
              x2={W - M.right}
              y1={sy(t)}
              y2={sy(t)}
              className="grid"
            />
            <text
              x={M.left - 8}
              y={sy(t)}
              className="tick"
              textAnchor="end"
              dominantBaseline="middle"
            >
              {t.toFixed(1)}
            </text>
          </g>
        ))}
        {xt.map((t) => (
          <g key={`x${t}`}>
            <line
              x1={sx(t)}
              x2={sx(t)}
              y1={M.top}
              y2={H - M.bottom}
              className="grid"
            />
            <text
              x={sx(t)}
              y={H - M.bottom + 16}
              className="tick"
              textAnchor="middle"
            >
              {tickLabel(t, xLog, xFormat)}
            </text>
          </g>
        ))}
        <line
          x1={M.left}
          x2={W - M.right}
          y1={H - M.bottom}
          y2={H - M.bottom}
          className="axis"
        />
        <text
          x={(M.left + W - M.right) / 2}
          y={H - 8}
          className="axis-label"
          textAnchor="middle"
        >
          {xLabel}
        </text>
        <text
          transform={`translate(14 ${(M.top + H - M.bottom) / 2}) rotate(-90)`}
          className="axis-label"
          textAnchor="middle"
        >
          {yLabel}
        </text>
        {refs.map((r) =>
          r.axis === "x" ? (
            <g key={r.label}>
              <line
                x1={sx(r.value)}
                x2={sx(r.value)}
                y1={M.top}
                y2={H - M.bottom}
                className="ref"
              />
              <text
                x={sx(r.value) + 4}
                y={H - M.bottom - 6}
                className="ref-label"
              >
                {r.label}
              </text>
            </g>
          ) : (
            <g key={r.label}>
              <line
                x1={M.left}
                x2={W - M.right}
                y1={sy(r.value)}
                y2={sy(r.value)}
                className="ref"
              />
              <text
                x={W - M.right - 4}
                y={sy(r.value) - 5}
                className="ref-label"
                textAnchor="end"
              >
                {r.label}
              </text>
            </g>
          ),
        )}
        {front.length > 1 && (
          <polyline
            className="frontier-line"
            points={front.map((p) => `${sx(p.x)},${sy(p.y)}`).join(" ")}
          />
        )}
        {ARCHS.map((a) => (
          <g key={a}>
            {points
              .filter((p) => p.series === a)
              .map((p) => (
                <g key={p.id} style={{ color: archVar(a) }}>
                  {p.ylo !== undefined && p.yhi !== undefined && (
                    <line
                      x1={sx(p.x)}
                      x2={sx(p.x)}
                      y1={sy(p.ylo)}
                      y2={sy(p.yhi)}
                      className="interval"
                    />
                  )}
                  {p.xlo !== undefined && p.xhi !== undefined && (
                    <line
                      x1={sx(p.xlo)}
                      x2={sx(p.xhi)}
                      y1={sy(p.y)}
                      y2={sy(p.y)}
                      className="interval"
                    />
                  )}
                </g>
              ))}
          </g>
        ))}
        {points.map((p) => {
          const isSel = selected === p.id;
          return (
            <g
              key={p.id}
              className="pt"
              style={{ color: archVar(p.series) }}
              onMouseEnter={() => setHover(p.id)}
              onFocus={() => setHover(p.id)}
              onBlur={() => setHover(null)}
              onClick={() => onSelect?.(p.id)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onSelect?.(p.id);
                }
              }}
              tabIndex={0}
              role="button"
              aria-pressed={onSelect ? isSel : undefined}
              aria-label={`${p.tooltip.join(", ")}`}
            >
              <circle cx={sx(p.x)} cy={sy(p.y)} r={12} className="hit" />
              {isSel && (
                <circle cx={sx(p.x)} cy={sy(p.y)} r={9} className="sel-ring" />
              )}
              <circle cx={sx(p.x)} cy={sy(p.y)} r={5} className="dot" />
              {labelled.has(p.id) && (
                <text x={sx(p.x) + 8} y={sy(p.y) - 7} className="direct-label">
                  {p.label}
                </text>
              )}
            </g>
          );
        })}
        {tip && (
          <g
            pointerEvents="none"
            transform={`translate(${Math.min(sx(tip.x) + 14, W - 220)} ${Math.max(sy(tip.y) - 20 - 16 * tip.tooltip.length, 4)})`}
          >
            <rect
              width={206}
              height={10 + 16 * tip.tooltip.length}
              rx={4}
              className="tooltip-bg"
            />
            {tip.tooltip.map((line, i) => (
              <text
                key={i}
                x={10}
                y={18 + 16 * i}
                className={i === 0 ? "tooltip-title" : "tooltip-line"}
              >
                {line}
              </text>
            ))}
          </g>
        )}
      </svg>
      <figcaption className="legend" aria-label="Legend">
        {ARCHS.map((a) => (
          <span key={a} className="arch-key">
            <span
              className="swatch"
              style={{ background: archVar(a) }}
              aria-hidden="true"
            />
            {ARCH_LABEL[a]}
          </span>
        ))}
        {front.length > 1 && (
          <span className="arch-key">
            <span className="swatch line" aria-hidden="true" />
            Pareto set
          </span>
        )}
      </figcaption>
    </figure>
  );
}
