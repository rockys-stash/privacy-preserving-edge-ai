import type { Arch } from "./api";

export const ARCHS: Arch[] = ["centralized", "edge", "local"];
export const ARCH_LABEL: Record<Arch, string> = {
  centralized: "Centralized",
  edge: "Edge",
  local: "Fully local",
};
export const ARCH_PATH: Record<Arch, string> = {
  centralized: "sensor → raw → cloud → AI",
  edge: "sensor → local features → cloud",
  local: "sensor → edge AI → decision",
};
/** Validated with the dataviz palette checker in both themes (docs/DESIGN.md). Colours live in CSS. */
export const archVar = (a: Arch) => `var(--series-${a})`;

export const LEVEL_NAMES = [
  "full",
  "minimise",
  "coarsen",
  "perturb",
  "aggregate",
];

export function num(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  return v.toFixed(digits);
}

export function pct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  return `${(v * 100).toFixed(digits)}%`;
}

export function signedPct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  const s = (v * 100).toFixed(digits);
  return v > 0 ? `+${s} pt` : `${s.replace("-", "−")} pt`;
}

export function bytes(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  if (v >= 1000) return `${(v / 1000).toFixed(v >= 10000 ? 0 : 1)} kB`;
  if (v >= 10) return `${v.toFixed(0)} B`;
  return `${v.toFixed(2)} B`;
}

export function compact(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  const a = Math.abs(v);
  if (a >= 1e9) return `${(v / 1e9).toFixed(1)} G`;
  if (a >= 1e6) return `${(v / 1e6).toFixed(1)} M`;
  if (a >= 1e3) return `${(v / 1e3).toFixed(1)} k`;
  if (a === 0) return "0";
  if (a < 0.01) return v.toExponential(1);
  return v.toFixed(a < 10 ? 2 : 0);
}

export function seconds(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  return v < 1
    ? `${(v * 1000).toFixed(v < 0.01 ? 1 : 0)} ms`
    : `${v.toFixed(2)} s`;
}

export function cell(v: unknown): string {
  if (v === null || v === undefined) return "–";
  if (typeof v === "boolean") return v ? "yes" : "";
  if (typeof v === "number") {
    const a = Math.abs(v);
    if (Number.isInteger(v)) return v.toLocaleString("en");
    if (a !== 0 && (a >= 1e5 || a < 1e-3)) return v.toPrecision(3);
    return a < 100
      ? v.toFixed(3)
      : v.toLocaleString("en", { maximumFractionDigits: 1 });
  }
  return String(v);
}
