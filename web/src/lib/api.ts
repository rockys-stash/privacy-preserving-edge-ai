import { useCallback, useEffect, useRef, useState } from "react";

/* ---------- types mirrored from the API (src/edgebench/api.py) ---------- */
export type Arch = "centralized" | "edge" | "local";

export interface Cell {
  cell: string;
  arch: Arch;
  level: number;
  level_name: string;
  accuracy: number;
  accuracy_lo: number;
  accuracy_hi: number;
  macro_f1: number;
  official_split_accuracy: number;
  exposure: number;
  exposure_lo: number;
  exposure_hi: number;
  advantage: number;
  segment_reid: number;
  bytes_per_window: number;
  subjects: number;
  fold_runs: number;
  delta_vs_arch_L0: number;
  delta_vs_centralized_L0: number;
  pareto: boolean;
}

export interface Baselines {
  majority_accuracy: number;
  chance_reidentification: number;
  subjects: number;
}

export interface CostRow {
  arch: Arch;
  level: number;
  bytes_per_window: number;
  zlib_bytes_per_window: number;
  uplink_bps: number;
  device_ops: number;
  cloud_ops: number;
  device_ms_measured: number;
  cloud_ms_measured: number;
  block_wait_s: number;
}

export interface Modelled {
  cell: string;
  arch: Arch;
  level: number;
  device: string;
  link: string;
  latency_s: number;
  energy_uj: number;
}

export interface Profiles {
  devices: { name: string; ops_per_s: number; joule_per_op: number }[];
  links: {
    name: string;
    rate_bps: number;
    one_way_s: number;
    joule_per_byte: number;
  }[];
}

export interface Provenance {
  experiment: string;
  run_id: string;
  commit: string;
  dirty: boolean;
  created_at: string;
  runtime_seconds: number;
  cpus: number;
  subjects: string | number[];
  dataset_sha256: string;
}

export interface Frontier {
  cells: Cell[];
  baselines: Baselines;
  costs: {
    measured: CostRow[];
    modelled: Modelled[];
    profiles: Profiles;
  } | null;
  provenance: Record<string, Provenance>;
}

export interface WindowEvent {
  window: number;
  segment: number;
  closed_at_s: number;
  truth: number;
  payload_kind: "raw" | "features" | "decision" | "";
  payload_bytes: number;
  sent: boolean;
  arrived_at_s: number | null;
  decision: number | null;
  preview: number[];
}

export interface LabSegment {
  segment: number;
  activity: number;
  n_windows: number;
  samples: number[][]; // 9 channels, every 2nd sample (25 Hz)
}

export interface Lab {
  subject: number;
  segments: LabSegment[];
  activities: string[];
  cells: Record<string, WindowEvent[]>;
  provenance: Provenance;
}

export interface SummaryTable {
  id: string;
  caption: string;
  columns: { key: string; label: string; align?: "l" | "r" }[];
  rows: Record<string, string | number | boolean | null>[];
}

export interface ExperimentSummary {
  title: string;
  question: string;
  findings: string[];
  tables: SummaryTable[];
  notes: string[];
  provenance: Provenance;
}

export interface ExperimentItem extends Partial<Provenance> {
  experiment: string;
  status: "complete" | "pending";
  title?: string;
  question?: string;
}

/* ---------- fetching ---------- */
export class ApiError extends Error {
  constructor(
    public url: string,
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

const cache = new Map<string, unknown>();

export async function getJson<T>(url: string): Promise<T> {
  if (cache.has(url)) return cache.get(url) as T;
  let res: Response;
  try {
    res = await fetch(url, { headers: { Accept: "application/json" } });
  } catch {
    throw new ApiError(url, 0, "The benchmark API could not be reached.");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(url, res.status, detail);
  }
  const data = (await res.json()) as T;
  cache.set(url, data);
  return data;
}

export interface ApiState<T> {
  data: T | undefined;
  error: ApiError | undefined;
  loading: boolean;
  retry: () => void;
}

export function useApi<T>(url: string | null): ApiState<T> {
  const [data, setData] = useState<T | undefined>(undefined);
  const [error, setError] = useState<ApiError | undefined>(undefined);
  const [loading, setLoading] = useState<boolean>(url !== null);
  const [nonce, setNonce] = useState(0);
  const current = useRef(url);

  useEffect(() => {
    current.current = url;
    if (url === null) {
      setLoading(false);
      return;
    }
    let alive = true;
    setLoading(true);
    setError(undefined);
    getJson<T>(url)
      .then((d) => {
        if (alive && current.current === url) setData(d);
      })
      .catch((e: unknown) => {
        if (alive && current.current === url)
          setError(e instanceof ApiError ? e : new ApiError(url, 0, String(e)));
      })
      .finally(() => {
        if (alive && current.current === url) setLoading(false);
      });
    return () => {
      alive = false;
    };
  }, [url, nonce]);

  const retry = useCallback(() => {
    if (url) cache.delete(url);
    setNonce((n) => n + 1);
  }, [url]);

  return { data, error, loading, retry };
}

export const api = {
  frontier: "/api/frontier",
  lab: "/api/lab",
  experiments: "/api/experiments",
  experiment: (exp: string) => `/api/experiments/${encodeURIComponent(exp)}`,
};
