import { Link, useParams } from "react-router-dom";
import {
  ErrorOrPending,
  Loading,
  PageHead,
  ProvenanceLine,
} from "../components/ui";
import {
  api,
  type ExperimentItem,
  type ExperimentSummary,
  useApi,
} from "../lib/api";
import { cell } from "../lib/format";

const COMMANDS: Record<string, string> = {
  e1_frontier: "make e1",
  e2_adversary: "make e2",
  e3_epsilon: "make e3",
  e4_cost: "make e4",
  e5_streaming: "make e5",
};

export function Experiments() {
  const { exp } = useParams();
  const list = useApi<{ items: ExperimentItem[] }>(api.experiments);
  if (list.loading && !list.data)
    return <Loading label="Loading experiments" />;
  if (list.error || !list.data)
    return list.error ? (
      <>
        <PageHead title="Experiments" />
        <ErrorOrPending
          error={list.error}
          retry={list.retry}
          what="experiments"
          command="make benchmark"
        />
      </>
    ) : null;
  const current =
    exp ??
    list.data.items.find((i) => i.status === "complete")?.experiment ??
    "e1_frontier";
  return (
    <>
      <PageHead title="Experiments">
        Each experiment's question, findings and tables, from its latest run.
      </PageHead>
      <nav className="subnav" aria-label="Experiments">
        {list.data.items.map((i) => (
          <Link
            key={i.experiment}
            to={`/experiments/${i.experiment}`}
            aria-current={i.experiment === current ? "page" : undefined}
          >
            <span className="mono">
              {i.experiment.slice(0, 2).toUpperCase()}
            </span>{" "}
            {i.title?.replace(/^E\d /, "") ??
              i.experiment.slice(3).replace("_", " ")}
            {i.status === "pending" && (
              <span className="muted"> · pending</span>
            )}
          </Link>
        ))}
      </nav>
      <ExperimentView exp={current} />
    </>
  );
}

function ExperimentView({ exp }: { exp: string }) {
  const { data, error, loading, retry } = useApi<ExperimentSummary>(
    api.experiment(exp),
  );
  if (loading && !data) return <Loading label={`Loading ${exp}`} />;
  if (error)
    return (
      <ErrorOrPending
        error={error}
        retry={retry}
        what={exp}
        command={COMMANDS[exp] ?? "make benchmark"}
      />
    );
  if (!data) return null;
  return (
    <article className="stack">
      <section className="panel">
        <div className="panel-head">
          <h2>{data.title}</h2>
        </div>
        <div className="panel-body stack">
          <p>
            <strong>Question.</strong> {data.question}
          </p>
          <h3>Findings</h3>
          <ul className="findings">
            {data.findings.map((f) => (
              <li key={f}>{f}</li>
            ))}
          </ul>
          {data.notes.length > 0 && (
            <>
              <h3>Notes</h3>
              <ul className="notes secondary">
                {data.notes.map((n) => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            </>
          )}
        </div>
      </section>
      {data.tables.map((t) => (
        <section key={t.id} className="panel" aria-label={t.caption}>
          <div className="panel-head">
            <h3>{t.caption}</h3>
          </div>
          <div
            className="table-wrap"
            tabIndex={0}
            role="region"
            aria-label={`${t.caption} (table)`}
          >
            <table>
              <thead>
                <tr>
                  {t.columns.map((c) => (
                    <th
                      key={c.key}
                      scope="col"
                      className={c.align === "r" ? "num" : undefined}
                    >
                      {c.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {t.rows.map((r, i) => (
                  <tr key={i}>
                    {t.columns.map((c) => (
                      <td
                        key={c.key}
                        className={c.align === "r" ? "num" : undefined}
                      >
                        {cell(r[c.key])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ))}
      <ProvenanceLine p={data.provenance} />
    </article>
  );
}
