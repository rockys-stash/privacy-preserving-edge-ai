import type { ReactNode } from "react";

export interface Column<R> {
  key: string;
  label: string;
  align?: "l" | "r";
  render: (row: R) => ReactNode;
  title?: string;
}

/** A plain table inside a focusable scroll region (tables scroll sideways on narrow screens). */
export function DataTable<R>({
  caption,
  columns,
  rows,
  rowKey,
  selected,
  onSelect,
}: {
  caption: string;
  columns: Column<R>[];
  rows: R[];
  rowKey: (r: R) => string;
  selected?: string | null;
  onSelect?: (key: string) => void;
}) {
  return (
    <div className="table-wrap" tabIndex={0} role="region" aria-label={caption}>
      <table>
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr>
            {columns.map((c) => (
              <th
                key={c.key}
                scope="col"
                className={c.align === "r" ? "num" : undefined}
                title={c.title}
              >
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const k = rowKey(r);
            return (
              <tr
                key={k}
                aria-selected={onSelect ? selected === k : undefined}
                className={selected === k ? "selected" : undefined}
              >
                {columns.map((c, i) => (
                  <td
                    key={c.key}
                    className={c.align === "r" ? "num" : undefined}
                  >
                    {i === 0 && onSelect ? (
                      <button
                        type="button"
                        className="linkish"
                        onClick={() => onSelect(k)}
                        aria-pressed={selected === k}
                      >
                        {c.render(r)}
                      </button>
                    ) : (
                      c.render(r)
                    )}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
