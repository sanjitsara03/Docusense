interface FieldsTableProps {
  docClass: string;
  fields: Record<string, unknown>;
}

export function FieldsTable({ docClass, fields }: FieldsTableProps) {
  const rows = flattenFields(fields);

  if (rows.length === 0) return null;

  return (
    <div>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
        marginBottom: '16px',
      }}>
        <span style={{
          fontFamily: 'var(--font-body)',
          fontSize: '11px',
          letterSpacing: '0.1em',
          textTransform: 'uppercase',
          color: 'var(--color-text-muted)',
        }}>
          Extracted fields
        </span>
        <span style={{
          fontSize: '10px',
          letterSpacing: '0.06em',
          textTransform: 'uppercase',
          fontFamily: 'var(--font-body)',
          color: 'var(--color-amber)',
          background: 'var(--color-amber-subtle)',
          padding: '2px 7px',
          borderRadius: '2px',
        }}>
          {docClass}
        </span>
      </div>

      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <caption className="sr-only">Extracted document fields for {docClass}</caption>
        <tbody>
          {rows.map(({ key, value }, i) => (
            <tr
              key={key}
              style={{
                borderTop: i === 0 ? 'none' : '1px solid var(--color-border-subtle)',
              }}
            >
              <th
                scope="row"
                style={{
                  fontFamily: 'var(--font-body)',
                  fontWeight: 400,
                  fontSize: '12px',
                  letterSpacing: '0.04em',
                  color: 'var(--color-text-muted)',
                  padding: '10px 12px 10px 0',
                  width: '38%',
                  verticalAlign: 'top',
                  textAlign: 'left',
                  overflowWrap: 'break-word',
                  minWidth: 0,
                }}
              >
                {humanize(key)}
              </th>
              <td style={{
                fontFamily: value && typeof value === 'string' && value.length > 60
                  ? 'var(--font-body)'
                  : 'var(--font-display)',
                fontSize: value && typeof value === 'string' && value.length > 60
                  ? '13px'
                  : '15px',
                fontWeight: 400,
                color: 'var(--color-text-primary)',
                padding: '10px 0',
                verticalAlign: 'top',
                lineHeight: 1.5,
                overflowWrap: 'break-word',
                minWidth: 0,
              }}>
                {value != null
                  ? <span style={{ whiteSpace: 'pre-line' }}>{value}</span>
                  : <em style={{ color: 'var(--color-text-muted)', fontStyle: 'italic' }}>—</em>
                }
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

interface Row {
  key: string;
  value: string | null;
}

function flattenFields(fields: Record<string, unknown>): Row[] {
  const rows: Row[] = [];

  for (const [key, val] of Object.entries(fields)) {
    if (Array.isArray(val)) {
      if (val.length === 0) continue;
      const items = val.map((item) => {
        if (typeof item === 'object' && item !== null) {
          return Object.values(item as Record<string, unknown>)
            .map((v) => String(v ?? ''))
            .filter(Boolean)
            .join(' · ');
        }
        return String(item);
      });
      rows.push({ key, value: items.join('\n') });
    } else if (typeof val === 'object' && val !== null) {
      for (const [subKey, subVal] of Object.entries(val as Record<string, unknown>)) {
        rows.push({ key: `${key} / ${subKey}`, value: subVal != null ? String(subVal) : null });
      }
    } else {
      rows.push({ key, value: val != null ? String(val) : null });
    }
  }

  return rows;
}

function humanize(key: string): string {
  return key
    .replace(/_/g, ' ')
    .replace(/\//g, ' / ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}
