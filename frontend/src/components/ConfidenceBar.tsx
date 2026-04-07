interface ConfidenceBarProps {
  value: number; // 0–1
}

export function ConfidenceBar({ value }: ConfidenceBarProps) {
  const pct = Math.round(value * 100);
  const isHigh = value >= 0.85;
  const isMed = value >= 0.70 && value < 0.85;

  const barColor = isHigh
    ? 'var(--color-success)'
    : isMed
    ? 'var(--color-amber)'
    : 'var(--color-danger)';

  return (
    <div>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'baseline',
        marginBottom: '8px',
      }}>
        <span
          id="confidence-label"
          style={{
            fontFamily: 'var(--font-body)',
            fontSize: '11px',
            letterSpacing: '0.1em',
            textTransform: 'uppercase',
            color: 'var(--color-text-muted)',
          }}
        >
          Confidence
        </span>
        <span style={{
          fontFamily: 'var(--font-display)',
          fontSize: '36px',
          fontWeight: 300,
          color: barColor,
          lineHeight: 1,
          letterSpacing: '-0.01em',
          transition: 'color 400ms ease',
        }}>
          {pct}
          <span style={{ fontSize: '18px', fontWeight: 300, color: 'var(--color-text-secondary)' }}>%</span>
        </span>
      </div>

      {/* Track — scaleX avoids layout thrash vs animating width */}
      <div
        role="progressbar"
        aria-labelledby="confidence-label"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
        style={{
          height: '2px',
          background: 'var(--color-border)',
          borderRadius: '1px',
          overflow: 'hidden',
        }}
      >
        <div style={{
          height: '100%',
          width: '100%',
          background: barColor,
          borderRadius: '1px',
          transform: `scaleX(${pct / 100})`,
          transformOrigin: 'left',
          transition: 'transform 600ms cubic-bezier(0.16, 1, 0.3, 1), background 400ms ease',
        }} />
      </div>
    </div>
  );
}
