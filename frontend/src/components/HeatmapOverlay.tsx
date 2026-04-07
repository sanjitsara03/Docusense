interface HeatmapOverlayProps {
  original: string;   // object URL from File
  heatmap: string;    // base64 PNG from API
}

export function HeatmapOverlay({ original, heatmap }: HeatmapOverlayProps) {
  return (
    <div className="heatmap-grid">
      <div style={{ position: 'relative', background: 'var(--color-surface)', minWidth: 0 }}>
        <img
          src={original}
          alt="Original document"
          decoding="async"
          style={{ width: '100%', display: 'block', objectFit: 'cover', aspectRatio: '3/4' }}
        />
        <Label text="Original" />
      </div>
      <div style={{ position: 'relative', background: 'var(--color-surface)', minWidth: 0 }}>
        <img
          src={`data:image/png;base64,${heatmap}`}
          alt="Grad-CAM attention heatmap — highlighted regions indicate where the model focused when classifying"
          decoding="async"
          style={{ width: '100%', display: 'block', objectFit: 'cover', aspectRatio: '3/4' }}
        />
        <Label text="Grad-CAM" amber />
      </div>
    </div>
  );
}

function Label({ text, amber }: { text: string; amber?: boolean }) {
  return (
    <span style={{
      position: 'absolute',
      bottom: '10px',
      left: '10px',
      fontSize: '10px',
      letterSpacing: '0.1em',
      textTransform: 'uppercase',
      fontFamily: 'var(--font-body)',
      fontWeight: 500,
      color: amber ? 'var(--color-amber)' : 'var(--color-text-secondary)',
      background: 'var(--color-canvas)',
      padding: '3px 7px',
      borderRadius: '2px',
    }}>
      {text}
    </span>
  );
}
