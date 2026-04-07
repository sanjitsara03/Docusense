import type { AnalyzeResponse } from '../types';
import { DOC_CLASS_LABELS } from '../types';
import { ConfidenceBar } from './ConfidenceBar';
import { HeatmapOverlay } from './HeatmapOverlay';
import { FieldsTable } from './FieldsTable';

interface ResultPanelProps {
  data: AnalyzeResponse;
  previewUrl: string;
  onReset: () => void;
}

export function ResultPanel({ data, previewUrl, onReset }: ResultPanelProps) {
  const isUnknown = data.doc_class === 'unknown' || data.confidence < 0.70;
  const label = DOC_CLASS_LABELS[data.doc_class] ?? data.doc_class;

  return (
    <div className="anim-fade-up-slow">
      {/* Header row */}
      <div className="result-header">
        <div>
          <p style={{
            fontFamily: 'var(--font-body)',
            fontSize: '11px',
            letterSpacing: '0.1em',
            textTransform: 'uppercase',
            color: 'var(--color-text-muted)',
            margin: '0 0 6px',
          }}>
            Document class
          </p>
          <h2 style={{
            fontFamily: 'var(--font-display)',
            fontSize: 'clamp(32px, 5vw, 48px)',
            fontWeight: isUnknown ? 300 : 400,
            fontStyle: isUnknown ? 'italic' : 'normal',
            color: isUnknown ? 'var(--color-danger)' : 'var(--color-text-primary)',
            margin: 0,
            lineHeight: 1,
            letterSpacing: isUnknown ? '0.02em' : '-0.01em',
          }}>
            {label}
          </h2>
        </div>

        <button className="ghost-btn" onClick={onReset}>
          New document
        </button>
      </div>

      {/* Confidence */}
      <div style={{ marginBottom: '32px' }}>
        <ConfidenceBar value={data.confidence} />
      </div>

      <hr className="divider" />

      {/* Heatmap */}
      {data.heatmap_b64 && !isUnknown && (
        <div style={{ marginBottom: '32px' }}>
          <HeatmapOverlay original={previewUrl} heatmap={data.heatmap_b64} />
        </div>
      )}

      {/* Unknown state — show original image only */}
      {isUnknown && (
        <div style={{ marginBottom: '32px' }}>
          <div style={{
            borderRadius: 'var(--radius-md)',
            overflow: 'hidden',
            border: '1px solid var(--color-border)',
          }}>
            <img
              src={previewUrl}
              alt="Uploaded document"
              decoding="async"
              style={{ width: '100%', display: 'block', aspectRatio: '3/4', maxHeight: '320px', objectFit: 'cover' }}
            />
          </div>
          <p style={{
            fontFamily: 'var(--font-display)',
            fontStyle: 'italic',
            fontSize: '15px',
            color: 'var(--color-text-muted)',
            marginTop: '12px',
            textAlign: 'center',
          }}>
            Confidence below threshold — classification skipped
          </p>
        </div>
      )}

      {/* Extracted fields */}
      {data.extracted_fields && !isUnknown && (
        <>
          <hr className="divider" />
          <FieldsTable docClass={label} fields={data.extracted_fields} />
        </>
      )}

      {/* Latency footer */}
      <div style={{
        display: 'flex',
        gap: '24px',
        marginTop: '40px',
        paddingTop: '20px',
        borderTop: '1px solid var(--color-border-subtle)',
      }}>
        <Latency label="SageMaker" ms={data.sagemaker_latency_ms} />
        {data.llm_latency_ms != null && (
          <Latency label="LLM extraction" ms={data.llm_latency_ms} />
        )}
      </div>
    </div>
  );
}

function Latency({ label, ms }: { label: string; ms: number }) {
  if (!Number.isFinite(ms)) return null;
  return (
    <div>
      <p style={{
        fontFamily: 'var(--font-body)',
        fontSize: '10px',
        letterSpacing: '0.1em',
        textTransform: 'uppercase',
        color: 'var(--color-text-muted)',
        margin: '0 0 3px',
      }}>
        {label}
      </p>
      <p style={{
        fontFamily: 'var(--font-display)',
        fontSize: '20px',
        fontWeight: 300,
        color: 'var(--color-text-secondary)',
        margin: 0,
        lineHeight: 1,
      }}>
        {ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(2)}s`}
      </p>
    </div>
  );
}
