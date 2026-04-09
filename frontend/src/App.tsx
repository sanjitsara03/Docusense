import { useCallback, useEffect, useRef, useState } from 'react';
import type { AppState, AnalyzeResponse } from './types';
import { DropZone } from './components/DropZone';
import { ResultPanel } from './components/ResultPanel';

const MAX_FILE_SIZE_MB = 10;

export default function App() {
  const [state, setState] = useState<AppState>({ status: 'idle' });
  const [announcement, setAnnouncement] = useState('');
  const previewUrlRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const mainRef = useRef<HTMLElement>(null);
  const isFirstRender = useRef(true);

  // Abort in-flight request on unmount
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    };
  }, []);

  // Focus management: move focus to new content after each state transition so
  // keyboard and screen-reader users don't get stranded on an unmounted element.
  useEffect(() => {
    if (isFirstRender.current) {
      isFirstRender.current = false;
      return;
    }
    if (state.status === 'loading') return; // no stable interactive target yet

    let frameId: number;
    frameId = requestAnimationFrame(() => {
      const el = mainRef.current;
      if (!el) return;
      // Prefer the first button in the new state; fall back to the main container.
      const firstButton = el.querySelector<HTMLElement>('button');
      (firstButton ?? el).focus();
    });
    return () => cancelAnimationFrame(frameId);
  }, [state.status]);

  const handleFile = useCallback(async (file: File) => {
    // File size guard
    if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      setAnnouncement(`Error: File too large. Maximum size is ${MAX_FILE_SIZE_MB}MB.`);
      setState((prev) =>
        prev.status === 'idle'
          ? { ...prev }
          : { status: 'error', preview: previewUrlRef.current ?? '', message: `File exceeds ${MAX_FILE_SIZE_MB}MB limit.` }
      );
      return;
    }

    // Cancel any previous in-flight request
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    const preview = URL.createObjectURL(file);
    previewUrlRef.current = preview;

    setState({ status: 'loading', preview });
    setAnnouncement('Analysing document…');

    try {
      const form = new FormData();
      form.append('file', file);

      const res = await fetch('/analyze', {
        method: 'POST',
        body: form,
        signal: controller.signal,
      });

      if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(text || `Server error (HTTP ${res.status})`);
      }

      const data: AnalyzeResponse = await res.json();
      setState({ status: 'result', preview, data });
      setAnnouncement(
        `Classification complete: ${data.doc_class}, ${Math.round(data.confidence * 100)}% confidence`
      );
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') return; // user navigated away
      const message = err instanceof Error ? err.message : 'Unknown error. Please try again.';
      setState({ status: 'error', preview, message });
      setAnnouncement(`Error: ${message}`);
    }
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
    setState({ status: 'idle' });
    setAnnouncement('');
  }, []);

  return (
    <div style={{
      minHeight: '100svh',
      display: 'grid',
      gridTemplateRows: 'auto 1fr auto',
    }}>
      {/* Skip link — visible only on keyboard focus */}
      <a href="#main-content" className="skip-link">Skip to main content</a>

      {/* Screen-reader live region for state transitions */}
      <div className="sr-only" aria-live="polite" aria-atomic="true">
        {announcement}
      </div>

      {/* Header */}
      <header style={{
        borderBottom: '1px solid var(--color-border)',
        padding: '0 var(--layout-inset)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        height: '56px',
      }}>
        {/* Persistent h1 anchors heading hierarchy for all states */}
        <h1 className="sr-only">DocuSense</h1>

        <div style={{ display: 'flex', alignItems: 'baseline', gap: '10px' }} aria-hidden="true">
          <span style={{
            fontFamily: 'var(--font-display)',
            fontSize: '20px',
            fontWeight: 500,
            color: 'var(--color-text-primary)',
            letterSpacing: '0.01em',
          }}>
            DocuSense
          </span>
          <span
            className="site-header-subtitle"
            style={{
              fontFamily: 'var(--font-body)',
              fontSize: '11px',
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              color: 'var(--color-text-muted)',
            }}
          >
            Document Intelligence
          </span>
        </div>

      </header>

      {/* Main */}
      <main
        ref={mainRef}
        id="main-content"
        tabIndex={-1}
        aria-busy={state.status === 'loading'}
        style={{
          padding: 'clamp(40px, 6vw, 80px) var(--layout-inset)',
          maxWidth: '900px',
          margin: '0 auto',
          width: '100%',
        }}
      >

        {/* Idle */}
        {state.status === 'idle' && (
          <div className="anim-fade-up">
            <div style={{ marginBottom: 'clamp(32px, 5vw, 56px)' }}>
              <h2 style={{
                fontFamily: 'var(--font-display)',
                fontSize: 'clamp(40px, 7vw, 72px)',
                fontWeight: 300,
                fontStyle: 'italic',
                color: 'var(--color-text-primary)',
                margin: '0 0 16px',
                lineHeight: 1.05,
                letterSpacing: '-0.01em',
              }}>
                Upload a document.
              </h2>
              <p style={{
                fontFamily: 'var(--font-body)',
                fontSize: '15px',
                color: 'var(--color-text-secondary)',
                maxWidth: '480px',
                lineHeight: 1.65,
                margin: 0,
              }}>
                EfficientNet-B0 classifies it and Claude extracts structured fields — all in one call.
              </p>
            </div>

            <DropZone onFile={handleFile} disabled={false} />
          </div>
        )}

        {/* Loading */}
        {state.status === 'loading' && (
          <div className="two-col-grid anim-fade-up">
            <div style={{
              borderRadius: 'var(--radius-md)',
              overflow: 'hidden',
              border: '1px solid var(--color-border)',
              minWidth: 0,
            }}>
              <img
                src={state.preview}
                alt="Uploaded document being analysed"
                width="600"
                height="800"
                decoding="async"
                style={{ width: '100%', display: 'block', aspectRatio: '3/4', objectFit: 'cover' }}
              />
            </div>

            <div style={{ paddingTop: '8px', minWidth: 0 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '28px' }}>
                <Spinner />
                <span style={{
                  fontFamily: 'var(--font-body)',
                  fontSize: '13px',
                  color: 'var(--color-text-muted)',
                  letterSpacing: '0.02em',
                }}>
                  Analysing…
                </span>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {(['Classifying document', 'Extracting fields'] as const).map((step, i) => (
                  <SkeletonStep key={step} label={step} delayMs={i * 180} />
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Result */}
        {state.status === 'result' && (
          <ResultPanel
            data={state.data}
            previewUrl={state.preview}
            onReset={reset}
          />
        )}

        {/* Error */}
        {state.status === 'error' && (
          <div className="two-col-grid anim-fade-up">
            <div style={{
              borderRadius: 'var(--radius-md)',
              overflow: 'hidden',
              border: '1px solid var(--color-border)',
              minWidth: 0,
            }}>
              <img
                src={state.preview}
                alt="Uploaded document"
                width="600"
                height="800"
                decoding="async"
                style={{ width: '100%', display: 'block', aspectRatio: '3/4', objectFit: 'cover' }}
              />
            </div>
            <div style={{ paddingTop: '8px', minWidth: 0 }}>
              <p style={{
                fontFamily: 'var(--font-body)',
                fontSize: '11px',
                letterSpacing: '0.1em',
                textTransform: 'uppercase',
                color: 'var(--color-danger)',
                margin: '0 0 12px',
              }}>
                Error
              </p>
              <p style={{
                fontFamily: 'var(--font-display)',
                fontStyle: 'italic',
                fontSize: '18px',
                color: 'var(--color-text-secondary)',
                margin: '0 0 28px',
                lineHeight: 1.5,
                overflowWrap: 'break-word',
              }}>
                {state.message}
              </p>
              <button className="ghost-btn" onClick={reset}>
                Try again
              </button>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="site-footer">
        <span style={{
          fontFamily: 'var(--font-body)',
          fontSize: '11px',
          color: 'var(--color-text-muted)',
          letterSpacing: '0.04em',
          whiteSpace: 'nowrap',
        }}>
          5 classes · RVL-CDIP · EfficientNet-B0
        </span>
        <span style={{
          fontFamily: 'var(--font-body)',
          fontSize: '11px',
          color: 'var(--color-text-muted)',
          letterSpacing: '0.04em',
          whiteSpace: 'nowrap',
        }}>
          Deployed on AWS SageMaker
        </span>
      </footer>
    </div>
  );
}

function Spinner() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      fill="none"
      aria-hidden="true"
      className="anim-spin"
      style={{ flexShrink: 0 }}
    >
      <circle cx="8" cy="8" r="6" stroke="var(--color-border)" strokeWidth="1.5" />
      <path d="M8 2a6 6 0 016 6" stroke="var(--color-amber)" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function SkeletonStep({ label, delayMs }: { label: string; delayMs: number }) {
  return (
    <div
      className="step-anim"
      style={{ '--anim-delay': `${delayMs}ms`, display: 'flex', alignItems: 'center', gap: '10px' } as React.CSSProperties}
    >
      <div
        className="pulse-dot"
        style={{ '--anim-delay': `${delayMs}ms` } as React.CSSProperties}
      />
      <span style={{
        fontFamily: 'var(--font-body)',
        fontSize: '13px',
        color: 'var(--color-text-muted)',
        letterSpacing: '0.02em',
      }}>
        {label}
      </span>
    </div>
  );
}
