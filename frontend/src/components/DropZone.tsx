import { useCallback, useEffect, useRef, useState } from 'react';

interface DropZoneProps {
  onFile: (file: File) => void;
  disabled: boolean;
}

const ACCEPTED_TYPES = new Set(['image/png', 'image/jpeg']);

export function DropZone({ onFile, disabled }: DropZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [typeError, setTypeError] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const typeErrorTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (typeErrorTimerRef.current !== null) clearTimeout(typeErrorTimerRef.current);
    };
  }, []);

  const handleFile = useCallback(
    (file: File) => {
      if (disabled) return;
      if (!ACCEPTED_TYPES.has(file.type)) {
        if (typeErrorTimerRef.current !== null) clearTimeout(typeErrorTimerRef.current);
        setTypeError(true);
        typeErrorTimerRef.current = setTimeout(() => {
          setTypeError(false);
          typeErrorTimerRef.current = null;
        }, 3000);
        return;
      }
      setTypeError(false);
      onFile(file);
    },
    [onFile, disabled]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) setIsDragging(true);
  }, [disabled]);

  const onDragLeave = useCallback(() => setIsDragging(false), []);

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = '';
  };

  const borderColor = typeError
    ? 'var(--color-danger)'
    : isDragging
    ? 'var(--color-amber)'
    : 'var(--color-border)';

  const bgColor = typeError
    ? 'var(--color-danger-surface)'
    : isDragging
    ? 'var(--color-amber-subtle)'
    : 'var(--color-surface)';

  return (
    <button
      type="button"
      aria-label="Upload document image — PNG or JPEG"
      aria-disabled={disabled || undefined}
      onClick={() => !disabled && inputRef.current?.click()}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      style={{
        width: '100%',
        border: `1px solid ${borderColor}`,
        borderRadius: 'var(--radius-lg)',
        background: bgColor,
        cursor: disabled ? 'default' : 'pointer',
        transition: 'border-color 150ms ease, background 150ms ease',
        padding: 'clamp(48px, 8vw, 80px) clamp(24px, 5vw, 64px)',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '20px',
        userSelect: 'none',
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg"
        style={{ display: 'none' }}
        onChange={onInputChange}
        disabled={disabled}
      />

      {/* Upload icon */}
      <svg
        width="40"
        height="40"
        viewBox="0 0 40 40"
        fill="none"
        aria-hidden="true"
        style={{
          color: typeError ? 'var(--color-danger)' : 'var(--color-amber)',
          opacity: isDragging || typeError ? 1 : 0.45,
          transition: 'opacity 150ms ease, color 150ms ease',
        }}
      >
        <rect x="8" y="4" width="18" height="24" rx="2" stroke="currentColor" strokeWidth="1.5" />
        <path d="M26 4l6 6v18a2 2 0 01-2 2H10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        <path d="M26 4v6h6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M20 22v-8M16 18l4-4 4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>

      <div style={{ textAlign: 'center' }}>
        <p style={{
          fontFamily: 'var(--font-display)',
          fontSize: 'clamp(20px, 3vw, 26px)',
          fontWeight: 400,
          color: typeError ? 'var(--color-danger)' : 'var(--color-text-primary)',
          margin: 0,
          letterSpacing: '0.01em',
          lineHeight: 1.2,
          transition: 'color 150ms ease',
        }}>
          {typeError
            ? 'PNG or JPEG only'
            : isDragging
            ? 'Release to analyse'
            : 'Drop a document here'}
        </p>
        <p style={{
          fontFamily: 'var(--font-body)',
          fontSize: '13px',
          color: 'var(--color-text-muted)',
          marginTop: '8px',
          letterSpacing: '0.02em',
        }}>
          {typeError ? 'That file type is not supported.' : 'or click to browse — PNG or JPEG'}
        </p>
      </div>

      <div style={{
        display: 'flex',
        gap: '8px',
        flexWrap: 'wrap',
        justifyContent: 'center',
        marginTop: '4px',
      }}>
        {['Letter', 'Form', 'Email', 'Budget', 'Invoice'].map((cls) => (
          <span
            key={cls}
            style={{
              fontSize: '11px',
              fontFamily: 'var(--font-body)',
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              color: 'var(--color-text-muted)',
              background: 'var(--color-surface-raised)',
              padding: '3px 8px',
              borderRadius: '2px',
              border: '1px solid var(--color-border-subtle)',
            }}
          >
            {cls}
          </span>
        ))}
      </div>
    </button>
  );
}
