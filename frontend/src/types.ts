export interface AnalyzeResponse {
  doc_class: string;
  confidence: number;
  heatmap_b64: string | null;
  extracted_fields: Record<string, unknown> | null;
  sagemaker_latency_ms: number;
  llm_latency_ms: number | null;
}

export type AppState =
  | { status: 'idle' }
  | { status: 'loading'; preview: string }
  | { status: 'result'; preview: string; data: AnalyzeResponse }
  | { status: 'error'; preview: string; message: string };

export const DOC_CLASS_LABELS: Record<string, string> = {
  letter: 'Letter',
  form: 'Form',
  email: 'Email',
  budget: 'Budget',
  invoice: 'Invoice',
  unknown: 'Unknown',
};
