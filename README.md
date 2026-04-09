# DocuSense

Intelligent document classification and field extraction pipeline. Upload a document image — EfficientNet-B0 classifies it, Claude extracts structured fields.

Built to demonstrate end-to-end MLOps on AWS SageMaker.

---

## Architecture

```
Document Image
    │
    ▼
SageMaker Real-Time Endpoint
    │  EfficientNet-B0 (fine-tuned)
    │  Returns: class + confidence score
    │
    ├── confidence < 0.70 → return "unknown", skip extraction
    │
    ▼
Claude Sonnet (Anthropic API)
    │  Structured extraction per document type
    │  Returns: typed JSON fields
    │
    ▼
FastAPI Backend → React Frontend
```

## ML Pipeline (SageMaker Pipelines)

```
Preprocess → Train → Evaluate → Condition (accuracy > 0.90) → Register Model
```

- **Preprocess** — resize images to 256×256, write train/val/test splits to S3
- **Train** — EfficientNet-B0 fine-tune on `ml.g4dn.4xlarge` (T4 GPU), cosine LR schedule, AdamW
- **Evaluate** — accuracy + per-class F1 + confusion matrix on held-out test set
- **Condition** — only registers the model if test accuracy exceeds 0.90
- **Register** — model stored in SageMaker Model Registry with `PendingManualApproval`

---

## Results

Trained on [RVL-CDIP](https://huggingface.co/datasets/aharley/rvl_cdip) filtered to 5 classes (~100k train / 12.5k val / 12.5k test images).

| Metric | Value |
|--------|-------|
| Test accuracy | 94.74% |
| Macro F1 | 94.72% |

| Class | F1 |
|-------|----|
| Letter | 95.64% |
| Form | 94.25% |
| Email | 98.79% |
| Budget | 91.98% |
| Invoice | 92.96% |

Training: 10 epochs, ~80 min on `ml.g4dn.4xlarge`.

---

## Document Classes

| Class | RVL-CDIP Index | Extracted Fields |
|-------|---------------|-----------------|
| Letter | 0 | sender, recipient, date, subject, summary |
| Form | 1 | form_title, fields (name/value pairs) |
| Email | 2 | sender, recipient, date, subject, summary, action_items |
| Budget | 10 | title, date, line_items, total, notes |
| Invoice | 11 | vendor, invoice_number, amount, currency, line_items, due_date |

---

## Stack

| Layer | Technology |
|-------|-----------|
| Model training | PyTorch, EfficientNet-B0 (ImageNet pretrained) |
| ML pipeline | AWS SageMaker Pipelines, Model Registry |
| Inference | SageMaker Real-Time Endpoint (`ml.m5.large`) |
| Field extraction | Claude Sonnet (Anthropic API), Pydantic schemas |
| Backend | FastAPI, Python |
| Frontend | React, Vite, TypeScript, Tailwind CSS v4 |
| Data | RVL-CDIP via HuggingFace, stored on S3 |
| Package manager | `uv` |

---

## Project Structure

```
Docusense/
  training/
    train.py          # EfficientNet-B0 fine-tune, checkpoint resumption
    dataset.py        # RVL-CDIP loader filtered to 5 classes
    evaluate.py       # metrics + confusion matrix on eval set
  pipeline/
    pipeline.py       # SageMaker Pipelines DAG (4 steps)
    preprocess.py     # resize + normalize, parallelized with multiprocessing
  backend/
    main.py           # FastAPI app — /classify, /extract, /analyze
    extractor.py      # Claude extraction agents + Pydantic schemas per doc type
  frontend/           # Vite + React + TypeScript + Tailwind
  scripts/
    deploy_endpoint.py
    delete_endpoint.py
    register_model.py
    run_evaluate.py
```

---

## Running Locally

### Prerequisites

- Python 3.12+, `uv`
- Node.js 18+
- AWS credentials configured (`~/.aws/credentials`)
- `ANTHROPIC_API_KEY` set in environment

### Deploy the endpoint

```bash
uv run scripts/deploy_endpoint.py
```

### Start the backend

```bash
cd backend
ANTHROPIC_API_KEY=<key> uv run uvicorn main:app --port 8000
```

### Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

### Tear down

```bash
uv run scripts/delete_endpoint.py
```

---

## Re-running the Pipeline

```bash
# Full pipeline (preprocess → train → evaluate → register)
uv run pipeline/pipeline.py --run

# Skip preprocessing (use existing S3 data)
uv run pipeline/pipeline.py --skip-preprocess --run
```

Preprocessing takes ~35 min on `ml.m5.2xlarge`. Training takes ~80 min on `ml.g4dn.4xlarge`.
