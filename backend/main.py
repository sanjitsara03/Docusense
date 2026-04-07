"""
DocuSense FastAPI backend.

Routes:
  POST /classify  — classify image via SageMaker endpoint
  POST /extract   — extract structured fields via LLM
  POST /analyze   — classify + extract in one call (main route)
"""

import io
import json
import logging
import os
import time

import boto3
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from extractor import extract
from gradcam import generate_heatmap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DocuSense")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "docusense-endpoint")
CONFIDENCE_THRESHOLD = 0.70
CLASS_NAMES = ["letter", "form", "email", "budget", "invoice"]

_runtime = boto3.client("sagemaker-runtime", region_name=os.environ.get("AWS_REGION", "us-east-2"))


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ClassifyResponse(BaseModel):
    doc_class: str
    confidence: float
    heatmap_b64: str | None  # None when confidence < threshold


class AnalyzeResponse(BaseModel):
    doc_class: str
    confidence: float
    heatmap_b64: str | None
    extracted_fields: dict | None  # None when confidence < threshold
    sagemaker_latency_ms: float
    llm_latency_ms: float | None


# ---------------------------------------------------------------------------
# Model loading (lazy, cached)
# ---------------------------------------------------------------------------

_model: torch.nn.Module | None = None


def _get_model() -> torch.nn.Module:
    global _model
    if _model is None:
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        import torch.nn as nn
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, len(CLASS_NAMES)),
        )
        model_path = os.environ.get("MODEL_PATH")
        if model_path:
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        _model = model
    return _model


# ---------------------------------------------------------------------------
# SageMaker inference
# ---------------------------------------------------------------------------

def _classify_via_sagemaker(image_bytes: bytes) -> tuple[str, float, float]:
    """
    Call SageMaker endpoint. Returns (class_name, confidence, latency_ms).
    """
    t0 = time.perf_counter()
    response = _runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="image/png",
        Body=image_bytes,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    result = json.loads(response["Body"].read())
    doc_class = result["class"]
    confidence = result["confidence"]

    logger.info(f"SageMaker: class={doc_class} confidence={confidence:.4f} latency={latency_ms:.1f}ms")
    return doc_class, confidence, latency_ms


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/classify", response_model=ClassifyResponse)
async def classify(file: UploadFile = File(...)) -> ClassifyResponse:
    image_bytes = await file.read()
    doc_class, confidence, _ = _classify_via_sagemaker(image_bytes)

    heatmap_b64 = None
    if confidence >= CONFIDENCE_THRESHOLD:
        image = Image.open(io.BytesIO(image_bytes))
        class_idx = CLASS_NAMES.index(doc_class)
        heatmap_b64 = generate_heatmap(image, _get_model(), class_idx)

    return ClassifyResponse(
        doc_class=doc_class,
        confidence=confidence,
        heatmap_b64=heatmap_b64,
    )


@app.post("/extract")
async def extract_fields(
    file: UploadFile = File(...),
    doc_class: str = Form(...),
) -> dict:
    if doc_class not in CLASS_NAMES:
        raise HTTPException(status_code=400, detail=f"Unknown doc_class: {doc_class}")
    image_bytes = await file.read()
    result = extract(doc_class, image_bytes)
    return result.model_dump()


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)) -> AnalyzeResponse:
    image_bytes = await file.read()

    # Step 1: Classify via SageMaker
    doc_class, confidence, sm_latency_ms = _classify_via_sagemaker(image_bytes)

    # Confidence routing — skip LLM if below threshold
    if confidence < CONFIDENCE_THRESHOLD:
        logger.info(f"Low confidence ({confidence:.4f}) — skipping LLM extraction")
        return AnalyzeResponse(
            doc_class="unknown",
            confidence=confidence,
            heatmap_b64=None,
            extracted_fields=None,
            sagemaker_latency_ms=sm_latency_ms,
            llm_latency_ms=None,
        )

    # Step 2: Grad-CAM heatmap
    image = Image.open(io.BytesIO(image_bytes))
    class_idx = CLASS_NAMES.index(doc_class)
    heatmap_b64 = generate_heatmap(image, _get_model(), class_idx)

    # Step 3: LLM extraction
    t0 = time.perf_counter()
    extracted = extract(doc_class, image_bytes)
    llm_latency_ms = (time.perf_counter() - t0) * 1000

    logger.info(f"LLM extraction: doc_class={doc_class} latency={llm_latency_ms:.1f}ms")

    return AnalyzeResponse(
        doc_class=doc_class,
        confidence=confidence,
        heatmap_b64=heatmap_b64,
        extracted_fields=extracted.model_dump(),
        sagemaker_latency_ms=sm_latency_ms,
        llm_latency_ms=llm_latency_ms,
    )
