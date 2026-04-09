"""
DocuSense FastAPI backend.

Routes:
  POST /classify  — classify image via SageMaker endpoint
  POST /extract   — extract structured fields via LLM
  POST /analyze   — classify + extract in one call (main route)
"""

import json
import logging
import os
import time

import boto3
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from extractor import extract

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




class ClassifyResponse(BaseModel):
    doc_class: str
    confidence: float


class AnalyzeResponse(BaseModel):
    doc_class: str
    confidence: float
    extracted_fields: dict | None  # None when confidence < threshold
    sagemaker_latency_ms: float
    llm_latency_ms: float | None



# SageMaker inference


def _classify_via_sagemaker(image_bytes: bytes) -> tuple[str, float, float]:
    """
    Call SageMaker endpoint. Returns class_name, confidence, latency_ms.
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


#
# Routes

@app.post("/classify", response_model=ClassifyResponse)
async def classify(file: UploadFile = File(...)) -> ClassifyResponse:
    image_bytes = await file.read()
    doc_class, confidence, _ = _classify_via_sagemaker(image_bytes)
    return ClassifyResponse(doc_class=doc_class, confidence=confidence)


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

    #Classify via SageMaker
    doc_class, confidence, sm_latency_ms = _classify_via_sagemaker(image_bytes)

    # Confidence routing — skip LLM if below threshold
    if confidence < CONFIDENCE_THRESHOLD:
        logger.info(f"Low confidence ({confidence:.4f}) — skipping LLM extraction")
        return AnalyzeResponse(
            doc_class="unknown",
            confidence=confidence,
            extracted_fields=None,
            sagemaker_latency_ms=sm_latency_ms,
            llm_latency_ms=None,
        )

    #LLM extraction
    t0 = time.perf_counter()
    extracted = extract(doc_class, image_bytes)
    llm_latency_ms = (time.perf_counter() - t0) * 1000

    logger.info(f"LLM extraction: doc_class={doc_class} latency={llm_latency_ms:.1f}ms")

    return AnalyzeResponse(
        doc_class=doc_class,
        confidence=confidence,
        extracted_fields=extracted.model_dump(),
        sagemaker_latency_ms=sm_latency_ms,
        llm_latency_ms=llm_latency_ms,
    )
