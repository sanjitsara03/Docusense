"""
LLM extraction agents 
one Pydantic schema per document type.

"""

import base64
import os
from typing import Union

import anthropic
from pydantic import BaseModel

_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
_MODEL = "claude-sonnet-4-6"




class LineItem(BaseModel):
    description: str
    amount: float


class InvoiceExtraction(BaseModel):
    vendor: str | None
    invoice_number: str | None
    amount: float | None
    currency: str | None
    line_items: list[LineItem]
    due_date: str | None


class FieldPair(BaseModel):
    name: str
    value: str


class FormExtraction(BaseModel):
    form_title: str | None
    fields: list[FieldPair]


class LetterExtraction(BaseModel):
    sender: str | None
    recipient: str | None
    date: str | None
    subject: str | None
    summary: str | None


class EmailExtraction(BaseModel):
    sender: str | None
    recipient: str | None
    date: str | None
    subject: str | None
    summary: str | None
    action_items: list[str]


class BudgetLineItem(BaseModel):
    category: str
    amount: float


class BudgetExtraction(BaseModel):
    title: str | None
    date: str | None
    line_items: list[BudgetLineItem]
    total: float | None
    notes: str | None


_SCHEMA_MAP: dict[str, type[BaseModel]] = {
    "invoice": InvoiceExtraction,
    "form": FormExtraction,
    "letter": LetterExtraction,
    "email": EmailExtraction,
    "budget": BudgetExtraction,
}

_PROMPTS: dict[str, str] = {
    "invoice": "Extract the vendor name, invoice number, total amount, currency, line items, and due date from this invoice image. Return null for any field you cannot find.",
    "form": "Extract the form title and all visible form fields with their values from this form image. Return null for the title if not visible.",
    "letter": "Extract the sender, recipient, date, subject, and a brief summary from this letter image. Return null for any field you cannot find.",
    "email": "Extract the sender, recipient, date, subject, a brief summary, and any action items from this email image. Return null for any field you cannot find.",
    "budget": "Extract the budget title, date, all line items with their categories and amounts, total amount, and any notes from this budget document. Return null for any field you cannot find.",
}



def extract(doc_class: str, image_bytes: bytes) -> BaseModel:
    """
    Extract structured fields from a document image using Claude.
    Returns: Populated Pydantic model for the given doc_class
    """
    schema = _SCHEMA_MAP[doc_class]
    prompt = _PROMPTS[doc_class]
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    # Detect media type from magic bytes
    if image_bytes[:3] == b"\xff\xd8\xff":
        media_type = "image/jpeg"
    elif image_bytes[:4] == b"\x89PNG":
        media_type = "image/png"
    elif image_bytes[:4] in (b"GIF8", b"GIF9"):
        media_type = "image/gif"
    else:
        media_type = "image/png"  # fallback

    response = _client.messages.create(
        model=_MODEL,
        max_tokens=1024,
        system="You are a document extraction assistant. Extract structured data from document images and return it as valid JSON matching the requested schema. Be precise and only include information clearly visible in the document.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": f"{prompt}\n\nRespond with ONLY a JSON object matching this schema:\n{schema.model_json_schema()}",
                    },
                ],
            }
        ],
    )

    import json, re
    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if match:
        raw = match.group(1).strip()
    return schema.model_validate_json(raw)
