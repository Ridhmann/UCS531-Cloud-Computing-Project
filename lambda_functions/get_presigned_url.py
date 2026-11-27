"""Lambda: get_presigned_url  (API: POST /upload-url)

RECONSTRUCTED from docs/2-Upload_Endpoint.md and the client contract in
index.ipynb. It is not the deployed source; run scripts/export_lambdas.sh
to pull the real code from AWS and replace this file.

Request:  {"filename": "paper.pdf"}
Response: {"upload_url": "<presigned PUT URL>", "document_id": "1a2b3c4d"}
"""

import json
import os
import re
import uuid

import boto3
from botocore.config import Config

REGION = os.environ.get("AWS_REGION", "ap-south-1")
BUCKET = os.environ.get("BUCKET_NAME", "rag-project-documents")
UPLOAD_PREFIX = "uploads/"  # must match the S3 trigger prefix
URL_EXPIRY_SECONDS = 3600

# SigV4 and the regional endpoint are required for presigned URLs to work
# in ap-south-1 (the global endpoint signs with the wrong region).
s3 = boto3.client(
    "s3",
    region_name=REGION,
    endpoint_url=f"https://s3.{REGION}.amazonaws.com",
    config=Config(signature_version="s3v4"),
)

CORS_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
}


def respond(status, payload):
    return {
        "statusCode": status,
        "headers": CORS_HEADERS,
        "body": json.dumps(payload),
    }


def parse_body(event):
    body = event.get("body", event) if isinstance(event, dict) else {}
    if isinstance(body, str):
        try:
            body = json.loads(body or "{}")
        except ValueError:
            return {}
    return body if isinstance(body, dict) else {}


def safe_filename(name):
    name = os.path.basename(name or "").strip()
    name = re.sub(r"[^A-Za-z0-9 ._()-]", "_", name)
    return name


def lambda_handler(event, context):
    filename = safe_filename(parse_body(event).get("filename"))
    if not filename:
        return respond(400, {"error": "filename is required"})
    if not filename.lower().endswith(".pdf"):
        return respond(400, {"error": "only .pdf files are supported"})

    document_id = uuid.uuid4().hex[:8]
    key = f"{UPLOAD_PREFIX}{document_id}/{filename}"

    upload_url = s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": BUCKET, "Key": key},
        ExpiresIn=URL_EXPIRY_SECONDS,
    )

    return respond(200, {
        "upload_url": upload_url,
        "document_id": document_id,
        "s3_key": key,
    })
