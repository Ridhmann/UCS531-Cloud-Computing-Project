"""Lambda: handle-textract-result  (trigger: SNS topic textract-job-complete)

RECONSTRUCTED from docs/3-Data_Ingestion.md. It is not the deployed source;
run scripts/export_lambdas.sh to pull the real code from AWS.

Fetches the full Textract result, splits the text into overlapping word
chunks and writes them to DynamoDB with status PENDING. The scheduled
generate-embeddings function then embeds them.

Timeout: 5 minutes (large PDFs need many Textract result pages).
"""

import json
import os

import boto3

TABLE_NAME = os.environ.get("TABLE_NAME", "document-chunks")
CHUNK_WORDS = int(os.environ.get("CHUNK_WORDS", "120"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "20"))

textract = boto3.client("textract")
table = boto3.resource("dynamodb").Table(TABLE_NAME)


def fetch_text(job_id):
    """Concatenate every LINE block across all result pages."""
    lines = []
    next_token = None
    while True:
        kwargs = {"JobId": job_id, "MaxResults": 1000}
        if next_token:
            kwargs["NextToken"] = next_token
        result = textract.get_document_text_detection(**kwargs)
        lines.extend(
            block["Text"]
            for block in result.get("Blocks", [])
            if block.get("BlockType") == "LINE" and block.get("Text")
        )
        next_token = result.get("NextToken")
        if not next_token:
            return "\n".join(lines)


def chunk_words(text, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def lambda_handler(event, context):
    written = 0

    for record in event.get("Records", []):
        message = json.loads(record["Sns"]["Message"])
        job_id = message["JobId"]
        status = message.get("Status")
        document_id = message.get("JobTag")
        key = message.get("DocumentLocation", {}).get("S3ObjectName", "")
        filename = os.path.basename(key) or "unknown.pdf"

        if status != "SUCCEEDED":
            print(f"Textract job {job_id} for {key} ended with {status}")
            continue
        if not document_id:
            print(f"Job {job_id} has no JobTag; cannot map it to a document")
            continue

        chunks = chunk_words(fetch_text(job_id))
        with table.batch_writer(overwrite_by_pkeys=["chunk_id", "document_id"]) as batch:
            for index, text in enumerate(chunks):
                batch.put_item(Item={
                    "chunk_id": f"{document_id}_{index:05d}",
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": index,
                    "text": text,
                    "status": "PENDING",
                })
        written += len(chunks)
        print(f"Wrote {len(chunks)} chunks for {filename} ({document_id})")

    return {"chunks_written": written}
