"""Lambda: handle-query  (API: POST /query)

RECONSTRUCTED from docs/4-RAG_Query_API.md and the client contract in
index.ipynb. It is not the deployed source; run scripts/export_lambdas.sh
to pull the real code from AWS.

Request:  {"question": "...", "document_id": "bad18322" | null, "top_k": 5}
Response: {"top_chunks": [{chunk_id, document_id, filename, chunk_index,
                           text, similarity_score}, ...]}

Retrieval is a full table scan plus cosine similarity in Python. That is fine
for a handful of papers, but API Gateway cuts requests off at 29 s, so
"all documents" queries will eventually time out as the table grows. The
fix at that point is a real vector index (OpenSearch Serverless, Aurora
pgvector, or S3 Vectors), not a longer Lambda timeout.
"""

import json
import math
import os

import boto3
from boto3.dynamodb.conditions import Attr

TABLE_NAME = os.environ.get("TABLE_NAME", "document-chunks")
EMBEDDING_MODEL_ID = os.environ.get(
    "EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"
)
MAX_TOP_K = 20

bedrock = boto3.client("bedrock-runtime")
table = boto3.resource("dynamodb").Table(TABLE_NAME)

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


def embed(text):
    response = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text}),
    )
    return json.loads(response["body"].read())["embedding"]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def completed_chunks(document_id):
    condition = Attr("status").eq("COMPLETED")
    if document_id:
        condition = condition & Attr("document_id").eq(document_id)
    kwargs = {
        "FilterExpression": condition,
        "ProjectionExpression":
            "chunk_id, document_id, filename, chunk_index, #t, embedding",
        "ExpressionAttributeNames": {"#t": "text"},
    }
    while True:
        page = table.scan(**kwargs)
        yield from page.get("Items", [])
        if "LastEvaluatedKey" not in page:
            return
        kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]


def lambda_handler(event, context):
    body = parse_body(event)
    question = (body.get("question") or "").strip()
    document_id = body.get("document_id") or None
    try:
        top_k = max(1, min(int(body.get("top_k", 5)), MAX_TOP_K))
    except (TypeError, ValueError):
        top_k = 5

    if not question:
        return respond(400, {"error": "question is required"})

    query_vector = embed(question)

    scored = []
    for item in completed_chunks(document_id):
        if "embedding" not in item:
            continue
        scored.append({
            "chunk_id": item["chunk_id"],
            "document_id": item["document_id"],
            "filename": item.get("filename", "unknown"),
            "chunk_index": int(item.get("chunk_index", 0)),
            "text": item.get("text", ""),
            "similarity_score": round(
                cosine(query_vector, json.loads(item["embedding"])), 4
            ),
        })

    scored.sort(key=lambda c: c["similarity_score"], reverse=True)
    return respond(200, {"top_chunks": scored[:top_k]})
