"""Lambda: generate-answer  (API: POST /answer)

RECONSTRUCTED from docs/4-RAG_Query_API.md and the client contract in
index.ipynb. It is not the deployed source; run scripts/export_lambdas.sh
to pull the real code from AWS.

Request:  {"question": "...", "top_chunks": [<chunks from /query>]}
Response: {"answer": "...", "sources": [{filename, chunk_index,
                                          similarity_score, excerpt}, ...]}

Uses the Bedrock Converse API, so ANSWER_MODEL_ID can be any chat model the
account has access to. The README names DeepSeek; set the model ID (or
inference-profile ID) that is enabled in your region. Reasoning models such
as DeepSeek-R1 can exceed API Gateway's 29 s limit on long contexts; keep
MAX_TOKENS modest or use a non-reasoning model if /answer returns 504.
"""

import json
import os

import boto3

ANSWER_MODEL_ID = os.environ.get("ANSWER_MODEL_ID", "deepseek.r1-v1:0")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "800"))
MAX_CONTEXT_CHARS = 12000
EXCERPT_CHARS = 150

bedrock = boto3.client("bedrock-runtime")

CORS_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
}

SYSTEM_PROMPT = (
    "You answer questions about research papers. Use only the provided "
    "context. If the context does not contain the answer, say so. Be "
    "specific and concise."
)


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


def build_context(chunks):
    parts = []
    used = 0
    for i, chunk in enumerate(chunks, 1):
        part = (
            f"[{i}] {chunk.get('filename', 'unknown')} "
            f"(chunk {chunk.get('chunk_index', '?')}):\n{chunk.get('text', '')}"
        )
        if used + len(part) > MAX_CONTEXT_CHARS:
            break
        parts.append(part)
        used += len(part)
    return "\n\n".join(parts)


def ask_model(question, context_text):
    response = bedrock.converse(
        modelId=ANSWER_MODEL_ID,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[{
            "role": "user",
            "content": [{
                "text": f"Context:\n{context_text}\n\nQuestion: {question}"
            }],
        }],
        inferenceConfig={"maxTokens": MAX_TOKENS, "temperature": 0.2},
    )
    # Reasoning models also return reasoningContent blocks; keep only text.
    blocks = response["output"]["message"]["content"]
    return "".join(b["text"] for b in blocks if "text" in b).strip()


def excerpt(text):
    text = " ".join((text or "").split())
    return text if len(text) <= EXCERPT_CHARS else text[:EXCERPT_CHARS] + "..."


def lambda_handler(event, context):
    body = parse_body(event)
    question = (body.get("question") or "").strip()
    chunks = body.get("top_chunks") or []

    if not question:
        return respond(400, {"error": "question is required"})
    if not isinstance(chunks, list) or not chunks:
        return respond(400, {"error": "top_chunks must be a non-empty list"})

    answer = ask_model(question, build_context(chunks))

    sources = [{
        "filename": c.get("filename", "unknown"),
        "chunk_index": c.get("chunk_index"),
        "similarity_score": c.get("similarity_score"),
        "excerpt": excerpt(c.get("text")),
    } for c in chunks]

    return respond(200, {"answer": answer or "No answer generated.",
                         "sources": sources})
