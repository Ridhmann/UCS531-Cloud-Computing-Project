"""Lambda: generate-embeddings  (trigger: EventBridge rate(5 minutes))

RECONSTRUCTED from docs/3-Data_Ingestion.md. It is not the deployed source;
run scripts/export_lambdas.sh to pull the real code from AWS.

Embeds every chunk with status PENDING or FAILED using Amazon Titan and marks
it COMPLETED. Stops early when the invocation is close to its timeout; the
next scheduled run picks up the rest.

The embedding is stored as a JSON string: DynamoDB rejects Python floats,
and a string keeps a 1024-dim vector at ~20 KB per item (limit 400 KB).
handle_query must use the same EMBEDDING_MODEL_ID.
"""

import json
import os

import boto3
from boto3.dynamodb.conditions import Attr

TABLE_NAME = os.environ.get("TABLE_NAME", "document-chunks")
EMBEDDING_MODEL_ID = os.environ.get(
    "EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"
)
SAFETY_MARGIN_MS = 20_000

bedrock = boto3.client("bedrock-runtime")
table = boto3.resource("dynamodb").Table(TABLE_NAME)


def embed(text):
    response = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text[:20000]}),
    )
    return json.loads(response["body"].read())["embedding"]


def pending_chunks():
    kwargs = {
        "FilterExpression": Attr("status").is_in(["PENDING", "FAILED"]),
        "ProjectionExpression": "chunk_id, document_id, #t",
        "ExpressionAttributeNames": {"#t": "text"},
    }
    while True:
        page = table.scan(**kwargs)
        yield from page.get("Items", [])
        if "LastEvaluatedKey" not in page:
            return
        kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]


def set_status(item, status, embedding=None):
    names = {"#s": "status"}
    values = {":s": status}
    expression = "SET #s = :s"
    if embedding is not None:
        expression += ", embedding = :e"
        values[":e"] = json.dumps(embedding)
    table.update_item(
        Key={"chunk_id": item["chunk_id"], "document_id": item["document_id"]},
        UpdateExpression=expression,
        ExpressionAttributeNames=names,
        ExpressionAttributeValues=values,
    )


def lambda_handler(event, context):
    done = failed = 0
    out_of_time = False

    for item in pending_chunks():
        if context and context.get_remaining_time_in_millis() < SAFETY_MARGIN_MS:
            out_of_time = True
            break
        try:
            set_status(item, "COMPLETED", embed(item["text"]))
            done += 1
        except Exception as e:
            print(f"Embedding failed for {item['chunk_id']}: {e}")
            set_status(item, "FAILED")
            failed += 1

    print(f"Embedded {done}, failed {failed}, stopped early: {out_of_time}")
    return {"embedded": done, "failed": failed, "stopped_early": out_of_time}
