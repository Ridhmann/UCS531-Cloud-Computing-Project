"""Lambda: process-document-batch  (trigger: S3 ObjectCreated, prefix uploads/, suffix .pdf)

RECONSTRUCTED from docs/3-Data_Ingestion.md. It is not the deployed source;
run scripts/export_lambdas.sh to pull the real code from AWS.

Starts an asynchronous Textract text-detection job for each uploaded PDF.
Textract publishes to the SNS topic when done, which triggers
handle-textract-result.

Environment variables (set these instead of editing the code):
  SNS_TOPIC_ARN      arn:aws:sns:ap-south-1:<ACCOUNT_ID>:textract-job-complete
  TEXTRACT_ROLE_ARN  arn:aws:iam::<ACCOUNT_ID>:role/lambda-rag-execution-role
"""

import os
import urllib.parse

import boto3

SNS_TOPIC_ARN = os.environ["SNS_TOPIC_ARN"]
TEXTRACT_ROLE_ARN = os.environ["TEXTRACT_ROLE_ARN"]

textract = boto3.client("textract")


def document_id_from_key(key):
    """uploads/<document_id>/<filename>.pdf -> <document_id>"""
    parts = key.split("/")
    if len(parts) >= 3 and parts[0] == "uploads":
        return parts[1]
    return None


def lambda_handler(event, context):
    started = []

    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])

        document_id = document_id_from_key(key)
        if not document_id:
            print(f"Skipping object outside uploads/<id>/: {key}")
            continue

        response = textract.start_document_text_detection(
            DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}},
            NotificationChannel={
                "SNSTopicArn": SNS_TOPIC_ARN,
                "RoleArn": TEXTRACT_ROLE_ARN,
            },
            # JobTag comes back in the SNS message so the result handler
            # knows which document the job belongs to.
            JobTag=document_id,
        )
        job_id = response["JobId"]
        print(f"Started Textract job {job_id} for {key}")
        started.append({"document_id": document_id, "job_id": job_id})

    return {"started": started}
