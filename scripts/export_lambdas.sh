#!/usr/bin/env bash
# Download the code that is actually deployed for each Lambda function and
# overwrite the reconstructed versions in lambda_functions/.
#
# Requires the AWS CLI, configured for the account that runs the pipeline:
#   aws configure            # or AWS_PROFILE=...
#   ./scripts/export_lambdas.sh
# Then review `git diff lambda_functions/` and commit the real code.
set -euo pipefail

REGION="${AWS_REGION:-ap-south-1}"
OUT_DIR="$(cd "$(dirname "$0")/.." && pwd)/lambda_functions"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

# deployed function name -> file in lambda_functions/
declare -A FUNCTIONS=(
  [get_presigned_url]=get_presigned_url.py
  [process-document-batch]=process_document_batch.py
  [handle-textract-result]=handle_textract_result.py
  [generate-embeddings]=generate_embeddings.py
  [handle-query]=handle_query.py
  [generate-answer]=generate_answer.py
)

for fn in "${!FUNCTIONS[@]}"; do
  target="${FUNCTIONS[$fn]}"
  echo "Exporting $fn -> lambda_functions/$target"
  url="$(aws lambda get-function --region "$REGION" --function-name "$fn" \
          --query 'Code.Location' --output text)" || {
    echo "  ! could not read $fn (wrong name, region or permissions?)" >&2
    continue
  }
  curl -sSfL "$url" -o "$TMP_DIR/$fn.zip"
  mkdir -p "$TMP_DIR/$fn"
  unzip -oq "$TMP_DIR/$fn.zip" -d "$TMP_DIR/$fn"

  handler="$(aws lambda get-function-configuration --region "$REGION" \
              --function-name "$fn" --query 'Handler' --output text)"
  src="$TMP_DIR/$fn/${handler%.*}.py"
  if [[ -f "$src" ]]; then
    cp "$src" "$OUT_DIR/$target"
  else
    echo "  ! handler file ${handler%.*}.py not found; files are in the zip:" >&2
    ls "$TMP_DIR/$fn" >&2
  fi
done

echo "Done. Review with: git diff lambda_functions/"
