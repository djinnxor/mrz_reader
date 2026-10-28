#!/bin/bash
# chmod +x predict.sh
# ./predict.sh test/test_pass_image.b64 passport
# ./predict.sh test/test_old_pass_image.b64 passport-old
# ./predict.sh test/test_res_old_pass_image.b64 passport-old

if [ -z "$1" ]; then
  echo "Usage: ./predict.sh <path-to-b64-file> [type]"
  echo "  type: passport (default), id_card, visa, etc."
  exit 1
fi

FILE=$1
TYPE=$2

# Create JSON payload using jq to properly escape the base64 content
jq -n --rawfile image "$FILE" --arg type "$TYPE" \
  '{image_base64: $image, type: $type}' | \
curl -X POST http://localhost:5006/predict \
  -H "Content-Type: application/json" \
  --data-binary @-