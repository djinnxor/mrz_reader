#!/bin/bash
# chmod +x test_api.sh

API="http://localhost:5006"

echo "=== Testing Health Endpoint ==="
curl -s $API/health | python -m json.tool

echo -e "\n=== Testing Root Endpoint ==="
curl -s $API/ | python -m json.tool

echo -e "\n=== Testing Predict (requires image base64) ==="
# If you have test image data
if [ -f "test/passport_test.b64" ]; then
  image_path="test/passport_test.b64"
elif [ -f "test/test_pass_image.b64" ]; then
  image_path="test/test_pass_image.b64"
else
  echo "Test image not found"
  exit 0
fi

base64_file=$(mktemp)
payload_file=$(mktemp)
tr -d '\r\n ' < "$image_path" > "$base64_file"
python - "$base64_file" "$payload_file" <<'PY'
import json
import sys

with open(sys.argv[1], 'r', encoding='utf-8') as handle:
    image_data = handle.read().strip()

payload = {"image_base64": image_data, "type": "passport"}
with open(sys.argv[2], 'w', encoding='utf-8') as handle:
    json.dump(payload, handle)
PY

curl -sS -X POST "$API/predict" \
  -H "Content-Type: application/json" \
  --data-binary @"$payload_file" \
  | python -m json.tool

rm -f "$base64_file" "$payload_file"