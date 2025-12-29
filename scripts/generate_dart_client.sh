#!/usr/bin/env bash
set -euo pipefail
# Génération du client Dart via OpenAPI Generator (Docker)
# Prérequis: Docker installé
# Usage: bash scripts/generate_dart_client.sh

GEN_IMAGE=openapitools/openapi-generator-cli:v7.6.0
CONFIG=openapi/dart-config.yaml

if [ ! -f "$CONFIG" ]; then
  echo "Config $CONFIG introuvable" >&2
  exit 1
fi

mkdir -p examples/flutter/openapi_client

docker run --rm \
  -v "$(pwd)":/local \
  $GEN_IMAGE generate \
  -g dart-dio-next \
  -i /local/api/openapi.json \
  -o /local/examples/flutter/openapi_client \
  --additional-properties pubName=geospatial_api_client,pubVersion=0.1.0,nullSafety=true,serializationLibrary=json_serializable

# Astuce: appliquer dart format si disponible
if command -v dart >/dev/null 2>&1; then
  (cd examples/flutter/openapi_client && dart format . || true)
fi

echo "Client Dart généré dans examples/flutter/openapi_client"
