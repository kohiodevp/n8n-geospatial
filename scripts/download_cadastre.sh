#!/bin/bash
# Téléchargement des données cadastrales depuis data.gouv.fr

set -e

COMMUNE="$1"
OUTPUT_DIR="${2:-/home/node/shared/cadastre}"

if [ -z "$COMMUNE" ]; then
    echo '{"status": "error", "message": "Code commune requis"}' >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Télécharger les parcelles (format GeoJSON)
echo "Téléchargement des parcelles pour $COMMUNE..." >&2
HTTP_CODE=$(curl -s -w "%{http_code}" -o "${OUTPUT_DIR}/parcelles_${COMMUNE}.json" \
    "https://cadastre.data.gouv.fr/bundler/cadastre-etalab/communes/${COMMUNE}/geojson/parcelles")

if [ "$HTTP_CODE" != "200" ]; then
    echo "{\"status\": \"error\", \"message\": \"Erreur HTTP $HTTP_CODE pour parcelles\"}"
    exit 1
fi

# Télécharger les bâtiments
echo "Téléchargement des bâtiments pour $COMMUNE..." >&2
HTTP_CODE=$(curl -s -w "%{http_code}" -o "${OUTPUT_DIR}/batiments_${COMMUNE}.json" \
    "https://cadastre.data.gouv.fr/bundler/cadastre-etalab/communes/${COMMUNE}/geojson/batiments")

if [ "$HTTP_CODE" != "200" ]; then
    echo "{\"status\": \"error\", \"message\": \"Erreur HTTP $HTTP_CODE pour batiments\"}"
    exit 1
fi

# Compter les entités
PARCELLES_COUNT=$(jq '.features | length' "${OUTPUT_DIR}/parcelles_${COMMUNE}.json" 2>/dev/null || echo "0")
BATIMENTS_COUNT=$(jq '.features | length' "${OUTPUT_DIR}/batiments_${COMMUNE}.json" 2>/dev/null || echo "0")

echo "{\"status\": \"success\", \"commune\": \"$COMMUNE\", \"output_dir\": \"$OUTPUT_DIR\", \"parcelles_count\": $PARCELLES_COUNT, \"batiments_count\": $BATIMENTS_COUNT}"
