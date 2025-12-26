#!/bin/bash
# cadastral-full-analysis.sh - Analyse complète pour n8n

set -e

INPUT_FILE="$1"
OUTPUT_DIR="${2:-/tmp/cadastral-output}"

mkdir -p "$OUTPUT_DIR"

echo "=== Analyse Cadastrale Complète ===" >&2

# 1. Validation
echo "1/4 Validation..." >&2
python3 cadastral_cli.py validate --file "$INPUT_FILE" > "$OUTPUT_DIR/validation.json"

# 2. Anomalies
echo "2/4 Détection anomalies..." >&2
python3 cadastral_cli.py anomalies --file "$INPUT_FILE" > "$OUTPUT_DIR/anomalies.json"

# 3. Prédiction valeurs
echo "3/4 Prédiction valeurs..." >&2
python3 cadastral_cli.py predict-values --file "$INPUT_FILE" > "$OUTPUT_DIR/values.json"

# 4. Rapport complet
echo "4/4 Génération rapport..." >&2
python3 cadastral_cli.py report --file "$INPUT_FILE" > "$OUTPUT_DIR/report.json"

# Consolider les résultats
echo "=== Consolidation ===" >&2

python3 << EOF
import json
from datetime import datetime

validation = json.load(open("$OUTPUT_DIR/validation.json"))
anomalies = json.load(open("$OUTPUT_DIR/anomalies.json"))
values = json.load(open("$OUTPUT_DIR/values.json"))
report = json.load(open("$OUTPUT_DIR/report.json"))

consolidated = {
    "timestamp": datetime.now().isoformat(),
    "summary": {
        "total_parcels": validation.get("total_parcels", 0),
        "parcels_with_issues": validation.get("parcels_with_issues", 0),
        "validation_passed": validation.get("validation_passed", False),
        "total_anomalies": anomalies.get("total_anomalies", 0),
        "total_estimated_value": values.get("summary", {}).get("total_value", 0)
    },
    "validation": validation,
    "anomalies": anomalies,
    "values": values,
    "report": report
}

print(json.dumps(consolidated, ensure_ascii=False, default=str))
EOF