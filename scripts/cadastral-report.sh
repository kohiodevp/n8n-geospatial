#!/bin/bash
# cadastral-report.sh - Rapport complet pour n8n

set -e

INPUT_FILE="${1:-}"
PROPS_FILE="${2:-}"

ARGS="report"

if [[ -n "$INPUT_FILE" && -f "$INPUT_FILE" ]]; then
    ARGS="$ARGS --file $INPUT_FILE"
fi

if [[ -n "$PROPS_FILE" && -f "$PROPS_FILE" ]]; then
    ARGS="$ARGS --properties-file $PROPS_FILE"
fi

python3 cadastral_cli.py $ARGS