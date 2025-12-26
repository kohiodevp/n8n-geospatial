#!/bin/bash
# cadastral-validate.sh - Validation rapide pour n8n

set -e

# Lire depuis stdin ou fichier
if [[ -n "$1" && -f "$1" ]]; then
    python3 cadastral_cli.py validate --file "$1"
else
    python3 cadastral_cli.py validate
fi