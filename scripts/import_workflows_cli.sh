#!/usr/bin/env bash
set -euo pipefail

# Import des workflows exportés depuis l'UI n8n (format v2 compatible CLI)
# - Répertoire source par défaut: workflows/exports-ui
# - Import individuel avec retries
# - N'active pas automatiquement les workflows

SRC_DIR="${1:-workflows/exports-ui}"
RETRIES="${RETRIES:-5}"
SLEEP_SEC="${SLEEP_SEC:-2}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker introuvable. Exécutez dans l'environnement où tourne n8n (docker compose)." >&2
  exit 1
fi

if ! docker ps --format '{{.Names}}' | grep -q '^n8n-geospatial$'; then
  echo "Le conteneur n8n-geospatial ne semble pas démarré. Lancez: docker compose up -d" >&2
  exit 1
fi

if [ ! -d "$SRC_DIR" ]; then
  echo "Répertoire introuvable: $SRC_DIR" >&2
  exit 1
fi

shopt -s nullglob
files=("$SRC_DIR"/*.json)
shopt -u nullglob

if [ ${#files[@]} -eq 0 ]; then
  echo "Aucun fichier .json trouvé dans $SRC_DIR" >&2
  exit 1
fi

echo "[import_workflows_cli] Import de ${#files[@]} workflow(s) depuis $SRC_DIR"

for f in "${files[@]}"; do
  echo "-- Import: $f"
  for i in $(seq 1 "$RETRIES"); do
    if docker exec n8n-geospatial n8n import:workflow --input "/home/node/.n8n/workflows/$(basename "$f")" --overwrite >/dev/null 2>&1; then
      echo "   OK"
      break
    else
      echo "   tentative $i/$RETRIES échouée, retry dans ${SLEEP_SEC}s..."
      sleep "$SLEEP_SEC"
    fi
    if [ "$i" -eq "$RETRIES" ]; then
      echo "   Échec import: $f" >&2
    fi
  done
done

echo "[import_workflows_cli] Terminé."
