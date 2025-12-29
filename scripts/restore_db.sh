#!/usr/bin/env bash
set -euo pipefail

# Restauration Postgres/PostGIS
# - Utilise DATABASE_URL si présent, sinon variables PG*
# - Attend un fichier dump en argument (.sql.gz ou .dump/.sql)
# - Crée la base si besoin (optionnel via CREATE_DB=true)

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <dump_file>"
  exit 1
fi

DUMP_FILE="$1"
CREATE_DB="${CREATE_DB:-false}"

if [[ ! -f "$DUMP_FILE" ]]; then
  echo "Fichier introuvable: $DUMP_FILE"
  exit 1
fi

# Résoudre la connexion
if [[ -n "${DATABASE_URL:-}" ]]; then
  export PGPASSWORD="$(python3 - <<'PY'
import os
from urllib.parse import urlparse, unquote
u = urlparse(os.environ['DATABASE_URL'])
print(unquote(u.password or ''))
PY
)"
  HOST=$(python3 - <<'PY'
import os
from urllib.parse import urlparse
u = urlparse(os.environ['DATABASE_URL'])
print(u.hostname or 'localhost')
PY
)
  PORT=$(python3 - <<'PY'
import os
from urllib.parse import urlparse
u = urlparse(os.environ['DATABASE_URL'])
print(u.port or 5432)
PY
)
  USER=$(python3 - <<'PY'
import os
from urllib.parse import urlparse, unquote
u = urlparse(os.environ['DATABASE_URL'])
print(unquote(u.username or 'postgres'))
PY
)
  DB=$(python3 - <<'PY'
import os
from urllib.parse import urlparse
u = urlparse(os.environ['DATABASE_URL'])
print((u.path or '/postgres').lstrip('/'))
PY
)
else
  HOST="${PGHOST:-localhost}"
  PORT="${PGPORT:-5432}"
  USER="${PGUSER:-postgres}"
  DB="${PGDATABASE:-postgres}"
  export PGPASSWORD="${PGPASSWORD:-}"
fi

echo "[restore_db] Restauration vers $DB@$HOST:$PORT depuis $DUMP_FILE"

if [[ "$CREATE_DB" == "true" ]]; then
  echo "[restore_db] Création de la base si nécessaire"
  createdb -h "$HOST" -p "$PORT" -U "$USER" "$DB" 2>/dev/null || true
fi

# Détecter format et restaurer
if [[ "$DUMP_FILE" == *.sql.gz ]]; then
  gunzip -c "$DUMP_FILE" | psql -h "$HOST" -p "$PORT" -U "$USER" -d "$DB"
elif [[ "$DUMP_FILE" == *.sql ]]; then
  psql -h "$HOST" -p "$PORT" -U "$USER" -d "$DB" -f "$DUMP_FILE"
else
  # Format pg_dump custom (pg_restore)
  pg_restore -h "$HOST" -p "$PORT" -U "$USER" -d "$DB" --create --clean "$DUMP_FILE" || \
  pg_restore -h "$HOST" -p "$PORT" -U "$USER" -d "$DB" "$DUMP_FILE"
fi

echo "[restore_db] Terminé"
