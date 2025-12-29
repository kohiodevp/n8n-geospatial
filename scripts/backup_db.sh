#!/usr/bin/env bash
set -euo pipefail

# Sauvegarde Postgres/PostGIS
# - Utilise DATABASE_URL si présent, sinon variables PG*
# - Sauvegarde dans ./backups avec timestamp
# - Rétention configurable via RETENTION_DAYS (défaut 7)

RETENTION_DAYS="${RETENTION_DAYS:-7}"
BACKUP_DIR="${BACKUP_DIR:-./backups}"
mkdir -p "$BACKUP_DIR"

# Résoudre la connexion
if [[ -n "${DATABASE_URL:-}" ]]; then
  export PGPASSWORD="$(python3 - <<'PY'
import os, sys
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
import os, sys
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

TS=$(date +"%Y%m%d_%H%M%S")
FILE="$BACKUP_DIR/${DB}_${TS}.sql.gz"

echo "[backup_db] Dump de $DB@$HOST:$PORT en $FILE"
pg_dump -h "$HOST" -p "$PORT" -U "$USER" -d "$DB" -Fc | gzip -c > "$FILE"
echo "[backup_db] Terminé"

# Rétention
find "$BACKUP_DIR" -type f -name "${DB}_*.sql.gz" -mtime +"$RETENTION_DAYS" -print -delete || true
