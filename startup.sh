#!/bin/bash
# Startup script for n8n geospatial runner on Render

# Ensure required directories exist and have proper permissions
mkdir -p /files /geodata /qgis-output /tmp/geodata-cache /tmp/runtime-node
chmod 777 /files /geodata /qgis-output /tmp/geodata-cache /tmp/runtime-node

# Set environment variables if not already set
export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-offscreen}
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp/runtime-node}
export GDAL_CACHEMAX=${GDAL_CACHEMAX:-1024}
export GDAL_NUM_THREADS=${GDAL_NUM_THREADS:-ALL_CPUS}
export PROJ_NETWORK=${PROJ_NETWORK:-ON}
export N8N_RUNNERS_MODE=${N8N_RUNNERS_MODE:-external}

# Convertit DATABASE_URL (Render) en variables DB_POSTGRESDB_* (n8n)
if [ -n "${DATABASE_URL:-}" ]; then
  export DB_POSTGRESDB_HOST="$(node -p "new URL(process.env.DATABASE_URL).hostname")"
  export DB_POSTGRESDB_PORT="$(node -p "new URL(process.env.DATABASE_URL).port || '5432'")"
  export DB_POSTGRESDB_DATABASE="$(node -p "new URL(process.env.DATABASE_URL).pathname.replace(/^\\//,'')")"
  export DB_POSTGRESDB_USER="$(node -p "decodeURIComponent(new URL(process.env.DATABASE_URL).username)")"
  export DB_POSTGRESDB_PASSWORD="$(node -p "decodeURIComponent(new URL(process.env.DATABASE_URL).password)")"
fi

# Start n8n
exec n8n