#!/bin/bash
# Startup script for n8n geospatial runner on Render

set -euo pipefail

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" >&2
}

log "Starting n8n geospatial runner..."

# Ensure required directories exist with appropriate permissions
# Use 755 instead of 777 for security (owner: rwx, group: rx, others: rx)
mkdir -p /files /geodata /qgis-output /tmp/geodata-cache /tmp/runtime-node
chmod 755 /files /geodata /qgis-output /tmp/geodata-cache /tmp/runtime-node

# Set environment variables if not already set
export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-offscreen}
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp/runtime-node}
export GDAL_CACHEMAX=${GDAL_CACHEMAX:-1024}
export GDAL_NUM_THREADS=${GDAL_NUM_THREADS:-ALL_CPUS}
export PROJ_NETWORK=${PROJ_NETWORK:-ON}
export N8N_RUNNERS_MODE=${N8N_RUNNERS_MODE:-external}

# Validate that n8n command is available
if ! command -v n8n &> /dev/null; then
    log "Error: n8n command is not available in PATH"
    exit 1
fi

# Convert DATABASE_URL (Render) to variables DB_POSTGRESDB_* (n8n)
if [ -n "${DATABASE_URL:-}" ]; then
    log "Configuring database connection from DATABASE_URL..."

    export DB_POSTGRESDB_HOST="$(node -p "new URL(process.env.DATABASE_URL).hostname")"
    export DB_POSTGRESDB_PORT="$(node -p "new URL(process.env.DATABASE_URL).port || '5432'")"
    export DB_POSTGRESDB_DATABASE="$(node -p "new URL(process.env.DATABASE_URL).pathname.replace(/^\\//,'')")"
    export DB_POSTGRESDB_USER="$(node -p "decodeURIComponent(new URL(process.env.DATABASE_URL).username)")"
    export DB_POSTGRESDB_PASSWORD="$(node -p "decodeURIComponent(new URL(process.env.DATABASE_URL).password)")"

    # Validate that we extracted the database parameters correctly
    if [ -z "$DB_POSTGRESDB_HOST" ] || [ -z "$DB_POSTGRESDB_DATABASE" ]; then
        log "Error: Failed to extract database parameters from DATABASE_URL"
        exit 1
    fi

    log "Database configuration completed: $DB_POSTGRESDB_HOST:$DB_POSTGRESDB_PORT/$DB_POSTGRESDB_DATABASE"
else
    log "DATABASE_URL not set, using default database configuration"
fi

# Check if QGIS is available for geospatial processing
if command -v qgis_process &> /dev/null; then
    log "QGIS is available, geospatial features enabled"
else
    log "Warning: QGIS is not available, some geospatial features may not work"
fi

# Check if GDAL is available
if command -v gdalinfo &> /dev/null; then
    log "GDAL is available, raster processing enabled"
else
    log "Warning: GDAL is not available, raster processing may not work"
fi

# Check if PostGIS is available (by checking for ogr2ogr which supports PostGIS)
if command -v ogr2ogr &> /dev/null; then
    log "GDAL/OGR is available, vector processing enabled"
else
    log "Warning: GDAL/OGR is not available, vector processing may not work"
fi

# Auto-deploy workflows in background if enabled
if [ "${ENABLE_AUTO_DEPLOY_WORKFLOWS:-false}" = "true" ]; then
  (
    log "Auto-deploy is enabled. Importing workflows individually with retries..."
    for i in $(seq 1 60); do
      if command -v n8n >/dev/null 2>&1; then
        success=0; fail=0
        for wf in /home/node/.n8n/workflows/*.json; do
          [ -e "$wf" ] || continue
          if n8n import:workflow --input "$wf" --overwrite >/dev/null 2>&1; then
            success=$((success+1))
          else
            fail=$((fail+1))
          fi
        done
        total=$((success+fail))
        if [ "$total" -gt 0 ] && [ "$fail" -eq 0 ]; then
          log "All workflows imported successfully ($success)"
          break
        else
          log "Imported=$success Failed=$fail (try $i); retrying in 2s..."
        fi
      else
        log "n8n CLI not found; cannot import workflows"
        break
      fi
      sleep 2
    done
  ) &
fi

log "Starting n8n with geospatial capabilities..."
exec n8n
