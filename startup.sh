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

# Start n8n
exec n8n