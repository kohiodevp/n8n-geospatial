#!/usr/bin/env bash
set -euo pipefail

# Simple monitoring script for n8n geospatial stack
# - Checks nginx (/health), MCP (/healthz), n8n (/healthz)
# - Prints JSON summary and exits non-zero on failure

NGINX_URL="${NGINX_URL:-http://localhost/health}"
MCP_URL="${MCP_URL:-http://localhost:5001/healthz}"
N8N_URL="${N8N_URL:-http://localhost:5678/healthz}"
TIMEOUT="${TIMEOUT:-5}"

check() {
  local url="$1" name="$2"
  local t0 ts status
  t0=$(date +%s%3N)
  if curl -fsS --max-time "$TIMEOUT" "$url" >/dev/null; then
    ts=$(($(date +%s%3N) - t0))
    echo "{\"name\":\"$name\",\"url\":\"$url\",\"ok\":true,\"latency_ms\":$ts}"
    return 0
  else
    ts=$(($(date +%s%3N) - t0))
    echo "{\"name\":\"$name\",\"url\":\"$url\",\"ok\":false,\"latency_ms\":$ts}"
    return 1
  fi
}

fail=0
out="["
res=$(check "$NGINX_URL" nginx) || fail=1; out+="$res,"
res=$(check "$MCP_URL" mcp) || fail=1; out+="$res,"
res=$(check "$N8N_URL" n8n) || fail=1; out+="$res]"

# Fix trailing comma if any
out=${out/,]/]}

echo "$out"
exit $fail
