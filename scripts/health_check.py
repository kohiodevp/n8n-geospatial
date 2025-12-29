#!/usr/bin/env python3
"""
Health check script for n8n geospatial runner
- Supporte sortie JSON (--json)
- Mesure latence
- URL configurable via --url
"""

import sys
import json
import time
import argparse
import requests
from requests.exceptions import RequestException

def check_health(url: str, timeout: float = 5.0):
    """Check if n8n is running and responding."""
    t0 = time.time()
    try:
        response = requests.get(url, timeout=timeout)
        latency_ms = int((time.time() - t0) * 1000)
        ok = (response.status_code == 200)
        return ok, {
            "url": url,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
        }
    except RequestException as e:
        latency_ms = int((time.time() - t0) * 1000)
        return False, {"url": url, "error": str(e), "latency_ms": latency_ms}

def main():
    parser = argparse.ArgumentParser(description="Health check n8n")
    parser.add_argument("--url", default="http://localhost:5678/healthz", help="URL de santé n8n")
    parser.add_argument("--json", action="store_true", help="Sortie JSON")
    parser.add_argument("--timeout", type=float, default=5.0, help="Délai (s)")
    args = parser.parse_args()

    ok, details = check_health(args.url, args.timeout)

    if args.json:
        print(json.dumps({"ok": ok, "details": details}, ensure_ascii=False))
    else:
        if ok:
            print(f"Health check OK: {details}")
        else:
            print(f"Health check FAIL: {details}")

    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
