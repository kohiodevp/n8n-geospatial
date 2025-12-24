#!/usr/bin/env python3
"""
Health check script for n8n geospatial runner
This script can be used to verify that the service is running correctly
"""

import sys
import requests
from requests.exceptions import RequestException

def check_health():
    """
    Check if n8n is running and responding
    """
    try:
        # Try to connect to the local n8n instance
        response = requests.get('http://localhost:5678/healthz', timeout=5)
        if response.status_code == 200:
            print("Health check passed: n8n is running")
            return True
        else:
            print(f"Health check failed: Unexpected status code {response.status_code}")
            return False
    except RequestException as e:
        print(f"Health check failed: Unable to connect to n8n - {e}")
        return False

if __name__ == "__main__":
    if check_health():
        sys.exit(0)
    else:
        sys.exit(1)