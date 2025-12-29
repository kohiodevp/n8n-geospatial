#!/usr/bin/env python3
"""
Register AI Agent Guide Workflow via n8n REST API
"""

import requests
import json
import os

N8N_URL = os.environ.get("N8N_URL", "http://n8n-geospatial:5678")

workflow_data = {
    "name": "AI Agent Guide",
    "active": False,  # Will activate after creation
    "nodes": [
        {
            "parameters": {
                "httpMethod": "POST",
                "path": "ai-guide",
                "responseMode": "lastNode",
                "options": {}
            },
            "id": "trigger",
            "name": "Webhook",
            "type": "n8n-nodes-base.webhook",
            "typeVersion": 1,
            "position": [220, 300]
        },
        {
            "parameters": {
                "mode": "runOnceForAllItems",
                "jsCode": """const userMessage = $input.first().json.body?.message || 'Pas de message reçu';
const response = {
  output: 'Bonjour! Je suis le Guide IA du systeme n8n-geo. Vous avez dit: \"' + userMessage + '\"',
  status: 'ok',
  timestamp: new Date().toISOString()
};
return [{ json: response }];"""
            },
            "id": "respondCode",
            "name": "Reponse Guide",
            "type": "n8n-nodes-base.code",
            "typeVersion": 2,
            "position": [460, 300]
        }
    ],
    "connections": {
        "Webhook": {
            "main": [[{"node": "Reponse Guide", "type": "main", "index": 0}]]
        }
    },
    "settings": {}
}

def main():
    # n8n Basic Auth credentials (from docker-compose.yml defaults)
    auth = ("admin", "cadastre2024")
    
    # Create workflow
    print(f"Creating workflow on {N8N_URL}...")
    try:
        create_resp = requests.post(
            f"{N8N_URL}/api/v1/workflows",
            json=workflow_data,
            headers={"Content-Type": "application/json"},
            auth=auth,
            timeout=30
        )
        
        if create_resp.status_code == 200:
            result = create_resp.json()
            workflow_id = result.get("id")
            print(f"Workflow created successfully! ID: {workflow_id}")
            
            # Activate workflow
            print("Activating workflow...")
            activate_resp = requests.patch(
                f"{N8N_URL}/api/v1/workflows/{workflow_id}",
                json={"active": True},
                headers={"Content-Type": "application/json"},
                auth=auth,
                timeout=30
            )
            
            if activate_resp.status_code == 200:
                print("Workflow activated successfully!")
                return True
            else:
                print(f"Error activating workflow: {activate_resp.status_code} - {activate_resp.text}")
        else:
            print(f"Error creating workflow: {create_resp.status_code}")
            print(create_resp.text)
            return False
            
    except Exception as e:
        print(f"Connection error: {e}")
        return False

if __name__ == "__main__":
    main()
