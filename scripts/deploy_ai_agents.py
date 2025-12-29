#!/usr/bin/env python3
"""
Master Deployment Script for AI Agents (v1+ Schema Compliant)
============================================================
Gère l'ordre d'insertion et les colonnes obligatoires (authors, autosaved, updatedAt)
pour n8n v1+.
"""

import json
import os
import sys
import subprocess
import uuid
from pathlib import Path

# Configuration
DB_CONTAINER = "n8n-postgis"
N8N_CONTAINER = "n8n-geospatial"
DB_USER = "n8n"
DB_NAME = "n8n"

# Workflows AI à déployer
AI_WORKFLOWS = {
    "guide": "ai_agent_guide.json",
    "cadastral": "ai_agent_cadastral.json",
    "domanial": "ai_agent_domanial.json",
    "administrateur": "ai_agent_administrateur.json",
}

def get_workflows_dir():
    return Path(__file__).parent.parent / "workflows"

def load_workflow(filename: str) -> dict:
    filepath = get_workflows_dir() / filename
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def escape_sql(text: str) -> str:
    if not isinstance(text, str):
        text = json.dumps(text)
    return text.replace("'", "''")

def main():
    print("🚀 Démarrage du déploiement des agents IA (n8n v1+)...")
    
    sql_lines = [
        "BEGIN;",
        "DELETE FROM webhook_entity;",
        "DELETE FROM workflow_history WHERE \"workflowId\" IN (SELECT id FROM workflow_entity WHERE name LIKE 'AI Agent%');",
        "DELETE FROM workflow_entity WHERE name LIKE 'AI Agent%';"
    ]

    for key, filename in AI_WORKFLOWS.items():
        print(f"  📦 Traitement de {filename}...")
        try:
            data = load_workflow(filename)
            name = data.get("name")
            nodes = data.get("nodes", [])
            connections = data.get("connections", {})
            
            nodes_json = json.dumps(nodes)
            connections_json = json.dumps(connections)
            
            w_id = str(uuid.uuid4())
            v_id = str(uuid.uuid4())
            
            # Étape 1: Insertion Workflow
            sql_lines.append(f"""
-- [{name}]
INSERT INTO workflow_entity (id, name, active, nodes, connections, settings, \"versionId\", \"createdAt\", \"updatedAt\", \"versionCounter\")
VALUES ('{w_id}', '{escape_sql(name)}', true, '{escape_sql(nodes_json)}', '{escape_sql(connections_json)}', '{{}}', '{v_id}', NOW(), NOW(), 1);

-- Étape 2: Insertion History (obligatoire pour activeVersionId)
-- Colonnes critiques: authors (json), autosaved (boolean), updatedAt (timestamp)
INSERT INTO workflow_history (\"versionId\", \"workflowId\", \"nodes\", \"connections\", \"createdAt\", \"updatedAt\", \"authors\", \"autosaved\")
VALUES ('{v_id}', '{w_id}', '{escape_sql(nodes_json)}', '{escape_sql(connections_json)}', NOW(), NOW(), '[]', false);

-- Étape 3: Liaison activeVersionId
UPDATE workflow_entity SET \"activeVersionId\" = '{v_id}' WHERE id = '{w_id}';
""")

            # Étape 4: Webhook Entity
            webhook_node = next((n for n in nodes if n["type"] == "n8n-nodes-base.webhook"), None)
            if webhook_node:
                path = webhook_node.get("parameters", {}).get("path")
                if path:
                    method = webhook_node.get("parameters", {}).get("httpMethod", "POST")
                    node_name = webhook_node.get("name", "Webhook")
                    path_len = len(path.split('/'))
                    w_webhook_id = str(uuid.uuid4())
                    sql_lines.append(f"""
INSERT INTO webhook_entity (\"webhookPath\", \"method\", \"node\", \"webhookId\", \"pathLength\", \"workflowId\")
VALUES ('{escape_sql(path)}', '{method}', '{escape_sql(node_name)}', '{w_webhook_id}', {path_len}, '{w_id}');
""")
            
        except Exception as e:
            print(f"  ❌ Erreur critique sur {filename}: {e}")
            sql_lines.append("ROLLBACK;")
            return 1

    sql_lines.append("COMMIT;")
    
    sql_file = Path("deploy_final_v1.sql")
    with open(sql_file, "w", encoding="utf-8") as f:
        f.write("\n".join(sql_lines))
    
    print(f"  💾 Injection SQL dans {DB_CONTAINER}...")
    try:
        with open(sql_file, "r", encoding="utf-8") as f:
            result = subprocess.run(
                ["docker", "exec", "-i", DB_CONTAINER, "psql", "-U", DB_USER, "-d", DB_NAME],
                input=f.read(),
                capture_output=True,
                text=True
            )
        
        if result.stderr and "ERROR" in result.stderr:
            print(f"    ❌ LOGS SQL ERROR:\n{result.stderr}")
            return 1
            
        if result.returncode == 0:
            print("  ✅ Injection SQL terminée avec succès !")
        else:
            print(f"  ❌ Échec de l'injection SQL (Code {result.returncode})")
            print(result.stderr)
            return 1
    except Exception as e:
        print(f"  ❌ Erreur lors de l'exécution: {e}")
        return 1
    finally:
        if sql_file.exists():
            sql_file.unlink()

    # Redémarrage n8n
    print(f"  🔄 Redémarrage de {N8N_CONTAINER}...")
    subprocess.run(["docker", "restart", N8N_CONTAINER], capture_output=True)
    
    print("\n🎉 Déploiement terminé. Vérifiez l'interface n8n ou utilisez le client chat.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
