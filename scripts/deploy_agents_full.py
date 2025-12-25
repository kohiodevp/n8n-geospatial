import os
import json
import psycopg2
import uuid

# Configuration
DB_HOST = os.environ.get('DB_POSTGRESDB_HOST', 'postgres')
DB_PORT = os.environ.get('DB_POSTGRESDB_PORT', '5432')
DB_NAME = os.environ.get('DB_POSTGRESDB_DATABASE', 'n8n')
DB_USER = os.environ.get('DB_POSTGRESDB_USER', 'n8n')
DB_PASS = os.environ.get('DB_POSTGRESDB_PASSWORD', 'n8n_password')

# Directory where workflows are mounted in the container
WORKFLOWS_DIR = "/home/node/workflows"

AGENTS = [
    {
        "id": "ai-agent-admin-001",
        "name": "AI Administrateur",
        "file": "ai_agent_administrateur.json"
    },
    {
        "id": "ai-agent-guide-001",
        "name": "AI Guide Utilisateur",
        "file": "ai_agent_guide.json"
    },
    {
        "id": "ai-agent-domanial-001",
        "name": "AI Agent Domanial",
        "file": "ai_agent_domanial.json"
    },
    {
        "id": "ai-agent-cadastral-001",
        "name": "AI Agent Cadastral",
        "file": "ai_agent_cadastral.json"
    }
]

def deploy():
    conn = None
    try:
        print(f"Connecting to database {DB_NAME} on {DB_HOST}:{DB_PORT}...")
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        conn.autocommit = True
        cur = conn.cursor()
        print("Connected successfully.")

        # Get a project ID to assume ownership
        # We try to find a personal project first (type='personal') or just any project
        cur.execute("SELECT id FROM project LIMIT 1;")
        row = cur.fetchone()
        project_id = row[0] if row else None
        
        if not project_id:
            print("WARNING: No project found in database. Workflows will be created but might not be visible to any user.")
        else:
            print(f"Using Project ID: {project_id}")

        for agent in AGENTS:
            file_path = os.path.join(WORKFLOWS_DIR, agent['file'])
            if not os.path.exists(file_path):
                print(f"ERROR: File not found: {file_path}")
                continue
                
            print(f"Reading {agent['file']}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                wf_data = json.load(f)
            
            nodes = json.dumps(wf_data.get('nodes', []))
            connections = json.dumps(wf_data.get('connections', {}))
            
            # Use fixed IDs from our list, but ensure they match
            wf_id = agent['id']
            wf_name = agent['name']
            
            print(f"Deploying {wf_name} (ID: {wf_id})...")
            
            # Upsert into workflow_entity
            # We use gen_random_uuid() for new entries
            
            sql_upsert = """
            INSERT INTO workflow_entity 
            (id, name, active, nodes, connections, "createdAt", "updatedAt", "versionId", "activeVersionId", "versionCounter")
            VALUES (%s, %s, %s, %s, %s, NOW(), NOW(), gen_random_uuid(), gen_random_uuid(), 1)
            ON CONFLICT (id) DO UPDATE SET 
                nodes = EXCLUDED.nodes,
                connections = EXCLUDED.connections,
                name = EXCLUDED.name,
                active = EXCLUDED.active,
                "updatedAt" = NOW();
            """
            
            cur.execute(sql_upsert, (wf_id, wf_name, True, nodes, connections))
            
            # Link to project if we have one
            if project_id:
                sql_share = """
                INSERT INTO "shared_workflow" ("workflowId", "projectId", "role", "createdAt", "updatedAt")
                VALUES (%s, %s, 'workflow:owner', NOW(), NOW())
                ON CONFLICT ("workflowId", "projectId") DO NOTHING;
                """
                cur.execute(sql_share, (wf_id, project_id))
            
            print(f"  -> SUCCESS: {wf_name} deployed.")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    deploy()
