#!/bin/bash
# SQL Script execution helper for n8n-postgis
cat <<EOF > /tmp/deploy_n8n.sql
DO \$\$ 
DECLARE 
    w_id_guide UUID := 'a1a1a1a1-a1a1-a1a1-a1a1-a1a1a1a1a1a1';
    v_id_guide UUID := gen_random_uuid();
    
    w_id_cad UUID := 'b2b2b2b2-b2b2-b2b2-b2b2-b2b2b2b2b2b2';
    v_id_cad UUID := gen_random_uuid();
    
    w_id_dom UUID := 'c3c3c3c3-c3c3-c3c3-c3c3-c3c3c3c3c3c3';
    v_id_dom UUID := gen_random_uuid();
    
    w_id_adm UUID := 'd4d4d4d4-d4d4-d4d4-d4d4-d4d4d4d4d4d4';
    v_id_adm UUID := gen_random_uuid();
BEGIN
    -- Nettoyage
    DELETE FROM workflow_entity WHERE name LIKE 'AI Agent%';
    DELETE FROM webhook_entity;

    -- GUIDE
    INSERT INTO workflow_entity (id, name, active, nodes, connections, settings, "versionId", "activeVersionId", "createdAt", "updatedAt")
    VALUES (w_id_guide, 'AI Agent Guide', true, '{{GUIDE_NODES}}', '{{GUIDE_CONNECTIONS}}', '{}', v_id_guide, v_id_guide, NOW(), NOW());
    
    INSERT INTO workflow_history ("versionId", "workflowId", "nodes", "connections", "createdAt")
    VALUES (v_id_guide, w_id_guide, '{{GUIDE_NODES}}', '{{GUIDE_CONNECTIONS}}', NOW());
    
    INSERT INTO webhook_entity ("webhookPath", "method", "node", "webhookId", "pathLength", "workflowId")
    VALUES ('ai-guide', 'POST', 'Webhook', gen_random_uuid(), 1, w_id_guide);

    -- CADASTRAL
    INSERT INTO workflow_entity (id, name, active, nodes, connections, settings, "versionId", "activeVersionId", "createdAt", "updatedAt")
    VALUES (w_id_cad, 'AI Agent Cadastral', true, '{{CADASTRAL_NODES}}', '{{CADASTRAL_CONNECTIONS}}', '{}', v_id_cad, v_id_cad, NOW(), NOW());
    
    INSERT INTO workflow_history ("versionId", "workflowId", "nodes", "connections", "createdAt")
    VALUES (v_id_cad, w_id_cad, '{{CADASTRAL_NODES}}', '{{CADASTRAL_CONNECTIONS}}', NOW());
    
    INSERT INTO webhook_entity ("webhookPath", "method", "node", "webhookId", "pathLength", "workflowId")
    VALUES ('ai-cadastral', 'POST', 'Webhook', gen_random_uuid(), 1, w_id_cad);

END \$\$;
EOF

psql -U n8n -d n8n -f /tmp/deploy_n8n.sql
rm /tmp/deploy_n8n.sql
