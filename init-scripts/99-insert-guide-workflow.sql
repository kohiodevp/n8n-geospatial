INSERT INTO workflow_entity (id, name, active, nodes, connections, settings, "createdAt", "updatedAt")
VALUES (
  'guide-agent-v1',
  'AI Agent Guide',
  true,
  '[{"parameters":{"httpMethod":"POST","path":"ai-guide","responseMode":"lastNode","options":{}},"id":"trigger","name":"Webhook","type":"n8n-nodes-base.webhook","typeVersion":1,"position":[220,300]},{"parameters":{"mode":"runOnceForAllItems","jsCode":"const userMessage = $input.first().json.body?.message || ''Pas de message''; return [{ json: { output: ''Bonjour! '' + userMessage } }];"},"id":"respondCode","name":"Reponse","type":"n8n-nodes-base.code","typeVersion":2,"position":[460,300]}]',
  '{"Webhook":{"main":[[{"node":"Reponse","type":"main","index":0}]]}}',
  '{}',
  NOW(),
  NOW()
)
ON CONFLICT (id) DO UPDATE SET
  nodes = EXCLUDED.nodes,
  connections = EXCLUDED.connections,
  active = EXCLUDED.active,
  "updatedAt" = NOW();
