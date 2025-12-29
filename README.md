# n8n Geospatial Workflow Runner

Ce projet fournit un runner de workflow géospatial pour n8n utilisant QGIS et des bibliothèques Python géospatiales. Il est conçu pour automatiser des traitements complexes dans les domaines du cadastre, de la gestion domaniale, de l'urbanisme, de l'environnement et de la planification territoriale.

## Fonctionnalités

- **Agents géospatiaux IA**: Cadastral, domanial, urbanisme, environnemental
- **Analyse spatiale avancée**: Validation géométrique, détection d'anomalies, clustering spatial
- **Intégration PostGIS**: Base de données spatiale pour le stockage et l'analyse
- **Traitement QGIS**: Outils d'analyse spatiale avancés
- **Machine Learning**: Algorithmes pour la classification, la prédiction et la détection d'anomalies
- **Automatisation complète**: Workflows n8n pour des processus complexes
- **Gestion avancée des workflows**: Orchestration, surveillance et statistiques
- **Déploiement automatique**: 22 workflows géospatiaux automatiquement chargés

## Architecture

Le projet utilise une architecture Docker composée de:

- **n8n-geospatial**: Service principal avec bibliothèques géospatiales
- **postgis**: Base de données PostgreSQL avec extension PostGIS
- **redis**: Gestion des queues pour les workflows
- **nginx**: Proxy inverse avec configuration optimisée pour les traitements géospatiaux

## Workflows Notables

Le projet inclut plusieurs workflows n8n pour l'automatisation. Voici quelques exemples clés :

- **`import_cadastre.json`**: Un workflow ETL (Extract, Transform, Load) qui télécharge les données cadastrales (parcelles, bâtiments) depuis `cadastre.data.gouv.fr`, les transforme et les charge dans la base de données PostGIS.
- **`ai_agent_cadastral.json`**: Un agent conversationnel basé sur l'IA (GPT-4) qui peut répondre à des questions sur le cadastre en utilisant des outils pour interroger la base de données et exécuter des scripts d'analyse spatiale.

## Agents Géospatiaux

### Agent Cadastral
- Validation géométrique des parcelles
- Détection d'anomalies cadastrales
- Consolidation de parcelles
- Prédiction de valeurs foncières
- Analyse de voisinage
- Détection de changements d'usage du sol

### Agent Domanial
- Gestion des propriétés domaniales
- Analyse des concessions
- Identification des zones stratégiques
- Optimisation de la gestion
- Suggestions d'optimisation de concessions

### Agent d'Urbanisme
- Analyse de densité urbaine
- Identification des opportunités de développement
- Évaluation de la capacité des infrastructures
- Prédiction de croissance urbaine
- Analyse d'accessibilité
- Simulation de scénarios de développement

### Agent Environnemental
- Surveillance environnementale
- Détection des risques environnementaux
- Analyse des hotspots de biodiversité
- Prédiction des tendances environnementales
- Évaluation des services écosystémiques
- Identification des priorités de conservation
- Analyse de l'impact de la pollution

## Prérequis

- Docker et Docker Compose
- Windows (scripts batch fournis) ou Linux/macOS
- 4 Go de RAM minimum (8 Go recommandés pour les traitements intensifs)
- 10 Go d'espace disque disponible

## Installation (Docker recommandé)

1. Clonez ce dépôt:
```bash
git clone https://github.com/votre-compte/n8n-geospatial.git
cd n8n-geospatial
```

2. Préparez l'environnement:
```bash
# Copier l'exemple d'environnement
cp .env.example .env

# Créer les dossiers locaux utilisés par les volumes
mkdir -p data geodata workflows tmp/geodata-cache
```

3. Démarrez le système Docker:
```bash
docker compose up -d --build
```

4. Vérifiez la santé (optionnel en local):
- Windows PowerShell: exécutez les commandes séparément (pas de `&&`).
```bash
python3 scripts/health_check.py
python3 scripts/verify_system.py
```

5. Accédez à l'interface:
   - URL: http://localhost:5678
   - Utilisateur: admin
   - Mot de passe: cadastre2024

## Utilisation

### Démarrage rapide
```bash
# Démarrer le système
.\scripts\start_n8n_geospatial.bat start

# Vérifier l'état du système
.\scripts\system_check.bat

# Exécuter les tests des agents
.\scripts\test_geospatial_agents.bat run-all
```

### Développement
```bash
# Démarrer en mode développement
.\scripts\dev_n8n_geospatial.bat dev-start

# Recharger les workflows
.\scripts\dev_n8n_geospatial.bat dev-reload

# Accéder au shell du conteneur
.\scripts\dev_n8n_geospatial.bat dev-shell
```

### Outils de gestion
- `start_n8n_geospatial.bat`: Gestion de base du système
- `dev_n8n_geospatial.bat`: Fonctionnalités de développement
- `test_geospatial_agents.bat`: Tests des agents géospatiaux
- `optimize_project.bat`: Outils d'optimisation
- `system_check.bat`: Vérification de l'état du système

## Scripts Principaux

Le répertoire `scripts/` contient les logiques de traitement personnalisées.

- **`analyse_spatiale.py`**: Fournit des fonctions d'analyse spatiale (buffer, calcul de surface, détection de chevauchement) qui peuvent être appelées depuis les workflows n8n.
- **`download_cadastre.sh`**: Un script shell pour télécharger les données cadastrales d'une commune spécifique depuis les services de `data.gouv.fr`.
- `cadastral_agent.py`: Agent pour l'analyse cadastrale
- `domain_agent.py`: Agent pour la gestion domaniale
- `urbanism_agent.py`: Agent pour l'urbanisme et l'aménagement
- `environmental_agent.py`: Agent pour l'environnement
- `workflow_manager.py`: Gestion avancée des workflows
- `main_runner.py`: Point d'entrée principal du système
- `example_agents.py`: Exemples d'utilisation des agents
- `integrated_demo.py`: Démonstration intégrée complète


## Configuration

Le projet est entièrement configurable via le fichier `.env` qui contient des paramètres pour:

- **Sécurité**: Clés de chiffrement, authentification, secrets JWT.
- **Base de données**: Les informations de connexion à la base de données PostGIS sont gérées via les variables d'environnement (ex: `DB_POSTGRESDB_HOST`, `DB_POSTGRESDB_USER`, `DB_POSTGRESDB_PASSWORD`).
- **Performance**: Limites mémoire, taille du cache.
- **Géospatial**: Paramètres pour GDAL et PROJ.
- **Développement**: Chemins de volumes, niveaux de logs.

## Déploiement

### Environnement de développement
Utilisez les scripts de développement pour un cycle de développement rapide.

### Production
- Remplacez la clé de chiffrement par une valeur sécurisée.
- Utilisez des secrets pour injecter les mots de passe et les clés d'API (plutôt que de les laisser dans le `.env`).
- Activez SSL/TLS sur le proxy inverse Nginx.
- Configurez des sauvegardes régulières de la base de données PostGIS.
- Mettez en place une surveillance des performances des conteneurs.

## Utiliser le MCP (Model Context Protocol)

Exemples rapides:

- Santé du service (hôte):
```bash
curl -sS http://localhost:5001/healthz
```

- Liste des outils disponibles (depuis n8n, dans le conteneur):
```bash
docker exec n8n-geospatial curl -sS http://mcp-server:5001/tools
```

- Invocation d’un outil (POST JSON):
```bash
docker exec n8n-geospatial sh -lc "curl -sS -H 'Content-Type: application/json' \
  -d '{\"tool_name\":\"ping\",\"params\":{\"source\":\"n8n\"}}' \
  http://mcp-server:5001/invoke"
```

Configuration du nœud n8n (HTTP Request):
- URL: http://mcp-server:5001/invoke
- Method: POST
- Send: JSON
- Body JSON: {"tool_name":"ping","params":{"source":"n8n"}}
- Response: JSON, Timeout: 10s

Conseils:
- Éviter les guillemets dans les noms de nœuds (ex: "Invoke MCP 'ping'"). Préférer: Invoke MCP ping.
- Si vous importez des workflows via CLI, utilisez des JSON exportés depuis l’UI n8n (v2) pour éviter les erreurs d’intégrité.

## Checklist post-import n8n v2 (éviter l’erreur nodeName)

- Renommez proprement les nœuds (sans guillemets ni caractères spéciaux).
- Ouvrez l’onglet “Connections” et supprimez/recréez les connexions orphelines.
- Vérifiez les expressions $node["..."] et mettez à jour le nom exact des nœuds référencés.
- Testez les nœuds critiques un par un (“Execute Node”).
- Import CLI: privilégiez des fichiers exportés depuis l’UI n8n. Évitez les JSON manuscrits.
- Auto-import: laissez désactivé tant que vos workflows ne sont pas au format d’export UI validé.

## Sauvegardes BD (PostgreSQL/PostGIS)

Restauration:
```bash
# Exemple de restauration
bash scripts/restore_db.sh ./backups/ma_base_20250101_120000.sql.gz
# Forcer la création de la base si nécessaire
CREATE_DB=true bash scripts/restore_db.sh ./backups/ma_base_20250101_120000.sql.gz
```


- Script: scripts/backup_db.sh
- Pré-requis: DATABASE_URL ou variables PG* (PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE)
- Exemple:
```bash
# Linux/macOS
bash scripts/backup_db.sh

# Variables optionnelles
RETENTION_DAYS=14 BACKUP_DIR=./backups bash scripts/backup_db.sh
```
- Planification: via cron (ex: quotidien)

## Dépannage (Troubleshooting)

- nginx unhealthy
  - Cause: healthcheck sur /
  - Action: endpoint /health ajouté; healthcheck pointe désormais /health
- n8n erreur "Cannot read properties of undefined (reading 'nodeName')"
  - Causes: renommage de nœud, connexion orpheline, import JSON non-UI
  - Actions: corriger les connexions, mettre à jour les expressions $node["..."], tester les nœuds clés, préférer export UI
- MCP "service refused connection" ou JSON invalide
  - Causes: serveur pas prêt, HEAD non géré, méthode non-POST, body non-JSON
  - Actions: MCP /healthz OK, nœud HTTP en POST JSON, CORS activé, parsing JSON durci

## Sécurité

- Secrets: .env (N8N_ENCRYPTION_KEY, DB creds). Ne pas commit des secrets.
- Réseau: exposer publiquement uniquement nginx; services internes sur réseau Docker.
- Rôles DB: utiliser un rôle applicatif dédié, permissions minimales.

## Observabilité & Logs

- n8n: logs persistants (volume) + nœuds d’error handling avec sortie JSON.
- MCP: logs structurés (option LOG_LEVEL, à ajouter si besoin), endpoints de santé.
- PostGIS: log_min_duration_statement pour traquer les requêtes lentes.

## Conventions & Bonnes pratiques n8n

- Noms de nœuds: simples, sans guillemets ni caractères spéciaux.
- Expressions: limiter $node["..."]; préférer des nœuds Set pour stocker les valeurs intermédiaires.
- Import/export: travailler en UI; exporter pour réimporter en CLI; désactiver l’auto-import sauf JSON UI validés.

## Makefile (DX)

### Créer une Pull Request automatiquement

```bash
# Depuis la racine du repo
bash scripts/create_pr.sh
# Variables optionnelles:
# BASE_BRANCH=main FEATURE_BRANCH=feature/ma-branche PR_TITLE="Mon titre" bash scripts/create_pr.sh
```


Commandes rapides:
```bash
make up          # build + up
make down        # stop + cleanup
make build       # build images
make health      # tests santé n8n + MCP
make backup      # pg_dump (./backups)
make restore file=./backups/xxx.sql.gz  # restauration
make mcp-ping    # appel ping MCP depuis n8n
```

## Activer HTTPS (nginx)

- Montez vos certificats dans ./ssl (fullchain.pem, privkey.pem)
- Décommentez le bloc HTTPS (port 443) dans nginx.conf
- Adaptez server_name et chemins si besoin
- Redéployez: docker compose up -d --build

Note: en local, vous pouvez générer des certs de test (mkcert/openssl). En prod, privilégiez Let’s Encrypt/traefik.

## Monitoring / Alerting (basique)

- Script de monitoring: scripts/monitor_services.sh
```bash
bash scripts/monitor_services.sh
# Variables:
# NGINX_URL=http://localhost/health MCP_URL=http://localhost:5001/healthz N8N_URL=http://localhost:5678/healthz bash scripts/monitor_services.sh
```
- Intégration cron (exemple):
```
*/5 * * * * cd /path/to/project && bash scripts/monitor_services.sh >> logs/monitor.log 2>&1
```
- Pour des alertes: redirigez la sortie JSON vers un webhook (Slack/Teams) via un petit script wrapper.

## Workflows MCP Agents (Cadastral/Domain/Urbanism/Environmental)

- Import depuis l’UI n8n (recommandé):
  - Importer les fichiers MCP (ex: workflows/import_cadastre_mcp.json) → ajuster les entrées → tester → Export UI (format v2 compatible)
  - Déposer les exports UI dans workflows/exports-ui/

- Import CLI (une fois les exports UI disponibles):
```bash
bash scripts/import_workflows_cli.sh workflows/exports-ui
```

- Notifications d’erreurs:
  - Renseignez SLACK_WEBHOOK_URL dans .env (facultatif). Les workflows MCP enverront une alerte sur les branches d’erreur.

- Bonnes pratiques:
  - Conservez des noms de nœuds simples (sans guillemets), vérifiez les connexions après import
  - Pour MCP: HTTP Request en POST JSON; body = JSON valide (Send: JSON)
  - Exportez depuis l’UI pour garantir la compatibilité CLI v2.

## Adaptateurs MCP (Agents)

Outils disponibles via MCP (/invoke):
- cadastral_agent
  - Actions: import, validate, analyze
  - Exemples payloads:
    - Import:
      {
        "tool_name":"cadastral_agent",
        "params":{
          "action":"import",
          "inputs":{ "file":"/files/sample_parcelles.geojson", "table":"cadastre.parcelles", "srid":2154 },
          "options":{ "if_exists":"append", "chunksize":5000 }
        }
      }
    - Validate:
      {
        "tool_name":"cadastral_agent",
        "params":{
          "action":"validate",
          "inputs":{ "table":"cadastre.parcelles" },
          "options":{ "fix_invalid":false }
        }
      }
    - Analyze:
      {
        "tool_name":"cadastral_agent",
        "params":{
          "action":"analyze",
          "inputs":{ "table":"cadastre.parcelles" },
          "options":{ "summary":true }
        }
      }
- domain_agent
  - Actions: inventory
  - Exemple:
    {
      "tool_name":"domain_agent",
      "params":{ "action":"inventory", "inputs":{ "table":"domaine.biens" } }
    }
- urbanism_agent
  - Actions: zoning_check
  - Exemple:
    {
      "tool_name":"urbanism_agent",
      "params":{ "action":"zoning_check", "inputs":{ "parcels_table":"cadastre.parcelles", "zoning_table":"urbanisme.zone_urba" } }
    }
- environmental_agent
  - Actions: buffer
  - Exemple:
    {
      "tool_name":"environmental_agent",
      "params":{ "action":"buffer", "inputs":{ "table":"environment.sites", "distance_m":200, "output_table":"environment.sites_buffer" } }
    }

Configuration du nœud n8n (HTTP Request):
- URL: http://mcp-server:5001/invoke
- Method: POST
- Send: JSON
- Body: le payload JSON ci-dessus
- Response: JSON; Timeout: adapté à la volumétrie

Pré-requis DB & notifications:
- Assurez-vous que les tables/schémas référencés existent (cadastre.parcelles, cadastre.batiments, domaine.biens, urbanisme.zone_urba, environment.sites, etc.).
- PostGIS doit être activé; les index spatiaux sont recommandés (create_spatial_index dans postgis_utils).
- Pour activer les notifications d’erreurs dans les workflows MCP, définissez SLACK_WEBHOOK_URL dans .env.

## Documentation

- [Documentation détaillée des agents géospatiaux](./docs/ia_geospatial_features.md)
- [Tutoriel d'utilisation](./tutoriel.md)

## Contribution

Les contributions sont les bienvenues ! Veuillez ouvrir une issue pour discuter des modifications que vous souhaitez apporter avant de créer une pull request.

## Licence

Ce projet est licencié sous la licence MIT.
