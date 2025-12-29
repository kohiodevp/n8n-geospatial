# Documentation des Workflows n8n

Ce dossier contient les définitions des workflows pour le projet n8n-geospatial.

## 🚀 Démarrage Rapide

### Tester les Agents IA

Utilisez le client de chat interactif pour dialoguer avec les agents :

```bash
# Depuis le conteneur n8n ou en local
python scripts/chat_client.py              # Agent Guide (par défaut)
python scripts/chat_client.py cadastre     # Agent Cadastral
python scripts/chat_client.py domaine      # Agent Domanial
python scripts/chat_client.py --list       # Liste les agents
```

---

## 🤖 Agents IA (Principaux)

Ces workflows utilisent l'intelligence artificielle (GPT-4) pour interagir avec les données géospatiales.

| Agent              | Webhook                 | Description                                   |
| ------------------ | ----------------------- | --------------------------------------------- |
| **Guide**          | `/webhook/ai-guide`     | Assistant de navigation et diagnostic système |
| **Cadastral**      | `/webhook/ai-cadastral` | Expert en données cadastrales et parcellaires |
| **Domanial**       | `/webhook/ai-domanial`  | Spécialiste gestion du domaine de l'État      |
| **Administrateur** | `/webhook/ai-admin`     | Supervision et maintenance du système         |

### Détails des Agents

#### AI Agent Guide (`ai_agent_guide.json`)

- **Outils** : `list_workflows`, `check_db_status`, `get_system_info`
- **Rôle** : Orienter les utilisateurs, expliquer le système, diagnostiquer l'état

#### AI Agent Cadastral (`ai_agent_cadastral.json`)

- **Outils** : `search_parcelle_by_id`, `find_parcelles_nearby`, `get_proprietaire_info`, `get_stats_commune`
- **Rôle** : Recherche de parcelles, propriétaires, analyses spatiales

#### AI Agent Domanial (`ai_agent_domanial.json`)

- **Outils** : `search_bien_by_ref`, `analyse_bien_details`, `search_dossier_by_ref`
- **Rôle** : Gestion des biens fonciers, évaluations, instruction des dossiers

---

## 📥 Import et Traitement de Données

- `api_import.json` + MCP `import_geodata` (NOUVEAU): import générique de fichiers géospatiaux (ZIP/SHP/GeoJSON/GPX/CSV) vers PostGIS.
- `import_cadastre_mcp.json` (Recommandé): import des données cadastrales via MCP (param: code commune, ex: 75056).
- `import_cadastre.json` (Legacy): ancienne version. Utilisez MCP de préférence.

---

## 🛠️ Maintenance et Utilitaires

| Workflow                   | Description                                      |
| -------------------------- | ------------------------------------------------ |
| `optimize_postgis.json`    | Maintenance automatique PostGIS (VACUUM ANALYZE) |
| `credentials_postgis.json` | Modèle pour les identifiants de connexion        |
| `debug_workflows.json`     | Outils de débogage et tests                      |

---

## 📊 Analyses Spécifiques

| Workflow                          | Description                        |
| --------------------------------- | ---------------------------------- |
| `calcul_valeur_locative.json`     | Estimation de la valeur locative   |
| `controle_occupation.json`        | Détection d'anomalies d'occupation |
| `planification_urbaine.json`      | Outils pour l'analyse urbaine      |
| `prediction_valeur_fonciere.json` | Prédiction de valeurs foncières    |

---

## 🌐 API Web (pour Flutter)

Base URL (dev): `http://localhost:5678`

Routes disponibles:
- GET `/webhook/api/parcelles/:id` → GeoJSON Feature (4326).
- GET `/webhook/api/parcelles?xmin&ymin&xmax&ymax&srid=4326&limit=100&offset=0` → GeoJSON FeatureCollection.
- POST `/webhook/api/import` (multipart: champ `data`, header `x-api-key`) → crée un `jobId` et enregistre le fichier dans `/files/uploads`.
- GET `/webhook/api/jobs/:id/status` → statut du job (queued/running/success/failed, message).
- GET `/webhook/api/jobs/:id/log` (header `x-api-key`) → contenu texte du log d’import (si disponible).

Exemples rapides (curl):
```bash
# Parcelle par ID
curl -sS "http://localhost:5678/webhook/api/parcelles/123" | jq .

# Parcelles par BBOX
curl -sS "http://localhost:5678/webhook/api/parcelles?xmin=2.33&ymin=48.85&xmax=2.35&ymax=48.86&limit=50" | jq .

# Import d'un fichier (GeoJSON, SHP en ZIP, GPX) — protégé par x-api-key
curl -sS -H "x-api-key: <YOUR_API_KEY>" -F "data=@/chemin/vers/fichier.geojson" "http://localhost:5678/webhook/api/import" | jq .

# Statut du job
curl -sS "http://localhost:5678/webhook/api/jobs/<job-uuid>/status" | jq .

# Télécharger le log d'import — protégé par x-api-key
curl -sS -H "x-api-key: <YOUR_API_KEY>" "http://localhost:5678/webhook/api/jobs/<job-uuid>/log"
```

Notes:
- Les réponses GeoJSON sont normalisées (Feature/FeatureCollection) en SRID 4326.
- Paramètres `limit` (1..1000), `offset` (>=0), `sort` (`id` ou `-id`).
- `includeTotal=true` ajoute le champ `totalCount` à la FeatureCollection.
- En cas de BBOX invalide → HTTP 400. Parcelle introuvable → HTTP 404.
- Créez les credentials Postgres n8n identifiés comme "PostGIS Credentials".

---

## 🧭 Intégration Flutter (guide rapide)

### Client Dart (OpenAPI)
- Générer le client:
  - bash scripts/generate_dart_client.sh (nécessite Docker)
- Utilisation (exemple avec Dio):
  - import 'package:geospatial_api_client/api.dart';
  - final api = GeospatialApi(dio: Dio()..interceptors.add(ApiKeyInterceptor('<YOUR_API_KEY>')));
  - Voir helper: examples/flutter/openapi_client/lib/auth_api_key_interceptor.dart


- Config: passez l’URL de base via `--dart-define=API_BASE_URL=http://10.0.2.2:5678`.
- Providers: utilisez `parcellesBboxProvider` avec `BboxQuery` (limit/offset/sort), et `parcelleByIdProvider`.
- Map: exemple dans `examples/flutter/lib/widgets/geojson_map_example.dart` (Flutter Map + bbox).
- Erreurs/API: 400 si bbox invalide, 404 si item absent; gérer ces cas (toasts/snackbars).
- Pagination: utilisez `offset` pour charger les pages suivantes, et `includeTotal=true` pour afficher le nombre total.

---

## 🔌 Intégration MCP

Ces workflows utilisent le serveur MCP Python (port 5001) pour des traitements avancés :

- `domain_agent_mcp.json`
- `environmental_agent_mcp.json`
- `urbanism_agent_mcp.json`
- `mcp_tool_runner.json`

---

## 🚀 Déploiement Render (gratuit)

Prérequis:
- Compte Render (https://render.com), repo GitHub connecté
- Fichier `render.yaml` à la racine (fourni dans ce dépôt)

Étapes:
1. Sur Render, cliquez “New +” → “Blueprint” → sélectionnez votre repo contenant `render.yaml`.
2. Configurez les variables sensibles dans l’interface Render (Service → Environment → Environment Variables):
   - N8N_ENCRYPTION_KEY (clé forte), N8N_USER_MANAGEMENT_JWT_SECRET (secret fort)
   - N8N_BASIC_AUTH_PASSWORD (accès UI n8n), N8N_RUNNERS_AUTH_TOKEN (fort)
3. Déployez: Render provisionne la base Postgres (plan free) et lance:
   - Service web n8n-geospatial (port 5678, startCommand: /startup.sh, health: /healthz)
   - Worker mcp-server (port 5001, startCommand: python /opt/geoscripts/mcp_server.py)
4. Récupérez l’URL publique (ex: https://<service>.onrender.com) et mettez à jour vos clients (Flutter) avec API_BASE_URL.
5. (Optionnel) Ajoutez un disque persistant au worker si vous souhaitez conserver des caches/logs.

Bonnes pratiques (gratuit):
- Restez léger: évitez les imports massifs; utilisez des limites/pagination côté API
- Sécurité: gardez N8N_ENABLE_EXECUTE_COMMAND=false en prod; forcez HTTPS (HSTS déjà activé dans nginx.conf)
- Observabilité: téléchargez les logs d’import via /webhook/api/jobs/:id/log pour diagnostiquer rapidement

---

## 📁 Structure des Fichiers

```
workflows/
├── ai_agent_*.json          # Agents IA principaux
├── *_mcp.json               # Workflows utilisant MCP
├── import_*.json            # Import de données
├── credentials_*.json       # Modèles de credentials
└── exports-ui/              # Exports de l'interface n8n
```
