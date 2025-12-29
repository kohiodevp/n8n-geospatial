# Documentation des Workflows n8n

Ce dossier contient les définitions des workflows pour le projet n8n-geospatial.

## 🤖 Agents IA (Recommandés)

Ces workflows utilisent l'intelligence artificielle pour interagir avec les données géospatiales.

- **`ai_agent_cadastral.json`** : Agent principal pour les requêtes cadastrales. Permet de chercher des parcelles, des propriétaires et d'effectuer des analyses spatiales simples (buffer).
- **`ai_agent_administrateur.json`** : Agent de supervision du système.
- **`ai_agent_domanial.json`** : Agent spécialisé dans la gestion du domaine de l'État.
- **`ai_agent_guide.json`** : Assistant pour guider les utilisateurs.

## 📥 Import et Traitement de Données

- **`import_cadastre_mcp.json`** (**Recommandé**) : Workflow avancé pour importer les données cadastrales (parcelles et bâtiments) d'une commune. Utilise le serveur MCP pour un traitement robuste et performant.
  - _Paramètre_ : Code commune (ex: 75056).
- **`import_cadastre.json`** (Legacy) : Ancienne version de l'import. **Utilisez la version MCP de préférence.**

## 🛠️ Maintenance et Utilitaires

- **`optimize_postgis.json`** (Nouveau) : Workflow de maintenance automatique pour optimiser les performances de la base de données PostGIS (VACUUM ANALYZE).
- **`credentials_postgis.json`** : Modèle pour les identifiants de connexion PostGIS.

## 📊 Analyses Spécifiques

- **`calcul_valeur_locative.json`** : Estimation de la valeur locative des parcelles.
- **`controle_occupation.json`** : Détections d'anomalies d'occupation.
- **`planification_urbaine.json`** : Outils pour l'analyse urbaine.
