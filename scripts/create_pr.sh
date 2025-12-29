#!/usr/bin/env bash
set -euo pipefail

# Script pour créer une Pull Request automatiquement
# Prérequis:
# - Dépôt git initialisé et remote "origin" configuré
# - GitHub CLI installé et authentifié: https://cli.github.com/ (commande: gh auth login)
# - Branche de base existante (par défaut: main)
#
# Variables optionnelles:
#   BASE_BRANCH (defaut: main)
#   FEATURE_BRANCH (defaut: feature/stabilisation-geo-n8n)
#   PR_TITLE (defaut: "Stabilisation géospatiale + n8n: infra, scripts, MCP, CI, sauvegardes, doc, monitoring")
#   PR_BODY_FILE (chemin vers un fichier de description si fourni)

BASE_BRANCH="${BASE_BRANCH:-main}"
FEATURE_BRANCH="${FEATURE_BRANCH:-feature/stabilisation-geo-n8n}"
PR_TITLE="${PR_TITLE:-Stabilisation géospatiale + n8n: infra, scripts, MCP, CI, sauvegardes, doc, monitoring}"
PR_BODY_FILE="${PR_BODY_FILE:-}"

# Vérifications
command -v git >/dev/null 2>&1 || { echo "git introuvable"; exit 1; }
if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "Ce répertoire n'est pas un dépôt git. Initialisez-le et configurez un remote origin."
  exit 1
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  echo "Remote 'origin' non configuré. Configurez-le avant de continuer."
  exit 1
fi

# Optionnel: vérifier que gh est disponible
if ! command -v gh >/dev/null 2>&1; then
  echo "Avertissement: GitHub CLI (gh) non installé. Le script fera un push de la branche et affichera l'URL pour créer la PR manuellement."
  USE_GH=false
else
  USE_GH=true
fi

# Fetch et création de branche
echo "[create_pr] Récupération des dernières références..."
git fetch origin "$BASE_BRANCH" --prune

echo "[create_pr] Création/changement de branche: $FEATURE_BRANCH"
if git rev-parse --verify "$FEATURE_BRANCH" >/dev/null 2>&1; then
  git checkout "$FEATURE_BRANCH"
else
  git checkout -b "$FEATURE_BRANCH" "origin/$BASE_BRANCH"
fi

# Ajouter tout et commit (skip si rien à committer)
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "[create_pr] Ajout des changements et commit..."
  git add -A
  git commit -m "$PR_TITLE" || true
else
  echo "[create_pr] Aucun changement à committer."
fi

# Push
echo "[create_pr] Push de la branche vers origin..."
git push -u origin "$FEATURE_BRANCH"

# Création de la PR
if [ "$USE_GH" = true ]; then
  echo "[create_pr] Création de la PR via GitHub CLI..."
  if [ -n "$PR_BODY_FILE" ] && [ -f "$PR_BODY_FILE" ]; then
    gh pr create --title "$PR_TITLE" --body-file "$PR_BODY_FILE" --base "$BASE_BRANCH" --head "$FEATURE_BRANCH"
  else
    gh pr create --title "$PR_TITLE" --body "$PR_TITLE" --base "$BASE_BRANCH" --head "$FEATURE_BRANCH"
  fi
else
  REPO_URL=$(git remote get-url origin)
  echo "GitHub CLI absent. Ouvrez une PR manuellement:"
  echo "- Branche source: $FEATURE_BRANCH"
  echo "- Branche cible:  $BASE_BRANCH"
  echo "- Titre:         $PR_TITLE"
  echo "Repo:           $REPO_URL"
fi

echo "[create_pr] Terminé."
