#!/bin/bash
# deploy_workflows.sh - Déploiement automatique des workflows n8n

set -e

echo "🚀 Déploiement automatique des workflows n8n"
echo "============================================="

# Définir l'URL de l'instance n8n
N8N_URL="${N8N_URL:-http://localhost:5678}"

# Vérifier si l'instance n8n est accessible
echo "🔍 Vérification de l'accessibilité de l'instance n8n..."
if ! curl -s -f "$N8N_URL/healthz" > /dev/null; then
    echo "❌ L'instance n8n n'est pas accessible à $N8N_URL"
    exit 1
fi

echo "✅ Instance n8n accessible"

# Déployer les workflows via le script Python
echo "📦 Déploiement des workflows..."
cd /opt/geoscripts
python3 deploy_workflows.py

echo "✅ Déploiement terminé"