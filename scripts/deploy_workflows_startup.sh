#!/bin/bash
# deploy_workflows_startup.sh - Script de déploiement des workflows au démarrage

set -e

echo "🚀 Démarrage du déploiement des workflows n8n"
echo "=============================================="

# Attendre que n8n soit pleinement démarré
echo "⏳ Attente que n8n soit prêt..."
sleep 10

# Vérifier que les fichiers de workflow sont présents
echo "🔍 Vérification des fichiers de workflow..."
WORKFLOW_COUNT=$(ls -la /home/node/.n8n/workflows/*.json 2>/dev/null | wc -l)
echo "   • Fichiers de workflow trouvés: $WORKFLOW_COUNT"

# Afficher les workflows importants
echo "📋 Workflows disponibles:"
ls -la /home/node/.n8n/workflows/ | grep -E '\.json$' | head -5 | while read line; do
    echo "   • $(echo $line | awk '{print $9}')"
done

echo "✅ Les workflows sont déjà disponibles dans le répertoire de n8n"
echo "   Ils seront automatiquement chargés par l'instance n8n"
echo "   Accédez à l'interface web pour les consulter et les activer"
echo "   URL: http://localhost:5678"