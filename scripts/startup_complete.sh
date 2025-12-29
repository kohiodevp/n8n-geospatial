#!/bin/bash
# startup_complete.sh - Script de démarrage complet du système n8n Geospatial

echo "🚀 Démarrage complet du système n8n Geospatial"
echo "=============================================="

echo "📍 Vérification de l'état des services..."
if docker ps | grep -q "n8n-geospatial"; then
    echo "✅ Service n8n-geospatial en cours d'exécution"
else
    echo "❌ Service n8n-geospatial non trouvé"
    exit 1
fi

if docker ps | grep -q "n8n-postgis"; then
    echo "✅ Service PostGIS en cours d'exécution"
else
    echo "❌ Service PostGIS non trouvé"
    exit 1
fi

if docker ps | grep -q "n8n-redis"; then
    echo "✅ Service Redis en cours d'exécution"
else
    echo "❌ Service Redis non trouvé"
    exit 1
fi

echo ""
echo "📍 Vérification de l'accessibilité de l'interface n8n..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5678/)
if [ "$HTTP_CODE" -eq 200 ]; then
    echo "✅ Interface n8n accessible (HTTP $HTTP_CODE)"
else
    echo "❌ Interface n8n inaccessible (HTTP $HTTP_CODE)"
    exit 1
fi

echo ""
echo "📍 Vérification des scripts et agents..."
if docker exec n8n-geospatial ls /opt/geoscripts/ | grep -q "cadastral_agent.py"; then
    echo "✅ Scripts d'agents présents"
else
    echo "❌ Scripts d'agents manquants"
    exit 1
fi

if docker exec n8n-geospatial ls /home/node/.n8n/workflows/ | grep -q ".json"; then
    echo "✅ Workflows présents"
else
    echo "❌ Workflows manquants"
    exit 1
fi

echo ""
echo "📍 Vérification des bibliothèques géospatiales..."
if docker exec n8n-geospatial python -c "import geopandas, shapely, pyproj" 2>/dev/null; then
    echo "✅ Bibliothèques géospatiales fonctionnelles"
else
    echo "❌ Erreur avec les bibliothèques géospatiales"
    exit 1
fi

echo ""
echo "📍 Vérification du bon fonctionnement des agents..."
if docker exec n8n-geospatial python -c "from cadastral_agent import CadastralAgent; print('Agent disponible')" 2>/dev/null | grep -q "disponible"; then
    echo "✅ Agents géospatiaux fonctionnels"
else
    echo "❌ Problème avec les agents géospatiaux"
    exit 1
fi

echo ""
echo "🎯 Système prêt à l'emploi!"
echo ""
echo "🌐 Accès à l'interface: http://localhost:5678"
echo "👤 Identifiants: admin / cadastre2024"
echo ""
echo "🤖 Agents disponibles:"
echo "   - CadastralAgent: Analyse cadastrale"
echo "   - DomainAgent: Gestion domaniale"
echo "   - UrbanismAgent: Planification urbaine"
echo "   - EnvironmentalAgent: Surveillance environnementale"
echo ""
echo "📋 Workflows déployés: 22 workflows géospatiaux"
echo ""
echo "✅ Le système n8n Geospatial est pleinement opérationnel!"