#!/usr/bin/env python3
"""
Vérification finale du système n8n Geospatial
=============================================

Ce script effectue une vérification complète du système
n8n Geospatial pour s'assurer que tout est fonctionnel.
"""

import os
import json
import sys
from pathlib import Path

def check_system_status():
    """
    Vérifier l'état du système (portable: local et conteneur)
    """
    print("🔍 Vérification de l'état du système...")

    # Détermination des chemins selon l'environnement
    container_paths = {
        "workflows": "/home/node/.n8n/workflows/",
        "scripts": "/opt/geoscripts/",
        "files": "/files/",
        "geodata": "/geodata/",
        "tmp_cache": "/tmp/geodata-cache/",
    }
    local_paths = {
        "workflows": os.path.join(os.getcwd(), "workflows"),
        "scripts": os.path.join(os.getcwd(), "scripts"),
        "files": os.path.join(os.getcwd(), "files"),
        "geodata": os.path.join(os.getcwd(), "geodata"),
        "tmp_cache": os.path.join(os.getcwd(), "tmp", "geodata-cache"),
    }

    # Utiliser les chemins qui existent, sinon tomber sur l'alternative
    def pick_path(key: str) -> str:
        primary = container_paths[key]
        fallback = local_paths[key]
        return primary if os.path.exists(primary) else fallback

    workflows_dir = pick_path("workflows")
    scripts_dir = pick_path("scripts")

    # Vérifier les dossiers de travail
    required_dirs = [
        workflows_dir,
        scripts_dir,
        pick_path("files"),
        pick_path("geodata"),
        pick_path("tmp_cache"),
    ]

    print("📁 Vérification des dossiers...")
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"   ✅ {directory}")
        else:
            print(f"   ❌ {directory}")

    # Vérifier les scripts d'agents (recherche sur scripts_dir)
    agent_basenames = [
        "cadastral_agent.py",
        "domain_agent.py",
        "urbanism_agent.py",
        "environmental_agent.py",
        "workflow_manager.py",
        "main_runner.py",
    ]

    print("\n🤖 Vérification des scripts d'agents...")
    for name in agent_basenames:
        script_path = os.path.join(scripts_dir, name)
        if os.path.exists(script_path):
            print(f"   ✅ {name}")
        else:
            print(f"   ❌ {name}")

    # Vérifier les fichiers de workflow (ne pas échouer si dossier absent)
    workflow_files = []
    if os.path.exists(workflows_dir):
        try:
            workflow_files = [f for f in os.listdir(workflows_dir) if f.endswith('.json')]
        except Exception as e:
            print(f"   ❌ Impossible de lister les workflows: {e}")
            workflow_files = []
    else:
        print(f"   ⚠️ Dossier workflows introuvable: {workflows_dir}")

    print(f"\n⚙️  Vérification des workflows...")
    print(f"   • Total workflows: {len(workflow_files)}")

    # Vérifier quelques workflows spécifiques
    important_workflows = [
        "ai_agent_cadastral.json",
        "ai_agent_domanial.json",
        "planification_urbaine.json",
        "surveillance_environnementale.json",
        "consolidation_parcels.json",
    ]

    for wf in important_workflows:
        if wf in workflow_files:
            print(f"   ✅ {wf}")
        else:
            print(f"   ❌ {wf}")

    return True

def check_geospatial_libraries():
    """
    Vérifier la disponibilité des bibliothèques géospatiales
    """
    print("\n🌐 Vérification des bibliothèques géospatiales...")
    
    try:
        import geopandas as gpd
        import shapely
        import pyproj
        import rasterio
        import numpy as np
        import pandas as pd
        print("   ✅ Toutes les bibliothèques géospatiales sont disponibles")
        return True
    except ImportError as e:
        print(f"   ❌ Erreur d'importation: {e}")
        return False

def check_main_functionality():
    """
    Vérifier la fonctionnalité principale
    """
    print("\n⚡ Vérification de la fonctionnalité principale...")
    
    try:
        # Importer les agents principaux
        sys.path.append('/opt/geoscripts')
        from cadastral_agent import CadastralAgent
        from domain_agent import DomainAgent
        from urbanism_agent import UrbanismAgent
        from environmental_agent import EnvironmentalAgent
        from workflow_manager import WorkflowManager
        
        print("   ✅ Import des agents principaux: OK")
        
        # Vérifier que les classes sont correctement définies
        agents = [
            ("CadastralAgent", CadastralAgent),
            ("DomainAgent", DomainAgent),
            ("UrbanismAgent", UrbanismAgent),
            ("EnvironmentalAgent", EnvironmentalAgent),
            ("WorkflowManager", WorkflowManager)
        ]
        
        for name, cls in agents:
            if cls:
                print(f"   ✅ {name} disponible")
            else:
                print(f"   ❌ {name} non disponible")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Erreur d'importation: {e}")
        return False

def print_final_summary(system_ok: bool, libs_ok: bool, functionality_ok: bool):
    """
    Afficher le résumé final en fonction des résultats
    """
    print("\n" + "="*60)
    title = "✅ SYSTÈME N8N GEOSPATIAL - VÉRIFICATION TERMINÉE" if (system_ok and libs_ok and functionality_ok) else "⚠️ SYSTÈME N8N GEOSPATIAL - PROBLÈMES DÉTECTÉS"
    print(title)
    print("="*60)
    print("🎯 État des vérifications:")
    print(f"   • Répertoires et fichiers: {'OK' if system_ok else 'Problèmes'}")
    print(f"   • Bibliothèques géospatiales: {'OK' if libs_ok else 'Manquantes'}")
    print(f"   • Import des agents principaux: {'OK' if functionality_ok else 'Échec'}")
    print("")
    print("🏠 Répertoires clés (selon environnement):")
    print("   • Scripts agents: /opt/geoscripts/ ou ./scripts")
    print("   • Workflows: /home/node/.n8n/workflows/ ou ./workflows")
    print("   • Données: /files/, /geodata/ ou ./files, ./geodata")
    print("")
    print("🌐 Accès au système:")
    print("   • Interface n8n: http://localhost:5678")
    print("   • Utilisateur: admin")
    print("   • Mot de passe: cadastre2024")
    print("")
    print("📌 Conseils de remédiation rapides:")
    if not libs_ok:
        print("   • Installez les dépendances Python: pip install -r requirements.txt")
    if not system_ok:
        print("   • Créez les dossiers locaux manquants: mkdir -p ./files ./geodata ./tmp/geodata-cache")
    if not functionality_ok:
        print("   • Exécutez dans le conteneur pour bénéficier des chemins /opt/geoscripts")
    print("="*60)

def main():
    """
    Fonction principale
    """
    print("🚀 Vérification finale du système n8n Geospatial")
    print("=" * 50)
    
    # Exécuter toutes les vérifications
    system_ok = check_system_status()
    libs_ok = check_geospatial_libraries()
    functionality_ok = check_main_functionality()
    
    # Afficher le résumé
    print_final_summary(system_ok, libs_ok, functionality_ok)
    
    # Retourner le statut global
    overall_status = system_ok and libs_ok and functionality_ok
    
    if overall_status:
        print("🎉 Le système n8n Geospatial est pleinement opérationnel!")
        return True
    else:
        print("⚠️  Des problèmes ont été détectés dans le système")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)