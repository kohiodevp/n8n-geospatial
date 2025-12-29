#!/usr/bin/env python3
"""
Point d'entrée principal pour le système n8n Geospatial Workflow Runner
===============================================================

Ce script fournit une interface unifiée pour démarrer et gérer
l'ensemble du système d'agents géospatiaux IA avec n8n.
"""

import sys
import os
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Configuration de la journalisation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ajouter le chemin des scripts pour l'import
sys.path.append('/opt/geoscripts')

from cadastral_agent import CadastralAgent
from domain_agent import DomainAgent
from urbanism_agent import UrbanismAgent
from environmental_agent import EnvironmentalAgent
from workflow_manager import WorkflowManager, GeospatialWorkflowOrchestrator


def initialize_agents():
    """
    Initialiser tous les agents géospatiaux
    
    Returns:
        Dict des agents initialisés
    """
    logger.info("Initialisation des agents géospatiaux...")
    
    agents = {
        'cadastral': CadastralAgent(target_crs="EPSG:2154"),
        'domainal': DomainAgent(),
        'urbanism': UrbanismAgent(),
        'environmental': EnvironmentalAgent()
    }
    
    logger.info(f"{len(agents)} agents géospatiaux initialisés avec succès")
    return agents


def run_system_health_check():
    """
    Exécuter une vérification de santé du système
    """
    logger.info("Exécution de la vérification de santé du système...")
    
    health_status = {
        'timestamp': datetime.now().isoformat(),
        'agents_initialized': True,
        'workflow_manager_available': True,
        'database_connection': True,  # Simulé pour cet exemple
        'geospatial_libraries_available': True,
        'overall_status': 'healthy'
    }
    
    # Vérifier la disponibilité des bibliothèques géospatiales
    try:
        import geopandas as gpd
        import shapely
        import numpy as np
        import pandas as pd
    except ImportError as e:
        logger.error(f"Bibliothèque géospatiale manquante: {e}")
        health_status['geospatial_libraries_available'] = False
        health_status['overall_status'] = 'degraded'
    
    # Vérifier la connexion à la base de données (simulé)
    try:
        # Dans une implémentation réelle, on testerait la connexion PostGIS
        pass
    except Exception as e:
        logger.error(f"Problème de connexion à la base de données: {e}")
        health_status['database_connection'] = False
        health_status['overall_status'] = 'degraded'
    
    logger.info(f"Vérification de santé terminée: {health_status['overall_status']}")
    return health_status


def start_geospatial_workflow_system():
    """
    Démarrer le système complet de workflows géospatiaux
    """
    logger.info("Démarrage du système n8n Geospatial Workflow Runner...")
    
    # Initialiser les agents
    agents = initialize_agents()
    
    # Initialiser le gestionnaire de workflows
    workflow_manager = WorkflowManager(n8n_url="http://localhost:5678")
    orchestrator = GeospatialWorkflowOrchestrator(workflow_manager)
    
    # Exécuter la vérification de santé
    health_status = run_system_health_check()
    
    # Créer des chaînes de workflows de démonstration
    logger.info("Création des chaînes de workflows de démonstration...")
    
    try:
        cadastral_chain_id = orchestrator.create_cadastral_analysis_chain()
        environmental_chain_id = orchestrator.create_environmental_monitoring_chain()
        
        logger.info(f"Chaînes de workflows créées: {cadastral_chain_id}, {environmental_chain_id}")
    except Exception as e:
        logger.error(f"Erreur lors de la création des chaînes de workflows: {e}")
    
    # Afficher le résumé du système
    system_summary = {
        'startup_time': datetime.now().isoformat(),
        'initialized_agents': list(agents.keys()),
        'workflow_templates_loaded': len(workflow_manager.workflow_templates),
        'health_status': health_status['overall_status'],
        'system_ready': True
    }
    
    logger.info("Système n8n Geospatial Workflow Runner démarré avec succès!")
    logger.info(f"Agents disponibles: {', '.join(agents.keys())}")
    logger.info(f"Modèles de workflows: {len(workflow_manager.workflow_templates)}")
    
    return {
        'agents': agents,
        'workflow_manager': workflow_manager,
        'orchestrator': orchestrator,
        'system_summary': system_summary
    }


def run_demonstration():
    """
    Exécuter une démonstration complète du système
    """
    logger.info("Démarrage de la démonstration complète...")
    
    # Démarrer le système
    system_components = start_geospatial_workflow_system()
    
    agents = system_components['agents']
    workflow_manager = system_components['workflow_manager']
    orchestrator = system_components['orchestrator']
    
    print("\n" + "="*60)
    print("DÉMONSTRATION COMPLETE DU SYSTEME GEOSPATIAL IA")
    print("="*60)
    
    print("\n1. AGENTS INITIALISÉS:")
    for name, agent in agents.items():
        print(f"   • {name.title()} Agent: OK")
    
    print(f"\n2. WORKFLOW MANAGER:")
    print(f"   • Modèles chargés: {len(workflow_manager.workflow_templates)}")
    print(f"   • Workflows actifs: {len(workflow_manager.list_active_workflows())}")
    
    print(f"\n3. ORCHESTRATOR:")
    print(f"   • Chaînes définies: {len(orchestrator.workflow_chains)}")
    
    print(f"\n4. STATISTIQUES:")
    stats = workflow_manager.get_workflow_statistics()
    for key, value in stats.items():
        print(f"   • {key}: {value}")
    
    print(f"\n5. EXEMPLES D'UTILISATION:")
    
    # Exemple simple d'utilisation de l'agent cadastral
    try:
        from shapely.geometry import Polygon
        import geopandas as gpd
        
        sample_parcel = gpd.GeoDataFrame([{
            'id': 'DEMO001',
            'geometry': Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            'area': 100,
            'perimeter': 40,
            'owner_id': 'DEMO_OWNER',
            'zone_type': 'urban'
        }], crs="EPSG:2154")
        
        agents['cadastral'].load_parcels(sample_parcel)
        report = agents['cadastral'].generate_cadastral_report()
        
        print(f"   • Agent cadastral: Analyse de {report['summary']['total_parcels']} parcelle(s)")
        print(f"   • Valeur prédite: {agents['cadastral'].predict_parcel_values()[0]['predicted_value']:.2f}€")
    except Exception as e:
        print(f"   • Erreur lors de l'exemple cadastral: {e}")
    
    # Exemple d'utilisation de l'agent environnemental
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        
        sample_env = gpd.GeoDataFrame([{
            'id': 'ENV_DEMO001',
            'geometry': Point(2.3522, 48.8566),
            'air_quality': 0.7,
            'water_quality': 0.8,
            'soil_quality': 0.6,
            'biodiversity_index': 0.75,
            'quality_score': 0.71
        }], crs="EPSG:4326")
        
        agents['environmental'].load_environmental_data(sample_env)
        env_report = agents['environmental'].assess_environmental_quality()
        
        print(f"   • Agent environnemental: Qualité moyenne {env_report['overall_quality_index']:.2f}")
    except Exception as e:
        print(f"   • Erreur lors de l'exemple environnemental: {e}")
    
    print(f"\n6. INTEGRATION AVEC N8N:")
    print(f"   • URL de l'API: {workflow_manager.n8n_url}")
    print(f"   • Type de workflows supportés: {len([e for e in dir(WorkflowType) if not e.startswith('_')])}")
    
    print("\n" + "="*60)
    print("DÉMONSTRATION TERMINÉE AVEC SUCCÈS")
    print("="*60)


def main():
    """
    Fonction principale du système
    """
    parser = argparse.ArgumentParser(
        description="Système n8n Geospatial Workflow Runner"
    )
    parser.add_argument(
        '--mode', 
        choices=['start', 'demo', 'health', 'version'], 
        default='demo',
        help='Mode d\'exécution (démarrage, démonstration, santé, version)'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Fichier de configuration optionnel'
    )
    
    args = parser.parse_args()
    
    print("🚀 n8n Geospatial Workflow Runner")
    print("   Système d'Agents IA Géospatiaux Avancés")
    print()
    
    if args.mode == 'version':
        print("Version: 2.1.1")
        print("Agents disponibles: Cadastral, Domanial, Urbanisme, Environnemental")
        print("Framework: n8n avec extensions géospatiales")
    elif args.mode == 'health':
        health_status = run_system_health_check()
        print(f"Statut du système: {health_status['overall_status']}")
        print(f"Vérifié à: {health_status['timestamp']}")
    elif args.mode == 'start':
        system_components = start_geospatial_workflow_system()
        print("Système démarré en mode service...")
        print("En attente de workflows...")
        # Dans une implémentation réelle, on aurait une boucle de service ici
    elif args.mode == 'demo':
        run_demonstration()


if __name__ == "__main__":
    main()