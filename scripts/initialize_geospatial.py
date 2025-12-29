#!/usr/bin/env python3
"""
Initialisation du système géospatial
===================================

Ce module fournit les fonctions d'initialisation pour le système
géospatial IA avec n8n.
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List

# Importer les agents géospatiaux
sys.path.append('/opt/geoscripts')

from cadastral_agent import CadastralAgent
from domain_agent import DomainAgent
from urbanism_agent import UrbanismAgent
from environmental_agent import EnvironmentalAgent
from workflow_manager import WorkflowManager, GeospatialWorkflowOrchestrator


def initialize_cadastral_agent() -> CadastralAgent:
    """
    Initialiser l'agent cadastral
    
    Returns:
        Instance d'agent cadastral initialisée
    """
    print("📦 Initialisation de l'agent cadastral...")
    agent = CadastralAgent(target_crs="EPSG:2154")
    print("✅ Agent cadastral initialisé")
    return agent


def initialize_domain_agent() -> DomainAgent:
    """
    Initialiser l'agent domanial
    
    Returns:
        Instance d'agent domanial initialisée
    """
    print("📦 Initialisation de l'agent domanial...")
    agent = DomainAgent()
    print("✅ Agent domanial initialisé")
    return agent


def initialize_urbanism_agent() -> UrbanismAgent:
    """
    Initialiser l'agent d'urbanisme
    
    Returns:
        Instance d'agent urbanisme initialisée
    """
    print("📦 Initialisation de l'agent d'urbanisme...")
    agent = UrbanismAgent()
    print("✅ Agent d'urbanisme initialisé")
    return agent


def initialize_environmental_agent() -> EnvironmentalAgent:
    """
    Initialiser l'agent environnemental
    
    Returns:
        Instance d'agent environnemental initialisée
    """
    print("📦 Initialisation de l'agent environnemental...")
    agent = EnvironmentalAgent()
    print("✅ Agent environnemental initialisé")
    return agent


def initialize_workflow_system() -> tuple:
    """
    Initialiser le système de gestion des workflows
    
    Returns:
        Tuple (workflow_manager, orchestrator)
    """
    print("⚙️  Initialisation du système de workflows...")
    workflow_manager = WorkflowManager(n8n_url="http://localhost:5678")
    orchestrator = GeospatialWorkflowOrchestrator(workflow_manager)
    print("✅ Système de workflows initialisé")
    return workflow_manager, orchestrator


def run_system_health_check() -> Dict[str, Any]:
    """
    Exécuter une vérification de santé du système
    
    Returns:
        Dictionnaire avec les résultats de la vérification
    """
    print("🔍 Vérification de santé du système...")
    
    health_status = {
        'timestamp': datetime.now().isoformat(),
        'agents_initialized': True,
        'workflow_system_available': True,
        'database_connection': True,  # À implémenter selon votre configuration
        'geospatial_libraries_available': True,
        'overall_status': 'healthy',
        'checks_performed': []
    }
    
    # Vérifier la disponibilité des bibliothèques géospatiales
    try:
        import geopandas as gpd
        import shapely
        import numpy as np
        import pandas as pd
        health_status['checks_performed'].append('geospatial_libraries')
    except ImportError as e:
        print(f"❌ Problème avec les bibliothèques géospatiales: {e}")
        health_status['geospatial_libraries_available'] = False
        health_status['overall_status'] = 'unhealthy'
    
    # Vérifier la connexion à la base de données (réel)
    try:
        import psycopg2
        import os
        from urllib.parse import urlparse
        db_url = os.getenv('DATABASE_URL')
        if not db_url:
            host = os.getenv('PGHOST') or os.getenv('DB_POSTGRESDB_HOST', 'postgis')
            port = os.getenv('PGPORT') or os.getenv('DB_POSTGRESDB_PORT', '5432')
            user = os.getenv('PGUSER') or os.getenv('DB_POSTGRESDB_USER', 'geo')
            pwd = os.getenv('PGPASSWORD') or os.getenv('DB_POSTGRESDB_PASSWORD', 'geo_password')
            db  = os.getenv('PGDATABASE') or os.getenv('DB_POSTGRESDB_DATABASE', 'cadastre')
            conn = psycopg2.connect(host=host, port=port, user=user, password=pwd, dbname=db)
        else:
            u = urlparse(db_url)
            conn = psycopg2.connect(host=u.hostname, port=u.port or 5432, user=u.username, password=u.password, dbname=u.path.lstrip('/'))
        with conn.cursor() as c:
            c.execute('SELECT version()')
            _ = c.fetchone()
        conn.close()
        health_status['checks_performed'].append('database_connection')
    except Exception as e:
        print(f"❌ Problème avec la connexion à la base de données: {e}")
        health_status['database_connection'] = False
        health_status['overall_status'] = 'unhealthy'
    
    print("✅ Vérification de santé terminée")
    return health_status


def initialize_all_agents() -> Dict[str, Any]:
    """
    Initialiser tous les agents géospatiaux
    
    Returns:
        Dictionnaire contenant toutes les instances d'agents
    """
    print("🚀 Initialisation de tous les agents géospatiaux...")
    
    agents = {}
    
    try:
        agents['cadastral'] = initialize_cadastral_agent()
        agents['domainal'] = initialize_domain_agent()
        agents['urbanism'] = initialize_urbanism_agent()
        agents['environmental'] = initialize_environmental_agent()
        
        print("✅ Tous les agents géospatiaux sont initialisés")
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation des agents: {e}")
        raise
    
    return agents


def initialize_system() -> Dict[str, Any]:
    """
    Initialiser le système géospatial complet
    
    Returns:
        Dictionnaire avec toutes les instances système
    """
    print("🌍 Initialisation du système géospatial IA")
    print("=" * 50)
    
    start_time = time.time()
    
    # Initialiser les agents
    agents = initialize_all_agents()
    
    # Initialiser le système de workflows
    workflow_manager, orchestrator = initialize_workflow_system()
    
    # Exécuter la vérification de santé
    health_status = run_system_health_check()
    
    # Résumé de l'initialisation
    initialization_summary = {
        'agents': agents,
        'workflow_manager': workflow_manager,
        'orchestrator': orchestrator,
        'health_status': health_status,
        'initialization_time': time.time() - start_time,
        'status': 'success'
    }
    
    print(f"\n⏱️  Temps d'initialisation: {initialization_summary['initialization_time']:.2f} secondes")
    print(f"📊 Agents initialisés: {len(agents)}")
    print(f"✅ Système prêt à traiter des workflows géospatiaux")
    
    return initialization_summary


def main():
    """
    Fonction principale d'initialisation
    """
    print("Initialisation du système géospatial IA avec n8n")
    print("=" * 55)
    
    try:
        # Initialiser le système
        system = initialize_system()
        
        print(f"\n🎉 Système géospatial initialisé avec succès!")
        print(f"   • Statut: {system['status']}")
        print(f"   • Agents disponibles: {list(system['agents'].keys())}")
        print(f"   • Santé du système: {system['health_status']['overall_status']}")
        
    except Exception as e:
        print(f"\n❌ Erreur lors de l'initialisation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()