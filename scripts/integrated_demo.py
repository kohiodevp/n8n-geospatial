#!/usr/bin/env python3
"""
Démonstration Intégrée des Agents Géospatiaux IA
================================================

Ce script démontre l'intégration complète des agents géospatiaux IA
avec le système de gestion des workflows dans l'application n8n.
"""

import sys
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
import argparse

# Ajouter le chemin des scripts pour l'import
sys.path.append('/opt/geoscripts')

from cadastral_agent import CadastralAgent
from domain_agent import DomainAgent
from urbanism_agent import UrbanismAgent
from environmental_agent import EnvironmentalAgent
from workflow_manager import WorkflowManager, WorkflowType, GeospatialWorkflowOrchestrator


def integrated_cadastral_domain_analysis():
    """
    Analyse intégrée cadastrale et domaniale
    """
    print("🌍 Analyse Intégrée Cadastrale et Domaniale")
    print("=" * 45)

    # Initialiser les agents
    cadastral_agent = CadastralAgent(target_crs="EPSG:2154")
    domain_agent = DomainAgent()

    # Créer des données de test
    import geopandas as gpd
    from shapely.geometry import Polygon

    # Données cadastrales
    cadastral_parcels = gpd.GeoDataFrame([
        {
            'id': 'PAR001',
            'geometry': Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            'area': 10000,  # 1 ha
            'perimeter': 400,
            'owner_id': 'STATE001',
            'zone_type': 'urban'
        },
        {
            'id': 'PAR002',
            'geometry': Polygon([(12, 0), (22, 0), (22, 10), (12, 10)]),
            'area': 5000,  # 0.5 ha
            'perimeter': 280,
            'owner_id': 'PRIVATE001',
            'zone_type': 'agricultural'
        }
    ], crs="EPSG:2154")

    # Données domaniales
    domain_properties = gpd.GeoDataFrame([
        {
            'id': 'DOM001',
            'geometry': Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            'category': 'coastal',
            'status': 'leased',
            'area_ha': 1.0,
            'value': 250000,
            'management_entity': 'state'
        }
    ], crs="EPSG:2154")

    domain_concessions = [
        {
            'id': 'CONC001',
            'property_id': 'DOM001',
            'type': 'tourism',
            'status': 'active',
            'start_date': '2020-01-01T00:00:00Z',
            'end_date': '2025-01-01T00:00:00Z',
            'annual_fee': 12000,
            'area_ha': 1.0
        }
    ]

    # Charger les données
    print("📁 Chargement des données...")
    cadastral_agent.load_parcels(cadastral_parcels)
    domain_agent.load_domain_properties(domain_properties)
    domain_agent.load_concessions(domain_concessions)

    # Analyse cadastrale
    print("🔍 Analyse cadastrale...")
    cadastral_report = cadastral_agent.generate_cadastral_report()
    print(f"   → {cadastral_report['summary']['total_parcels']} parcelles analysées")
    print(f"   → {len(cadastral_report['recommendations'])} recommandations")

    # Analyse domaniale
    print("🏛️  Analyse domaniale...")
    domain_report = domain_agent.generate_domain_report()
    print(f"   → {domain_report['total_properties']} propriétés analysées")
    print(f"   → {len(domain_report['recommendations'])} recommandations")

    # Intégration des résultats
    print("🔗 Intégration des analyses...")
    integrated_insights = {
        'cadastral_summary': cadastral_report['summary'],
        'domain_summary': domain_report['concessions_analysis'],
        'cross_analysis': {
            'state_owned_parcels': len([p for _, p in cadastral_parcels.iterrows() if p['owner_id'].startswith('STATE')]),
            'leased_domain_properties': len([c for c in domain_concessions if c['status'] == 'active'])
        }
    }

    print(f"   → {integrated_insights['cross_analysis']['state_owned_parcels']} parcelles domaniales")
    print(f"   → {integrated_insights['cross_analysis']['leased_domain_properties']} propriétés en concession")

    return integrated_insights


def integrated_urbanism_environmental_assessment():
    """
    Évaluation intégrée urbanisme-environnement
    """
    print("\n🌍 Évaluation Intégrée Urbanisme-Environnement")
    print("=" * 50)

    # Initialiser les agents
    urbanism_agent = UrbanismAgent()
    environmental_agent = EnvironmentalAgent()

    # Créer des données de test
    import geopandas as gpd
    from shapely.geometry import Polygon, Point, LineString

    # Données d'urbanisme
    planning_zones = gpd.GeoDataFrame([
        {
            'id': 'ZONE001',
            'geometry': Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            'zone_type': 'residential',
            'density_limit': 200,
            'usage_type': 'housing',
            'current_density': 180
        },
        {
            'id': 'ZONE002',
            'geometry': Polygon([(12, 0), (22, 0), (22, 10), (12, 10)]),
            'zone_type': 'industrial',
            'density_limit': 100,
            'usage_type': 'manufacturing',
            'current_density': 80
        }
    ], crs="EPSG:2154")

    infrastructure = gpd.GeoDataFrame([
        {
            'id': 'ROAD001',
            'geometry': LineString([(0, 5), (25, 5)]),
            'type': 'road',
            'capacity': 2000,
            'current_flow': 1800
        },
        {
            'id': 'PARK001',
            'geometry': Polygon([(15, 15), (20, 15), (20, 20), (15, 20)]),
            'type': 'green_space',
            'area_ha': 2.5,
            'ecosystem_service': 'air_purification'
        }
    ], crs="EPSG:2154")

    # Données environnementales
    environmental_data = gpd.GeoDataFrame([
        {
            'id': 'ENV001',
            'geometry': Point(5, 5),
            'air_quality': 0.7,
            'water_quality': 0.8,
            'soil_quality': 0.6,
            'biodiversity_index': 0.75,
            'quality_score': 0.71
        },
        {
            'id': 'ENV002',
            'geometry': Point(17, 5),
            'air_quality': 0.4,
            'water_quality': 0.5,
            'soil_quality': 0.3,
            'biodiversity_index': 0.45,
            'quality_score': 0.42
        }
    ], crs="EPSG:2154")

    biodiversity_data = [
        {
            'id': 'SITE001',
            'species_richness': 120,
            'endemic_species': 15,
            'threatened_species': 8,
            'conservation_status': 'protected',
            'ecosystem_type': 'urban_green_space',
            'area_ha': 2.5,
            'human_pressure_index': 0.3
        }
    ]

    # Charger les données
    print("📁 Chargement des données...")
    urbanism_agent.load_planning_zones(planning_zones)
    urbanism_agent.load_infrastructure(infrastructure)
    environmental_agent.load_environmental_data(environmental_data)
    environmental_agent.load_biodiversity_data(biodiversity_data)

    # Analyse d'urbanisme
    print("🏙️  Analyse d'urbanisme...")
    urbanism_report = urbanism_agent.generate_urbanism_report()
    print(f"   → {urbanism_report['analysis_summary']['total_zones']} zones analysées")
    print(f"   → {len(urbanism_report['recommendations'])} recommandations")

    # Analyse environnementale
    print("🌿 Analyse environnementale...")
    environmental_report = environmental_agent.generate_environmental_report()
    print(f"   → {environmental_report['analysis_summary']['total_biodiversity_sites']} sites analysés")
    print(f"   → {len(environmental_report['recommendations'])} recommandations")

    # Intégration des résultats
    print("🔗 Intégration des analyses...")
    integrated_assessment = {
        'urbanism_summary': urbanism_report['analysis_summary'],
        'environmental_summary': {
            'total_monitoring_stations': environmental_report['analysis_summary']['total_monitoring_stations'],
            'environmental_quality_index': environmental_report['environmental_quality']['overall_quality_index']
        },
        'integrated_insights': {
            'green_space_ratio': len(infrastructure[infrastructure['type'] == 'green_space']) / len(planning_zones),
            'environmental_urban_conflicts': [],
            'sustainable_development_opportunities': []
        }
    }

    print(f"   → Ratio espaces verts: {integrated_assessment['integrated_insights']['green_space_ratio']:.2f}")
    print(f"   → Index qualité environnementale: {integrated_assessment['environmental_summary']['environmental_quality_index']:.2f}")

    return integrated_assessment


def workflow_orchestration_example():
    """
    Exemple d'orchestration de workflows
    """
    print("\n⚙️  Orchestration de Workflows")
    print("=" * 30)

    # Initialiser le gestionnaire et l'orchestrateur
    workflow_manager = WorkflowManager(n8n_url="http://localhost:5678")
    orchestrator = GeospatialWorkflowOrchestrator(workflow_manager)

    print("📁 Chargement des modèles de workflows...")
    print(f"   → {len(workflow_manager.workflow_templates)} modèles disponibles")

    # Créer une chaîne d'analyse cadastrale
    print("🏗️  Création d'une chaîne de workflows...")
    chain_id = orchestrator.create_cadastral_analysis_chain()
    print(f"   → Chaîne créée: {chain_id}")

    # Créer une chaîne de surveillance environnementale
    env_chain_id = orchestrator.create_environmental_monitoring_chain()
    print(f"   → Chaîne environnementale: {env_chain_id}")

    # Simuler l'exécution d'une chaîne
    print("▶️  Simulation d'exécution de chaîne...")
    initial_params = {
        "area_id": "TEST001",
        "analysis_level": "comprehensive",
        "output_format": "detailed_report"
    }

    # Pour cet exemple, nous n'exécutons pas réellement les workflows
    # car cela nécessiterait une instance n8n active
    print("   → Simulation d'exécution terminée (sans exécution réelle)")
    print("   → Dans un environnement réel, les workflows seraient exécutés via l'API n8n")

    # Afficher les statistiques
    stats = workflow_manager.get_workflow_statistics()
    print(f"\n📊 Statistiques des workflows:")
    print(f"   → Historique: {stats['total_history']} workflows")
    print(f"   → Taux de succès: {stats['success_rate']:.1f}%")

    return {
        'chain_ids': [chain_id, env_chain_id],
        'workflow_stats': stats
    }


def generate_comprehensive_report():
    """
    Générer un rapport complet intégrant toutes les analyses
    """
    print("\n📋 Génération d'un Rapport Complet")
    print("=" * 35)

    # Exécuter les différentes analyses
    cadastral_domain_results = integrated_cadastral_domain_analysis()
    urbanism_env_results = integrated_urbanism_environmental_assessment()
    workflow_results = workflow_orchestration_example()

    # Créer un rapport intégré
    comprehensive_report = {
        'report_date': datetime.now().isoformat(),
        'report_type': 'comprehensive_geospatial_analysis',
        'executive_summary': {
            'total_analyses_performed': 3,
            'integration_points': 5,
            'key_findings': [
                f"{cadastral_domain_results['cross_analysis']['state_owned_parcels']} parcelles domaniales identifiées",
                f"Qualité environnementale moyenne: {urbanism_env_results['environmental_summary']['environmental_quality_index']:.2f}",
                f"Taux de succès des workflows: {workflow_results['workflow_stats']['success_rate']:.1f}%"
            ]
        },
        'detailed_analyses': {
            'cadastral_domain': cadastral_domain_results,
            'urbanism_environmental': urbanism_env_results
        },
        'workflow_implementation': workflow_results,
        'recommendations': [
            "Renforcer l'intégration entre les systèmes cadastraux et domaniaux",
            "Améliorer la surveillance environnementale dans les zones à forte densité",
            "Optimiser les workflows pour une exécution plus efficace"
        ],
        'next_steps': [
            "Développer des indicateurs de performance spécifiques",
            "Mettre en place une surveillance continue",
            "Automatiser davantage les processus d'analyse"
        ]
    }

    # Sauvegarder le rapport
    report_filename = f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_filename, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_report, f, ensure_ascii=False, indent=2)

    print(f"✅ Rapport complet généré: {report_filename}")
    print(f"   → {len(comprehensive_report['recommendations'])} recommandations")
    print(f"   → {len(comprehensive_report['next_steps'])} étapes suivantes")

    return comprehensive_report


def main():
    """
    Fonction principale pour exécuter la démonstration intégrée
    """
    print("Démonstration Intégrée des Agents Géospatiaux IA")
    print("=" * 55)
    print(f"Démarré à: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        # Générer le rapport complet
        report = generate_comprehensive_report()

        print(f"\n🎉 Démonstration intégrée terminée avec succès!")
        print(f"📦 {report['executive_summary']['total_analyses_performed']} analyses complètes")
        print(f"🔗 {len(report['executive_summary']['key_findings'])} points d'intégration clés")
        print(f"💡 {len(report['recommendations'])} recommandations stratégiques")

    except Exception as e:
        print(f"\n❌ Erreur lors de la démonstration: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()