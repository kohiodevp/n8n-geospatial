#!/usr/bin/env python3
"""
Scripts d'exemple pour les agents géospatiaux IA
=================================================

Ce module fournit des scripts d'exemple pour démontrer l'utilisation
des différents agents géospatiaux : cadastral, domanial, urbanisme et environnemental.
"""

import sys
import os
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List

# Ajouter le chemin des scripts pour l'import
sys.path.append('/opt/geoscripts')

from cadastral_agent import CadastralAgent
from domain_agent import DomainAgent
from urbanism_agent import UrbanismAgent
from environmental_agent import EnvironmentalAgent
from workflow_manager import WorkflowManager, WorkflowType


def example_cadastral_agent():
    """
    Exemple d'utilisation de l'agent cadastral
    """
    print("🧪 Exemple - Agent Cadastral")
    print("-" * 30)

    # Créer l'agent
    agent = CadastralAgent(target_crs="EPSG:2154")

    # Créer des données de test
    from shapely.geometry import Polygon
    import geopandas as gpd

    sample_parcels = gpd.GeoDataFrame([
        {
            'id': 'PAR001',
            'geometry': Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            'area': 100,
            'perimeter': 40,
            'owner_id': 'OWNER001',
            'zone_type': 'urban'
        },
        {
            'id': 'PAR002',
            'geometry': Polygon([(12, 0), (22, 0), (22, 10), (12, 10)]),
            'area': 100,
            'perimeter': 40,
            'owner_id': 'OWNER002',
            'zone_type': 'agricultural'
        },
        {
            'id': 'PAR003',
            'geometry': Polygon([(5, 12), (15, 12), (15, 22), (5, 22)]),
            'area': 100,
            'perimeter': 40,
            'owner_id': 'OWNER001',
            'zone_type': 'urban'
        }
    ], crs="EPSG:2154")

    sample_properties = gpd.GeoDataFrame([
        {
            'parcel_id': 'PAR001',
            'declared_area': 95,
            'building_area': 50,
            'usage': 'residential'
        },
        {
            'parcel_id': 'PAR002',
            'declared_area': 120,
            'building_area': 10,
            'usage': 'agricultural'
        }
    ], crs="EPSG:2154")

    # Charger les données
    print("📁 Chargement des données...")
    agent.load_parcels(sample_parcels)
    agent.load_properties(sample_properties)

    # Validation
    print("🔍 Validation des parcelles...")
    validation_results = agent.validate_parcels()
    print(f"   → {len(validation_results)} parcelle(s) avec problèmes")

    # Anomalies
    print("🚨 Détection d'anomalies...")
    anomalies = agent.detect_cadastral_anomalies()
    print(f"   → {len(anomalies)} anomalie(s) détectée(s)")

    # Prédictions de valeur
    print("💰 Prédictions de valeur...")
    predictions = agent.predict_parcel_values()
    for pred in predictions[:2]:  # Afficher les 2 premières
        print(f"   • {pred['parcel_id']}: {pred['predicted_value']:,.2f}€")

    # Consolidation
    print("🔗 Consolidation des parcelles...")
    from cadastral_agent import ConsolidationCriteria
    criteria = ConsolidationCriteria(
        owner_similarity=True,
        zone_type_similarity=True,
        max_distance=50
    )
    consolidated = agent.consolidate_parcels(criteria)
    print(f"   → {len(sample_parcels)} parcelles → {len(consolidated)} après consolidation")

    # Rapport
    print("📊 Génération du rapport...")
    report = agent.generate_cadastral_report()
    print(f"   • Total parcelles: {report['summary']['total_parcels']}")
    print(f"   • Recommandations: {len(report['recommendations'])}")

    print("\n✅ Exemple d'agent cadastral terminé\n")


def example_domain_agent():
    """
    Exemple d'utilisation de l'agent domanial
    """
    print("🧪 Exemple - Agent Domanial")
    print("-" * 30)

    # Créer l'agent
    agent = DomainAgent()

    # Créer des données de test
    import geopandas as gpd
    from shapely.geometry import Polygon

    sample_properties = gpd.GeoDataFrame([
        {
            'id': 'DOM001',
            'geometry': Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            'category': 'coastal',
            'status': 'available',
            'area_ha': 50.0,
            'value': 250000,
            'management_entity': 'state'
        },
        {
            'id': 'DOM002',
            'geometry': Polygon([(12, 0), (22, 0), (22, 10), (12, 10)]),
            'category': 'urban_perimeter',
            'status': 'leased',
            'area_ha': 25.0,
            'value': 150000,
            'management_entity': 'municipality'
        }
    ], crs="EPSG:2154")

    sample_concessions = [
        {
            'id': 'CONC001',
            'property_id': 'DOM001',
            'type': 'tourism',
            'status': 'active',
            'start_date': '2020-01-01T00:00:00Z',
            'end_date': '2025-01-01T00:00:00Z',
            'annual_fee': 12000,
            'area_ha': 50.0
        },
        {
            'id': 'CONC002',
            'property_id': 'DOM002',
            'type': 'agricultural',
            'status': 'active',
            'start_date': '2018-01-01T00:00:00Z',
            'end_date': '2024-01-01T00:00:00Z',
            'annual_fee': 5000,
            'area_ha': 25.0
        }
    ]

    # Charger les données
    print("📁 Chargement des données...")
    agent.load_domain_properties(sample_properties)
    agent.load_concessions(sample_concessions)

    # Analyse des concessions
    print("📊 Analyse des concessions...")
    concessions_analysis = agent.analyze_concessions()
    print(f"   → {concessions_analysis['total_concessions']} concessions")
    print(f"   → Revenu potentiel: {concessions_analysis['revenue_potential']:,.2f}€")

    # Zones stratégiques
    print("🎯 Identification des zones stratégiques...")
    strategic_zones = agent.identify_strategic_zones()
    print(f"   → {len(strategic_zones)} zone(s) stratégique(s) identifiée(s)")

    # Opportunités d'optimisation
    print("🔍 Détection des opportunités d'optimisation...")
    opportunities = agent.detect_optimization_opportunities()
    print(f"   → {len(opportunities)} opportunité(s) détectée(s)")

    # Suggestions de concession
    print("💡 Suggestions de concession...")
    suggestions = agent.suggest_concession_optimization()
    print(f"   → {len(suggestions)} suggestion(s) générée(s)")

    # Rapport complet
    print("📋 Génération du rapport...")
    report = agent.generate_domain_report()
    print(f"   • Total propriétés: {report['total_properties']}")
    print(f"   • Recommandations: {len(report['recommendations'])}")

    print("\n✅ Exemple d'agent domanial terminé\n")


def example_urbanism_agent():
    """
    Exemple d'utilisation de l'agent d'urbanisme
    """
    print("🧪 Exemple - Agent d'Urbanisme")
    print("-" * 30)

    # Créer l'agent
    agent = UrbanismAgent()

    # Créer des données de test
    import geopandas as gpd
    from shapely.geometry import Polygon, Point, LineString

    sample_zones = gpd.GeoDataFrame([
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
            'zone_type': 'commercial',
            'density_limit': 500,
            'usage_type': 'retail',
            'current_density': 300
        }
    ], crs="EPSG:2154")

    sample_infrastructure = gpd.GeoDataFrame([
        {
            'id': 'ROAD001',
            'geometry': LineString([(0, 5), (25, 5)]),
            'type': 'road',
            'capacity': 2000,
            'current_flow': 1800
        },
        {
            'id': 'SCHOOL001',
            'geometry': Point(5, 5),
            'type': 'school',
            'capacity': 500,
            'current_enrollment': 400
        }
    ], crs="EPSG:2154")

    sample_population = [
        {
            'year': 2020,
            'population': 100000,
            'area_id': 'ZONE001'
        },
        {
            'year': 2021,
            'population': 105000,
            'area_id': 'ZONE001'
        }
    ]

    # Charger les données
    print("📁 Chargement des données...")
    agent.load_planning_zones(sample_zones)
    agent.load_infrastructure(sample_infrastructure)
    agent.load_population_data(sample_population)

    # Analyse de densité
    print("📊 Analyse de densité urbaine...")
    density_analysis = agent.analyze_urban_density()
    print(f"   → Densité globale: {density_analysis['overall_density']:.2f} hab/ha")

    # Opportunités de développement
    print("🏗️  Identification des opportunités...")
    opportunities = agent.identify_development_opportunities()
    print(f"   → {len(opportunities)} opportunité(s) identifiée(s)")

    # Évaluation des infrastructures
    print("🛣️  Évaluation des infrastructures...")
    infra_assessment = agent.assess_infrastructure_capacity()
    print(f"   → Analyse des capacités terminée")

    # Prédiction de croissance
    print("📈 Prédiction de croissance...")
    growth_predictions = agent.predict_urban_growth()
    print(f"   → Taux de croissance: {growth_predictions['overall_growth_rate']:.4f}")

    # Analyse d'accessibilité
    print("🗺️  Analyse d'accessibilité...")
    accessibility = agent.analyze_accessibility()
    print(f"   → {len(accessibility['accessibility_index'])} zones analysées")

    # Rapport complet
    print("📋 Génération du rapport...")
    report = agent.generate_urbanism_report()
    print(f"   • Total zones: {report['analysis_summary']['total_zones']}")
    print(f"   • Recommandations: {len(report['recommendations'])}")

    print("\n✅ Exemple d'agent d'urbanisme terminé\n")


def example_environmental_agent():
    """
    Exemple d'utilisation de l'agent environnemental
    """
    print("🧪 Exemple - Agent Environnemental")
    print("-" * 30)

    # Créer l'agent
    agent = EnvironmentalAgent()

    # Créer des données de test
    import geopandas as gpd
    from shapely.geometry import Point, Polygon

    sample_environmental = gpd.GeoDataFrame([
        {
            'id': 'ENV001',
            'geometry': Point(2.3522, 48.8566),
            'air_quality': 0.7,
            'water_quality': 0.8,
            'soil_quality': 0.6,
            'biodiversity_index': 0.75,
            'quality_score': 0.71
        },
        {
            'id': 'ENV002',
            'geometry': Point(2.3500, 48.8500),
            'air_quality': 0.4,
            'water_quality': 0.5,
            'soil_quality': 0.3,
            'biodiversity_index': 0.45,
            'quality_score': 0.42
        }
    ], crs="EPSG:4326")

    sample_monitoring = gpd.GeoDataFrame([
        {
            'id': 'STATION001',
            'geometry': Point(2.3522, 48.8566),
            'station_type': 'air_quality',
            'last_reading': datetime.now().isoformat(),
            'status': 'active'
        }
    ], crs="EPSG:4326")

    sample_risk_zones = gpd.GeoDataFrame([
        {
            'id': 'RISK001',
            'geometry': Polygon([(2.34, 48.84), (2.36, 48.84), (2.36, 48.86), (2.34, 48.86)]),
            'risk_type': 'pollution',
            'probability': 0.7,
            'impact_level': 'high',
            'severity': 'high'
        }
    ], crs="EPSG:4326")

    sample_biodiversity = [
        {
            'id': 'SITE001',
            'species_richness': 120,
            'endemic_species': 15,
            'threatened_species': 8,
            'conservation_status': 'protected',
            'ecosystem_type': 'forest',
            'area_ha': 150,
            'human_pressure_index': 0.3
        }
    ]

    sample_pollution = gpd.GeoDataFrame([
        {
            'id': 'POLL001',
            'geometry': Point(2.3550, 48.8550),
            'pollution_type': 'industrial',
            'severity': 'high',
            'impact_radius': 2000,
            'emission_level': 0.9
        }
    ], crs="EPSG:4326")

    # Charger les données
    print("📁 Chargement des données...")
    agent.load_environmental_data(sample_environmental)
    agent.load_monitoring_stations(sample_monitoring)
    agent.load_risk_zones(sample_risk_zones)
    agent.load_biodiversity_data(sample_biodiversity)
    agent.load_pollution_sources(sample_pollution)

    # Évaluation de la qualité
    print("📊 Évaluation de la qualité environnementale...")
    quality = agent.assess_environmental_quality()
    print(f"   → Index de qualité global: {quality['overall_quality_index']:.2f}")

    # Détection des risques
    print("🚨 Détection des risques environnementaux...")
    risks = agent.detect_environmental_risks()
    print(f"   → {len(risks)} risque(s) détecté(s)")

    # Analyse de la biodiversité
    print("🌿 Analyse des hotspots de biodiversité...")
    hotspots = agent.analyze_biodiversity_hotspots()
    print(f"   → {len(hotspots)} hotspot(s) identifié(s)")

    # Prédiction des tendances
    print("📈 Prédiction des tendances environnementales...")
    trends = agent.predict_environmental_trends()
    print(f"   → Analyse des tendances terminée")

    # Évaluation des services écosystémiques
    print("🌍 Évaluation des services écosystémiques...")
    services = agent.assess_ecosystem_services()
    print(f"   → Évaluation des services terminée")

    # Priorités de conservation
    print("🎯 Identification des priorités de conservation...")
    priorities = agent.identify_conservation_priorities()
    print(f"   → {len(priorities)} priorité(s) identifiée(s)")

    # Analyse de l'impact de la pollution
    print("☣️  Analyse de l'impact de la pollution...")
    pollution_impact = agent.analyze_pollution_impact()
    print(f"   → {len(pollution_impact['affected_areas'])} zone(s) affectée(s)")

    # Rapport complet
    print("📋 Génération du rapport...")
    report = agent.generate_environmental_report()
    print(f"   • Total sites: {report['analysis_summary']['total_biodiversity_sites']}")
    print(f"   • Recommandations: {len(report['recommendations'])}")

    print("\n✅ Exemple d'agent environnemental terminé\n")


def example_workflow_integration():
    """
    Exemple d'intégration avec le gestionnaire de workflows
    """
    print("🧪 Exemple - Intégration avec le gestionnaire de workflows")
    print("-" * 55)

    # Créer le gestionnaire de workflows
    workflow_manager = WorkflowManager(n8n_url="http://localhost:5678")

    print("📁 Chargement des modèles de workflows...")
    print(f"   → {len(workflow_manager.workflow_templates)} modèles chargés")

    # Créer un workflow d'analyse cadastrale
    print("🏗️  Création d'un workflow d'analyse...")
    workflow_id = workflow_manager.create_workflow_instance(
        "ai_agent_cadastral",
        {
            "parcel_id": "TEST001",
            "analysis_type": "comprehensive",
            "output_format": "detailed_report"
        },
        WorkflowType.CADASTRAL
    )
    print(f"   → Instance créée: {workflow_id}")

    # Afficher les workflows actifs
    active_workflows = workflow_manager.list_active_workflows()
    print(f"   → Workflows actifs: {len(active_workflows)}")

    # Afficher les statistiques
    stats = workflow_manager.get_workflow_statistics()
    print(f"📊 Statistiques:")
    print(f"   → Terminés: {stats['total_completed']}")
    print(f"   → Taux de succès: {stats['success_rate']:.1f}%")

    print("\n✅ Exemple d'intégration avec les workflows terminé\n")


def main():
    """
    Fonction principale pour exécuter tous les exemples
    """
    print("Scripts d'exemple pour les agents géospatiaux IA")
    print("=" * 55)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Exécuter tous les exemples
    example_cadastral_agent()
    example_domain_agent()
    example_urbanism_agent()
    example_environmental_agent()
    example_workflow_integration()

    print("🎉 Tous les exemples ont été exécutés avec succès!")


if __name__ == "__main__":
    main()