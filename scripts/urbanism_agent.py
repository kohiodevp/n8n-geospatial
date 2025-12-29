#!/usr/bin/env python3
"""
Agent d'Urbanisme pour les Workflows IA Géospatiaux
===================================================

Cet agent est spécialisé dans les traitements liés à l'urbanisme,
l'aménagement du territoire, la planification urbaine et l'analyse
des dynamiques de développement urbain.
"""

import json
import os
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from shapely.geometry import Point, Polygon, LineString, MultiPolygon
from shapely.ops import unary_union
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings


class UrbanismAgent:
    """
    Agent spécialisé dans l'urbanisme et l'aménagement du territoire
    """

    def __init__(self):
        self.planning_zones = gpd.GeoDataFrame()
        self.infrastructure = gpd.GeoDataFrame()
        self.population_data = pd.DataFrame()
        self.development_projects = []
        self.urban_indicators = {}
        self.accessibility_matrix = None

    def load_planning_zones(self, gdf: gpd.GeoDataFrame) -> bool:
        """
        Charger les zones de planification
        """
        try:
            required_cols = ['id', 'geometry', 'zone_type', 'density_limit', 'usage_type']
            if not all(col in gdf.columns for col in required_cols):
                raise ValueError(f"Colonnes requises manquantes: {set(required_cols) - set(gdf.columns)}")

            self.planning_zones = gdf
            print(f"Chargé {len(self.planning_zones)} zones de planification")
            return True
        except Exception as e:
            print(f"Erreur lors du chargement des zones de planification: {e}")
            return False

    def load_infrastructure(self, gdf: gpd.GeoDataFrame) -> bool:
        """
        Charger les données d'infrastructure
        """
        try:
            self.infrastructure = gdf
            print(f"Chargé {len(self.infrastructure)} éléments d'infrastructure")
            return True
        except Exception as e:
            print(f"Erreur lors du chargement des infrastructures: {e}")
            return False

    def load_population_data(self, df: pd.DataFrame) -> bool:
        """
        Charger les données de population
        """
        try:
            self.population_data = df
            print(f"Chargé {len(self.population_data)} enregistrements de population")
            return True
        except Exception as e:
            print(f"Erreur lors du chargement des données de population: {e}")
            return False

    def load_development_projects(self, projects_data: List[Dict]) -> bool:
        """
        Charger les projets d'aménagement
        """
        try:
            self.development_projects = projects_data
            print(f"Chargé {len(self.development_projects)} projets d'aménagement")
            return True
        except Exception as e:
            print(f"Erreur lors du chargement des projets d'aménagement: {e}")
            return False

    def analyze_urban_density(self) -> Dict[str, Any]:
        """
        Analyser la densité urbaine
        """
        if self.planning_zones.empty:
            return {}

        density_analysis = {
            'overall_density': 0,
            'density_by_zone': {},
            'overcrowded_zones': [],
            'underutilized_zones': [],
            'density_trends': {}
        }

        total_area = self.planning_zones.geometry.area.sum()
        if total_area > 0 and not self.population_data.empty and 'population' in self.population_data.columns:
            total_population = self.population_data['population'].sum()
            density_analysis['overall_density'] = (total_population / total_area) * 10000  # par ha

            # Analyse par type de zone
            for zone_type in self.planning_zones['zone_type'].unique():
                zone_subset = self.planning_zones[self.planning_zones['zone_type'] == zone_type]
                if not zone_subset.empty:
                    zone_area = zone_subset.geometry.area.sum()
                    
                    # Calculer la population pour cette zone (simplifié)
                    zone_population = int(total_population * (zone_area / total_area)) if total_area > 0 else 0
                    zone_density = (zone_population / zone_area * 10000) if zone_area > 0 else 0
                    
                    density_analysis['density_by_zone'][zone_type] = {
                        'population': zone_population,
                        'area_ha': zone_area / 10000,
                        'density': zone_density
                    }

                    # Identifier les zones surpeuplées ou sous-utilisées
                    density_limit = zone_subset['density_limit'].mean() if 'density_limit' in zone_subset.columns else 500
                    if zone_density > density_limit * 1.2:  # 20% au-dessus de la limite
                        density_analysis['overcrowded_zones'].append({
                            'zone_id': zone_subset.iloc[0]['id'],
                            'zone_type': zone_type,
                            'current_density': zone_density,
                            'density_limit': density_limit
                        })
                    elif zone_density < density_limit * 0.3:  # Moins de 30% de la limite
                        density_analysis['underutilized_zones'].append({
                            'zone_id': zone_subset.iloc[0]['id'],
                            'zone_type': zone_type,
                            'current_density': zone_density,
                            'density_limit': density_limit
                        })

        return density_analysis

    def identify_development_opportunities(self) -> List[Dict[str, Any]]:
        """
        Identifier les opportunités de développement
        """
        opportunities = []

        if self.planning_zones.empty:
            return opportunities

        for idx, zone in self.planning_zones.iterrows():
            # Calculer le potentiel de développement
            potential = self._calculate_development_potential(zone)

            if potential > 0.7:  # Seulement les zones avec haut potentiel
                opportunities.append({
                    'zone_id': zone.get('id'),
                    'geometry': zone.geometry,
                    'development_potential': potential,
                    'recommended_type': self._recommend_development_type(zone, potential),
                    'accessibility_score': self._calculate_accessibility_score(zone),
                    'infrastructure_readiness': self._assess_infrastructure_readiness(zone)
                })

        return opportunities

    def _calculate_development_potential(self, zone: pd.Series) -> float:
        """
        Calculer le potentiel de développement pour une zone
        """
        # Facteurs: densité actuelle, proximité des infrastructures, type de zone
        current_density = zone.get('current_density', 0) / zone.get('density_limit', 1000) if 'density_limit' in zone and zone.get('density_limit', 1000) > 0 else 0
        proximity_score = self._calculate_infrastructure_proximity(zone)
        zone_type_factor = self._get_zone_type_factor(zone.get('zone_type', ''))

        # Score combiné (normalisé entre 0 et 1)
        potential = (1 - min(current_density, 1.0)) * 0.4 + proximity_score * 0.4 + zone_type_factor * 0.2
        return min(potential, 1.0)

    def _calculate_infrastructure_proximity(self, zone: pd.Series) -> float:
        """
        Calculer la proximité aux infrastructures
        """
        if self.infrastructure.empty:
            return 0.5  # Score moyen par défaut

        # Calculer la distance moyenne aux infrastructures principales
        zone_centroid = zone.geometry.centroid
        min_distance = float('inf')

        for infra in self.infrastructure.itertuples():
            dist = zone_centroid.distance(infra.geometry)
            if dist < min_distance:
                min_distance = dist

        # Convertir la distance en score (plus c'est proche, plus le score est élevé)
        max_distance = 5000  # 5km
        proximity_score = max(0, (max_distance - min(min_distance, max_distance)) / max_distance)
        return proximity_score

    def _get_zone_type_factor(self, zone_type: str) -> float:
        """
        Obtenir le facteur pour un type de zone
        """
        factors = {
            'residential': 0.9,
            'mixed_use': 0.8,
            'commercial': 0.7,
            'industrial': 0.5,
            'recreational': 0.6,
            'agricultural': 0.3,
            'protected': 0.1
        }
        return factors.get(zone_type, 0.4)

    def _recommend_development_type(self, zone: pd.Series, potential: float) -> str:
        """
        Recommander le type de développement optimal
        """
        if potential > 0.8:
            return 'high_density_residential' if zone.get('zone_type') == 'residential' else 'mixed_use_development'
        elif potential > 0.6:
            return 'medium_density_residential' if zone.get('zone_type') == 'residential' else 'commercial_expansion'
        elif potential > 0.3:
            return 'planned_development' if zone.get('zone_type') in ['residential', 'commercial'] else 'infrastructure_development'
        else:
            return 'conservation' if potential < 0.3 else 'planned_development'

    def _calculate_accessibility_score(self, zone: pd.Series) -> float:
        """
        Calculer le score d'accessibilité
        """
        # Simulation basée sur la proximité des transports en commun et des services
        proximity_score = self._calculate_infrastructure_proximity(zone)
        return min(proximity_score + np.random.uniform(-0.1, 0.1), 1.0)

    def _assess_infrastructure_readiness(self, zone: pd.Series) -> Dict[str, float]:
        """
        Évaluer la préparation des infrastructures pour le développement
        """
        readiness = {
            'transport': 0.5,
            'utilities': 0.5,
            'services': 0.5,
            'overall': 0.5
        }

        # Calculer la disponibilité des infrastructures autour de la zone
        zone_buffer = zone.geometry.buffer(1000)  # 1km buffer
        nearby_infra = self.infrastructure[self.infrastructure.geometry.intersects(zone_buffer)]

        if not nearby_infra.empty:
            transport_count = len(nearby_infra[nearby_infra['type'].isin(['road', 'public_transport', 'railway'])])
            utility_count = len(nearby_infra[nearby_infra['type'].isin(['water', 'electricity', 'gas'])])
            service_count = len(nearby_infra[nearby_infra['type'].isin(['school', 'hospital', 'commercial'])])

            readiness['transport'] = min(transport_count / 5, 1.0)  # 5 infrastructures max = 100%
            readiness['utilities'] = min(utility_count / 3, 1.0)   # 3 infrastructures max = 100%
            readiness['services'] = min(service_count / 4, 1.0)   # 4 infrastructures max = 100%
            readiness['overall'] = (readiness['transport'] + readiness['utilities'] + readiness['services']) / 3

        return readiness

    def assess_infrastructure_capacity(self) -> Dict[str, Any]:
        """
        Évaluer la capacité des infrastructures
        """
        if self.infrastructure.empty:
            return {}

        capacity_assessment = {
            'transport_capacity': {},
            'utility_capacity': {},
            'service_capacity': {},
            'overload_risks': [],
            'development_impact': {}
        }

        # Évaluer les infrastructures par type
        for infra_type in self.infrastructure.get('type', pd.Series()).unique():
            infra_subset = self.infrastructure[self.infrastructure['type'] == infra_type]
            if not infra_subset.empty:
                usage_rate = self._calculate_usage_rate(infra_subset)

                capacity_info = {
                    'count': len(infra_subset),
                    'average_usage_rate': usage_rate,
                    'capacity_status': 'adequate' if usage_rate < 0.8 else 'overloaded'
                }

                if infra_type in ['road', 'public_transport', 'railway']:
                    capacity_assessment['transport_capacity'][infra_type] = capacity_info
                elif infra_type in ['water', 'electricity', 'gas']:
                    capacity_assessment['utility_capacity'][infra_type] = capacity_info
                else:
                    capacity_assessment['service_capacity'][infra_type] = capacity_info

                if usage_rate > 0.9:
                    capacity_assessment['overload_risks'].append({
                        'type': infra_type,
                        'usage_rate': usage_rate,
                        'location': 'multiple'
                    })

        return capacity_assessment

    def _calculate_usage_rate(self, infra_subset: gpd.GeoDataFrame) -> float:
        """
        Calculer le taux d'utilisation des infrastructures (simulation)
        """
        # Simulation basée sur la population desservie et la capacité
        return np.random.uniform(0.4, 0.95)

    def predict_urban_growth(self) -> Dict[str, Any]:
        """
        Prédire la croissance urbaine à partir de modèles ML
        """
        if self.population_data.empty:
            return {}

        growth_predictions = {
            'overall_growth_rate': 0.0,
            'growth_by_zone': {},
            'capacity_requirements': {},
            'development_timeline': {}
        }

        # Simulation de prédiction de croissance
        if len(self.population_data) > 1 and 'population' in self.population_data.columns and 'year' in self.population_data.columns:
            # Calculer le taux de croissance historique
            pop_values = self.population_data['population'].values
            years = self.population_data['year'].values
            if len(pop_values) >= 2 and len(years) >= 2:
                time_span = years[-1] - years[0]
                if time_span > 0:
                    historical_growth = (pop_values[-1] - pop_values[0]) / time_span / pop_values[0]
                    growth_predictions['overall_growth_rate'] = historical_growth * 1.1  # Légère accélération

        # Prédire la croissance par zone
        if not self.planning_zones.empty and 'population' in self.population_data.columns:
            for idx, zone in self.planning_zones.iterrows():
                zone_growth = self._predict_zone_growth(zone)
                growth_predictions['growth_by_zone'][zone.get('id')] = {
                    'predicted_growth': zone_growth,
                    'required_capacity': zone_growth * zone.get('current_density', 100),
                    'development_priority': 'high' if zone_growth > 0.05 else 'medium' if zone_growth > 0.02 else 'low'
                }

        return growth_predictions

    def _predict_zone_growth(self, zone: pd.Series) -> float:
        """
        Prédire la croissance pour une zone spécifique (simulation)
        """
        # Simulation basée sur plusieurs facteurs
        proximity_score = self._calculate_infrastructure_proximity(zone)
        zone_type_factor = self._get_zone_type_factor(zone.get('zone_type', ''))
        accessibility_score = self._calculate_accessibility_score(zone)

        # Facteur aléatoire pour simuler d'autres variables
        random_factor = np.random.uniform(0.8, 1.2)

        predicted_growth = (proximity_score * 0.3 + zone_type_factor * 0.3 + accessibility_score * 0.2 + 0.2) * random_factor * 0.08
        return min(predicted_growth, 0.15)  # Limiter à 15% de croissance annuelle

    def analyze_accessibility(self) -> Dict[str, Any]:
        """
        Analyser l'accessibilité des différents points de la ville
        """
        accessibility_analysis = {
            'accessibility_index': {},
            'service_coverage': {},
            'transport_connectivity': {},
            'barriers_identified': []
        }

        if self.infrastructure.empty or self.planning_zones.empty:
            return accessibility_analysis

        # Analyser l'accessibilité aux services
        services = self.infrastructure[self.infrastructure['type'].isin(['school', 'hospital', 'commercial'])]
        
        for idx, zone in self.planning_zones.iterrows():
            zone_centroid = zone.geometry.centroid
            accessibility_score = 0
            
            for service in services.itertuples():
                dist = zone_centroid.distance(service.geometry)
                # Plus la distance est courte, plus le score est élevé
                if dist < 5000:  # 5km max
                    accessibility_score += max(0, (5000 - dist) / 5000)
            
            accessibility_analysis['accessibility_index'][zone.get('id')] = min(accessibility_score, 1.0)

        return accessibility_analysis

    def generate_urbanism_report(self) -> Dict[str, Any]:
        """
        Générer un rapport complet sur l'urbanisme
        """
        report = {
            'report_date': datetime.now().isoformat(),
            'analysis_summary': {
                'total_zones': len(self.planning_zones),
                'total_infrastructure': len(self.infrastructure),
                'population_covered': len(self.population_data)
            },
            'density_analysis': self.analyze_urban_density(),
            'development_opportunities': self.identify_development_opportunities(),
            'infrastructure_assessment': self.assess_infrastructure_capacity(),
            'growth_predictions': self.predict_urban_growth(),
            'accessibility_analysis': self.analyze_accessibility(),
            'recommendations': self._generate_urbanism_recommendations()
        }

        return report

    def _generate_urbanism_recommendations(self) -> List[Dict[str, str]]:
        """
        Générer des recommandations d'aménagement urbain
        """
        recommendations = []

        # Recommandations basées sur la densité
        density_analysis = self.analyze_urban_density()
        if density_analysis.get('overcrowded_zones'):
            recommendations.append({
                'priority': 'high',
                'category': 'density_management',
                'description': f'Planifier la décongestion de {len(density_analysis["overcrowded_zones"])} zone(s) surpeuplée(s)'
            })

        if density_analysis.get('underutilized_zones'):
            recommendations.append({
                'priority': 'medium',
                'category': 'development',
                'description': f'Explorer les opportunités de développement pour {len(density_analysis["underutilized_zones"])} zone(s) sous-utilisée(s)'
            })

        # Recommandations basées sur les opportunités de développement
        opportunities = self.identify_development_opportunities()
        if opportunities:
            recommendations.append({
                'priority': 'high',
                'category': 'planning',
                'description': f'Prioriser {len(opportunities)} opportunité(s) de développement à haut potentiel'
            })

        # Recommandations basées sur les infrastructures
        infrastructure_assessment = self.assess_infrastructure_capacity()
        if infrastructure_assessment.get('overload_risks'):
            recommendations.append({
                'priority': 'high',
                'category': 'infrastructure',
                'description': f'Investir dans {len(infrastructure_assessment["overload_risks"])} infrastructure(s) en surcharge'
            })

        return recommendations

    def simulate_development_scenarios(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simuler différents scénarios de développement urbain
        """
        simulation_results = {
            'scenarios_evaluated': len(scenarios),
            'scenario_impacts': {},
            'best_scenario': None,
            'recommendations': []
        }

        best_score = -1
        best_scenario_name = None

        for scenario in scenarios:
            scenario_name = scenario.get('name', 'unnamed_scenario')
            scenario_impact = self._evaluate_scenario(scenario)
            simulation_results['scenario_impacts'][scenario_name] = scenario_impact

            # Déterminer le meilleur scénario basé sur un score composite
            score = scenario_impact.get('overall_score', 0)
            if score > best_score:
                best_score = score
                best_scenario_name = scenario_name

        simulation_results['best_scenario'] = best_scenario_name

        return simulation_results

    def _evaluate_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Évaluer un scénario de développement spécifique
        """
        # Simulation d'évaluation basée sur différents critères
        population_impact = scenario.get('expected_population_growth', 0) * 0.3
        infrastructure_impact = scenario.get('infrastructure_investment', 0) * 0.2
        environmental_impact = scenario.get('environmental_considerations', 0.5) * 0.3
        economic_impact = scenario.get('expected_economic_benefit', 0) * 0.2

        overall_score = population_impact + infrastructure_impact + environmental_impact + economic_impact

        return {
            'population_impact': population_impact,
            'infrastructure_impact': infrastructure_impact,
            'environmental_impact': environmental_impact,
            'economic_impact': economic_impact,
            'overall_score': overall_score,
            'feasibility': 'high' if overall_score > 0.7 else 'medium' if overall_score > 0.4 else 'low'
        }


def main():
    """
    Fonction principale pour démontrer l'agent d'urbanisme
    """
    agent = UrbanismAgent()

    print("Agent d'Urbanisme pour les Workflows IA Géospatiaux")
    print("=" * 55)

    # Charger des données de test
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
        },
        {
            'id': 'ZONE003',
            'geometry': Polygon([(0, 12), (10, 12), (10, 22), (0, 22)]),
            'zone_type': 'industrial',
            'density_limit': 100,
            'usage_type': 'manufacturing',
            'current_density': 80
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
            'id': 'TRANSIT001',
            'geometry': LineString([(2, 2), (8, 20)]),
            'type': 'public_transport',
            'capacity': 15000,
            'current_flow': 12000
        },
        {
            'id': 'SCHOOL001',
            'geometry': Point(5, 5),
            'type': 'school',
            'capacity': 500,
            'current_enrollment': 400
        },
        {
            'id': 'HOSPITAL001',
            'geometry': Point(15, 15),
            'type': 'hospital',
            'capacity': 300,
            'current_patients': 250
        }
    ], crs="EPSG:2154")

    sample_population = pd.DataFrame([
        {
            'year': 2020,
            'population': 100000,
            'area_id': 'ZONE001'
        },
        {
            'year': 2021,
            'population': 105000,
            'area_id': 'ZONE001'
        },
        {
            'year': 2022,
            'population': 110000,
            'area_id': 'ZONE001'
        }
    ])

    # Charger les données
    agent.load_planning_zones(sample_zones)
    agent.load_infrastructure(sample_infrastructure)
    agent.load_population_data(sample_population)

    # Exécuter les analyses
    print("\n🔍 Analyse de la densité urbaine...")
    density_analysis = agent.analyze_urban_density()
    print(f"   → Densité globale: {density_analysis['overall_density']:.2f} hab/ha")

    print("\n🏗️  Identification des opportunités de développement...")
    opportunities = agent.identify_development_opportunities()
    print(f"   → {len(opportunities)} opportunité(s) identifiée(s)")

    print("\n📊 Évaluation de la capacité des infrastructures...")
    infra_assessment = agent.assess_infrastructure_capacity()
    print(f"   → Analyse complète des infrastructures")

    print("\n📈 Prédiction de la croissance urbaine...")
    growth_predictions = agent.predict_urban_growth()
    print(f"   → Taux de croissance global: {growth_predictions['overall_growth_rate']:.4f}")

    print("\n🗺️  Analyse de l'accessibilité...")
    accessibility = agent.analyze_accessibility()
    print(f"   → {len(accessibility['accessibility_index'])} zones analysées")

    print("\n📋 Génération du rapport complet...")
    report = agent.generate_urbanism_report()
    print(f"   → {len(report['recommendations'])} recommandations générées")

    # Simulation de scénarios
    print("\n🔮 Simulation de scénarios de développement...")
    scenarios = [
        {
            'name': 'scenario_residential_expansion',
            'expected_population_growth': 0.15,
            'infrastructure_investment': 0.8,
            'environmental_considerations': 0.7,
            'expected_economic_benefit': 0.9
        },
        {
            'name': 'scenario_commercial_development',
            'expected_population_growth': 0.05,
            'infrastructure_investment': 0.6,
            'environmental_considerations': 0.6,
            'expected_economic_benefit': 0.8
        }
    ]
    
    simulation_results = agent.simulate_development_scenarios(scenarios)
    print(f"   → {simulation_results['scenarios_evaluated']} scénario(s) évalué(s)")
    print(f"   → Meilleur scénario: {simulation_results['best_scenario']}")

    print("\n✅ Démonstration de l'agent d'urbanisme terminée")


if __name__ == "__main__":
    main()