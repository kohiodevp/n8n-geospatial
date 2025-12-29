#!/usr/bin/env python3
"""
Agent Environnemental pour les Workflows IA Géospatiaux
========================================================

Cet agent est spécialisé dans la surveillance et l'analyse environnementales,
la gestion des risques écologiques, la conservation de la biodiversité,
et l'évaluation des impacts environnementaux.
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
from scipy import stats


class EnvironmentalAgent:
    """
    Agent spécialisé dans la surveillance et l'analyse environnementales
    """

    def __init__(self):
        self.environmental_data = gpd.GeoDataFrame()
        self.monitoring_stations = gpd.GeoDataFrame()
        self.risk_zones = gpd.GeoDataFrame()
        self.biodiversity_data = pd.DataFrame()
        self.environmental_indicators = {}
        self.pollution_sources = gpd.GeoDataFrame()
        self.ecosystem_services = pd.DataFrame()
        self.climate_data = pd.DataFrame()

    def load_environmental_data(self, gdf: gpd.GeoDataFrame) -> bool:
        """
        Charger les données environnementales
        """
        try:
            self.environmental_data = gdf
            print(f"Chargé {len(self.environmental_data)} enregistrements environnementaux")
            return True
        except Exception as e:
            print(f"Erreur lors du chargement des données environnementales: {e}")
            return False

    def load_monitoring_stations(self, gdf: gpd.GeoDataFrame) -> bool:
        """
        Charger les stations de surveillance
        """
        try:
            self.monitoring_stations = gdf
            print(f"Chargé {len(self.monitoring_stations)} stations de surveillance")
            return True
        except Exception as e:
            print(f"Erreur lors du chargement des stations de surveillance: {e}")
            return False

    def load_risk_zones(self, gdf: gpd.GeoDataFrame) -> bool:
        """
        Charger les zones à risque environnemental
        """
        try:
            self.risk_zones = gdf
            print(f"Chargé {len(self.risk_zones)} zones à risque")
            return True
        except Exception as e:
            print(f"Erreur lors du chargement des zones à risque: {e}")
            return False

    def load_biodiversity_data(self, df: pd.DataFrame) -> bool:
        """
        Charger les données de biodiversité
        """
        try:
            self.biodiversity_data = df
            print(f"Chargé {len(self.biodiversity_data)} enregistrements de biodiversité")
            return True
        except Exception as e:
            print(f"Erreur lors du chargement des données de biodiversité: {e}")
            return False

    def load_pollution_sources(self, gdf: gpd.GeoDataFrame) -> bool:
        """
        Charger les sources de pollution
        """
        try:
            self.pollution_sources = gdf
            print(f"Chargé {len(self.pollution_sources)} sources de pollution")
            return True
        except Exception as e:
            print(f"Erreur lors du chargement des sources de pollution: {e}")
            return False

    def load_ecosystem_services(self, df: pd.DataFrame) -> bool:
        """
        Charger les données sur les services écosystémiques
        """
        try:
            self.ecosystem_services = df
            print(f"Chargé {len(self.ecosystem_services)} enregistrements de services écosystémiques")
            return True
        except Exception as e:
            print(f"Erreur lors du chargement des services écosystémiques: {e}")
            return False

    def load_climate_data(self, df: pd.DataFrame) -> bool:
        """
        Charger les données climatiques
        """
        try:
            self.climate_data = df
            print(f"Chargé {len(self.climate_data)} enregistrements climatiques")
            return True
        except Exception as e:
            print(f"Erreur lors du chargement des données climatiques: {e}")
            return False

    def assess_environmental_quality(self) -> Dict[str, Any]:
        """
        Évaluer la qualité environnementale
        """
        if self.environmental_data.empty:
            return {}

        quality_assessment = {
            'overall_quality_index': 0.0,
            'quality_by_parameter': {},
            'pollution_hotspots': [],
            'improvement_recommendations': [],
            'trend_analysis': {}
        }

        # Calculer l'index de qualité global
        if 'quality_score' in self.environmental_data.columns:
            quality_assessment['overall_quality_index'] = self.environmental_data['quality_score'].mean()

        # Évaluer par paramètre environnemental
        for param in ['air_quality', 'water_quality', 'soil_quality', 'biodiversity_index']:
            if param in self.environmental_data.columns:
                avg_score = self.environmental_data[param].mean()
                quality_assessment['quality_by_parameter'][param] = {
                    'average': avg_score,
                    'min': self.environmental_data[param].min(),
                    'max': self.environmental_data[param].max(),
                    'std': self.environmental_data[param].std()
                }

                # Identifier les points critiques
                if param == 'air_quality':
                    hotspots = self.environmental_data[self.environmental_data[param] < 0.3]
                    for idx, hotspot in hotspots.iterrows():
                        quality_assessment['pollution_hotspots'].append({
                            'type': 'air_pollution',
                            'location': hotspot.get('id', f'point_{idx}'),
                            'severity': 'high' if hotspot[param] < 0.2 else 'medium',
                            'value': hotspot[param],
                            'geometry': hotspot.geometry
                        })
                elif param == 'water_quality':
                    hotspots = self.environmental_data[self.environmental_data[param] < 0.4]
                    for idx, hotspot in hotspots.iterrows():
                        quality_assessment['pollution_hotspots'].append({
                            'type': 'water_pollution',
                            'location': hotspot.get('id', f'point_{idx}'),
                            'severity': 'high' if hotspot[param] < 0.25 else 'medium',
                            'value': hotspot[param],
                            'geometry': hotspot.geometry
                        })

        # Analyse des tendances
        if not self.climate_data.empty and 'date' in self.climate_data.columns:
            self.climate_data['date'] = pd.to_datetime(self.climate_data['date'])
            recent_data = self.climate_data.sort_values('date').tail(10)  # 10 derniers points
            
            if 'temperature' in recent_data.columns:
                temp_trend = recent_data['temperature'].iloc[-1] - recent_data['temperature'].iloc[0]
                quality_assessment['trend_analysis']['temperature_trend'] = temp_trend
            
            if 'precipitation' in recent_data.columns:
                precip_trend = recent_data['precipitation'].iloc[-1] - recent_data['precipitation'].iloc[0]
                quality_assessment['trend_analysis']['precipitation_trend'] = precip_trend

        return quality_assessment

    def detect_environmental_risks(self) -> List[Dict[str, Any]]:
        """
        Détecter les risques environnementaux
        """
        risks = []

        # Détecter les risques basés sur les zones à risque
        if not self.risk_zones.empty:
            for idx, risk_zone in self.risk_zones.iterrows():
                risks.append({
                    'type': risk_zone.get('risk_type', 'unknown'),
                    'zone_id': risk_zone.get('id'),
                    'geometry': risk_zone.geometry,
                    'probability': risk_zone.get('probability', 0.5),
                    'potential_impact': risk_zone.get('impact_level', 'medium'),
                    'recommended_action': self._recommend_risk_action(risk_zone.get('risk_type')),
                    'severity': risk_zone.get('severity', 'medium')
                })

        # Détecter les risques basés sur les données environnementales
        if not self.environmental_data.empty:
            # Détecter les tendances environnementales préoccupantes
            for param in ['contamination_level', 'pollution_index', 'hazard_level']:
                if param in self.environmental_data.columns:
                    high_risk = self.environmental_data[self.environmental_data[param] > 0.7]
                    for idx, data_point in high_risk.iterrows():
                        risks.append({
                            'type': f'{param}_risk',
                            'location': data_point.get('id', f'point_{idx}'),
                            'geometry': data_point.geometry,
                            'severity': 'high',
                            'current_level': data_point[param],
                            'recommended_action': f'monitoring_and_remediation_for_{param}',
                            'probability': data_point.get('probability', 0.8)
                        })

        # Détecter les risques liés aux sources de pollution
        if not self.pollution_sources.empty:
            for idx, source in self.pollution_sources.iterrows():
                # Créer une zone de risque autour de la source
                buffer_zone = source.geometry.buffer(source.get('impact_radius', 1000))
                
                # Trouver les zones environnementales affectées
                affected_areas = self.environmental_data[self.environmental_data.geometry.intersects(buffer_zone)]
                
                for affected_idx, affected in affected_areas.iterrows():
                    risks.append({
                        'type': 'pollution_spread',
                        'location': affected.get('id', f'affected_{affected_idx}'),
                        'source_id': source.get('id'),
                        'geometry': affected.geometry,
                        'severity': source.get('severity', 'medium'),
                        'pollution_type': source.get('pollution_type', 'unknown'),
                        'recommended_action': 'pollution_control_and_monitoring',
                        'probability': 0.9 if source.get('severity') == 'high' else 0.7
                    })

        return risks

    def _recommend_risk_action(self, risk_type: str) -> str:
        """
        Recommander une action basée sur le type de risque
        """
        recommendations = {
            'flood': 'flood_prevention_measures',
            'landslide': 'slope_stabilization',
            'pollution': 'contamination_control',
            'fire': 'fire_prevention_systems',
            'erosion': 'erosion_control_measures',
            'drought': 'water_conservation_programs',
            'storm': 'infrastructure_hardening',
            'heat_wave': 'urban_cooling_initiatives'
        }
        return recommendations.get(risk_type, 'assessment_required')

    def analyze_biodiversity_hotspots(self) -> List[Dict[str, Any]]:
        """
        Analyser les hotspots de biodiversité
        """
        hotspots = []

        if self.biodiversity_data.empty:
            return hotspots

        # Identifier les zones avec haute richesse spécifique
        if 'species_richness' in self.biodiversity_data.columns:
            # Calculer le seuil pour les hotspots (80e percentile)
            threshold = self.biodiversity_data['species_richness'].quantile(0.8)

            high_richness = self.biodiversity_data[self.biodiversity_data['species_richness'] > threshold]

            for idx, site in high_richness.iterrows():
                hotspots.append({
                    'site_id': site.get('id', f'site_{idx}'),
                    'species_richness': site['species_richness'],
                    'endemic_species': site.get('endemic_species', 0),
                    'threatened_species': site.get('threatened_species', 0),
                    'conservation_status': site.get('conservation_status', 'unknown'),
                    'protection_level': self._determine_protection_level(site),
                    'ecosystem_type': site.get('ecosystem_type', 'mixed'),
                    'area_ha': site.get('area_ha', 0)
                })

        return hotspots

    def _determine_protection_level(self, site_data: pd.Series) -> str:
        """
        Déterminer le niveau de protection requis pour un site
        """
        richness = site_data.get('species_richness', 0)
        endemic_count = site_data.get('endemic_species', 0)
        threatened_count = site_data.get('threatened_species', 0)

        if richness > 150 or endemic_count > 20 or threatened_count > 10:
            return 'strict_protection'
        elif richness > 100 or endemic_count > 10 or threatened_count > 5:
            return 'protected_area'
        elif richness > 50 or endemic_count > 5:
            return 'conservation_monitoring'
        else:
            return 'habitat_management'

    def predict_environmental_trends(self) -> Dict[str, Any]:
        """
        Prédire les tendances environnementales futures
        """
        predictions = {
            'climate_trends': {},
            'pollution_trends': {},
            'biodiversity_trends': {},
            'recommendation_priority': 'medium',
            'confidence_intervals': {}
        }

        # Analyser les données climatiques
        if not self.climate_data.empty and 'date' in self.climate_data.columns:
            self.climate_data['date'] = pd.to_datetime(self.climate_data['date'])
            df_with_dates = self.climate_data.sort_values('date')

            for param in ['temperature', 'precipitation', 'humidity']:
                if param in df_with_dates.columns:
                    values = df_with_dates[param].dropna()
                    if len(values) > 2:  # Besoin de minimum 3 points pour tendance
                        # Calculer la tendance avec régression linéaire
                        x = np.arange(len(values))
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                        
                        predictions['climate_trends'][param] = {
                            'slope': slope,
                            'r_squared': r_value ** 2,
                            'p_value': p_value,
                            'current_value': values.iloc[-1],
                            'projected_change_5y': slope * 5  # Projection sur 5 ans
                        }

        # Analyser les données de pollution
        if not self.environmental_data.empty:
            for param in ['air_quality', 'water_quality', 'soil_contamination']:
                if param in self.environmental_data.columns:
                    current_avg = self.environmental_data[param].mean()
                    trend_direction = 'improving' if current_avg > 0.7 else 'deteriorating' if current_avg < 0.3 else 'stable'
                    
                    predictions['pollution_trends'][param] = {
                        'current_average': current_avg,
                        'trend': trend_direction,
                        'concern_level': 'low' if current_avg > 0.7 else 'high' if current_avg < 0.3 else 'medium'
                    }

        # Analyser les tendances de biodiversité
        if not self.biodiversity_data.empty and 'species_richness' in self.biodiversity_data.columns:
            richness_values = self.biodiversity_data['species_richness']
            avg_richness = richness_values.mean()
            richness_trend = 'stable' if abs(avg_richness - 100) < 20 else 'concerning'  # 100 = seuil de référence
            
            predictions['biodiversity_trends'] = {
                'average_richness': avg_richness,
                'trend': richness_trend,
                'endemic_ratio': self.biodiversity_data['endemic_species'].sum() / richness_values.sum() if richness_values.sum() > 0 else 0
            }

        return predictions

    def assess_ecosystem_services(self) -> Dict[str, Any]:
        """
        Évaluer les services écosystémiques fournis
        """
        if self.ecosystem_services.empty:
            return {}

        services_assessment = {
            'provisioning_services': {},
            'regulating_services': {},
            'cultural_services': {},
            'supporting_services': {},
            'economic_value': 0,
            'spatial_distribution': {}
        }

        # Calculer la valeur économique des services écosystémiques
        if 'economic_value' in self.ecosystem_services.columns:
            services_assessment['economic_value'] = self.ecosystem_services['economic_value'].sum()

        # Répartir les services par type
        for service_type in ['provisioning', 'regulating', 'cultural', 'supporting']:
            type_services = self.ecosystem_services[self.ecosystem_services['service_type'] == service_type]
            if not type_services.empty:
                services_assessment[f'{service_type}_services'] = {
                    'count': len(type_services),
                    'total_value': type_services['economic_value'].sum() if 'economic_value' in type_services.columns else 0,
                    'average_value': type_services['economic_value'].mean() if 'economic_value' in type_services.columns else 0,
                    'services': type_services['service_name'].tolist() if 'service_name' in type_services.columns else []
                }

        return services_assessment

    def identify_conservation_priorities(self) -> List[Dict[str, Any]]:
        """
        Identifier les priorités de conservation
        """
        priorities = []

        # Analyser les données de biodiversité pour les priorités
        if not self.biodiversity_data.empty:
            for idx, site in self.biodiversity_data.iterrows():
                priority_score = self._calculate_conservation_priority(site)
                
                if priority_score > 0.7:  # Seulement les sites à haute priorité
                    priorities.append({
                        'site_id': site.get('id', f'site_{idx}'),
                        'priority_score': priority_score,
                        'conservation_category': self._categorize_conservation_site(site),
                        'threat_level': self._assess_threat_level(site),
                        'recommended_actions': self._generate_conservation_actions(site),
                        'funding_requirement': self._estimate_funding_requirement(site, priority_score)
                    })

        # Analyser les zones à risque pour les priorités de conservation
        if not self.risk_zones.empty:
            for idx, risk_zone in self.risk_zones.iterrows():
                if risk_zone.get('severity') == 'high':
                    priorities.append({
                        'zone_id': risk_zone.get('id'),
                        'priority_score': 0.9,
                        'conservation_category': 'emergency_intervention',
                        'threat_level': risk_zone.get('severity'),
                        'recommended_actions': ['immediate_protection_measures', 'risk_mitigation'],
                        'funding_requirement': 'high'
                    })

        return priorities

    def _calculate_conservation_priority(self, site: pd.Series) -> float:
        """
        Calculer le score de priorité de conservation pour un site
        """
        # Facteurs: richesse spécifique, espèces endémiques, espèces menacées, rareté
        richness_factor = min(site.get('species_richness', 0) / 200, 1.0)  # Max 200 espèces
        endemic_factor = min(site.get('endemic_species', 0) / 50, 1.0)    # Max 50 espèces endémiques
        threatened_factor = min(site.get('threatened_species', 0) / 30, 1.0)  # Max 30 espèces menacées
        rarity_factor = 1.0 - (site.get('area_ha', 1000) / 10000)  # Plus petit = plus rare
        rarity_factor = max(0, min(rarity_factor, 1.0))

        # Calculer le score combiné
        priority = (richness_factor * 0.3 + endemic_factor * 0.3 + 
                   threatened_factor * 0.25 + rarity_factor * 0.15)
        
        return min(priority, 1.0)

    def _categorize_conservation_site(self, site: pd.Series) -> str:
        """
        Catégoriser un site de conservation
        """
        priority_score = self._calculate_conservation_priority(site)
        
        if priority_score > 0.8:
            return 'critical_site'
        elif priority_score > 0.6:
            return 'high_priority_site'
        elif priority_score > 0.4:
            return 'medium_priority_site'
        else:
            return 'monitoring_site'

    def _assess_threat_level(self, site: pd.Series) -> str:
        """
        Évaluer le niveau de menace pour un site
        """
        threatened_count = site.get('threatened_species', 0)
        human_pressure = site.get('human_pressure_index', 0.5)  # 0-1 scale
        
        if threatened_count > 15 or human_pressure > 0.8:
            return 'critical'
        elif threatened_count > 8 or human_pressure > 0.6:
            return 'high'
        elif threatened_count > 3 or human_pressure > 0.4:
            return 'medium'
        else:
            return 'low'

    def _generate_conservation_actions(self, site: pd.Series) -> List[str]:
        """
        Générer des actions de conservation recommandées
        """
        actions = []
        
        if site.get('threatened_species', 0) > 5:
            actions.append('species_recovery_program')
        
        if site.get('human_pressure_index', 0.5) > 0.6:
            actions.append('habitat_protection')
        
        if site.get('area_ha', 0) < 100:
            actions.append('habitat_corridor_creation')
        
        if 'endemic_species' in site and site.get('endemic_species', 0) > 10:
            actions.append('endemic_species_conservation')
        
        if not actions:
            actions.append('routine_monitoring')
        
        return actions

    def _estimate_funding_requirement(self, site: pd.Series, priority_score: float) -> str:
        """
        Estimer le besoin de financement pour un site
        """
        if priority_score > 0.8:
            return 'very_high'
        elif priority_score > 0.6:
            return 'high'
        elif priority_score > 0.4:
            return 'medium'
        else:
            return 'low'

    def analyze_pollution_impact(self) -> Dict[str, Any]:
        """
        Analyser l'impact de la pollution sur l'environnement
        """
        impact_analysis = {
            'pollution_sources_impact': {},
            'affected_areas': [],
            'contamination_levels': {},
            'mitigation_recommendations': []
        }

        if self.pollution_sources.empty or self.environmental_data.empty:
            return impact_analysis

        # Analyser chaque source de pollution
        for idx, source in self.pollution_sources.iterrows():
            source_impact = {
                'source_id': source.get('id'),
                'pollution_type': source.get('pollution_type', 'unknown'),
                'severity': source.get('severity', 'medium'),
                'impact_radius': source.get('impact_radius', 1000),
                'affected_area_count': 0,
                'max_contamination_level': 0
            }

            # Créer une zone tampon autour de la source
            buffer_zone = source.geometry.buffer(source_impact['impact_radius'])
            
            # Trouver les zones environnementales affectées
            affected_areas = self.environmental_data[self.environmental_data.geometry.intersects(buffer_zone)]
            
            source_impact['affected_area_count'] = len(affected_areas)
            
            if not affected_areas.empty and 'contamination_level' in affected_areas.columns:
                max_contamination = affected_areas['contamination_level'].max()
                source_impact['max_contamination_level'] = max_contamination

                # Ajouter les zones affectées à l'analyse globale
                for affected_idx, affected in affected_areas.iterrows():
                    impact_analysis['affected_areas'].append({
                        'area_id': affected.get('id'),
                        'source_id': source.get('id'),
                        'contamination_level': affected.get('contamination_level', 0),
                        'distance_to_source': source.geometry.distance(affected.geometry),
                        'geometry': affected.geometry
                    })

            impact_analysis['pollution_sources_impact'][source.get('id')] = source_impact

        # Calculer les niveaux de contamination globaux
        if 'contamination_level' in self.environmental_data.columns:
            contamination_levels = self.environmental_data['contamination_level']
            impact_analysis['contamination_levels'] = {
                'mean': float(contamination_levels.mean()),
                'median': float(contamination_levels.median()),
                'max': float(contamination_levels.max()),
                'min': float(contamination_levels.min()),
                'std': float(contamination_levels.std())
            }

        # Générer des recommandations de mitigation
        if len(impact_analysis['affected_areas']) > 0:
            impact_analysis['mitigation_recommendations'].append('pollution_source_control')
        
        if impact_analysis['contamination_levels'].get('mean', 0) > 0.5:
            impact_analysis['mitigation_recommendations'].append('environmental_remediation')
        
        if any(area['contamination_level'] > 0.8 for area in impact_analysis['affected_areas']):
            impact_analysis['mitigation_recommendations'].append('immediate_containment_measures')

        return impact_analysis

    def generate_environmental_report(self) -> Dict[str, Any]:
        """
        Générer un rapport complet sur l'environnement
        """
        report = {
            'report_date': datetime.now().isoformat(),
            'analysis_summary': {
                'total_monitoring_stations': len(self.monitoring_stations),
                'total_risk_zones': len(self.risk_zones),
                'total_biodiversity_sites': len(self.biodiversity_data),
                'total_pollution_sources': len(self.pollution_sources)
            },
            'environmental_quality': self.assess_environmental_quality(),
            'environmental_risks': self.detect_environmental_risks(),
            'biodiversity_hotspots': self.analyze_biodiversity_hotspots(),
            'environmental_trends': self.predict_environmental_trends(),
            'ecosystem_services': self.assess_ecosystem_services(),
            'conservation_priorities': self.identify_conservation_priorities(),
            'pollution_impact': self.analyze_pollution_impact(),
            'recommendations': self._generate_environmental_recommendations()
        }

        return report

    def _generate_environmental_recommendations(self) -> List[Dict[str, str]]:
        """
        Générer des recommandations environnementales
        """
        recommendations = []

        # Recommandations basées sur la qualité environnementale
        quality_assessment = self.assess_environmental_quality()
        if quality_assessment.get('overall_quality_index', 1.0) < 0.5:
            recommendations.append({
                'priority': 'high',
                'category': 'quality_improvement',
                'description': f'Améliorer la qualité environnementale (actuel: {quality_assessment["overall_quality_index"]:.2f})'
            })

        # Recommandations basées sur les risques
        risks = self.detect_environmental_risks()
        high_risk_count = sum(1 for r in risks if r.get('severity') == 'high')
        if high_risk_count > 0:
            recommendations.append({
                'priority': 'high',
                'category': 'risk_management',
                'description': f'Gérer {high_risk_count} risque(s) environnemental(aux) critique(s)'
            })

        # Recommandations basées sur la biodiversité
        hotspots = self.analyze_biodiversity_hotspots()
        if hotspots:
            recommendations.append({
                'priority': 'high',
                'category': 'biodiversity_conservation',
                'description': f'Protéger {len(hotspots)} hotspot(s) de biodiversité'
            })

        # Recommandations basées sur les priorités de conservation
        priorities = self.identify_conservation_priorities()
        if priorities:
            critical_sites = sum(1 for p in priorities if p.get('conservation_category') == 'critical_site')
            if critical_sites > 0:
                recommendations.append({
                    'priority': 'high',
                    'category': 'conservation',
                    'description': f'Prioriser la conservation de {critical_sites} site(s) critique(s)'
                })

        # Recommandations basées sur la pollution
        pollution_impact = self.analyze_pollution_impact()
        if pollution_impact.get('affected_areas'):
            recommendations.append({
                'priority': 'medium',
                'category': 'pollution_control',
                'description': f'Contrôler la pollution affectant {len(pollution_impact["affected_areas"])} zone(s)'
            })

        return recommendations

    def simulate_environmental_scenarios(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simuler différents scénarios environnementaux
        """
        simulation_results = {
            'scenarios_evaluated': len(scenarios),
            'scenario_impacts': {},
            'best_scenario': None,
            'environmental_score_evolution': {}
        }

        best_score = -1
        best_scenario_name = None

        for scenario in scenarios:
            scenario_name = scenario.get('name', 'unnamed_scenario')
            scenario_impact = self._evaluate_environmental_scenario(scenario)
            simulation_results['scenario_impacts'][scenario_name] = scenario_impact

            # Déterminer le meilleur scénario basé sur un score environnemental
            score = scenario_impact.get('environmental_score', 0)
            if score > best_score:
                best_score = score
                best_scenario_name = scenario_name

        simulation_results['best_scenario'] = best_scenario_name

        # Simuler l'évolution des scores dans le temps
        for scenario in scenarios:
            scenario_name = scenario.get('name', 'unnamed_scenario')
            time_evolution = self._simulate_score_evolution(scenario)
            simulation_results['environmental_score_evolution'][scenario_name] = time_evolution

        return simulation_results

    def _evaluate_environmental_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Évaluer un scénario environnemental spécifique
        """
        # Calculer un score environnemental basé sur différents critères
        conservation_impact = scenario.get('conservation_impact', 0) * 0.3
        pollution_reduction = scenario.get('pollution_reduction', 0) * 0.25
        biodiversity_enhancement = scenario.get('biodiversity_enhancement', 0) * 0.25
        ecosystem_service_improvement = scenario.get('ecosystem_service_improvement', 0) * 0.2

        environmental_score = conservation_impact + pollution_reduction + \
                             biodiversity_enhancement + ecosystem_service_improvement

        return {
            'conservation_impact': conservation_impact,
            'pollution_reduction': pollution_reduction,
            'biodiversity_enhancement': biodiversity_enhancement,
            'ecosystem_service_improvement': ecosystem_service_improvement,
            'environmental_score': environmental_score,
            'feasibility': scenario.get('feasibility', 'medium'),
            'cost_estimate': scenario.get('cost_estimate', 'unknown')
        }

    def _simulate_score_evolution(self, scenario: Dict[str, Any]) -> List[Dict[str, float]]:
        """
        Simuler l'évolution du score environnemental dans le temps
        """
        evolution = []
        base_score = scenario.get('initial_environmental_score', 0.5)
        improvement_rate = scenario.get('improvement_rate', 0.05)  # 5% par période

        for year in range(1, 6):  # Projection sur 5 ans
            score = base_score + (improvement_rate * year) + (np.random.normal(0, 0.02) * year)  # Ajouter un peu de variabilité
            score = max(0, min(1, score))  # S'assurer que le score est entre 0 et 1
            evolution.append({
                'year': year,
                'predicted_score': round(score, 3),
                'cumulative_improvement': round(score - base_score, 3)
            })

        return evolution


def main():
    """
    Fonction principale pour démontrer l'agent environnemental
    """
    agent = EnvironmentalAgent()

    print("Agent Environnemental pour les Workflows IA Géospatiaux")
    print("=" * 55)

    # Charger des données de test
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
        },
        {
            'id': 'ENV003',
            'geometry': Point(2.3600, 48.8600),
            'air_quality': 0.9,
            'water_quality': 0.9,
            'soil_quality': 0.85,
            'biodiversity_index': 0.88,
            'quality_score': 0.88
        }
    ], crs="EPSG:4326")

    sample_monitoring = gpd.GeoDataFrame([
        {
            'id': 'STATION001',
            'geometry': Point(2.3522, 48.8566),
            'station_type': 'air_quality',
            'last_reading': datetime.now().isoformat(),
            'status': 'active'
        },
        {
            'id': 'STATION002',
            'geometry': Point(2.3500, 48.8500),
            'station_type': 'water_quality',
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
        },
        {
            'id': 'RISK002',
            'geometry': Polygon([(2.37, 48.87), (2.39, 48.87), (2.39, 48.89), (2.37, 48.89)]),
            'risk_type': 'flood',
            'probability': 0.3,
            'impact_level': 'medium',
            'severity': 'medium'
        }
    ], crs="EPSG:4326")

    sample_biodiversity = pd.DataFrame([
        {
            'id': 'SITE001',
            'species_richness': 120,
            'endemic_species': 15,
            'threatened_species': 8,
            'conservation_status': 'protected',
            'ecosystem_type': 'forest',
            'area_ha': 150,
            'human_pressure_index': 0.3
        },
        {
            'id': 'SITE002',
            'species_richness': 80,
            'endemic_species': 5,
            'threatened_species': 12,
            'conservation_status': 'vulnerable',
            'ecosystem_type': 'wetland',
            'area_ha': 80,
            'human_pressure_index': 0.7
        }
    ])

    sample_pollution = gpd.GeoDataFrame([
        {
            'id': 'POLL001',
            'geometry': Point(2.3550, 48.8550),
            'pollution_type': 'industrial',
            'severity': 'high',
            'impact_radius': 2000,
            'emission_level': 0.9
        },
        {
            'id': 'POLL002',
            'geometry': Point(2.3450, 48.8450),
            'pollution_type': 'urban_runoff',
            'severity': 'medium',
            'impact_radius': 1000,
            'emission_level': 0.6
        }
    ], crs="EPSG:4326")

    # Charger les données
    agent.load_environmental_data(sample_environmental)
    agent.load_monitoring_stations(sample_monitoring)
    agent.load_risk_zones(sample_risk_zones)
    agent.load_biodiversity_data(sample_biodiversity)
    agent.load_pollution_sources(sample_pollution)

    # Exécuter les analyses
    print("\n🔍 Évaluation de la qualité environnementale...")
    quality = agent.assess_environmental_quality()
    print(f"   → Index de qualité global: {quality['overall_quality_index']:.2f}")

    print("\n🚨 Détection des risques environnementaux...")
    risks = agent.detect_environmental_risks()
    print(f"   → {len(risks)} risque(s) détecté(s)")

    print("\n🌿 Analyse des hotspots de biodiversité...")
    hotspots = agent.analyze_biodiversity_hotspots()
    print(f"   → {len(hotspots)} hotspot(s) identifié(s)")

    print("\n📊 Prédiction des tendances environnementales...")
    trends = agent.predict_environmental_trends()
    print(f"   → Analyse des tendances terminée")

    print("\n🌍 Évaluation des services écosystémiques...")
    services = agent.assess_ecosystem_services()
    print(f"   → Évaluation des services écosystémiques terminée")

    print("\n🎯 Identification des priorités de conservation...")
    priorities = agent.identify_conservation_priorities()
    print(f"   → {len(priorities)} priorité(s) identifiée(s)")

    print("\n☣️  Analyse de l'impact de la pollution...")
    pollution_impact = agent.analyze_pollution_impact()
    print(f"   → {len(pollution_impact['affected_areas'])} zone(s) affectée(s)")

    print("\n📋 Génération du rapport complet...")
    report = agent.generate_environmental_report()
    print(f"   → {len(report['recommendations'])} recommandations générées")

    # Simulation de scénarios
    print("\n🔮 Simulation de scénarios environnementaux...")
    scenarios = [
        {
            'name': 'scenario_conservation_focus',
            'conservation_impact': 0.9,
            'pollution_reduction': 0.6,
            'biodiversity_enhancement': 0.8,
            'ecosystem_service_improvement': 0.7,
            'initial_environmental_score': 0.5,
            'improvement_rate': 0.12,
            'feasibility': 'high',
            'cost_estimate': 'high'
        },
        {
            'name': 'scenario_pollution_control',
            'conservation_impact': 0.5,
            'pollution_reduction': 0.9,
            'biodiversity_enhancement': 0.4,
            'ecosystem_service_improvement': 0.6,
            'initial_environmental_score': 0.4,
            'improvement_rate': 0.08,
            'feasibility': 'medium',
            'cost_estimate': 'medium'
        }
    ]
    
    simulation_results = agent.simulate_environmental_scenarios(scenarios)
    print(f"   → {simulation_results['scenarios_evaluated']} scénario(s) évalué(s)")
    print(f"   → Meilleur scénario: {simulation_results['best_scenario']}")

    print("\n✅ Démonstration de l'agent environnemental terminée")


if __name__ == "__main__":
    main()