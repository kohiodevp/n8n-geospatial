#!/usr/bin/env python3
"""
Agent Domanial pour les Workflows IA Géospatiaux
================================================

Cet agent est spécialisé dans les traitements liés aux propriétés domaniales
(propriétés de l'État, collectivités publiques). Il inclut des capacités
d'analyse spatiale, de gestion des droits domaniaux, et de suivi des concessions.
"""

import json
import os
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from shapely.geometry import Point, Polygon
from geopy.distance import geodesic
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

class DomainAgent:
    """
    Agent spécialisé dans les propriétés domaniales
    """
    
    def __init__(self):
        self.domain_properties = []
        self.concessions = []
        self.zones = []
        self.revenue_model = None
        
    def load_domain_properties(self, gdf: gpd.GeoDataFrame) -> bool:
        """
        Charger les données des propriétés domaniales
        """
        try:
            # Vérifier que les propriétés requises existent
            required_cols = ['id', 'geometry', 'category', 'status', 'area_ha', 'value', 'management_entity']
            if not all(col in gdf.columns for col in required_cols):
                raise ValueError(f"Colonnes requises manquantes: {set(required_cols) - set(gdf.columns)}")
            
            self.domain_properties = gdf
            print(f"Chargé {len(self.domain_properties)} propriétés domaniales")
            return True
        except Exception as e:
            print(f"Erreur lors du chargement des propriétés domaniales: {e}")
            return False
    
    def load_concessions(self, concessions_data: List[Dict]) -> bool:
        """
        Charger les données des concessions domaniales
        """
        try:
            self.concessions = concessions_data
            print(f"Chargé {len(self.concessions)} concessions")
            return True
        except Exception as e:
            print(f"Erreur lors du chargement des concessions: {e}")
            return False
    
    def identify_strategic_zones(self) -> gpd.GeoDataFrame:
        """
        Identifier les zones stratégiques pour la gestion domaniale
        """
        if self.domain_properties.empty:
            return gpd.GeoDataFrame()
        
        # Calculer les zones avec potentiel de développement
        strategic_zones = []
        
        for idx, prop in self.domain_properties.iterrows():
            if prop['geometry'] and prop['status'] == 'available':
                # Calculer le potentiel basé sur la localisation et la taille
                potential_score = self._calculate_potential_score(prop)
                
                if potential_score > 0.7:  # Seulement les zones à haut potentiel
                    strategic_zones.append({
                        'id': prop['id'],
                        'geometry': prop['geometry'],
                        'potential_score': potential_score,
                        'category': prop['category'],
                        'recommended_use': self._recommend_use(prop, potential_score),
                        'accessibility_score': self._calculate_accessibility_score(prop)
                    })
        
        if strategic_zones:
            return gpd.GeoDataFrame(strategic_zones)
        else:
            return gpd.GeoDataFrame()
    
    def _calculate_potential_score(self, property_data: pd.Series) -> float:
        """
        Calculer le score de potentiel pour une propriété
        """
        # Facteurs: superficie, localisation, catégorie, accessibilité
        area_factor = min(property_data['area_ha'] / 100, 1.0)  # Normaliser la superficie
        category_factor = self._get_category_weight(property_data['category'])
        
        # Calculer la distance aux infrastructures (simulation)
        location_factor = self._calculate_location_score(property_data)
        
        # Score combiné
        potential = (area_factor * 0.3 + category_factor * 0.4 + location_factor * 0.3)
        return min(potential, 1.0)
    
    def _get_category_weight(self, category: str) -> float:
        """
        Obtenir le poids pour une catégorie de propriété
        """
        category_weights = {
            'coastal': 0.9,      # Côtes - potentiel élevé
            'riverbank': 0.8,    # Rives - potentiel élevé
            'urban_perimeter': 0.7,  # Périmètre urbain
            'agricultural': 0.5, # Agricole
            'forest': 0.6,       # Forêt
            'industrial': 0.4    # Industriel
        }
        return category_weights.get(category, 0.3)
    
    def _calculate_location_score(self, property_data: pd.Series) -> float:
        """
        Calculer le score de localisation (simulation)
        """
        # Calculer la distance à des points d'intérêt (simulation)
        # Dans une implémentation réelle, on utiliserait des données d'infrastructures
        return np.random.uniform(0.4, 0.9)  # Simulation aléatoire pondérée
    
    def _recommend_use(self, property_data: pd.Series, potential_score: float) -> str:
        """
        Recommander l'usage optimal pour une propriété
        """
        if potential_score > 0.8:
            if property_data['category'] in ['coastal', 'riverbank']:
                return 'tourism_development'
            elif property_data['category'] == 'urban_perimeter':
                return 'urban_expansion'
            else:
                return 'strategic_concession'
        elif potential_score > 0.6:
            return 'sustainable_management'
        else:
            return 'conservation'
    
    def _calculate_accessibility_score(self, property_data: pd.Series) -> float:
        """
        Calculer le score d'accessibilité
        """
        # Simulation basée sur la proximité des routes et infrastructures
        return np.random.uniform(0.3, 0.9)
    
    def analyze_concessions(self) -> Dict[str, Any]:
        """
        Analyser les concessions domaniales en cours
        """
        if not self.concessions:
            return {}
        
        analysis = {
            'total_concessions': len(self.concessions),
            'by_status': {},
            'by_type': {},
            'revenue_potential': 0,
            'expiration_warnings': [],
            'performance_metrics': {}
        }
        
        # Compter par statut
        for concession in self.concessions:
            status = concession.get('status', 'unknown')
            concession_type = concession.get('type', 'unknown')
            
            analysis['by_status'][status] = analysis['by_status'].get(status, 0) + 1
            analysis['by_type'][concession_type] = analysis['by_type'].get(concession_type, 0) + 1
            
            # Calculer le potentiel de revenu
            if 'annual_fee' in concession:
                analysis['revenue_potential'] += float(concession['annual_fee'])
            
            # Vérifier les échéances
            if 'end_date' in concession:
                end_date = datetime.fromisoformat(concession['end_date'].replace('Z', '+00:00'))
                if end_date < datetime.now() + timedelta(days=365):  # 1 an avant échéance
                    analysis['expiration_warnings'].append({
                        'id': concession.get('id'),
                        'end_date': concession['end_date'],
                        'days_left': (end_date - datetime.now()).days
                    })
        
        # Calculer les métriques de performance
        total_area = sum(c.get('area_ha', 0) for c in self.concessions)
        active_concessions = sum(1 for c in self.concessions if c.get('status') == 'active')
        
        analysis['performance_metrics'] = {
            'total_area_ha': total_area,
            'active_concessions': active_concessions,
            'utilization_rate': active_concessions / len(self.concessions) if self.concessions else 0,
            'average_fee_per_ha': analysis['revenue_potential'] / total_area if total_area > 0 else 0
        }
        
        return analysis
    
    def detect_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """
        Détecter les opportunités d'optimisation de la gestion domaniale
        """
        opportunities = []
        
        # Analyse des clusters de propriétés
        if not self.domain_properties.empty:
            clusters = self._find_property_clusters()
            for cluster in clusters:
                if len(cluster['properties']) >= 3:  # Groupe significatif
                    opportunities.append({
                        'type': 'property_cluster_optimization',
                        'cluster_id': cluster['id'],
                        'properties': cluster['properties'],
                        'recommended_action': 'integrated_management',
                        'estimated_benefit': self._calculate_cluster_benefit(cluster)
                    })
        
        # Analyse des zones sous-exploitées
        for idx, prop in self.domain_properties.iterrows():
            if prop['status'] == 'available' and prop['area_ha'] > 10:
                opportunities.append({
                    'type': 'underutilized_property',
                    'property_id': prop['id'],
                    'area_ha': prop['area_ha'],
                    'recommended_action': 'market_exploration',
                    'estimated_benefit': prop['area_ha'] * 500  # Simulation de valeur
                })
        
        return opportunities
    
    def _find_property_clusters(self) -> List[Dict[str, Any]]:
        """
        Trouver les clusters de propriétés domaniales
        """
        if self.domain_properties.empty or len(self.domain_properties) < 3:
            return []
        
        # Extraire les coordonnées centrales des propriétés
        centroids = []
        prop_ids = []
        
        for idx, prop in self.domain_properties.iterrows():
            if prop['geometry']:
                centroid = prop['geometry'].centroid
                centroids.append([centroid.y, centroid.x])  # [lat, lon]
                prop_ids.append(prop['id'])
        
        if len(centroids) < 3:
            return []
        
        # Convertir en numpy array
        centroids = np.array(centroids)
        
        # Normaliser les coordonnées pour une distance homogène
        scaler = StandardScaler()
        centroids_scaled = scaler.fit_transform(centroids)
        
        # Appliquer DBSCAN pour trouver les clusters
        clustering = DBSCAN(eps=0.01, min_samples=2)  # Ajuster selon la zone
        cluster_labels = clustering.fit_predict(centroids_scaled)
        
        # Organiser les résultats par cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label != -1:  # -1 indique les points bruit
                if label not in clusters:
                    clusters[label] = {
                        'id': label,
                        'properties': [],
                        'centroid': centroids[i].tolist()
                    }
                clusters[label]['properties'].append({
                    'id': prop_ids[i],
                    'geometry': self.domain_properties.iloc[i]['geometry']
                })
        
        return [cluster_info for cluster_info in clusters.values()]
    
    def _calculate_cluster_benefit(self, cluster: Dict[str, Any]) -> float:
        """
        Calculer le bénéfice estimé pour un cluster de propriétés
        """
        # Simulation : plus grand cluster = plus grand bénéfice
        cluster_size = len(cluster['properties'])
        return cluster_size * 10000  # Valeur simulée
    
    def generate_domain_report(self) -> Dict[str, Any]:
        """
        Générer un rapport complet sur la gestion domaniale
        """
        report = {
            'report_date': datetime.now().isoformat(),
            'total_properties': len(self.domain_properties) if not self.domain_properties.empty else 0,
            'total_area_ha': self.domain_properties['area_ha'].sum() if not self.domain_properties.empty else 0,
            'concessions_analysis': self.analyze_concessions(),
            'strategic_zones': [],
            'optimization_opportunities': self.detect_optimization_opportunities(),
            'recommendations': []
        }
        
        # Ajouter les zones stratégiques
        strategic_zones = self.identify_strategic_zones()
        if not strategic_zones.empty:
            report['strategic_zones'] = strategic_zones.to_json()
        
        # Générer des recommandations
        report['recommendations'] = self._generate_recommendations(report)
        
        return report
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Générer des recommandations basées sur l'analyse
        """
        recommendations = []
        
        # Recommandations basées sur les opportunités d'optimisation
        opportunities = report['optimization_opportunities']
        if opportunities:
            if any(op['type'] == 'property_cluster_optimization' for op in opportunities):
                recommendations.append({
                    'priority': 'high',
                    'category': 'management',
                    'description': 'Implémenter une gestion intégrée pour les clusters de propriétés'
                })
            
            if any(op['type'] == 'underutilized_property' for op in opportunities):
                recommendations.append({
                    'priority': 'medium',
                    'category': 'development',
                    'description': 'Explorer les opportunités de valorisation pour les propriétés sous-exploitées'
                })
        
        # Recommandations basées sur les concessions
        concessions_analysis = report['concessions_analysis']
        if concessions_analysis.get('expiration_warnings'):
            recommendations.append({
                'priority': 'high',
                'category': 'renewal',
                'description': f'Renouvellement imminent de {len(concessions_analysis["expiration_warnings"])} concessions'
            })
        
        if concessions_analysis.get('performance_metrics', {}).get('utilization_rate', 0) < 0.7:
            recommendations.append({
                'priority': 'medium',
                'category': 'optimization',
                'description': 'Améliorer le taux d\'utilisation des concessions'
            })
        
        return recommendations
    
    def suggest_concession_optimization(self) -> List[Dict[str, Any]]:
        """
        Suggérer des optimisations pour les concessions existantes
        """
        suggestions = []
        
        for concession in self.concessions:
            if concession.get('status') == 'active':
                # Calculer des indicateurs de performance
                performance_score = self._calculate_concession_performance(concession)
                
                if performance_score < 0.5:  # Sous-performance
                    suggestions.append({
                        'concession_id': concession.get('id'),
                        'current_performance': performance_score,
                        'suggested_action': 'performance_review',
                        'justification': 'Performance inférieure au seuil minimal'
                    })
                elif performance_score > 0.8:  # Haute performance
                    suggestions.append({
                        'concession_id': concession.get('id'),
                        'current_performance': performance_score,
                        'suggested_action': 'extension_consideration',
                        'justification': 'Haute performance justifiant une extension éventuelle'
                    })
        
        return suggestions
    
    def _calculate_concession_performance(self, concession: Dict) -> float:
        """
        Calculer la performance d'une concession (simulation)
        """
        # Simulation basée sur plusieurs facteurs
        factors = [
            np.random.uniform(0.3, 1.0),  # Respect des obligations
            np.random.uniform(0.4, 1.0),  # Paiement des redevances
            np.random.uniform(0.2, 1.0),  # Impact environnemental
            np.random.uniform(0.5, 1.0)   # Valeur économique
        ]
        return sum(factors) / len(factors)


def main():
    """
    Fonction principale pour démontrer l'agent domanial
    """
    agent = DomainAgent()
    
    print("Agent Domanial pour les Workflows IA Géospatiaux")
    print("=" * 50)
    
    # Charger des données de test (simulation)
    # Dans une application réelle, ces données viendraient d'une base de données ou d'un fichier
    sample_properties = gpd.GeoDataFrame([
        {
            'id': 'DOM001',
            'geometry': Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),  # Simulation
            'category': 'coastal',
            'status': 'available',
            'area_ha': 50.0,
            'value': 250000,
            'management_entity': 'state'
        },
        {
            'id': 'DOM002',
            'geometry': Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
            'category': 'urban_perimeter',
            'status': 'leased',
            'area_ha': 25.0,
            'value': 150000,
            'management_entity': 'municipality'
        }
    ])
    
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
            'status': 'expired',
            'start_date': '2018-01-01T00:00:00Z',
            'end_date': '2023-01-01T00:00:00Z',
            'annual_fee': 5000,
            'area_ha': 25.0
        }
    ]
    
    # Charger les données
    agent.load_domain_properties(sample_properties)
    agent.load_concessions(sample_concessions)
    
    # Exécuter des analyses
    print("Analyse des concessions:")
    concessions_analysis = agent.analyze_concessions()
    print(json.dumps(concessions_analysis, indent=2, default=str))
    
    print("\nZones stratégiques identifiées:")
    strategic_zones = agent.identify_strategic_zones()
    print(f"Trouvé {len(strategic_zones)} zones stratégiques")
    
    print("\nOpportunités d'optimisation:")
    opportunities = agent.detect_optimization_opportunities()
    print(f"Identifié {len(opportunities)} opportunités")
    
    print("\nSuggestions de concession:")
    suggestions = agent.suggest_concession_optimization()
    print(f"{len(suggestions)} suggestions générées")
    
    print("\nRapport complet:")
    report = agent.generate_domain_report()
    print(f"Rapport généré le: {report['report_date']}")
    print(f"Recommandations: {len(report['recommendations'])}")


if __name__ == "__main__":
    main()