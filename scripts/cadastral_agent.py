#!/usr/bin/env python3
"""
Agent Cadastral pour les Workflows IA Géospatiaux
=================================================

Cet agent est spécialisé dans les traitements liés au cadastre et aux parcelles.
Il inclut des capacités d'analyse spatiale, de validation géométrique,
de détection de problèmes cadastraux, et de gestion des données foncières.
"""

import json
import os
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.validation import make_valid
from shapely.ops import unary_union
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
import warnings

class CadastralAgent:
    """
    Agent spécialisé dans les données cadastrales et les parcelles
    """
    
    def __init__(self):
        self.parcels = gpd.GeoDataFrame()
        self.properties = gpd.GeoDataFrame()
        self.ownerships = []
        self.cadastral_issues = []
        self.validation_rules = self._initialize_validation_rules()
    
    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """
        Initialiser les règles de validation cadastrale
        """
        return {
            'geometry_validity': True,
            'area_threshold': (10, 1000000),  # 10m² à 1km²
            'perimeter_area_ratio': 0.1,  # Pour détecter les formes étranges
            'minimum_adjacent_parcel_ratio': 0.05,  # 5% du périmètre doit toucher d'autres parcelles
            'overlap_threshold': 0.01  # 1% de chevauchement autorisé
        }
    
    def load_parcels(self, gdf: gpd.GeoDataFrame) -> bool:
        """
        Charger les données des parcelles cadastrales
        """
        try:
            # Vérifier que les propriétés requises existent
            required_cols = ['id', 'geometry', 'area', 'perimeter', 'owner_id', 'zone_type']
            if not all(col in gdf.columns for col in required_cols):
                missing_cols = set(required_cols) - set(gdf.columns)
                raise ValueError(f"Colonnes requises manquantes: {missing_cols}")
            
            # Convertir en GeoDataFrame si ce n'en est pas un
            if not isinstance(gdf, gpd.GeoDataFrame):
                gdf = gpd.GeoDataFrame(gdf)
            
            # S'assurer que la géométrie est valide
            gdf['geometry'] = gdf['geometry'].apply(lambda geom: make_valid(geom) if geom and not geom.is_valid else geom)
            
            self.parcels = gdf
            print(f"Chargé {len(self.parcels)} parcelles cadastrales")
            return True
        except Exception as e:
            print(f"Erreur lors du chargement des parcelles cadastrales: {e}")
            return False
    
    def load_properties(self, gdf: gpd.GeoDataFrame) -> bool:
        """
        Charger les données des propriétés associées aux parcelles
        """
        try:
            self.properties = gdf
            print(f"Chargé {len(self.properties)} propriétés")
            return True
        except Exception as e:
            print(f"Erreur lors du chargement des propriétés: {e}")
            return False
    
    def load_ownerships(self, ownerships_data: List[Dict]) -> bool:
        """
        Charger les données de propriété/droit
        """
        try:
            self.ownerships = ownerships_data
            print(f"Chargé {len(self.ownerships)} enregistrements de propriété")
            return True
        except Exception as e:
            print(f"Erreur lors du chargement des droits de propriété: {e}")
            return False
    
    def validate_parcels(self) -> List[Dict[str, Any]]:
        """
        Valider la géométrie et les propriétés des parcelles cadastrales
        """
        if self.parcels.empty:
            return []
        
        invalid_parcels = []
        
        for idx, parcel in self.parcels.iterrows():
            issues = []
            
            # Vérifier la validité de la géométrie
            if not parcel.geometry or not parcel.geometry.is_valid:
                issues.append({
                    'type': 'geometry_invalid',
                    'description': 'Géométrie invalide',
                    'severity': 'high'
                })
            
            # Vérifier la superficie
            area = parcel.get('area') or (parcel.geometry.area if parcel.geometry else 0)
            min_area, max_area = self.validation_rules['area_threshold']
            
            if area < min_area:
                issues.append({
                    'type': 'area_too_small',
                    'description': f'Superficie trop petite: {area}m²',
                    'severity': 'medium'
                })
            elif area > max_area:
                issues.append({
                    'type': 'area_too_large',
                    'description': f'Superficie trop grande: {area}m²',
                    'severity': 'medium'
                })
            
            # Vérifier le rapport périmètre/superficie (détecte formes étranges)
            perimeter = parcel.get('perimeter') or (parcel.geometry.length if parcel.geometry else 0)
            if area > 0:
                ratio = perimeter / (area ** 0.5)
                if ratio > self.validation_rules['perimeter_area_ratio']:
                    issues.append({
                        'type': 'irregular_shape',
                        'description': f'Forme irégulière - ratio périmètre/√aire: {ratio:.2f}',
                        'severity': 'low'
                    })
            
            # Vérifier les chevauchements avec d'autres parcelles
            overlapping_parcels = self._find_overlapping_parcels(idx, parcel)
            if overlapping_parcels:
                issues.append({
                    'type': 'overlapping_parcels',
                    'description': f'Chevauchement avec {len(overlapping_parcels)} autres parcelles: {", ".join(overlapping_parcels)}',
                    'severity': 'high'
                })
            
            # Vérifier les frontières avec d'autres parcelles
            adjacent_parcels = self._find_adjacent_parcels(idx, parcel)
            if len(adjacent_parcels) == 0 and area > 1000:  # Pour les grandes parcelles
                issues.append({
                    'type': 'island_parcel',
                    'description': 'Parcelle isolée sans voisin adjacent',
                    'severity': 'low'
                })
            
            if issues:
                invalid_parcels.append({
                    'parcel_id': parcel.get('id', f'parcel_{idx}'),
                    'issues': issues,
                    'geometry': parcel.geometry
                })
        
        self.cadastral_issues = invalid_parcels
        return invalid_parcels
    
    def _find_overlapping_parcels(self, current_idx: int, current_parcel) -> List[str]:
        """
        Trouver les parcelles qui se chevauchent avec la parcelle courante
        """
        overlapping = []
        
        if current_parcel.geometry:
            for idx, other_parcel in self.parcels.iterrows():
                if idx != current_idx and other_parcel.geometry:
                    intersection = current_parcel.geometry.intersection(other_parcel.geometry)
                    if intersection.area > self.validation_rules['overlap_threshold'] * current_parcel.geometry.area:
                        overlapping.append(other_parcel.get('id', f'parcel_{idx}'))
        
        return overlapping
    
    def _find_adjacent_parcels(self, current_idx: int, current_parcel) -> List[str]:
        """
        Trouver les parcelles adjacentes à la parcelle courante
        """
        adjacent = []
        
        if current_parcel.geometry:
            for idx, other_parcel in self.parcels.iterrows():
                if idx != current_idx and other_parcel.geometry:
                    # Vérifier si les parcelles se touchent
                    if current_parcel.geometry.touches(other_parcel.geometry) or \
                       not current_parcel.geometry.intersection(other_parcel.geometry).is_empty:
                        adjacent.append(other_parcel.get('id', f'parcel_{idx}'))
        
        return adjacent
    
    def detect_cadastral_anomalies(self) -> List[Dict[str, Any]]:
        """
        Détecter les anomalies cadastrales avancées
        """
        anomalies = []
        
        # Anomalies basées sur la topologie
        topology_anomalies = self._detect_topology_anomalies()
        anomalies.extend(topology_anomalies)
        
        # Anomalies basées sur les propriétés
        property_anomalies = self._detect_property_anomalies()
        anomalies.extend(property_anomalies)
        
        # Anomalies basées sur les clusters
        cluster_anomalies = self._detect_cluster_anomalies()
        anomalies.extend(cluster_anomalies)
        
        return anomalies
    
    def _detect_topology_anomalies(self) -> List[Dict[str, Any]]:
        """
        Détecter les anomalies topologiques
        """
        anomalies = []
        
        # Détecter les trous dans le tissu cadastral
        gaps = self._find_cadastral_gaps()
        for gap in gaps:
            anomalies.append({
                'type': 'cadastral_gap',
                'description': 'Trou dans le tissu cadastral',
                'geometry': gap,
                'severity': 'high'
            })
        
        # Détecter les superpositions multiples
        for idx, parcel in self.parcels.iterrows():
            overlapping = self._find_overlapping_parcels(idx, parcel)
            if len(overlapping) > 1:
                anomalies.append({
                    'type': 'multiple_overlap',
                    'description': f'Parcelle {parcel.get("id")} chevauchant {len(overlapping)} autres parcelles',
                    'parcel_id': parcel.get('id'),
                    'overlapping_parcels': overlapping,
                    'severity': 'high'
                })
        
        return anomalies
    
    def _find_cadastral_gaps(self) -> List[Polygon]:
        """
        Trouver les trous dans le tissu cadastral
        """
        if self.parcels.empty:
            return []
        
        # Union de toutes les parcelles pour former l'ensemble cadastrale
        try:
            union_geom = unary_union(self.parcels.geometry)
            
            if isinstance(union_geom, Polygon):
                # Si c'est un polygone simple, il n'y a pas de trous
                return []
            elif hasattr(union_geom, 'geoms'):  # MultiPolygon
                # Les trous seraient dans les polygones intérieurs
                gaps = []
                for geom in union_geom.geoms:
                    if isinstance(geom, Polygon) and len(geom.interiors) > 0:
                        for interior in geom.interiors:
                            gaps.append(Polygon(interior.coords))
                return gaps
            else:
                return []
        except Exception:
            return []
    
    def _detect_property_anomalies(self) -> List[Dict[str, Any]]:
        """
        Détecter les anomalies liées aux propriétés
        """
        anomalies = []
        
        # Vérifier les incohérences entre superficie cadastrale et déclarée
        if not self.properties.empty and not self.parcels.empty:
            for prop_idx, property in self.properties.iterrows():
                if 'parcel_id' in property and 'declared_area' in property:
                    parcel = self.parcels[self.parcels['id'] == property['parcel_id']]
                    if not parcel.empty:
                        cadastral_area = parcel.iloc[0]['area']
                        declared_area = property['declared_area']
                        
                        if abs(cadastral_area - declared_area) / declared_area > 0.1:  # 10% de différence
                            anomalies.append({
                                'type': 'area_mismatch',
                                'description': f'Différence d\'aire: cadastrale={cadastral_area:.2f}m², déclarée={declared_area:.2f}m²',
                                'parcel_id': property['parcel_id'],
                                'severity': 'medium'
                            })
        
        return anomalies
    
    def _detect_cluster_anomalies(self) -> List[Dict[str, Any]]:
        """
        Détecter les anomalies basées sur des clusters de parcelles
        """
        anomalies = []
        
        if self.parcels.empty or len(self.parcels) < 3:
            return anomalies
        
        # Extraire les coordonnées centrales des parcelles
        centroids = []
        parcel_ids = []
        
        for idx, parcel in self.parcels.iterrows():
            if parcel['geometry']:
                centroid = parcel['geometry'].centroid
                centroids.append([centroid.y, centroid.x])  # [lat, lon]
                parcel_ids.append(parcel.get('id', f'parcel_{idx}'))
        
        if len(centroids) < 3:
            return anomalies
        
        # Convertir en numpy array
        centroids = np.array(centroids)
        
        # Normaliser les coordonnées pour une distance homogène
        scaler = StandardScaler()
        centroids_scaled = scaler.fit_transform(centroids)
        
        # Appliquer DBSCAN pour trouver les clusters
        clustering = DBSCAN(eps=0.005, min_samples=2)  # Ajuster selon la zone
        cluster_labels = clustering.fit_predict(centroids_scaled)
        
        # Analyser les clusters pour trouver des anomalies
        for i, label in enumerate(cluster_labels):
            if label != -1:  # -1 indique les points bruit
                # Pour chaque cluster, vérifier l'homogénéité des types
                cluster_parcels = self.parcels.iloc[np.where(cluster_labels == label)[0]]
                
                # Vérifier la diversité des types de zones
                zone_types = cluster_parcels['zone_type'].unique()
                
                if len(zone_types) > 2:  # Trop de diversité pour un cluster
                    anomalies.append({
                        'type': 'heterogeneous_cluster',
                        'description': f'Cluster {label} contient {len(zone_types)} types de zones différents',
                        'parcel_ids': cluster_parcels['id'].tolist(),
                        'zone_types': zone_types.tolist(),
                        'severity': 'low'
                    })
        
        return anomalies
    
    def consolidate_parcels(self, consolidation_criteria: Dict[str, Any] = None) -> gpd.GeoDataFrame:
        """
        Consolider les parcelles selon des critères spécifiques
        """
        if self.parcels.empty:
            return gpd.GeoDataFrame()
        
        if consolidation_criteria is None:
            consolidation_criteria = {
                'owner_similarity': True,
                'zone_type_similarity': True,
                'max_distance': 50,  # mètres
                'area_threshold': 1000  # m²
            }
        
        consolidated_parcels = []
        
        # Créer des groupes de parcelles à consolider
        groups = self._create_consolidation_groups(consolidation_criteria)
        
        for group_id, parcel_indices in groups.items():
            if len(parcel_indices) > 1:  # Seulement consolider s'il y a plusieurs parcelles
                # Fusionner les géométries
                group_geoms = [self.parcels.iloc[i]['geometry'] for i in parcel_indices]
                if group_geoms:
                    try:
                        consolidated_geom = unary_union(group_geoms)
                        
                        # Calculer les propriétés consolidées
                        total_area = sum(self.parcels.iloc[i]['area'] for i in parcel_indices)
                        avg_owner = self.parcels.iloc[parcel_indices[0]].get('owner_id', 'unknown')
                        avg_zone_type = self.parcels.iloc[parcel_indices[0]].get('zone_type', 'mixed')
                        
                        consolidated_parcels.append({
                            'id': f'consolidated_{group_id}',
                            'geometry': consolidated_geom,
                            'area': total_area,
                            'perimeter': consolidated_geom.length if hasattr(consolidated_geom, 'length') else 0,
                            'owner_id': avg_owner,
                            'zone_type': avg_zone_type,
                            'original_parcels': [self.parcels.iloc[i]['id'] for i in parcel_indices],
                            'consolidation_date': datetime.now().isoformat()
                        })
                    except Exception:
                        # Si la consolidation échoue, conserver les parcelles originales
                        for idx in parcel_indices:
                            consolidated_parcels.append(self.parcels.iloc[idx])
            else:
                # Si le groupe n'a qu'une parcelle, l'ajouter telle quelle
                for idx in parcel_indices:
                    consolidated_parcels.append(self.parcels.iloc[idx])
        
        if consolidated_parcels:
            # Convertir en GeoDataFrame
            consolidated_gdf = gpd.GeoDataFrame(consolidated_parcels)
            return consolidated_gdf
        else:
            return gpd.GeoDataFrame()
    
    def _create_consolidation_groups(self, criteria: Dict[str, Any]) -> Dict[str, List[int]]:
        """
        Créer des groupes de parcelles à consolider
        """
        groups = {}
        group_id = 0
        
        # Pour simplifier, on va regrouper les parcelles adjacentes avec des propriétés similaires
        for idx, parcel in self.parcels.iterrows():
            # Vérifier si cette parcelle peut être ajoutée à un groupe existant
            added_to_group = False
            
            for existing_group_id, existing_indices in groups.items():
                # Vérifier les critères de regroupement
                can_group = True
                
                # Si on vérifie la similitude du propriétaire
                if criteria.get('owner_similarity', True):
                    existing_parcel_idx = existing_indices[0]
                    if self.parcels.iloc[existing_parcel_idx]['owner_id'] != parcel['owner_id']:
                        can_group = False
                
                # Si on vérifie la similitude du type de zone
                if can_group and criteria.get('zone_type_similarity', True):
                    existing_parcel_idx = existing_indices[0]
                    if self.parcels.iloc[existing_parcel_idx]['zone_type'] != parcel['zone_type']:
                        can_group = False
                
                # Si on vérifie la distance
                if can_group and criteria.get('max_distance', 50) > 0:
                    adjacent = False
                    for existing_idx in existing_indices:
                        existing_parcel = self.parcels.iloc[existing_idx]
                        if parcel['geometry'].distance(existing_parcel['geometry']) <= criteria['max_distance']:
                            adjacent = True
                            break
                    if not adjacent:
                        can_group = False
                
                if can_group:
                    groups[existing_group_id].append(idx)
                    added_to_group = True
                    break
            
            # Si la parcelle n'a pas été ajoutée à un groupe existant, créer un nouveau groupe
            if not added_to_group:
                groups[group_id] = [idx]
                group_id += 1
        
        return groups
    
    def generate_cadastral_report(self) -> Dict[str, Any]:
        """
        Générer un rapport complet sur le cadastre
        """
        report = {
            'report_date': datetime.now().isoformat(),
            'total_parcels': len(self.parcels),
            'total_area_ha': self.parcels['area'].sum() / 10000 if not self.parcels.empty else 0,
            'validation_results': self.validate_parcels(),
            'cadastral_anomalies': self.detect_cadastral_anomalies(),
            'consolidation_opportunities': len(self._create_consolidation_groups({})),
            'statistics': self._calculate_cadastral_statistics(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _calculate_cadastral_statistics(self) -> Dict[str, Any]:
        """
        Calculer les statistiques cadastrales
        """
        if self.parcels.empty:
            return {}
        
        stats = {
            'average_area': self.parcels['area'].mean(),
            'median_area': self.parcels['area'].median(),
            'total_area': self.parcels['area'].sum(),
            'area_std': self.parcels['area'].std(),
            'zone_type_distribution': self.parcels['zone_type'].value_counts().to_dict(),
            'owner_distribution': self.parcels['owner_id'].value_counts().to_dict() if 'owner_id' in self.parcels.columns else {}
        }
        
        return stats
    
    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """
        Générer des recommandations pour l'amélioration du cadastre
        """
        recommendations = []
        
        # Vérifier les problèmes identifiés
        validation_issues = self.validate_parcels()
        anomalies = self.detect_cadastral_anomalies()
        
        if validation_issues:
            recommendations.append({
                'priority': 'high',
                'category': 'data_quality',
                'description': f'Corriger {len(validation_issues)} parcelles avec des problèmes de validation'
            })
        
        if anomalies:
            recommendations.append({
                'priority': 'medium',
                'category': 'data_integrity',
                'description': f'Analyser {len(anomalies)} anomalies cadastrales identifiées'
            })
        
        # Vérifier les opportunités de consolidation
        consolidation_groups = self._create_consolidation_groups({})
        if len(consolidation_groups) > len(self.parcels) * 0.7:  # Si beaucoup de petits groupes
            recommendations.append({
                'priority': 'low',
                'category': 'optimization',
                'description': 'Explorer les opportunités de consolidation des parcelles adjacentes'
            })
        
        # Vérifier la distribution des tailles
        if not self.parcels.empty:
            avg_area = self.parcels['area'].mean()
            if avg_area < 100:  # Moins de 100m² en moyenne
                recommendations.append({
                    'priority': 'medium',
                    'category': 'parcelation',
                    'description': 'Analyser la fragmentation excessive des parcelles (taille moyenne < 100m²)'
                })
        
        return recommendations
    
    def predict_parcel_values(self) -> List[Dict[str, Any]]:
        """
        Prédire les valeurs des parcelles basé sur divers facteurs
        """
        predictions = []
        
        for idx, parcel in self.parcels.iterrows():
            value = self._calculate_parcel_value(parcel)
            predictions.append({
                'parcel_id': parcel.get('id'),
                'predicted_value': value,
                'factors': {
                    'area_factor': parcel.get('area', 0) * 10,  # Simulation
                    'location_factor': np.random.uniform(0.8, 1.2),  # Simulation
                    'zone_factor': self._get_zone_value_multiplier(parcel.get('zone_type', ''))
                }
            })
        
        return predictions
    
    def _calculate_parcel_value(self, parcel: pd.Series) -> float:
        """
        Calculer la valeur estimée d'une parcelle
        """
        # Simulation de calcul de valeur basé sur plusieurs facteurs
        base_value = parcel.get('area', 0) * 20  # 20€/m² de base
        location_factor = np.random.uniform(0.5, 2.0)  # Facteur de localisation
        zone_multiplier = self._get_zone_value_multiplier(parcel.get('zone_type', ''))
        owner_factor = 1.0  # Facteur basé sur le type de propriétaire (simplifié)
        
        estimated_value = base_value * location_factor * zone_multiplier * owner_factor
        return max(estimated_value, 1000)  # Valeur minimum de 1000€
    
    def _get_zone_value_multiplier(self, zone_type: str) -> float:
        """
        Obtenir le multiplicateur de valeur basé sur le type de zone
        """
        multipliers = {
            'urban': 5.0,
            'urban_perimeter': 3.0,
            'agricultural': 1.0,
            'forest': 0.5,
            'industrial': 2.0,
            'coastal': 4.0,
            'protected': 0.1
        }
        return multipliers.get(zone_type, 1.0)
    
    def detect_land_use_changes(self, previous_parcels: gpd.GeoDataFrame = None) -> List[Dict[str, Any]]:
        """
        Détecter les changements d'usage du sol entre deux jeux de données
        """
        changes = []
        
        if previous_parcels is None or previous_parcels.empty:
            # Si aucune donnée précédente, on ne peut pas détecter de changements
            return changes
        
        # Pour une implémentation complète, on devrait comparer les parcelles
        # par ID ou par géométrie pour identifier les modifications
        # Pour cette simulation, nous allons générer des changements aléatoires
        
        current_ids = set(self.parcels['id']) if 'id' in self.parcels.columns else set()
        previous_ids = set(previous_parcels['id']) if 'id' in previous_parcels.columns else set()
        
        # Nouvelles parcelles
        new_parcels = current_ids - previous_ids
        for parcel_id in new_parcels:
            changes.append({
                'type': 'new_parcel',
                'parcel_id': parcel_id,
                'change_description': 'Nouvelle parcelle cadastrale',
                'severity': 'information'
            })
        
        # Parcelles supprimées
        removed_parcels = previous_ids - current_ids
        for parcel_id in removed_parcels:
            changes.append({
                'type': 'removed_parcel',
                'parcel_id': parcel_id,
                'change_description': 'Parcelle supprimée du cadastre',
                'severity': 'information'
            })
        
        # Pour les parcelles existantes dans les deux jeux, vérifier les changements
        common_ids = current_ids & previous_ids
        for parcel_id in common_ids:
            current_parcel = self.parcels[self.parcels['id'] == parcel_id].iloc[0]
            previous_parcel = previous_parcels[previous_parcels['id'] == parcel_id].iloc[0]
            
            # Comparer les types de zones
            if current_parcel.get('zone_type') != previous_parcel.get('zone_type'):
                changes.append({
                    'type': 'zone_change',
                    'parcel_id': parcel_id,
                    'change_description': f'Changement de type de zone: {previous_parcel.get("zone_type")} → {current_parcel.get("zone_type")}',
                    'severity': 'medium'
                })
        
        return changes


def main():
    """
    Fonction principale pour démontrer l'agent cadastral
    """
    agent = CadastralAgent()
    
    print("Agent Cadastral pour les Workflows IA Géospatiaux")
    print("=" * 50)
    
    # Charger des données de test (simulation)
    sample_parcels = gpd.GeoDataFrame([
        {
            'id': 'PAR001',
            'geometry': Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),  # 100m²
            'area': 100,
            'perimeter': 40,
            'owner_id': 'OWNER001',
            'zone_type': 'urban'
        },
        {
            'id': 'PAR002',
            'geometry': Polygon([(12, 0), (22, 0), (22, 10), (12, 10)]),  # 100m²
            'area': 100,
            'perimeter': 40,
            'owner_id': 'OWNER002',
            'zone_type': 'agricultural'
        },
        {
            'id': 'PAR003',
            'geometry': Polygon([(5, 12), (15, 12), (15, 22), (5, 22)]),  # 100m²
            'area': 100,
            'perimeter': 40,
            'owner_id': 'OWNER001',
            'zone_type': 'urban'
        }
    ])
    
    sample_properties = gpd.GeoDataFrame([
        {
            'parcel_id': 'PAR001',
            'declared_area': 95,
            'building_area': 50,
            'usage': 'residential'
        },
        {
            'parcel_id': 'PAR002',
            'declared_area': 100,
            'building_area': 10,
            'usage': 'agricultural'
        }
    ])
    
    sample_ownerships = [
        {
            'parcel_id': 'PAR001',
            'owner_id': 'OWNER001',
            'ownership_type': 'full',
            'registration_date': '2020-01-01T00:00:00Z'
        },
        {
            'parcel_id': 'PAR002',
            'owner_id': 'OWNER002',
            'ownership_type': 'full',
            'registration_date': '2019-05-15T00:00:00Z'
        }
    ]
    
    # Charger les données
    agent.load_parcels(sample_parcels)
    agent.load_properties(sample_properties)
    agent.load_ownerships(sample_ownerships)
    
    # Exécuter des analyses
    print("Validation des parcelles:")
    validation_results = agent.validate_parcels()
    print(f"{len(validation_results)} parcelles avec des problèmes identifiés")
    
    print("\nAnomalies cadastrales:")
    anomalies = agent.detect_cadastral_anomalies()
    print(f"{len(anomalies)} anomalies identifiées")
    
    print("\nPrédictions de valeur:")
    predictions = agent.predict_parcel_values()
    print(f"Valeur moyenne prédite: {np.mean([p['predicted_value'] for p in predictions]):.2f}€")
    
    print("\nRapport cadastral:")
    report = agent.generate_cadastral_report()
    print(f"Rapport généré le: {report['report_date']}")
    print(f"Recommandations: {len(report['recommendations'])}")
    
    print("\nOpportunités de consolidation:")
    consolidated = agent.consolidate_parcels()
    print(f"{len(consolidated)} parcelles après consolidation")


if __name__ == "__main__":
    main()