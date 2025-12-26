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
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.validation import make_valid
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings


# =============================================================================
# ENUMS ET CONSTANTES
# =============================================================================

class ZoneType(Enum):
    """Types de zones cadastrales"""
    URBAN = "urban"
    URBAN_PERIMETER = "urban_perimeter"
    AGRICULTURAL = "agricultural"
    FOREST = "forest"
    INDUSTRIAL = "industrial"
    COASTAL = "coastal"
    PROTECTED = "protected"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class IssueSeverity(Enum):
    """Niveaux de sévérité des problèmes"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    INFORMATION = "information"


class AnomalyType(Enum):
    """Types d'anomalies cadastrales"""
    GEOMETRY_INVALID = "geometry_invalid"
    AREA_TOO_SMALL = "area_too_small"
    AREA_TOO_LARGE = "area_too_large"
    IRREGULAR_SHAPE = "irregular_shape"
    OVERLAPPING_PARCELS = "overlapping_parcels"
    ISLAND_PARCEL = "island_parcel"
    CADASTRAL_GAP = "cadastral_gap"
    MULTIPLE_OVERLAP = "multiple_overlap"
    AREA_MISMATCH = "area_mismatch"
    HETEROGENEOUS_CLUSTER = "heterogeneous_cluster"
    ZONE_CHANGE = "zone_change"
    NEW_PARCEL = "new_parcel"
    REMOVED_PARCEL = "removed_parcel"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ValidationRules:
    """Règles de validation cadastrale"""
    geometry_validity: bool = True
    min_area: float = 10.0  # m²
    max_area: float = 1_000_000.0  # m² (1 km²)
    perimeter_area_ratio_max: float = 8.0  # Un cercle ~3.54, carré ~4, formes irrégulières > 6-8
    minimum_adjacent_parcel_ratio: float = 0.05  # 5% du périmètre
    overlap_threshold: float = 0.01  # 1% de chevauchement autorisé
    area_mismatch_tolerance: float = 0.10  # 10% de différence tolérée


@dataclass
class CadastralIssue:
    """Représente un problème cadastral"""
    issue_type: AnomalyType
    description: str
    severity: IssueSeverity
    parcel_id: Optional[str] = None
    geometry: Optional[Any] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsolidationCriteria:
    """Critères de consolidation des parcelles"""
    owner_similarity: bool = True
    zone_type_similarity: bool = True
    max_distance: float = 50.0  # mètres
    min_area_threshold: float = 1000.0  # m²


# =============================================================================
# AGENT CADASTRAL
# =============================================================================

class CadastralAgent:
    """
    Agent spécialisé dans les données cadastrales et les parcelles
    """
    
    # Colonnes requises pour les parcelles
    REQUIRED_PARCEL_COLUMNS = {'id', 'geometry', 'area', 'perimeter', 'owner_id', 'zone_type'}
    
    # CRS par défaut (Lambert 93 pour la France)
    DEFAULT_CRS = "EPSG:2154"
    
    def __init__(self, target_crs: str = None):
        """
        Initialiser l'agent cadastral
        
        Args:
            target_crs: Système de coordonnées cible (défaut: EPSG:2154 Lambert 93)
        """
        self.parcels: gpd.GeoDataFrame = gpd.GeoDataFrame()
        self.properties: gpd.GeoDataFrame = gpd.GeoDataFrame()
        self.ownerships: List[Dict] = []
        self.target_crs = target_crs or self.DEFAULT_CRS
        
        # Cache pour les résultats de validation
        self._validation_cache: Optional[List[Dict[str, Any]]] = None
        self._anomalies_cache: Optional[List[Dict[str, Any]]] = None
        self._cache_valid: bool = False
        
        # Règles de validation
        self.validation_rules = ValidationRules()
        
        # Multiplicateurs de valeur par type de zone
        self._zone_value_multipliers: Dict[str, float] = {
            ZoneType.URBAN.value: 5.0,
            ZoneType.URBAN_PERIMETER.value: 3.0,
            ZoneType.AGRICULTURAL.value: 1.0,
            ZoneType.FOREST.value: 0.5,
            ZoneType.INDUSTRIAL.value: 2.0,
            ZoneType.COASTAL.value: 4.0,
            ZoneType.PROTECTED.value: 0.1,
            ZoneType.MIXED.value: 1.5,
            ZoneType.UNKNOWN.value: 1.0,
        }
    
    def _invalidate_cache(self) -> None:
        """Invalider le cache après modification des données"""
        self._validation_cache = None
        self._anomalies_cache = None
        self._cache_valid = False
    
    def _ensure_spatial_index(self) -> None:
        """S'assurer que l'index spatial est créé"""
        if not self.parcels.empty and not hasattr(self.parcels, 'sindex'):
            # Force la création de l'index spatial
            _ = self.parcels.sindex
    
    # =========================================================================
    # CHARGEMENT DES DONNÉES
    # =========================================================================
    
    def load_parcels(
        self, 
        gdf: gpd.GeoDataFrame, 
        target_crs: str = None,
        validate_columns: bool = True
    ) -> bool:
        """
        Charger les données des parcelles cadastrales
        
        Args:
            gdf: GeoDataFrame des parcelles
            target_crs: Système de coordonnées cible (optionnel)
            validate_columns: Vérifier la présence des colonnes requises
            
        Returns:
            True si le chargement a réussi, False sinon
        """
        try:
            # Convertir en GeoDataFrame si nécessaire
            if not isinstance(gdf, gpd.GeoDataFrame):
                gdf = gpd.GeoDataFrame(gdf)
            
            # Vérifier les colonnes requises
            if validate_columns:
                missing_cols = self.REQUIRED_PARCEL_COLUMNS - set(gdf.columns)
                if missing_cols:
                    raise ValueError(f"Colonnes requises manquantes: {missing_cols}")
            
            # Gérer le CRS
            crs = target_crs or self.target_crs
            if gdf.crs is None:
                warnings.warn("CRS non défini, assumant EPSG:4326")
                gdf = gdf.set_crs("EPSG:4326")
            
            if str(gdf.crs) != crs:
                gdf = gdf.to_crs(crs)
            
            # Valider et corriger les géométries
            gdf = gdf.copy()
            gdf['geometry'] = gdf['geometry'].apply(self._validate_geometry)
            
            self.parcels = gdf
            self.target_crs = crs
            self._invalidate_cache()
            self._ensure_spatial_index()
            
            print(f"Chargé {len(self.parcels)} parcelles cadastrales (CRS: {crs})")
            return True
            
        except Exception as e:
            print(f"Erreur lors du chargement des parcelles cadastrales: {e}")
            return False
    
    def _validate_geometry(self, geom) -> Optional[Any]:
        """Valider et corriger une géométrie si nécessaire"""
        if geom is None:
            return None
        if not geom.is_valid:
            return make_valid(geom)
        return geom
    
    def load_properties(self, gdf: gpd.GeoDataFrame) -> bool:
        """
        Charger les données des propriétés associées aux parcelles
        
        Args:
            gdf: GeoDataFrame des propriétés
            
        Returns:
            True si le chargement a réussi, False sinon
        """
        try:
            if not isinstance(gdf, gpd.GeoDataFrame):
                gdf = gpd.GeoDataFrame(gdf)
            
            self.properties = gdf
            self._invalidate_cache()
            print(f"Chargé {len(self.properties)} propriétés")
            return True
            
        except Exception as e:
            print(f"Erreur lors du chargement des propriétés: {e}")
            return False
    
    def load_ownerships(self, ownerships_data: List[Dict]) -> bool:
        """
        Charger les données de propriété/droit
        
        Args:
            ownerships_data: Liste des enregistrements de propriété
            
        Returns:
            True si le chargement a réussi, False sinon
        """
        try:
            self.ownerships = ownerships_data.copy()
            self._invalidate_cache()
            print(f"Chargé {len(self.ownerships)} enregistrements de propriété")
            return True
            
        except Exception as e:
            print(f"Erreur lors du chargement des droits de propriété: {e}")
            return False
    
    # =========================================================================
    # VALIDATION DES PARCELLES
    # =========================================================================
    
    def validate_parcels(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Valider la géométrie et les propriétés des parcelles cadastrales
        
        Args:
            force_refresh: Forcer le recalcul même si le cache est valide
            
        Returns:
            Liste des parcelles invalides avec leurs problèmes
        """
        # Utiliser le cache si disponible
        if self._validation_cache is not None and not force_refresh:
            return self._validation_cache
        
        if self.parcels.empty:
            return []
        
        self._ensure_spatial_index()
        invalid_parcels = []
        
        for idx, parcel in self.parcels.iterrows():
            issues = self._validate_single_parcel(idx, parcel)
            
            if issues:
                invalid_parcels.append({
                    'parcel_id': parcel.get('id', f'parcel_{idx}'),
                    'issues': [self._issue_to_dict(issue) for issue in issues],
                    'geometry': parcel.geometry
                })
        
        self._validation_cache = invalid_parcels
        return invalid_parcels
    
    def _validate_single_parcel(self, idx: int, parcel: pd.Series) -> List[CadastralIssue]:
        """
        Valider une seule parcelle
        
        Args:
            idx: Index de la parcelle
            parcel: Données de la parcelle
            
        Returns:
            Liste des problèmes détectés
        """
        issues = []
        geometry = parcel.get('geometry')
        parcel_id = parcel.get('id', f'parcel_{idx}')
        
        # 1. Vérifier la validité de la géométrie
        if not geometry or not geometry.is_valid:
            issues.append(CadastralIssue(
                issue_type=AnomalyType.GEOMETRY_INVALID,
                description='Géométrie invalide ou absente',
                severity=IssueSeverity.HIGH,
                parcel_id=parcel_id
            ))
            return issues  # Pas besoin de continuer si la géométrie est invalide
        
        # 2. Vérifier la superficie
        area = parcel.get('area') or geometry.area
        
        if area < self.validation_rules.min_area:
            issues.append(CadastralIssue(
                issue_type=AnomalyType.AREA_TOO_SMALL,
                description=f'Superficie trop petite: {area:.2f}m² (min: {self.validation_rules.min_area}m²)',
                severity=IssueSeverity.MEDIUM,
                parcel_id=parcel_id,
                additional_data={'area': area}
            ))
        elif area > self.validation_rules.max_area:
            issues.append(CadastralIssue(
                issue_type=AnomalyType.AREA_TOO_LARGE,
                description=f'Superficie trop grande: {area:.2f}m² (max: {self.validation_rules.max_area}m²)',
                severity=IssueSeverity.MEDIUM,
                parcel_id=parcel_id,
                additional_data={'area': area}
            ))
        
        # 3. Vérifier le rapport périmètre/superficie (formes irrégulières)
        perimeter = parcel.get('perimeter') or geometry.length
        if area > 0:
            # Un cercle a un ratio de ~3.54, un carré ~4
            # Des formes très irrégulières ont des ratios > 6-8
            ratio = perimeter / (area ** 0.5)
            if ratio > self.validation_rules.perimeter_area_ratio_max:
                issues.append(CadastralIssue(
                    issue_type=AnomalyType.IRREGULAR_SHAPE,
                    description=f'Forme irrégulière - ratio: {ratio:.2f} (seuil: {self.validation_rules.perimeter_area_ratio_max})',
                    severity=IssueSeverity.LOW,
                    parcel_id=parcel_id,
                    additional_data={'ratio': ratio}
                ))
        
        # 4. Vérifier les chevauchements
        overlapping_parcels = self._find_overlapping_parcels(idx, parcel)
        if overlapping_parcels:
            issues.append(CadastralIssue(
                issue_type=AnomalyType.OVERLAPPING_PARCELS,
                description=f'Chevauchement avec {len(overlapping_parcels)} parcelle(s): {", ".join(overlapping_parcels[:5])}{"..." if len(overlapping_parcels) > 5 else ""}',
                severity=IssueSeverity.HIGH,
                parcel_id=parcel_id,
                additional_data={'overlapping_ids': overlapping_parcels}
            ))
        
        # 5. Vérifier l'isolation (parcelles sans voisins)
        if area > 1000:  # Pour les grandes parcelles seulement
            adjacent_parcels = self._find_adjacent_parcels(idx, parcel)
            if len(adjacent_parcels) == 0:
                issues.append(CadastralIssue(
                    issue_type=AnomalyType.ISLAND_PARCEL,
                    description='Parcelle isolée sans voisin adjacent',
                    severity=IssueSeverity.LOW,
                    parcel_id=parcel_id
                ))
        
        return issues
    
    def _issue_to_dict(self, issue: CadastralIssue) -> Dict[str, Any]:
        """Convertir un CadastralIssue en dictionnaire"""
        return {
            'type': issue.issue_type.value,
            'description': issue.description,
            'severity': issue.severity.value,
            **issue.additional_data
        }
    
    def _find_overlapping_parcels(self, current_idx: int, current_parcel: pd.Series) -> List[str]:
        """
        Trouver les parcelles qui se chevauchent avec la parcelle courante
        Utilise l'index spatial R-tree pour l'optimisation
        
        Args:
            current_idx: Index de la parcelle courante
            current_parcel: Données de la parcelle courante
            
        Returns:
            Liste des IDs des parcelles en chevauchement
        """
        overlapping = []
        geometry = current_parcel.get('geometry')
        
        if not geometry:
            return overlapping
        
        # Utiliser l'index spatial pour trouver les candidats potentiels
        possible_matches_idx = list(self.parcels.sindex.intersection(geometry.bounds))
        
        for idx in possible_matches_idx:
            if idx == current_idx:
                continue
                
            other_parcel = self.parcels.iloc[idx]
            other_geometry = other_parcel.get('geometry')
            
            if not other_geometry:
                continue
            
            try:
                intersection = geometry.intersection(other_geometry)
                overlap_ratio = intersection.area / geometry.area if geometry.area > 0 else 0
                
                if overlap_ratio > self.validation_rules.overlap_threshold:
                    overlapping.append(other_parcel.get('id', f'parcel_{idx}'))
            except Exception:
                # Gérer les erreurs de géométrie silencieusement
                continue
        
        return overlapping
    
    def _find_adjacent_parcels(self, current_idx: int, current_parcel: pd.Series) -> List[str]:
        """
        Trouver les parcelles adjacentes à la parcelle courante
        Utilise l'index spatial R-tree pour l'optimisation
        
        Args:
            current_idx: Index de la parcelle courante
            current_parcel: Données de la parcelle courante
            
        Returns:
            Liste des IDs des parcelles adjacentes
        """
        adjacent = []
        geometry = current_parcel.get('geometry')
        
        if not geometry:
            return adjacent
        
        # Utiliser l'index spatial avec un buffer pour trouver les candidats
        buffered_bounds = geometry.buffer(1).bounds  # Buffer de 1m pour les adjacences
        possible_matches_idx = list(self.parcels.sindex.intersection(buffered_bounds))
        
        for idx in possible_matches_idx:
            if idx == current_idx:
                continue
                
            other_parcel = self.parcels.iloc[idx]
            other_geometry = other_parcel.get('geometry')
            
            if not other_geometry:
                continue
            
            try:
                # Vérifier si les parcelles sont adjacentes (se touchent ou ont une intersection linéaire)
                if geometry.intersects(other_geometry):
                    intersection = geometry.intersection(other_geometry)
                    # Adjacent si l'intersection n'a pas de surface (ligne, point)
                    if intersection.area == 0 or intersection.area < self.validation_rules.overlap_threshold * geometry.area:
                        adjacent.append(other_parcel.get('id', f'parcel_{idx}'))
            except Exception:
                continue
        
        return adjacent
    
    # =========================================================================
    # DÉTECTION D'ANOMALIES
    # =========================================================================
    
    def detect_cadastral_anomalies(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Détecter les anomalies cadastrales avancées
        
        Args:
            force_refresh: Forcer le recalcul même si le cache est valide
            
        Returns:
            Liste des anomalies détectées
        """
        if self._anomalies_cache is not None and not force_refresh:
            return self._anomalies_cache
        
        anomalies = []
        
        # Anomalies topologiques
        anomalies.extend(self._detect_topology_anomalies())
        
        # Anomalies de propriétés
        anomalies.extend(self._detect_property_anomalies())
        
        # Anomalies de clusters
        anomalies.extend(self._detect_cluster_anomalies())
        
        self._anomalies_cache = anomalies
        return anomalies
    
    def _detect_topology_anomalies(self) -> List[Dict[str, Any]]:
        """Détecter les anomalies topologiques"""
        anomalies = []
        
        if self.parcels.empty:
            return anomalies
        
        # Détecter les trous dans le tissu cadastral
        gaps = self._find_cadastral_gaps()
        for i, gap in enumerate(gaps):
            anomalies.append({
                'type': AnomalyType.CADASTRAL_GAP.value,
                'description': f'Trou dans le tissu cadastral (#{i+1})',
                'geometry': gap,
                'severity': IssueSeverity.HIGH.value,
                'area': gap.area if gap else 0
            })
        
        # Détecter les superpositions multiples
        self._ensure_spatial_index()
        for idx, parcel in self.parcels.iterrows():
            overlapping = self._find_overlapping_parcels(idx, parcel)
            if len(overlapping) > 1:
                anomalies.append({
                    'type': AnomalyType.MULTIPLE_OVERLAP.value,
                    'description': f'Parcelle chevauchant {len(overlapping)} autres parcelles',
                    'parcel_id': parcel.get('id'),
                    'overlapping_parcels': overlapping,
                    'severity': IssueSeverity.HIGH.value
                })
        
        return anomalies
    
    def _find_cadastral_gaps(self) -> List[Polygon]:
        """Trouver les trous dans le tissu cadastral"""
        if self.parcels.empty:
            return []
        
        try:
            # Union de toutes les parcelles
            valid_geoms = [g for g in self.parcels.geometry if g is not None and g.is_valid]
            if not valid_geoms:
                return []
            
            union_geom = unary_union(valid_geoms)
            gaps = []
            
            if isinstance(union_geom, Polygon):
                # Extraire les trous intérieurs du polygone
                for interior in union_geom.interiors:
                    gaps.append(Polygon(interior.coords))
            elif hasattr(union_geom, 'geoms'):  # MultiPolygon
                for geom in union_geom.geoms:
                    if isinstance(geom, Polygon):
                        for interior in geom.interiors:
                            gaps.append(Polygon(interior.coords))
            
            return gaps
            
        except Exception as e:
            warnings.warn(f"Erreur lors de la recherche des trous cadastraux: {e}")
            return []
    
    def _detect_property_anomalies(self) -> List[Dict[str, Any]]:
        """Détecter les anomalies liées aux propriétés"""
        anomalies = []
        
        if self.properties.empty or self.parcels.empty:
            return anomalies
        
        # Vérifier les incohérences entre superficie cadastrale et déclarée
        for _, prop in self.properties.iterrows():
            parcel_id = prop.get('parcel_id')
            declared_area = prop.get('declared_area')
            
            if not parcel_id or not declared_area:
                continue
            
            matching_parcels = self.parcels[self.parcels['id'] == parcel_id]
            if matching_parcels.empty:
                continue
            
            cadastral_area = matching_parcels.iloc[0].get('area', 0)
            
            if declared_area > 0:
                difference_ratio = abs(cadastral_area - declared_area) / declared_area
                
                if difference_ratio > self.validation_rules.area_mismatch_tolerance:
                    anomalies.append({
                        'type': AnomalyType.AREA_MISMATCH.value,
                        'description': f'Différence d\'aire: cadastrale={cadastral_area:.2f}m², déclarée={declared_area:.2f}m² ({difference_ratio*100:.1f}%)',
                        'parcel_id': parcel_id,
                        'cadastral_area': cadastral_area,
                        'declared_area': declared_area,
                        'difference_ratio': difference_ratio,
                        'severity': IssueSeverity.MEDIUM.value
                    })
        
        return anomalies
    
    def _detect_cluster_anomalies(self) -> List[Dict[str, Any]]:
        """Détecter les anomalies basées sur des clusters de parcelles"""
        anomalies = []
        
        if self.parcels.empty or len(self.parcels) < 3:
            return anomalies
        
        # Extraire les coordonnées des centroïdes
        centroids = []
        parcel_indices = []
        
        for idx, parcel in self.parcels.iterrows():
            geometry = parcel.get('geometry')
            if geometry and geometry.is_valid:
                centroid = geometry.centroid
                centroids.append([centroid.y, centroid.x])
                parcel_indices.append(idx)
        
        if len(centroids) < 3:
            return anomalies
        
        centroids = np.array(centroids)
        
        # Normaliser les coordonnées
        scaler = StandardScaler()
        centroids_scaled = scaler.fit_transform(centroids)
        
        # Calculer eps dynamiquement basé sur la distribution des distances
        eps = self._calculate_dbscan_eps(centroids_scaled)
        
        # Appliquer DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=2)
        cluster_labels = clustering.fit_predict(centroids_scaled)
        
        # Analyser les clusters
        unique_labels = set(cluster_labels) - {-1}  # Exclure le bruit
        
        for label in unique_labels:
            cluster_mask = cluster_labels == label
            cluster_indices = [parcel_indices[i] for i, is_in_cluster in enumerate(cluster_mask) if is_in_cluster]
            cluster_parcels = self.parcels.iloc[cluster_indices]
            
            # Vérifier la diversité des types de zones
            zone_types = cluster_parcels['zone_type'].unique()
            
            if len(zone_types) > 2:
                anomalies.append({
                    'type': AnomalyType.HETEROGENEOUS_CLUSTER.value,
                    'description': f'Cluster contient {len(zone_types)} types de zones différents',
                    'cluster_id': int(label),
                    'parcel_ids': cluster_parcels['id'].tolist(),
                    'zone_types': zone_types.tolist(),
                    'severity': IssueSeverity.LOW.value
                })
        
        return anomalies
    
    def _calculate_dbscan_eps(self, data: np.ndarray) -> float:
        """
        Calculer automatiquement le paramètre eps pour DBSCAN
        basé sur la distribution des distances au plus proche voisin
        
        Args:
            data: Données normalisées
            
        Returns:
            Valeur optimale de eps
        """
        if len(data) < 2:
            return 0.5
        
        try:
            nn = NearestNeighbors(n_neighbors=min(2, len(data)))
            nn.fit(data)
            distances, _ = nn.kneighbors(data)
            
            # Utiliser le 90e percentile des distances au plus proche voisin
            eps = np.percentile(distances[:, -1], 90)
            return max(eps, 0.01)  # Valeur minimum
            
        except Exception:
            return 0.5
    
    # =========================================================================
    # CONSOLIDATION DES PARCELLES
    # =========================================================================
    
    def consolidate_parcels(
        self, 
        criteria: ConsolidationCriteria = None
    ) -> gpd.GeoDataFrame:
        """
        Consolider les parcelles selon des critères spécifiques
        
        Args:
            criteria: Critères de consolidation
            
        Returns:
            GeoDataFrame des parcelles consolidées
        """
        if self.parcels.empty:
            return gpd.GeoDataFrame()
        
        if criteria is None:
            criteria = ConsolidationCriteria()
        
        # Créer les groupes de consolidation
        groups = self._create_consolidation_groups(criteria)
        
        consolidated_parcels = []
        
        for group_id, parcel_indices in groups.items():
            if len(parcel_indices) > 1:
                # Fusionner les parcelles du groupe
                consolidated = self._merge_parcel_group(group_id, parcel_indices)
                if consolidated:
                    consolidated_parcels.append(consolidated)
            else:
                # Garder la parcelle telle quelle
                idx = parcel_indices[0]
                parcel = self.parcels.iloc[idx].to_dict()
                consolidated_parcels.append(parcel)
        
        if not consolidated_parcels:
            return gpd.GeoDataFrame()
        
        result = gpd.GeoDataFrame(consolidated_parcels, crs=self.target_crs)
        return result
    
    def _create_consolidation_groups(
        self, 
        criteria: ConsolidationCriteria
    ) -> Dict[int, List[int]]:
        """
        Créer des groupes de parcelles à consolider
        
        Args:
            criteria: Critères de consolidation
            
        Returns:
            Dictionnaire {group_id: [indices des parcelles]}
        """
        groups: Dict[int, List[int]] = {}
        assigned: Set[int] = set()
        group_id = 0
        
        self._ensure_spatial_index()
        
        for idx, parcel in self.parcels.iterrows():
            if idx in assigned:
                continue
            
            # Créer un nouveau groupe avec cette parcelle
            current_group = [idx]
            assigned.add(idx)
            
            # Chercher les parcelles compatibles
            self._expand_consolidation_group(
                current_group, 
                parcel, 
                criteria, 
                assigned
            )
            
            groups[group_id] = current_group
            group_id += 1
        
        return groups
    
    def _expand_consolidation_group(
        self,
        group: List[int],
        seed_parcel: pd.Series,
        criteria: ConsolidationCriteria,
        assigned: Set[int]
    ) -> None:
        """Étendre un groupe de consolidation avec les parcelles compatibles"""
        
        geometry = seed_parcel.get('geometry')
        if not geometry:
            return
        
        # Trouver les candidats proches
        buffer = geometry.buffer(criteria.max_distance)
        candidate_indices = list(self.parcels.sindex.intersection(buffer.bounds))
        
        for idx in candidate_indices:
            if idx in assigned:
                continue
            
            candidate = self.parcels.iloc[idx]
            
            if self._can_consolidate(seed_parcel, candidate, criteria):
                group.append(idx)
                assigned.add(idx)
    
    def _can_consolidate(
        self,
        parcel1: pd.Series,
        parcel2: pd.Series,
        criteria: ConsolidationCriteria
    ) -> bool:
        """Vérifier si deux parcelles peuvent être consolidées"""
        
        # Vérifier la distance
        geom1 = parcel1.get('geometry')
        geom2 = parcel2.get('geometry')
        
        if not geom1 or not geom2:
            return False
        
        if geom1.distance(geom2) > criteria.max_distance:
            return False
        
        # Vérifier la similitude du propriétaire
        if criteria.owner_similarity:
            if parcel1.get('owner_id') != parcel2.get('owner_id'):
                return False
        
        # Vérifier la similitude du type de zone
        if criteria.zone_type_similarity:
            if parcel1.get('zone_type') != parcel2.get('zone_type'):
                return False
        
        return True
    
    def _merge_parcel_group(
        self, 
        group_id: int, 
        parcel_indices: List[int]
    ) -> Optional[Dict[str, Any]]:
        """
        Fusionner un groupe de parcelles
        
        Args:
            group_id: ID du groupe
            parcel_indices: Indices des parcelles à fusionner
            
        Returns:
            Dictionnaire de la parcelle fusionnée ou None en cas d'erreur
        """
        try:
            group_parcels = [self.parcels.iloc[i] for i in parcel_indices]
            geometries = [p.get('geometry') for p in group_parcels if p.get('geometry')]
            
            if not geometries:
                return None
            
            merged_geom = unary_union(geometries)
            total_area = sum(p.get('area', 0) for p in group_parcels)
            
            # Prendre les propriétés de la première parcelle comme référence
            first_parcel = group_parcels[0]
            
            return {
                'id': f'consolidated_{group_id}',
                'geometry': merged_geom,
                'area': total_area,
                'perimeter': merged_geom.length if hasattr(merged_geom, 'length') else 0,
                'owner_id': first_parcel.get('owner_id', 'unknown'),
                'zone_type': first_parcel.get('zone_type', 'mixed'),
                'original_parcels': [self.parcels.iloc[i].get('id') for i in parcel_indices],
                'consolidation_date': datetime.now().isoformat(),
                'parcel_count': len(parcel_indices)
            }
            
        except Exception as e:
            warnings.warn(f"Erreur lors de la fusion du groupe {group_id}: {e}")
            return None
    
    # =========================================================================
    # RAPPORTS ET STATISTIQUES
    # =========================================================================
    
    def generate_cadastral_report(self) -> Dict[str, Any]:
        """
        Générer un rapport complet sur le cadastre
        
        Returns:
            Dictionnaire contenant le rapport complet
        """
        # Calculer une seule fois les résultats
        validation_results = self.validate_parcels()
        anomalies = self.detect_cadastral_anomalies()
        statistics = self._calculate_cadastral_statistics()
        
        report = {
            'report_date': datetime.now().isoformat(),
            'crs': self.target_crs,
            'summary': {
                'total_parcels': len(self.parcels),
                'total_area_ha': self.parcels['area'].sum() / 10000 if not self.parcels.empty else 0,
                'parcels_with_issues': len(validation_results),
                'total_anomalies': len(anomalies)
            },
            'validation_results': validation_results,
            'cadastral_anomalies': anomalies,
            'statistics': statistics,
            'recommendations': self._generate_recommendations(validation_results, anomalies)
        }
        
        return report
    
    def _calculate_cadastral_statistics(self) -> Dict[str, Any]:
        """Calculer les statistiques cadastrales"""
        if self.parcels.empty:
            return {}
        
        areas = self.parcels['area']
        
        return {
            'area_statistics': {
                'mean': float(areas.mean()),
                'median': float(areas.median()),
                'std': float(areas.std()),
                'min': float(areas.min()),
                'max': float(areas.max()),
                'total': float(areas.sum())
            },
            'zone_type_distribution': self.parcels['zone_type'].value_counts().to_dict(),
            'owner_distribution': self.parcels['owner_id'].value_counts().to_dict() if 'owner_id' in self.parcels.columns else {},
            'parcel_count_by_size': {
                'small_(<100m²)': int((areas < 100).sum()),
                'medium_(100-1000m²)': int(((areas >= 100) & (areas < 1000)).sum()),
                'large_(1000-10000m²)': int(((areas >= 1000) & (areas < 10000)).sum()),
                'very_large_(>10000m²)': int((areas >= 10000).sum())
            }
        }
    
    def _generate_recommendations(
        self, 
        validation_issues: List[Dict], 
        anomalies: List[Dict]
    ) -> List[Dict[str, str]]:
        """
        Générer des recommandations pour l'amélioration du cadastre
        
        Args:
            validation_issues: Résultats de validation (pré-calculés)
            anomalies: Anomalies détectées (pré-calculées)
            
        Returns:
            Liste des recommandations
        """
        recommendations = []
        
        # Recommandations basées sur les problèmes de validation
        if validation_issues:
            high_severity = sum(
                1 for issue in validation_issues 
                for i in issue.get('issues', []) 
                if i.get('severity') == IssueSeverity.HIGH.value
            )
            
            if high_severity > 0:
                recommendations.append({
                    'priority': 'high',
                    'category': 'data_quality',
                    'description': f'Corriger {high_severity} problème(s) de sévérité haute (géométries invalides, chevauchements)'
                })
            
            if len(validation_issues) > high_severity:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'data_quality',
                    'description': f'Réviser {len(validation_issues) - high_severity} parcelle(s) avec des problèmes mineurs'
                })
        
        # Recommandations basées sur les anomalies
        if anomalies:
            gap_count = sum(1 for a in anomalies if a.get('type') == AnomalyType.CADASTRAL_GAP.value)
            if gap_count > 0:
                recommendations.append({
                    'priority': 'high',
                    'category': 'topology',
                    'description': f'Combler {gap_count} trou(s) dans le tissu cadastral'
                })
            
            mismatch_count = sum(1 for a in anomalies if a.get('type') == AnomalyType.AREA_MISMATCH.value)
            if mismatch_count > 0:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'data_integrity',
                    'description': f'Vérifier {mismatch_count} incohérence(s) de superficie'
                })
        
        # Recommandations basées sur la fragmentation
        if not self.parcels.empty:
            avg_area = self.parcels['area'].mean()
            small_parcels_ratio = (self.parcels['area'] < 100).sum() / len(self.parcels)
            
            if small_parcels_ratio > 0.3:  # Plus de 30% de petites parcelles
                recommendations.append({
                    'priority': 'low',
                    'category': 'optimization',
                    'description': f'{small_parcels_ratio*100:.1f}% des parcelles font moins de 100m² - envisager une consolidation'
                })
        
        return recommendations
    
    # =========================================================================
    # PRÉDICTION DE VALEUR
    # =========================================================================
    
    def predict_parcel_values(self) -> List[Dict[str, Any]]:
        """
        Prédire les valeurs des parcelles basé sur divers facteurs
        
        Returns:
            Liste des prédictions de valeur pour chaque parcelle
        """
        predictions = []
        
        for idx, parcel in self.parcels.iterrows():
            value, factors = self._calculate_parcel_value(parcel)
            predictions.append({
                'parcel_id': parcel.get('id'),
                'predicted_value': round(value, 2),
                'factors': factors
            })
        
        return predictions
    
    def _calculate_parcel_value(self, parcel: pd.Series) -> Tuple[float, Dict[str, float]]:
        """
        Calculer la valeur estimée d'une parcelle de manière déterministe
        
        Args:
            parcel: Données de la parcelle
            
        Returns:
            Tuple (valeur estimée, dictionnaire des facteurs)
        """
        area = parcel.get('area', 0)
        base_value_per_m2 = 20.0  # €/m² de base
        base_value = area * base_value_per_m2
        
        # Facteur de localisation déterministe basé sur la position
        geometry = parcel.get('geometry')
        if geometry and geometry.is_valid:
            centroid = geometry.centroid
            # Simulation déterministe basée sur la position
            # Crée une variation de ±50% basée sur les coordonnées
            location_factor = 1.0 + 0.5 * np.sin(centroid.x * 0.001) * np.cos(centroid.y * 0.001)
        else:
            location_factor = 1.0
        
        # Facteur de type de zone
        zone_type = parcel.get('zone_type', '')
        zone_multiplier = self._zone_value_multipliers.get(zone_type, 1.0)
        
        # Facteur de forme (les formes régulières valent plus)
        perimeter = parcel.get('perimeter', 0)
        if area > 0 and perimeter > 0:
            # Plus le ratio est proche de 4 (carré), meilleure est la forme
            shape_ratio = perimeter / (area ** 0.5)
            shape_factor = max(0.7, min(1.0, 1.0 - (shape_ratio - 4) * 0.05))
        else:
            shape_factor = 1.0
        
        # Calcul final
        estimated_value = base_value * location_factor * zone_multiplier * shape_factor
        estimated_value = max(estimated_value, 1000.0)  # Valeur minimum
        
        factors = {
            'base_value': round(base_value, 2),
            'location_factor': round(location_factor, 3),
            'zone_multiplier': round(zone_multiplier, 2),
            'shape_factor': round(shape_factor, 3),
            'zone_type': zone_type
        }
        
        return estimated_value, factors
    
    # =========================================================================
    # DÉTECTION DE CHANGEMENTS
    # =========================================================================
    
    def detect_land_use_changes(
        self, 
        previous_parcels: gpd.GeoDataFrame
    ) -> List[Dict[str, Any]]:
        """
        Détecter les changements d'usage du sol entre deux jeux de données
        
        Args:
            previous_parcels: GeoDataFrame des parcelles précédentes
            
        Returns:
            Liste des changements détectés
        """
        changes = []
        
        if previous_parcels is None or previous_parcels.empty:
            return changes
        
        current_ids = set(self.parcels['id']) if 'id' in self.parcels.columns else set()
        previous_ids = set(previous_parcels['id']) if 'id' in previous_parcels.columns else set()
        
        # Nouvelles parcelles
        for parcel_id in (current_ids - previous_ids):
            parcel = self.parcels[self.parcels['id'] == parcel_id].iloc[0]
            changes.append({
                'type': AnomalyType.NEW_PARCEL.value,
                'parcel_id': parcel_id,
                'description': 'Nouvelle parcelle cadastrale',
                'severity': IssueSeverity.INFORMATION.value,
                'zone_type': parcel.get('zone_type'),
                'area': parcel.get('area')
            })
        
        # Parcelles supprimées
        for parcel_id in (previous_ids - current_ids):
            prev_parcel = previous_parcels[previous_parcels['id'] == parcel_id].iloc[0]
            changes.append({
                'type': AnomalyType.REMOVED_PARCEL.value,
                'parcel_id': parcel_id,
                'description': 'Parcelle supprimée du cadastre',
                'severity': IssueSeverity.INFORMATION.value,
                'previous_zone_type': prev_parcel.get('zone_type'),
                'previous_area': prev_parcel.get('area')
            })
        
        # Changements sur les parcelles existantes
        for parcel_id in (current_ids & previous_ids):
            current = self.parcels[self.parcels['id'] == parcel_id].iloc[0]
            previous = previous_parcels[previous_parcels['id'] == parcel_id].iloc[0]
            
            # Changement de type de zone
            current_zone = current.get('zone_type')
            previous_zone = previous.get('zone_type')
            
            if current_zone != previous_zone:
                changes.append({
                    'type': AnomalyType.ZONE_CHANGE.value,
                    'parcel_id': parcel_id,
                    'description': f'Changement de zone: {previous_zone} → {current_zone}',
                    'severity': IssueSeverity.MEDIUM.value,
                    'previous_zone': previous_zone,
                    'current_zone': current_zone
                })
            
            # Changement significatif de superficie
            current_area = current.get('area', 0)
            previous_area = previous.get('area', 0)
            
            if previous_area > 0:
                area_change_ratio = abs(current_area - previous_area) / previous_area
                if area_change_ratio > 0.1:  # Plus de 10% de changement
                    changes.append({
                        'type': 'area_change',
                        'parcel_id': parcel_id,
                        'description': f'Changement de superficie: {previous_area:.2f}m² → {current_area:.2f}m² ({area_change_ratio*100:.1f}%)',
                        'severity': IssueSeverity.MEDIUM.value,
                        'previous_area': previous_area,
                        'current_area': current_area,
                        'change_ratio': area_change_ratio
                    })
        
        return changes


# =============================================================================
# TESTS UNITAIRES
# =============================================================================

def run_tests():
    """Exécuter les tests unitaires de l'agent cadastral"""
    import traceback
    
    print("=" * 60)
    print("TESTS UNITAIRES - Agent Cadastral")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    def assert_test(condition: bool, test_name: str, details: str = ""):
        nonlocal tests_passed, tests_failed
        if condition:
            print(f"✅ {test_name}")
            tests_passed += 1
        else:
            print(f"❌ {test_name}")
            if details:
                print(f"   Détails: {details}")
            tests_failed += 1
    
    try:
        # Test 1: Création de l'agent
        agent = CadastralAgent()
        assert_test(agent is not None, "Création de l'agent")
        
        # Test 2: Chargement de parcelles valides
        valid_parcels = gpd.GeoDataFrame([
            {
                'id': 'P1',
                'geometry': Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
                'area': 100,
                'perimeter': 40,
                'owner_id': 'O1',
                'zone_type': 'urban'
            }
        ])
        result = agent.load_parcels(valid_parcels, validate_columns=True)
        assert_test(result == True, "Chargement de parcelles valides")
        
        # Test 3: Validation d'une parcelle valide
        issues = agent.validate_parcels()
        assert_test(len(issues) == 0, "Validation d'une parcelle valide (pas d'erreurs)")
        
        # Test 4: Détection de chevauchement
        overlapping_parcels = gpd.GeoDataFrame([
            {
                'id': 'P1',
                'geometry': Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
                'area': 100,
                'perimeter': 40,
                'owner_id': 'O1',
                'zone_type': 'urban'
            },
            {
                'id': 'P2',
                'geometry': Polygon([(5, 5), (15, 5), (15, 15), (5, 15)]),
                'area': 100,
                'perimeter': 40,
                'owner_id': 'O2',
                'zone_type': 'urban'
            }
        ])
        agent.load_parcels(overlapping_parcels)
        issues = agent.validate_parcels(force_refresh=True)
        has_overlap = any(
            any(i.get('type') == 'overlapping_parcels' for i in issue.get('issues', []))
            for issue in issues
        )
        assert_test(has_overlap, "Détection de chevauchement")
        
        # Test 5: Ratio périmètre/aire correct
        # Un carré de 10x10 a un ratio de 40/10 = 4, ce qui est normal
        agent2 = CadastralAgent()
        square_parcel = gpd.GeoDataFrame([
            {
                'id': 'SQUARE',
                'geometry': Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
                'area': 100,
                'perimeter': 40,
                'owner_id': 'O1',
                'zone_type': 'urban'
            }
        ])
        agent2.load_parcels(square_parcel)
        issues = agent2.validate_parcels()
        has_irregular_shape = any(
            any(i.get('type') == 'irregular_shape' for i in issue.get('issues', []))
            for issue in issues
        )
        assert_test(not has_irregular_shape, "Ratio périmètre/aire - carré normal non signalé")
        
        # Test 6: Forme irrégulière détectée
        # Une forme très allongée a un ratio élevé
        irregular_parcel = gpd.GeoDataFrame([
            {
                'id': 'IRREGULAR',
                'geometry': Polygon([(0, 0), (100, 0), (100, 1), (0, 1)]),  # 100x1
                'area': 100,
                'perimeter': 202,  # ratio = 202/10 = 20.2
                'owner_id': 'O1',
                'zone_type': 'urban'
            }
        ])
        agent2.load_parcels(irregular_parcel)
        issues = agent2.validate_parcels(force_refresh=True)
        has_irregular_shape = any(
            any(i.get('type') == 'irregular_shape' for i in issue.get('issues', []))
            for issue in issues
        )
        assert_test(has_irregular_shape, "Ratio périmètre/aire - forme irrégulière détectée")
        
        # Test 7: Prédiction de valeur déterministe
        agent3 = CadastralAgent()
        agent3.load_parcels(valid_parcels)
        predictions1 = agent3.predict_parcel_values()
        predictions2 = agent3.predict_parcel_values()
        values_equal = predictions1[0]['predicted_value'] == predictions2[0]['predicted_value']
        assert_test(values_equal, "Prédiction de valeur déterministe (reproductible)")
        
        # Test 8: Cache de validation
        agent4 = CadastralAgent()
        agent4.load_parcels(valid_parcels)
        result1 = agent4.validate_parcels()
        result2 = agent4.validate_parcels()  # Devrait utiliser le cache
        assert_test(result1 is result2, "Cache de validation fonctionnel")
        
        # Test 9: Invalidation du cache après modification
        agent4.load_parcels(valid_parcels)  # Recharge les données
        result3 = agent4.validate_parcels()
        assert_test(result1 is not result3, "Invalidation du cache après modification")
        
        # Test 10: Génération de rapport
        agent5 = CadastralAgent()
        agent5.load_parcels(valid_parcels)
        report = agent5.generate_cadastral_report()
        has_required_keys = all(
            key in report 
            for key in ['report_date', 'summary', 'validation_results', 'recommendations']
        )
        assert_test(has_required_keys, "Génération de rapport complète")
        
        # Test 11: Détection de changements
        previous_parcels = gpd.GeoDataFrame([
            {
                'id': 'P1',
                'geometry': Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
                'area': 100,
                'perimeter': 40,
                'owner_id': 'O1',
                'zone_type': 'agricultural'  # Différent de 'urban'
            }
        ])
        agent5.load_parcels(valid_parcels)
        changes = agent5.detect_land_use_changes(previous_parcels)
        has_zone_change = any(c.get('type') == 'zone_change' for c in changes)
        assert_test(has_zone_change, "Détection de changement de zone")
        
    except Exception as e:
        print(f"\n❌ ERREUR INATTENDUE: {e}")
        traceback.print_exc()
        tests_failed += 1
    
    print("\n" + "=" * 60)
    print(f"RÉSULTATS: {tests_passed} réussis, {tests_failed} échoués")
    print("=" * 60)
    
    return tests_failed == 0


# =============================================================================
# DÉMONSTRATION
# =============================================================================

def main():
    """Fonction principale pour démontrer l'agent cadastral"""
    
    print("\n" + "=" * 60)
    print("Agent Cadastral pour les Workflows IA Géospatiaux")
    print("=" * 60)
    
    # Créer l'agent
    agent = CadastralAgent(target_crs="EPSG:2154")
    
    # Créer des données de test
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
        },
        {
            'id': 'PAR004',
            'geometry': Polygon([(8, 8), (18, 8), (18, 18), (8, 18)]),  # Chevauche PAR003
            'area': 100,
            'perimeter': 40,
            'owner_id': 'OWNER003',
            'zone_type': 'industrial'
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
            'declared_area': 120,  # Incohérence avec cadastre (100m²)
            'building_area': 10,
            'usage': 'agricultural'
        }
    ])
    
    # Charger les données
    print("\n📁 Chargement des données...")
    agent.load_parcels(sample_parcels)
    agent.load_properties(sample_properties)
    
    # Validation
    print("\n🔍 Validation des parcelles...")
    validation_results = agent.validate_parcels()
    print(f"   → {len(validation_results)} parcelle(s) avec problèmes")
    for result in validation_results:
        print(f"   • {result['parcel_id']}: {len(result['issues'])} problème(s)")
        for issue in result['issues']:
            print(f"     - [{issue['severity']}] {issue['type']}: {issue['description']}")
    
    # Anomalies
    print("\n🚨 Détection d'anomalies...")
    anomalies = agent.detect_cadastral_anomalies()
    print(f"   → {len(anomalies)} anomalie(s) détectée(s)")
    for anomaly in anomalies:
        print(f"   • [{anomaly['severity']}] {anomaly['type']}: {anomaly['description']}")
    
    # Prédictions de valeur
    print("\n💰 Prédictions de valeur...")
    predictions = agent.predict_parcel_values()
    for pred in predictions:
        print(f"   • {pred['parcel_id']}: {pred['predicted_value']:,.2f}€")
        print(f"     Facteurs: zone={pred['factors']['zone_type']} (×{pred['factors']['zone_multiplier']})")
    
    # Consolidation
    print("\n🔗 Opportunités de consolidation...")
    criteria = ConsolidationCriteria(
        owner_similarity=True,
        zone_type_similarity=True,
        max_distance=50
    )
    consolidated = agent.consolidate_parcels(criteria)
    print(f"   → {len(sample_parcels)} parcelles → {len(consolidated)} après consolidation")
    
    # Rapport
    print("\n📊 Génération du rapport...")
    report = agent.generate_cadastral_report()
    print(f"   • Date: {report['report_date']}")
    print(f"   • Total parcelles: {report['summary']['total_parcels']}")
    print(f"   • Surface totale: {report['summary']['total_area_ha']:.4f} ha")
    print(f"   • Parcelles avec problèmes: {report['summary']['parcels_with_issues']}")
    print(f"   • Recommandations: {len(report['recommendations'])}")
    
    for rec in report['recommendations']:
        print(f"     - [{rec['priority']}] {rec['category']}: {rec['description']}")
    
    # Détection de changements
    print("\n🔄 Détection de changements...")
    previous_parcels = sample_parcels.copy()
    previous_parcels.loc[previous_parcels['id'] == 'PAR001', 'zone_type'] = 'agricultural'
    
    changes = agent.detect_land_use_changes(previous_parcels)
    print(f"   → {len(changes)} changement(s) détecté(s)")
    for change in changes:
        print(f"   • [{change['severity']}] {change['type']}: {change['description']}")
    
    print("\n" + "=" * 60)
    print("Démonstration terminée")
    print("=" * 60)


if __name__ == "__main__":
    # Exécuter les tests d'abord
    if run_tests():
        print("\n✅ Tous les tests ont réussi!\n")
    else:
        print("\n⚠️ Certains tests ont échoué.\n")
    
    # Puis la démonstration
    main()