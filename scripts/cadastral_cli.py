#!/usr/bin/env python3
"""
CLI Agent Cadastral pour n8n
============================
Interface en ligne de commande pour l'agent cadastral.
Conçu pour être appelé depuis n8n via le node "Execute Command".
"""

import sys
import json
import argparse
from typing import Dict, Any, Optional
from pathlib import Path
import tempfile
import os

import geopandas as gpd
from shapely.geometry import shape, mapping

# Importer l'agent cadastral
from cadastral_agent import (
    CadastralAgent,
    ConsolidationCriteria,
    ValidationRules
)


# =============================================================================
# UTILITAIRES
# =============================================================================

def read_input() -> Dict[str, Any]:
    """Lire les données JSON depuis stdin"""
    try:
        input_data = sys.stdin.read()
        if input_data.strip():
            return json.loads(input_data)
        return {}
    except json.JSONDecodeError as e:
        error_exit(f"Erreur de parsing JSON: {e}")


def output_json(data: Dict[str, Any]) -> None:
    """Écrire les données JSON sur stdout"""
    print(json.dumps(data, ensure_ascii=False, default=str))


def error_exit(message: str, code: int = 1) -> None:
    """Afficher une erreur et quitter"""
    output_json({
        "success": False,
        "error": message
    })
    sys.exit(code)


def load_geojson_file(filepath: str) -> gpd.GeoDataFrame:
    """Charger un fichier GeoJSON"""
    try:
        return gpd.read_file(filepath)
    except Exception as e:
        error_exit(f"Erreur lecture fichier {filepath}: {e}")


def json_to_geodataframe(data: Dict[str, Any]) -> gpd.GeoDataFrame:
    """Convertir des données JSON en GeoDataFrame"""
    parcels = data.get('parcels', [])
    crs = data.get('crs', 'EPSG:4326')
    
    if not parcels:
        error_exit("Aucune parcelle fournie")
    
    records = []
    for parcel in parcels:
        geom = shape(parcel['geometry']) if 'geometry' in parcel else None
        records.append({
            'id': parcel.get('id', ''),
            'geometry': geom,
            'area': parcel.get('area', 0),
            'perimeter': parcel.get('perimeter', 0),
            'owner_id': parcel.get('owner_id', ''),
            'zone_type': parcel.get('zone_type', '')
        })
    
    return gpd.GeoDataFrame(records, crs=crs)


def geodataframe_to_json(gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    """Convertir un GeoDataFrame en dictionnaire JSON-compatible"""
    if gdf.empty:
        return {"type": "FeatureCollection", "features": []}
    return json.loads(gdf.to_json())


# =============================================================================
# COMMANDES
# =============================================================================

def cmd_validate(args: argparse.Namespace) -> None:
    """Valider des parcelles cadastrales"""
    agent = CadastralAgent()
    
    # Charger les données
    if args.file:
        gdf = load_geojson_file(args.file)
        # S'assurer des colonnes requises
        for col in ['id', 'area', 'perimeter', 'owner_id', 'zone_type']:
            if col not in gdf.columns:
                gdf[col] = 'unknown' if col in ['id', 'owner_id', 'zone_type'] else 0
    else:
        input_data = read_input()
        gdf = json_to_geodataframe(input_data)
    
    # Configurer les règles si spécifiées
    if args.min_area:
        agent.validation_rules.min_area = args.min_area
    if args.max_area:
        agent.validation_rules.max_area = args.max_area
    if args.ratio_max:
        agent.validation_rules.perimeter_area_ratio_max = args.ratio_max
    
    # Charger et valider
    agent.load_parcels(gdf, validate_columns=False)
    issues = agent.validate_parcels()
    
    # Préparer la sortie
    results = []
    for issue in issues:
        result = {
            "parcel_id": issue["parcel_id"],
            "issues": issue["issues"],
            "issue_count": len(issue["issues"])
        }
        if args.include_geometry and issue.get("geometry"):
            result["geometry"] = mapping(issue["geometry"])
        results.append(result)
    
    output_json({
        "success": True,
        "total_parcels": len(gdf),
        "parcels_with_issues": len(results),
        "validation_passed": len(results) == 0,
        "issues": results
    })


def cmd_anomalies(args: argparse.Namespace) -> None:
    """Détecter les anomalies cadastrales"""
    agent = CadastralAgent()
    
    # Charger les données
    if args.file:
        gdf = load_geojson_file(args.file)
        for col in ['id', 'area', 'perimeter', 'owner_id', 'zone_type']:
            if col not in gdf.columns:
                gdf[col] = 'unknown' if col in ['id', 'owner_id', 'zone_type'] else 0
    else:
        input_data = read_input()
        gdf = json_to_geodataframe(input_data)
    
    agent.load_parcels(gdf, validate_columns=False)
    
    # Charger les propriétés si fournies
    if args.properties_file:
        props = gpd.read_file(args.properties_file)
        agent.load_properties(props)
    
    anomalies = agent.detect_cadastral_anomalies()
    
    # Préparer la sortie
    results = []
    for anomaly in anomalies:
        result = dict(anomaly)
        if "geometry" in result and result["geometry"]:
            if args.include_geometry:
                result["geometry"] = mapping(result["geometry"])
            else:
                del result["geometry"]
        results.append(result)
    
    # Compter par type
    type_counts = {}
    for r in results:
        t = r.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    
    output_json({
        "success": True,
        "total_anomalies": len(results),
        "anomalies_by_type": type_counts,
        "anomalies": results
    })


def cmd_consolidate(args: argparse.Namespace) -> None:
    """Consolider des parcelles"""
    agent = CadastralAgent()
    
    # Charger les données
    if args.file:
        gdf = load_geojson_file(args.file)
        for col in ['id', 'area', 'perimeter', 'owner_id', 'zone_type']:
            if col not in gdf.columns:
                gdf[col] = 'unknown' if col in ['id', 'owner_id', 'zone_type'] else 0
    else:
        input_data = read_input()
        gdf = json_to_geodataframe(input_data)
    
    agent.load_parcels(gdf, validate_columns=False)
    
    # Configurer les critères
    criteria = ConsolidationCriteria(
        owner_similarity=args.by_owner,
        zone_type_similarity=args.by_zone,
        max_distance=args.max_distance
    )
    
    consolidated = agent.consolidate_parcels(criteria)
    
    output_json({
        "success": True,
        "original_count": len(gdf),
        "consolidated_count": len(consolidated),
        "reduction": len(gdf) - len(consolidated),
        "parcels": geodataframe_to_json(consolidated) if not consolidated.empty else {"features": []}
    })


def cmd_predict_values(args: argparse.Namespace) -> None:
    """Prédire les valeurs des parcelles"""
    agent = CadastralAgent()
    
    # Charger les données
    if args.file:
        gdf = load_geojson_file(args.file)
        for col in ['id', 'area', 'perimeter', 'owner_id', 'zone_type']:
            if col not in gdf.columns:
                gdf[col] = 'unknown' if col in ['id', 'owner_id', 'zone_type'] else 0
    else:
        input_data = read_input()
        gdf = json_to_geodataframe(input_data)
    
    agent.load_parcels(gdf, validate_columns=False)
    predictions = agent.predict_parcel_values()
    
    total_value = sum(p["predicted_value"] for p in predictions)
    avg_value = total_value / len(predictions) if predictions else 0
    
    output_json({
        "success": True,
        "predictions": predictions,
        "summary": {
            "total_value": round(total_value, 2),
            "average_value": round(avg_value, 2),
            "parcel_count": len(predictions),
            "min_value": round(min(p["predicted_value"] for p in predictions), 2) if predictions else 0,
            "max_value": round(max(p["predicted_value"] for p in predictions), 2) if predictions else 0
        }
    })


def cmd_report(args: argparse.Namespace) -> None:
    """Générer un rapport cadastral complet"""
    agent = CadastralAgent()
    
    # Charger les données
    if args.file:
        gdf = load_geojson_file(args.file)
        for col in ['id', 'area', 'perimeter', 'owner_id', 'zone_type']:
            if col not in gdf.columns:
                gdf[col] = 'unknown' if col in ['id', 'owner_id', 'zone_type'] else 0
    else:
        input_data = read_input()
        gdf = json_to_geodataframe(input_data)
    
    agent.load_parcels(gdf, validate_columns=False)
    
    # Charger les propriétés si fournies
    if args.properties_file:
        props = gpd.read_file(args.properties_file)
        agent.load_properties(props)
    
    report = agent.generate_cadastral_report()
    
    # Nettoyer les géométries du rapport
    if "validation_results" in report:
        for item in report["validation_results"]:
            if "geometry" in item:
                del item["geometry"]
    
    output_json({
        "success": True,
        **report
    })


def cmd_changes(args: argparse.Namespace) -> None:
    """Détecter les changements entre deux jeux de données"""
    agent = CadastralAgent()
    
    # Charger les données actuelles
    if args.current_file:
        current_gdf = load_geojson_file(args.current_file)
    else:
        error_exit("Fichier des parcelles actuelles requis (--current)")
    
    # Charger les données précédentes
    if args.previous_file:
        previous_gdf = load_geojson_file(args.previous_file)
    else:
        error_exit("Fichier des parcelles précédentes requis (--previous)")
    
    # S'assurer des colonnes requises
    for gdf in [current_gdf, previous_gdf]:
        for col in ['id', 'area', 'perimeter', 'owner_id', 'zone_type']:
            if col not in gdf.columns:
                gdf[col] = 'unknown' if col in ['id', 'owner_id', 'zone_type'] else 0
    
    agent.load_parcels(current_gdf, validate_columns=False)
    changes = agent.detect_land_use_changes(previous_gdf)
    
    # Compter par type
    type_counts = {}
    for c in changes:
        t = c.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    
    output_json({
        "success": True,
        "total_changes": len(changes),
        "changes_by_type": type_counts,
        "changes": changes
    })


def cmd_stats(args: argparse.Namespace) -> None:
    """Calculer les statistiques cadastrales"""
    agent = CadastralAgent()
    
    # Charger les données
    if args.file:
        gdf = load_geojson_file(args.file)
        for col in ['id', 'area', 'perimeter', 'owner_id', 'zone_type']:
            if col not in gdf.columns:
                gdf[col] = 'unknown' if col in ['id', 'owner_id', 'zone_type'] else 0
    else:
        input_data = read_input()
        gdf = json_to_geodataframe(input_data)
    
    agent.load_parcels(gdf, validate_columns=False)
    stats = agent._calculate_cadastral_statistics()
    
    output_json({
        "success": True,
        "parcel_count": len(gdf),
        "statistics": stats
    })


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CLI Agent Cadastral pour n8n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Valider depuis un fichier
  cadastral-cli validate --file parcelles.geojson
  
  # Valider depuis stdin
  cat parcelles.json | cadastral-cli validate
  
  # Générer un rapport complet
  cadastral-cli report --file parcelles.geojson --properties-file props.geojson
  
  # Détecter les changements
  cadastral-cli changes --current new.geojson --previous old.geojson
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commande à exécuter")
    
    # --- Commande: validate ---
    p_validate = subparsers.add_parser("validate", help="Valider les parcelles")
    p_validate.add_argument("--file", "-f", help="Fichier GeoJSON des parcelles")
    p_validate.add_argument("--min-area", type=float, help="Superficie minimum (m²)")
    p_validate.add_argument("--max-area", type=float, help="Superficie maximum (m²)")
    p_validate.add_argument("--ratio-max", type=float, help="Ratio périmètre/√aire maximum")
    p_validate.add_argument("--include-geometry", action="store_true", help="Inclure les géométries")
    p_validate.set_defaults(func=cmd_validate)
    
    # --- Commande: anomalies ---
    p_anomalies = subparsers.add_parser("anomalies", help="Détecter les anomalies")
    p_anomalies.add_argument("--file", "-f", help="Fichier GeoJSON des parcelles")
    p_anomalies.add_argument("--properties-file", "-p", help="Fichier des propriétés")
    p_anomalies.add_argument("--include-geometry", action="store_true", help="Inclure les géométries")
    p_anomalies.set_defaults(func=cmd_anomalies)
    
    # --- Commande: consolidate ---
    p_consolidate = subparsers.add_parser("consolidate", help="Consolider les parcelles")
    p_consolidate.add_argument("--file", "-f", help="Fichier GeoJSON des parcelles")
    p_consolidate.add_argument("--by-owner", action="store_true", default=True, help="Grouper par propriétaire")
    p_consolidate.add_argument("--by-zone", action="store_true", default=True, help="Grouper par type de zone")
    p_consolidate.add_argument("--max-distance", type=float, default=50, help="Distance max entre parcelles (m)")
    p_consolidate.set_defaults(func=cmd_consolidate)
    
    # --- Commande: predict-values ---
    p_values = subparsers.add_parser("predict-values", help="Prédire les valeurs")
    p_values.add_argument("--file", "-f", help="Fichier GeoJSON des parcelles")
    p_values.set_defaults(func=cmd_predict_values)
    
    # --- Commande: report ---
    p_report = subparsers.add_parser("report", help="Générer un rapport complet")
    p_report.add_argument("--file", "-f", help="Fichier GeoJSON des parcelles")
    p_report.add_argument("--properties-file", "-p", help="Fichier des propriétés")
    p_report.set_defaults(func=cmd_report)
    
    # --- Commande: changes ---
    p_changes = subparsers.add_parser("changes", help="Détecter les changements")
    p_changes.add_argument("--current", "-c", dest="current_file", required=True, help="Fichier parcelles actuelles")
    p_changes.add_argument("--previous", "-p", dest="previous_file", required=True, help="Fichier parcelles précédentes")
    p_changes.set_defaults(func=cmd_changes)
    
    # --- Commande: stats ---
    p_stats = subparsers.add_parser("stats", help="Calculer les statistiques")
    p_stats.add_argument("--file", "-f", help="Fichier GeoJSON des parcelles")
    p_stats.set_defaults(func=cmd_stats)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except Exception as e:
        error_exit(str(e))


if __name__ == "__main__":
    main()