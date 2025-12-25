#!/usr/bin/env python3
"""Analyses spatiales pour le cadastre et le domaine"""

import sys
import json
import os
import argparse
import geopandas as gpd
from shapely.geometry import shape
from sqlalchemy import create_engine, text

def get_engine():
    """Créer une connexion à la base de données"""
    return create_engine(
        f"postgresql://{os.environ.get('PGUSER', 'geo')}:{os.environ.get('PGPASSWORD', 'geo_password')}"
        f"@{os.environ.get('PGHOST', 'postgis')}:{os.environ.get('PGPORT', '5432')}"
        f"/{os.environ.get('PGDATABASE', 'cadastre')}"
    )

def get_parcelles_in_buffer(point_wkt, distance, code_commune=None):
    """Trouver les parcelles dans un rayon autour d'un point"""
    engine = get_engine()
    
    query = """
        SELECT 
            id_parcelle,
            section,
            numero,
            contenance,
            ST_Distance(geom, ST_GeomFromText(:point, 2154)) as distance,
            ST_AsGeoJSON(geom) as geojson
        FROM cadastre.parcelles
        WHERE ST_DWithin(geom, ST_GeomFromText(:point, 2154), :distance)
    """
    
    params = {"point": point_wkt, "distance": distance}
    
    if code_commune:
        query += " AND code_commune = :commune"
        params["commune"] = code_commune
    
    query += " ORDER BY distance"
    
    with engine.connect() as conn:
        result = conn.execute(text(query), params)
        
        results = []
        for row in result:
            results.append({
                "id_parcelle": row[0],
                "section": row[1],
                "numero": row[2],
                "contenance": float(row[3]) if row[3] else None,
                "distance": float(row[4]),
                "geometry": json.loads(row[5])
            })
    
    return results

def calculate_surface(geojson_str):
    """Calculer la surface d'une géométrie"""
    geom = shape(json.loads(geojson_str))
    return {
        "surface_m2": round(geom.area, 2),
        "surface_ha": round(geom.area / 10000, 4),
        "perimetre_m": round(geom.length, 2)
    }

def find_overlapping_parcelles(bien_id):
    """Trouver les parcelles qui chevauchent un bien domanial"""
    engine = get_engine()
    
    query = """
        SELECT 
            p.id_parcelle,
            p.section,
            p.numero,
            p.contenance,
            ST_Area(ST_Intersection(p.geom, b.geom)) as surface_intersection,
            ST_Area(ST_Intersection(p.geom, b.geom)) / NULLIF(ST_Area(p.geom), 0) * 100 as pourcentage_parcelle,
            ST_AsGeoJSON(ST_Intersection(p.geom, b.geom)) as geom_intersection
        FROM cadastre.parcelles p
        JOIN domaine.biens b ON ST_Intersects(p.geom, b.geom)
        WHERE b.id = :bien_id
        ORDER BY surface_intersection DESC
    """
    
    with engine.connect() as conn:
        result = conn.execute(text(query), {"bien_id": bien_id})
        
        results = []
        for row in result:
            results.append({
                "id_parcelle": row[0],
                "section": row[1],
                "numero": row[2],
                "contenance": float(row[3]) if row[3] else None,
                "surface_intersection": float(row[4]) if row[4] else 0,
                "pourcentage_parcelle": float(row[5]) if row[5] else 0,
                "geometry": json.loads(row[6]) if row[6] else None
            })
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyses spatiales")
    parser.add_argument('--action', required=True, 
                       choices=['buffer', 'surface', 'overlap'],
                       help="Type d'analyse")
    parser.add_argument('--params', type=str, default='{}',
                       help="Paramètres JSON")
    
    args = parser.parse_args()
    
    try:
        params = json.loads(args.params)
        
        if args.action == 'buffer':
            if 'point' not in params:
                raise ValueError("Paramètre 'point' requis (format WKT)")
            result = get_parcelles_in_buffer(
                params['point'],
                params.get('distance', 100),
                params.get('commune')
            )
        elif args.action == 'surface':
            if 'geojson' not in params:
                raise ValueError("Paramètre 'geojson' requis")
            result = calculate_surface(params['geojson'])
        elif args.action == 'overlap':
            if 'bien_id' not in params:
                raise ValueError("Paramètre 'bien_id' requis")
            result = find_overlapping_parcelles(params['bien_id'])
        
        print(json.dumps(result, ensure_ascii=False))
        
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False))
        sys.exit(1)
