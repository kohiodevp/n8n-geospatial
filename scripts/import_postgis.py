#!/usr/bin/env python3
"""Import de données géospatiales dans PostGIS"""

import sys
import json
import os
import geopandas as gpd
from sqlalchemy import create_engine

def get_engine():
    """Créer une connexion à la base de données"""
    return create_engine(
        f"postgresql://{os.environ.get('PGUSER', 'geo')}:{os.environ.get('PGPASSWORD', 'geo_password')}"
        f"@{os.environ.get('PGHOST', 'postgis')}:{os.environ.get('PGPORT', '5432')}"
        f"/{os.environ.get('PGDATABASE', 'cadastre')}"
    )

def import_geojson(filepath, table_name, schema='cadastre'):
    """Importer un fichier GeoJSON dans PostGIS"""
    
    try:
        # Lire avec GeoPandas
        gdf = gpd.read_file(filepath)
        
        if len(gdf) == 0:
            return {"status": "warning", "message": "Fichier vide", "count": 0}
        
        # Reprojeter en Lambert 93 si nécessaire
        if gdf.crs and gdf.crs.to_epsg() != 2154:
            gdf = gdf.to_crs(epsg=2154)
        elif not gdf.crs:
            gdf = gdf.set_crs(epsg=4326).to_crs(epsg=2154)
        
        # Connexion et import
        engine = get_engine()
        
        gdf.to_postgis(
            table_name, 
            engine, 
            schema=schema,
            if_exists='append',
            index=False
        )
        
        return {
            "status": "success",
            "table": f"{schema}.{table_name}",
            "count": len(gdf)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(json.dumps({"status": "error", "message": "Usage: import_postgis.py <fichier> <table> [schema]"}))
        sys.exit(1)
    
    filepath = sys.argv[1]
    table_name = sys.argv[2]
    schema = sys.argv[3] if len(sys.argv) > 3 else 'cadastre'
    
    if not os.path.exists(filepath):
        print(json.dumps({"status": "error", "message": f"Fichier non trouvé: {filepath}"}))
        sys.exit(1)
    
    result = import_geojson(filepath, table_name, schema)
    print(json.dumps(result, ensure_ascii=False))
    
    if result.get("status") == "error":
        sys.exit(1)
