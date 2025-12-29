#!/usr/bin/env python3
"""Import de données géospatiales dans PostGIS

- Lecture de fichiers via GeoPandas (GeoJSON, Shapefile, etc.)
- Reprojection optionnelle vers un SRID cible (par défaut EPSG:2154)
- Import dans PostGIS avec gestion des options d'écrasement

Compatible exécution locale et dans le conteneur.
"""

import sys
import json
import os
import argparse
import logging
from typing import Optional, Dict, Any

import geopandas as gpd
from sqlalchemy import create_engine

# Configuration logging simple et lisible
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def get_database_url() -> str:
    """Construire l'URL de connexion à la base.
    Priorité: DATABASE_URL > variables PG* > valeurs par défaut.
    """
    if os.getenv("DATABASE_URL"):
        return os.environ["DATABASE_URL"]

    user = os.getenv("PGUSER") or os.getenv("DB_POSTGRESDB_USER", "geo")
    password = os.getenv("PGPASSWORD") or os.getenv("DB_POSTGRESDB_PASSWORD", "geo_password")
    host = os.getenv("PGHOST") or os.getenv("DB_POSTGRESDB_HOST", "postgis")
    port = os.getenv("PGPORT") or os.getenv("DB_POSTGRESDB_PORT", "5432")
    dbname = os.getenv("PGDATABASE") or os.getenv("DB_POSTGRESDB_DATABASE", "cadastre")
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


def get_engine():
    """Créer une connexion à la base de données"""
    url = get_database_url()
    logger.debug(f"Using database URL: {url.replace(password := url.split(':')[2].split('@')[0].replace('//', ''), '***')}")
    return create_engine(url)


def import_geodata(
    filepath: str,
    table_name: str,
    schema: str = "cadastre",
    target_epsg: int = 2154,
    if_exists: str = "append",
    index: bool = False,
    chunksize: Optional[int] = None,
    geometry: Optional[str] = None,
) -> Dict[str, Any]:
    """Importer un fichier Geo dans PostGIS.

    Args:
        filepath: chemin vers le fichier géospatial (GeoJSON, Shapefile, etc.)
        table_name: nom de la table cible
        schema: schéma cible (par défaut 'cadastre')
        target_epsg: SRID cible pour reprojection
        if_exists: comportement si table existe: {'fail','replace','append'}
        index: créer un index de DataFrame (pas index spatial)
        chunksize: taille des chunks pour l'upload
        geometry: nom de la colonne géométrique si besoin de forcer
    """
    try:
        if not os.path.exists(filepath):
            return {"status": "error", "message": f"Fichier non trouvé: {filepath}"}

        logger.info(f"Lecture du fichier: {filepath}")
        gdf = gpd.read_file(filepath)

        if gdf.empty:
            return {"status": "warning", "message": "Fichier géospatial vide", "count": 0}

        # Déterminer/forcer la colonne géométrique si spécifiée
        if geometry and geometry in gdf.columns:
            gdf.set_geometry(geometry, inplace=True)

        # Reprojection
        if gdf.crs is None:
            logger.warning("CRS non défini dans la source, supposition EPSG:4326 avant reprojection")
            gdf = gdf.set_crs(epsg=4326)
        if gdf.crs.to_epsg() != target_epsg:
            logger.info(f"Reprojection vers EPSG:{target_epsg}")
            gdf = gdf.to_crs(epsg=target_epsg)

        # Import
        engine = get_engine()
        logger.info(f"Import vers {schema}.{table_name} (if_exists={if_exists})")
        gdf.to_postgis(
            table_name,
            engine,
            schema=schema,
            if_exists=if_exists,
            index=index,
            chunksize=chunksize,
        )

        return {
            "status": "success",
            "table": f"{schema}.{table_name}",
            "count": int(len(gdf)),
            "srid": target_epsg,
        }

    except Exception as e:
        logger.exception("Erreur lors de l'import PostGIS")
        return {"status": "error", "message": str(e)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Importer des données géospatiales dans PostGIS")
    parser.add_argument("file", help="Chemin du fichier géospatial (GeoJSON, SHP, etc.)")
    parser.add_argument("table", help="Nom de la table cible")
    parser.add_argument("schema", nargs="?", default="cadastre", help="Schéma cible (par défaut cadastre)")
    parser.add_argument("--srid", type=int, default=2154, help="SRID cible (par défaut 2154 - Lambert 93)")
    parser.add_argument("--if-exists", choices=["fail", "replace", "append"], default="append", help="Comportement si la table existe")
    parser.add_argument("--chunksize", type=int, default=None, help="Taille des chunks pour l'upload")
    parser.add_argument("--geometry", type=str, default=None, help="Nom de la colonne géométrique à utiliser")
    return parser


if __name__ == '__main__':
    # Assurer l'import local du module si besoin
    sys.path.append(os.path.dirname(__file__))

    parser = build_arg_parser()
    args = parser.parse_args()

    result = import_geodata(
        filepath=args.file,
        table_name=args.table,
        schema=args.schema,
        target_epsg=args.srid,
        if_exists=args.if_exists,
        chunksize=args.chunksize,
        geometry=args.geometry,
    )

    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0 if result.get("status") == "success" else 1)
