#!/usr/bin/env python3
"""
Utilitaires PostGIS pour n8n
"""
import os
import json
from typing import Optional, List, Dict, Any, Union
import psycopg2
from psycopg2.extras import RealDictCursor
import geopandas as gpd
from sqlalchemy import create_engine
from geoalchemy2 import Geometry, WKTElement

# Configuration par défaut depuis les variables d'environnement
DEFAULT_CONFIG = {
    'host': os.getenv('PGHOST', 'postgres'),
    'port': int(os.getenv('PGPORT', 5432)),
    'database': os.getenv('PGDATABASE', 'geodatabase'),
    'user': os.getenv('PGUSER', 'n8n'),
    'password': os.getenv('PGPASSWORD', ''),
}


class PostGISClient:
    """Client PostGIS pour opérations géospatiales"""

    def __init__(self, **kwargs):
        self.config = {**DEFAULT_CONFIG, **kwargs}
        self._conn = None
        self._engine = None

    @property
    def conn(self):
        """Connexion psycopg2 lazy"""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(**self.config)
        return self._conn

    @property
    def engine(self):
        """Engine SQLAlchemy lazy"""
        if self._engine is None:
            url = f"postgresql://{self.config['user']}:{self.config['password']}@{self.config['host']}:{self.config['port']}/{self.config['database']}"
            self._engine = create_engine(url)
        return self._engine

    def execute(self, query: str, params: tuple = None) -> List[Dict]:
        """Exécute une requête SQL"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            if cur.description:
                return [dict(row) for row in cur.fetchall()]
            self.conn.commit()
            return []

    def read_postgis(self, table: str, geom_col: str = 'geom',
                     columns: List[str] = None, where: str = None) -> gpd.GeoDataFrame:
        """Lit une table PostGIS en GeoDataFrame"""
        cols = ', '.join(columns) if columns else '*'
        query = f"SELECT {cols} FROM {table}"
        if where:
            query += f" WHERE {where}"
        return gpd.read_postgis(query, self.engine, geom_col=geom_col)

    def write_postgis(self, gdf: gpd.GeoDataFrame, table: str,
                      if_exists: str = 'replace', schema: str = 'public') -> bool:
        """Écrit un GeoDataFrame dans PostGIS"""
        try:
            gdf.to_postgis(table, self.engine, if_exists=if_exists,
                          schema=schema, index=False)
            return True
        except Exception as e:
            print(f"Erreur écriture PostGIS: {e}")
            return False

    # ========================
    # Fonctions spatiales
    # ========================

    def buffer(self, table: str, distance: float, output_table: str,
               geom_col: str = 'geom') -> bool:
        """Crée des buffers dans PostGIS"""
        query = f"""
        CREATE TABLE {output_table} AS
        SELECT *, ST_Buffer({geom_col}::geography, {distance})::geometry as buffer_geom
        FROM {table};
        """
        self.execute(query)
        return True

    def intersection(self, table1: str, table2: str, output_table: str) -> bool:
        """Intersection de deux tables"""
        query = f"""
        CREATE TABLE {output_table} AS
        SELECT a.*, ST_Intersection(a.geom, b.geom) as geom_intersection
        FROM {table1} a, {table2} b
        WHERE ST_Intersects(a.geom, b.geom);
        """
        self.execute(query)
        return True

    def nearest_neighbor(self, from_table: str, to_table: str,
                         k: int = 1) -> List[Dict]:
        """Trouve les k plus proches voisins"""
        query = f"""
        SELECT
            a.id as from_id,
            b.id as to_id,
            ST_Distance(a.geom::geography, b.geom::geography) as distance_meters
        FROM {from_table} a
        CROSS JOIN LATERAL (
            SELECT id, geom
            FROM {to_table}
            ORDER BY a.geom <-> geom
            LIMIT {k}
        ) b;
        """
        return self.execute(query)

    def create_spatial_index(self, table: str, geom_col: str = 'geom') -> bool:
        """Crée un index spatial"""
        query = f"CREATE INDEX IF NOT EXISTS idx_{table}_{geom_col} ON {table} USING GIST ({geom_col});"
        self.execute(query)
        return True

    def get_extent(self, table: str, geom_col: str = 'geom') -> Dict:
        """Retourne l'emprise d'une table"""
        query = f"""
        SELECT
            ST_XMin(extent) as xmin,
            ST_YMin(extent) as ymin,
            ST_XMax(extent) as xmax,
            ST_YMax(extent) as ymax
        FROM (SELECT ST_Extent({geom_col}) as extent FROM {table}) sub;
        """
        result = self.execute(query)
        return result[0] if result else {}

    def to_geojson(self, table: str, geom_col: str = 'geom',
                   limit: int = None) -> Dict:
        """Exporte une table en GeoJSON"""
        limit_clause = f"LIMIT {limit}" if limit else ""
        query = f"""
        SELECT jsonb_build_object(
            'type', 'FeatureCollection',
            'features', jsonb_agg(ST_AsGeoJSON(t.*)::jsonb)
        )
        FROM (SELECT * FROM {table} {limit_clause}) t;
        """
        result = self.execute(query)
        return result[0] if result else {}

    def close(self):
        """Ferme les connexions"""
        if self._conn:
            self._conn.close()
        if self._engine:
            self._engine.dispose()


# Instance globale
_client = None

def get_client(**kwargs) -> PostGISClient:
    """Retourne un client PostGIS"""
    global _client
    if _client is None:
        _client = PostGISClient(**kwargs)
    return _client


# Fonctions directes pour n8n
def postgis_query(sql: str, params: tuple = None) -> List[Dict]:
    """Exécute une requête PostGIS"""
    return get_client().execute(sql, params)


def postgis_read(table: str, where: str = None) -> Dict:
    """Lit une table PostGIS et retourne du GeoJSON"""
    gdf = get_client().read_postgis(table, where=where)
    return json.loads(gdf.to_json())


def postgis_write(geojson: Dict, table: str, if_exists: str = 'replace') -> bool:
    """Écrit du GeoJSON dans PostGIS"""
    gdf = gpd.GeoDataFrame.from_features(geojson['features'])
    return get_client().write_postgis(gdf, table, if_exists=if_exists)