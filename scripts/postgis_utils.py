#!/usr/bin/env python3
"""
Utilitaires PostGIS pour n8n (robustes et idempotents)
- Connexion via DATABASE_URL ou variables PG*
- Exécution avec transactions et erreurs structurées
- Export GeoJSON canonique (FeatureCollection)
- Opérations spatiales idempotentes (buffer/intersection) + index
"""
import os
import json
import logging
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor
import geopandas as gpd
from sqlalchemy import create_engine

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def _db_url_from_env() -> str:
    if os.getenv("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    host = os.getenv("PGHOST") or os.getenv("DB_POSTGRESDB_HOST", "postgis")
    port = os.getenv("PGPORT") or os.getenv("DB_POSTGRESDB_PORT", "5432")
    user = os.getenv("PGUSER") or os.getenv("DB_POSTGRESDB_USER", "geo")
    password = os.getenv("PGPASSWORD") or os.getenv("DB_POSTGRESDB_PASSWORD", "geo_password")
    db = os.getenv("PGDATABASE") or os.getenv("DB_POSTGRESDB_DATABASE", "cadastre")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def _conn_kwargs_from_url(url: str) -> Dict[str, Any]:
    from urllib.parse import urlparse
    u = urlparse(url)
    return {
        "host": u.hostname or "localhost",
        "port": u.port or 5432,
        "user": (u.username or "postgres"),
        "password": (u.password or ""),
        "dbname": (u.path.lstrip("/") or "postgres"),
    }


def _safe_ident(name: str) -> str:
    # Quoting simple des identifiants (schema/table/col). Pour des besoins avancés, utiliser psycopg2.sql.Identifier
    return '"' + name.replace('"', '""') + '"'


class PostGISClient:
    """Client PostGIS pour opérations géospatiales"""

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or _db_url_from_env()
        self._pool = None
        self._engine = None

    @property
    def pool(self):
        """Pool de connexion psycopg2 lazy"""
        if self._pool is None or self._pool.closed:
            from psycopg2 import pool
            cfg = _conn_kwargs_from_url(self.db_url)
            # minconn=1, maxconn=10
            self._pool = pool.SimpleConnectionPool(1, 10, **cfg)
        return self._pool

    @contextmanager
    def get_conn_cursor(self):
        """Context manager pour obtenir une connexion et un curseur du pool"""
        conn = self.pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                yield conn, cur
        finally:
            self.pool.putconn(conn)

    @property
    def engine(self):
        """Engine SQLAlchemy lazy"""
        if self._engine is None:
            self._engine = create_engine(self.db_url)
        return self._engine

    def execute(self, query: str, params: Optional[tuple] = None) -> List[Dict]:
        """Exécute une requête SQL avec gestion d'erreur et commit."""
        try:
            with self.get_conn_cursor() as (conn, cur):
                cur.execute(query, params)
                try:
                    if cur.description:
                        rows = [dict(row) for row in cur.fetchall()]
                        conn.commit()
                        return rows
                except psycopg2.ProgrammingError:
                     # Cas où la requête ne retourne rien
                     pass
                conn.commit()
                return []
        except Exception:
            logger.exception("Erreur SQL")
            raise

    def read_postgis(self, table: str, geom_col: str = 'geom',
                     columns: Optional[List[str]] = None, where: Optional[str] = None) -> gpd.GeoDataFrame:
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
        except Exception:
            logger.exception("Erreur écriture PostGIS")
            return False

    # ========================
    # Fonctions spatiales
    # ========================

    def buffer(self, table: str, distance: float, output_table: str,
               geom_col: str = 'geom') -> bool:
        """Crée des buffers (idempotent)"""
        t = _safe_ident(table)
        out = _safe_ident(output_table)
        gc = _safe_ident(geom_col)
        q = f"""
        DROP TABLE IF EXISTS {out};
        CREATE TABLE {out} AS
        SELECT *, ST_Buffer({gc}::geography, %s)::geometry as buffer_geom
        FROM {t};
        CREATE INDEX IF NOT EXISTS idx_{output_table}_buffer_geom ON {out} USING GIST(buffer_geom);
        """
        self.execute(q, (distance,))
        return True

    def intersection(self, table1: str, table2: str, output_table: str,
                     geom_col1: str = 'geom', geom_col2: str = 'geom') -> bool:
        """Intersection de deux tables (idempotent)"""
        t1 = _safe_ident(table1)
        t2 = _safe_ident(table2)
        out = _safe_ident(output_table)
        g1 = _safe_ident(geom_col1)
        g2 = _safe_ident(geom_col2)
        q = f"""
        DROP TABLE IF EXISTS {out};
        CREATE TABLE {out} AS
        SELECT a.*, ST_Intersection(a.{g1}, b.{g2}) as geom_intersection
        FROM {t1} a
        JOIN {t2} b ON ST_Intersects(a.{g1}, b.{g2});
        CREATE INDEX IF NOT EXISTS idx_{output_table}_geom_intersection ON {out} USING GIST(geom_intersection);
        """
        self.execute(q)
        return True

    def nearest_neighbor(self, from_table: str, to_table: str,
                         k: int = 1, from_id_col: str = 'id', to_id_col: str = 'id',
                         geom_col_from: str = 'geom', geom_col_to: str = 'geom') -> List[Dict]:
        """Trouve les k plus proches voisins (paramétrable)."""
        q = f"""
        SELECT
            a.{from_id_col} as from_id,
            b.{to_id_col} as to_id,
            ST_Distance(a.{geom_col_from}::geography, b.{geom_col_to}::geography) as distance_meters
        FROM {from_table} a
        CROSS JOIN LATERAL (
            SELECT {to_id_col}, {geom_col_to}
            FROM {to_table}
            ORDER BY a.{geom_col_from} <-> {geom_col_to}
            LIMIT %s
        ) b;
        """
        return self.execute(q, (k,))

    def create_spatial_index(self, table: str, geom_col: str = 'geom') -> bool:
        t = _safe_ident(table)
        gc = _safe_ident(geom_col)
        q = f"CREATE INDEX IF NOT EXISTS idx_{table}_{geom_col} ON {t} USING GIST ({gc});"
        self.execute(q)
        return True

    def get_extent(self, table: str, geom_col: str = 'geom') -> Dict:
        q = f"""
        SELECT
            ST_XMin(extent) as xmin,
            ST_YMin(extent) as ymin,
            ST_XMax(extent) as xmax,
            ST_YMax(extent) as ymax
        FROM (SELECT ST_Extent({geom_col}) as extent FROM {table}) sub;
        """
        result = self.execute(q)
        return result[0] if result else {}

    def to_geojson(self, table: str, geom_col: str = 'geom',
                   limit: Optional[int] = None, include_bbox: bool = False) -> Dict:
        """Exporte une table en FeatureCollection canonique."""
        limit_clause = f"LIMIT {int(limit)}" if limit else ""
        bbox_part = ", 'bbox', (SELECT jsonb_build_array(ST_XMin(env), ST_YMin(env), ST_XMax(env), ST_YMax(env)) FROM (SELECT ST_Extent({geom_col}) AS env FROM {table}) e)" if include_bbox else ""
        q = f"""
        SELECT jsonb_build_object(
          'type','FeatureCollection',
          'features', jsonb_agg(
            jsonb_build_object(
              'type','Feature',
              'properties', to_jsonb(t) - '{{geom}}',
              'geometry', ST_AsGeoJSON({geom_col})::jsonb
            )
          ){bbox_part}
        ) as collection
        FROM (
          SELECT * FROM {table} {limit_clause}
        ) t;
        """
        res = self.execute(q)
        return res[0]["collection"] if res and res[0].get("collection") else {"type":"FeatureCollection","features":[]}

    def close(self):
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
        if self._engine:
            try:
                self._engine.dispose()
            except Exception:
                pass


_client = None

def get_client(**kwargs) -> PostGISClient:
    global _client
    if _client is None:
        _client = PostGISClient(**kwargs)
    return _client


def postgis_query(sql: str, params: Optional[tuple] = None) -> List[Dict]:
    return get_client().execute(sql, params)


def postgis_read(table: str, where: Optional[str] = None) -> Dict:
    gdf = get_client().read_postgis(table, where=where)
    return json.loads(gdf.to_json())


def postgis_write(geojson: Dict, table: str, if_exists: str = 'replace') -> bool:
    gdf = gpd.GeoDataFrame.from_features(geojson['features'])
    return get_client().write_postgis(gdf, table, if_exists=if_exists)
