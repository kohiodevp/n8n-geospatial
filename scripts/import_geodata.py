#!/usr/bin/env python3
"""
Tool: import_geodata
- Dézippe si nécessaire
- Détecte le format (SHP/GeoJSON/GPX/CSV)
- Importe en PostGIS via ogr2ogr
- Met à jour api.job_status (status, message)

Params attendus:
{
  "job_id": "<uuid>",
  "file_path": "/files/uploads/<name>",
  "target_schema": "cadastre",
  "target_table": "parcelles",
  "srid": 4326,
  "strategy": "append" | "replace"
}
"""
import os
import json
import shutil
import tempfile
import subprocess
import psycopg2
from datetime import datetime

TOOL_SCHEMA = {
    "name": "import_geodata",
    "description": "Importe un fichier géospatial (ZIP/SHP/GeoJSON/GPX/CSV) dans PostGIS via ogr2ogr",
    "params": {
        "job_id": {"type": "string", "required": True},
        "file_path": {"type": "string", "required": True},
        "target_schema": {"type": "string", "required": True},
        "target_table": {"type": "string", "required": True},
        "srid": {"type": "integer", "required": False, "default": 4326},
        "strategy": {"type": "string", "enum": ["append", "replace"], "default": "append"}
    }
}

PG_DSN = {
    "host": os.getenv("PGHOST", "postgis"),
    "port": os.getenv("PGPORT", "5432"),
    "dbname": os.getenv("PGDATABASE", "n8n"),
    "user": os.getenv("PGUSER", "n8n"),
    "password": os.getenv("PGPASSWORD", "n8npassword"),
}

def _pg_conn():
    return psycopg2.connect(
        host=PG_DSN["host"], port=PG_DSN["port"], dbname=PG_DSN["dbname"],
        user=PG_DSN["user"], password=PG_DSN["password"]
    )

def _ensure_job_table(cur):
    cur.execute(
        """
        CREATE SCHEMA IF NOT EXISTS api;
        CREATE TABLE IF NOT EXISTS api.job_status (
          id UUID PRIMARY KEY,
          filename TEXT,
          status TEXT NOT NULL DEFAULT 'queued',
          message TEXT,
          created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )

def _update_job(job_id: str, status: str, filename: str = None, message: str = None):
    with _pg_conn() as conn:
        with conn.cursor() as cur:
            _ensure_job_table(cur)
            cur.execute(
                """
                INSERT INTO api.job_status (id, filename, status, message)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                  status = EXCLUDED.status,
                  filename = COALESCE(EXCLUDED.filename, api.job_status.filename),
                  message = EXCLUDED.message,
                  updated_at = NOW();
                """,
                (job_id, filename, status, message)
            )


def _detect_source_file(path: str) -> str:
    """Retourne le fichier source à ingérer (généralement .shp/.geojson/.gpx/.csv)."""
    if os.path.isdir(path):
        # Chercher un fichier prioritaire
        for ext in (".shp", ".geojson", ".gpx", ".csv"):
            for root, _dirs, files in os.walk(path):
                for f in files:
                    if f.lower().endswith(ext):
                        return os.path.join(root, f)
        return None
    return path


def _unzip_if_needed(file_path: str) -> str:
    lower = file_path.lower()
    if lower.endswith(".zip"):
        tmpdir = tempfile.mkdtemp(prefix="import_geodata_")
        shutil.unpack_archive(file_path, tmpdir)
        return tmpdir
    return file_path


def _ogr2ogr_import(src: str, schema: str, table: str, srid: int, strategy: str) -> subprocess.CompletedProcess:
    # Connexion OGR
    pg_conn_str = (
        f"PG:host={PG_DSN['host']} port={PG_DSN['port']} dbname={PG_DSN['dbname']} "
        f"user={PG_DSN['user']} password={PG_DSN['password']}"
    )
    layer_name = f"{schema}.{table}"
    args = [
        "ogr2ogr", "-f", "PostgreSQL", pg_conn_str, src,
        "-nln", layer_name,
        "-lco", "GEOMETRY_NAME=geom",
        "-lco", "FID=gid",
        "-skipfailures",
    ]
    if strategy == "replace":
        args.append("-overwrite")
    else:
        args.append("-append")
    # Reprojection vers SRID cible (4326 par défaut)
    if srid and int(srid) != 0:
        args.extend(["-t_srs", f"EPSG:{int(srid)}"])
    # Tenter d'ajouter SRS si absent
    # args.extend(["-a_srs", f"EPSG:{int(srid)}"])  # optionnel selon données
    return subprocess.run(args, capture_output=True, text=True)


def run_tool(params: dict):
    job_id = params.get("job_id")
    file_path = params.get("file_path")
    schema = params.get("target_schema")
    table = params.get("target_table")
    srid = int(params.get("srid", 4326))
    strategy = params.get("strategy", "append")

    if not all([job_id, file_path, schema, table]):
        return {"ok": False, "error": "Missing required params: job_id, file_path, target_schema, target_table"}

    logs_dir = "/files/logs"
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"job_{job_id}.log")

    _update_job(job_id, "running", os.path.basename(file_path), "Starting import")

    try:
        path = _unzip_if_needed(file_path)
        source = _detect_source_file(path)
        if not source or not os.path.exists(source):
            _update_job(job_id, "failed", message="Unsupported or missing geodata source")
            return {"ok": False, "error": "Unsupported or missing geodata source", "log_path": log_path}

        proc = _ogr2ogr_import(source, schema, table, srid, strategy)
        # Write full logs
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("# ogr2ogr import log\n")
                f.write(f"source={source}\n")
                f.write(f"schema={schema} table={table} srid={srid} strategy={strategy}\n\n")
                f.write("[STDOUT]\n" + (proc.stdout or "") + "\n\n")
                f.write("[STDERR]\n" + (proc.stderr or "") + "\n")
        except Exception:
            pass

        if proc.returncode == 0:
            _update_job(job_id, "success", message=f"Imported successfully. See log: {log_path}")
            return {"ok": True, "message": "Import success", "log_path": log_path}
        else:
            _update_job(job_id, "failed", message=f"ogr2ogr error. See log: {log_path}")
            return {"ok": False, "error": "ogr2ogr failed", "log_path": log_path}
    except Exception as e:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[EXCEPTION] {e}\n")
        except Exception:
            pass
        _update_job(job_id, "failed", message=str(e))
        return {"ok": False, "error": str(e), "log_path": log_path}
