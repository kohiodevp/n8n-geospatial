#!/usr/bin/env python3
"""
Adaptateur MCP pour l'Agent Cadastral
- Expose run_tool(params) pour /invoke du MCP
- Actions supportées: import, validate, analyze
- Retour JSON standard: {status, message, results, metrics}

Dépendances:
- scripts/import_postgis.py pour l'import bulk
- scripts/postgis_utils.py pour les requêtes PostGIS
"""
from __future__ import annotations
import os
import json
from typing import Dict, Any, Optional

# Import locaux
from scripts import import_postgis  # type: ignore
from scripts.postgis_utils import postgis_query  # type: ignore


TOOL_SCHEMA = {
    "name": "cadastral_agent",
    "description": "Agent cadastral (import, validation, analyse) via MCP",
    "params": {
        "action": {"type": "string", "required": True, "enum": ["import", "validate", "analyze"]},
        "inputs": {
            "type": "object",
            "required": True,
            "properties": {
                "file": {"type": "string", "description": "Chemin du fichier à importer (GeoJSON, SHP, etc.)"},
                "table": {"type": "string", "description": "Table cible pour import/validation/analyse"},
                "schema": {"type": "string", "default": "cadastre"},
                "srid": {"type": "integer", "default": 2154},
            }
        },
        "options": {
            "type": "object",
            "required": False,
            "properties": {
                "if_exists": {"type": "string", "enum": ["append", "replace", "fail"], "default": "append"},
                "chunksize": {"type": "integer", "default": 5000},
                "fix_invalid": {"type": "boolean", "default": False},
                "summary": {"type": "boolean", "default": True}
            }
        }
    },
    "example": {
        "tool_name": "cadastral_agent",
        "params": {
            "action": "import",
            "inputs": {"file": "/files/sample_parcelles.geojson", "table": "cadastre.parcelles", "srid": 2154},
            "options": {"if_exists": "append", "chunksize": 5000}
        }
    }
}


def _ok(message: str, results: Optional[Dict[str, Any]] = None, **metrics) -> Dict[str, Any]:
    return {"status": "success", "message": message, "results": results or {}, "metrics": metrics}


def _err(message: str, **extra) -> Dict[str, Any]:
    return {"status": "error", "message": message, **({"results": extra} if extra else {})}


def _validate_table(table: str, fix_invalid: bool = False) -> Dict[str, Any]:
    # Compter total, geom NULL, invalides, srid
    total = postgis_query(f"SELECT COUNT(*)::int AS n FROM {table}")[0]["n"]
    nulls = postgis_query(f"SELECT COUNT(*)::int AS n FROM {table} WHERE geom IS NULL")[0]["n"]
    invalid = postgis_query(f"SELECT COUNT(*)::int AS n FROM {table} WHERE geom IS NOT NULL AND NOT ST_IsValid(geom)")[0]["n"]
    srid = postgis_query(f"SELECT COALESCE(ST_SRID(geom),0)::int AS srid FROM {table} WHERE geom IS NOT NULL LIMIT 1")
    srid_val = srid[0]["srid"] if srid else 0

    fixed = 0
    if fix_invalid and invalid > 0:
        postgis_query(f"UPDATE {table} SET geom = ST_MakeValid(geom) WHERE geom IS NOT NULL AND NOT ST_IsValid(geom)")
        fixed = postgis_query(f"SELECT COUNT(*)::int AS n FROM {table} WHERE geom IS NOT NULL AND NOT ST_IsValid(geom)")[0]["n"]
        fixed = invalid - fixed

    results = {
        "table": table,
        "total": total,
        "geom_null": nulls,
        "invalid_before": invalid,
        "invalid_fixed": fixed,
        "srid": srid_val,
    }
    return _ok("Validation terminée", results, total=total, invalid=invalid, fixed=fixed)


def _analyze_table(table: str, summary: bool = True) -> Dict[str, Any]:
    # Stats simples (surface en m² et nombre de sections si présent)
    stats = postgis_query(
        f"SELECT COUNT(*)::int AS total, COALESCE(SUM(ST_Area(geom::geography))::double precision,0) AS area_m2 FROM {table}"
    )[0]
    sections = 0
    try:
        sections = postgis_query(f"SELECT COUNT(DISTINCT section)::int AS n FROM {table}")[0]["n"]
    except Exception:
        pass
    results = {
        "table": table,
        "total": stats["total"],
        "area_m2": float(stats["area_m2"] or 0),
        "nb_sections": sections,
    }
    return _ok("Analyse terminée", results, total=results["total"], area_m2=results["area_m2"], nb_sections=sections)


def run_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Exécuter une action de l'agent cadastral via MCP.
    params = { action, inputs:{file/table/schema/srid}, options:{...} }
    """
    try:
        action = (params.get("action") or "").strip().lower()
        inputs = params.get("inputs") or {}
        options = params.get("options") or {}

        if action not in {"import", "validate", "analyze"}:
            return _err(f"Action inconnue: {action}")

        table = inputs.get("table")
        schema = inputs.get("schema", "cadastre")
        if action in {"validate", "analyze"} and not table:
            return _err("'table' est requis pour validate/analyze")

        if action == "import":
            file_path = inputs.get("file")
            if not file_path or not os.path.exists(file_path):
                return _err(f"Fichier introuvable: {file_path}")
            srid = int(inputs.get("srid", 2154))
            if_exists = options.get("if_exists", "append")
            chunksize = options.get("chunksize")
            # Déterminer nom de table (peut inclure schéma)
            tbl = table or "parcelles"
            tbl_schema = schema or "cadastre"
            res = import_postgis.import_geodata(
                filepath=file_path,
                table_name=tbl,
                schema=tbl_schema,
                target_epsg=srid,
                if_exists=if_exists,
                chunksize=chunksize,
            )
            if res.get("status") != "success":
                return _err(res.get("message", "Échec import"))
            return _ok(
                "Import terminé",
                {"table": f"{tbl_schema}.{tbl}", "srid": srid, "count": res.get("count", 0)},
                count=res.get("count", 0), srid=srid,
            )

        if action == "validate":
            fix_invalid = bool(options.get("fix_invalid", False))
            return _validate_table(table, fix_invalid)

        if action == "analyze":
            return _analyze_table(table, summary=bool(options.get("summary", True)))

        return _err("Action non gérée")
    except Exception as e:
        return _err(str(e))


if __name__ == "__main__":
    # Exécution simple via CLI pour debug
    import argparse
    parser = argparse.ArgumentParser(description="Cadastral MCP Adapter")
    parser.add_argument("action", choices=["import", "validate", "analyze"])  # type: ignore
    parser.add_argument("--file", dest="file", help="Chemin du fichier à importer")
    parser.add_argument("--table", dest="table", help="Table cible")
    parser.add_argument("--schema", dest="schema", default="cadastre")
    parser.add_argument("--srid", dest="srid", type=int, default=2154)
    parser.add_argument("--if-exists", dest="ifexists", choices=["append","replace","fail"], default="append")
    parser.add_argument("--chunksize", dest="chunksize", type=int, default=None)
    parser.add_argument("--fix-invalid", dest="fix_invalid", action="store_true")
    args = parser.parse_args()

    payload = {
        "action": args.action,
        "inputs": {"file": args.file, "table": args.table, "schema": args.schema, "srid": args.srid},
        "options": {"if_exists": args.ifexists, "chunksize": args.chunksize, "fix_invalid": args.fix_invalid},
    }
    print(json.dumps(run_tool(payload), ensure_ascii=False))
