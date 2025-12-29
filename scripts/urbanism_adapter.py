#!/usr/bin/env python3
"""
Adaptateur MCP pour UrbanismAgent
- Action principale: zoning_check (vérification croisement parcelles/zones)
"""
from __future__ import annotations
import json
from typing import Dict, Any, Optional
from scripts.postgis_utils import postgis_query  # type: ignore

TOOL_SCHEMA = {
    "name": "urbanism_agent",
    "description": "Agent d'urbanisme (zoning check)",
    "params": {
        "action": {"type": "string", "required": True, "enum": ["zoning_check"]},
        "inputs": {"type": "object", "required": True, "properties": {"parcels_table": {"type": "string"}, "zoning_table": {"type": "string"}}},
        "options": {"type": "object", "required": False, "properties": {"spatial_index": {"type": "boolean", "default": True}}}
    },
    "example": {"tool_name": "urbanism_agent", "params": {"action": "zoning_check", "inputs": {"parcels_table": "cadastre.parcelles", "zoning_table": "urbanisme.zone_urba"}}}
}


def _ok(message: str, results: Optional[Dict[str, Any]] = None, **metrics) -> Dict[str, Any]:
    return {"status": "success", "message": message, "results": results or {}, "metrics": metrics}


def _err(message: str, **extra) -> Dict[str, Any]:
    return {"status": "error", "message": message, **({"results": extra} if extra else {})}


def run_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        action = (params.get("action") or "").strip().lower()
        inputs = params.get("inputs") or {}
        if action != "zoning_check":
            return _err(f"Action inconnue: {action}")
        pt = inputs.get("parcels_table")
        zt = inputs.get("zoning_table")
        if not pt or not zt:
            return _err("'parcels_table' et 'zoning_table' sont requis")
        # Comptage des parcelles intersectant des zones
        q = f"""
        SELECT COUNT(*)::int AS n
        FROM {pt} p
        JOIN {zt} z ON ST_Intersects(p.geom, z.geom);
        """
        n = postgis_query(q)[0]["n"]
        return _ok("Zoning check terminé", {"parcels_table": pt, "zoning_table": zt, "intersections": n}, intersections=n)
    except Exception as e:
        return _err(str(e))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--parcels_table", required=True)
    p.add_argument("--zoning_table", required=True)
    args = p.parse_args()
    print(json.dumps(run_tool({"action":"zoning_check","inputs":{"parcels_table":args.parcels_table, "zoning_table":args.zoning_table}}), ensure_ascii=False))
