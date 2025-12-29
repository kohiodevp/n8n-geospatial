#!/usr/bin/env python3
"""
Adaptateur MCP pour EnvironmentalAgent
- Action principale: buffer (créer une zone tampon)
"""
from __future__ import annotations
import json
from typing import Dict, Any, Optional
from scripts.postgis_utils import postgis_query, PostGISClient  # type: ignore

TOOL_SCHEMA = {
    "name": "environmental_agent",
    "description": "Agent environnemental (buffers/alertes)",
    "params": {
        "action": {"type": "string", "required": True, "enum": ["buffer"]},
        "inputs": {"type": "object", "required": True, "properties": {"table": {"type": "string"}, "distance_m": {"type": "number"}, "output_table": {"type": "string"}}},
        "options": {"type": "object", "required": False, "properties": {}}
    },
    "example": {"tool_name": "environmental_agent", "params": {"action": "buffer", "inputs": {"table": "environment.sites", "distance_m": 200, "output_table": "environment.sites_buffer"}}}
}


def _ok(message: str, results: Optional[Dict[str, Any]] = None, **metrics) -> Dict[str, Any]:
    return {"status": "success", "message": message, "results": results or {}, "metrics": metrics}


def _err(message: str, **extra) -> Dict[str, Any]:
    return {"status": "error", "message": message, **({"results": extra} if extra else {})}


def run_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        action = (params.get("action") or "").strip().lower()
        inputs = params.get("inputs") or {}
        if action != "buffer":
            return _err(f"Action inconnue: {action}")
        table = inputs.get("table")
        dist = float(inputs.get("distance_m", 100))
        out = inputs.get("output_table", f"{table}_buffer")
        if not table:
            return _err("'table' est requis")
        # Utiliser client pour buffer idempotent (DROP+CREATE) + index
        client = PostGISClient()
        client.buffer(table=table, distance=dist, output_table=out, geom_col='geom')
        cnt = postgis_query(f"SELECT COUNT(*)::int AS n FROM {out}")[0]["n"]
        return _ok("Buffer terminé", {"table": table, "output_table": out, "count": cnt, "distance_m": dist}, count=cnt, distance_m=dist)
    except Exception as e:
        return _err(str(e))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--table", required=True)
    p.add_argument("--distance_m", type=float, default=200)
    p.add_argument("--output_table", default=None)
    args = p.parse_args()
    print(json.dumps(run_tool({"action":"buffer","inputs":{"table":args.table, "distance_m": args.distance_m, "output_table": args.output_table}}), ensure_ascii=False))
