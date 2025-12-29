#!/usr/bin/env python3
"""
Adaptateur MCP pour DomainAgent
- Action principale: inventory (liste/compte des biens)
- Retour JSON standard: {status, message, results, metrics}
"""
from __future__ import annotations
import json
from typing import Dict, Any, Optional
from scripts.postgis_utils import postgis_query  # type: ignore

TOOL_SCHEMA = {
    "name": "domain_agent",
    "description": "Agent domanial (inventaire des biens)",
    "params": {
        "action": {"type": "string", "required": True, "enum": ["inventory"]},
        "inputs": {"type": "object", "required": True, "properties": {"table": {"type": "string"}}},
        "options": {"type": "object", "required": False, "properties": {"filters": {"type": "object"}}}
    },
    "example": {"tool_name": "domain_agent", "params": {"action": "inventory", "inputs": {"table": "domaine.biens"}}}
}


def _ok(message: str, results: Optional[Dict[str, Any]] = None, **metrics) -> Dict[str, Any]:
    return {"status": "success", "message": message, "results": results or {}, "metrics": metrics}


def _err(message: str, **extra) -> Dict[str, Any]:
    return {"status": "error", "message": message, **({"results": extra} if extra else {})}


def run_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        action = (params.get("action") or "").strip().lower()
        inputs = params.get("inputs") or {}
        if action != "inventory":
            return _err(f"Action inconnue: {action}")
        table = inputs.get("table")
        if not table:
            return _err("'table' est requis")
        stats = postgis_query(f"SELECT COUNT(*)::int AS total FROM {table}")[0]
        return _ok("Inventaire terminé", {"table": table, "total": stats["total"]}, total=stats["total"])
    except Exception as e:
        return _err(str(e))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--table", required=True)
    args = p.parse_args()
    print(json.dumps(run_tool({"action":"inventory","inputs":{"table":args.table}}), ensure_ascii=False))
