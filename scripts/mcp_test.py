#!/usr/bin/env python3
"""
MCP test helper script
- Permet de tester les adaptateurs MCP (domain/cadastral/urbanism/environmental)
- Envoie des requêtes HTTP JSON vers MCP /invoke

Usage:
  python3 scripts/mcp_test.py domain inventory --table domaine.biens
  python3 scripts/mcp_test.py urbanism zoning_check --parcels cadastre.parcelles --zoning urbanisme.zone_urba
  python3 scripts/mcp_test.py environmental buffer --table environment.sites --distance 200 --output environment.sites_buffer
  python3 scripts/mcp_test.py cadastral validate --table cadastre.parcelles --fix-invalid
  python3 scripts/mcp_test.py cadastral import --file /files/sample.geojson --table parcelles --schema cadastre --srid 2154 --if-exists append --chunksize 5000
  python3 scripts/mcp_test.py cadastral analyze --table cadastre.parcelles

Options:
  --url pour définir l'URL MCP (défaut: http://localhost:5001/invoke)
"""
from __future__ import annotations
import argparse
import json
import sys
import requests

DEFAULT_URL = "http://localhost:5001/invoke"


def post(url: str, payload: dict, timeout: int = 60):
    r = requests.post(url, json=payload, timeout=timeout)
    print(r.status_code)
    try:
        print(json.dumps(r.json(), ensure_ascii=False, indent=2))
    except Exception:
        print(r.text)
    return r.status_code, r.text


def main():
    p = argparse.ArgumentParser(description="MCP test helper")
    p.add_argument("agent", choices=["domain", "urbanism", "environmental", "cadastral"], help="Agent MCP")
    p.add_argument("action", help="Action de l'agent")
    p.add_argument("--url", default=DEFAULT_URL, help="URL MCP /invoke")

    # Arguments communs
    p.add_argument("--table", help="Nom de table (selon agent)")

    # Cadastral import
    p.add_argument("--file", help="Chemin du fichier à importer")
    p.add_argument("--schema", default="cadastre", help="Schéma cible")
    p.add_argument("--srid", type=int, default=2154, help="SRID cible")
    p.add_argument("--if-exists", dest="if_exists", choices=["append", "replace", "fail"], default="append")
    p.add_argument("--chunksize", type=int, default=None)

    # Cadastral validate
    p.add_argument("--fix-invalid", dest="fix_invalid", action="store_true")

    # Urbanism
    p.add_argument("--parcels", help="Table des parcelles (urbanism)")
    p.add_argument("--zoning", help="Table des zones (urbanism)")

    # Environmental
    p.add_argument("--distance", type=float, default=None, help="Distance en mètres (environmental)")
    p.add_argument("--output", help="Table de sortie (environmental)")

    args = p.parse_args()

    tool_map = {
        "domain": "domain_agent",
        "urbanism": "urbanism_agent",
        "environmental": "environmental_agent",
        "cadastral": "cadastral_agent",
    }
    tool_name = tool_map[args.agent]
    action = args.action

    payload = {"tool_name": tool_name, "params": {"action": action, "inputs": {}, "options": {}}}

    if args.agent == "domain":
        if action != "inventory":
            print("Action supportée: inventory", file=sys.stderr)
            sys.exit(2)
        if not args.table:
            print("--table requis", file=sys.stderr)
            sys.exit(2)
        payload["params"]["inputs"] = {"table": args.table}

    elif args.agent == "urbanism":
        if action != "zoning_check":
            print("Action supportée: zoning_check", file=sys.stderr)
            sys.exit(2)
        if not args.parcels or not args.zoning:
            print("--parcels et --zoning requis", file=sys.stderr)
            sys.exit(2)
        payload["params"]["inputs"] = {"parcels_table": args.parcels, "zoning_table": args.zoning}

    elif args.agent == "environmental":
        if action != "buffer":
            print("Action supportée: buffer", file=sys.stderr)
            sys.exit(2)
        if not args.table:
            print("--table requis", file=sys.stderr)
            sys.exit(2)
        inputs = {"table": args.table}
        if args.distance is not None:
            inputs["distance_m"] = args.distance
        if args.output:
            inputs["output_table"] = args.output
        payload["params"]["inputs"] = inputs

    elif args.agent == "cadastral":
        if action == "import":
            if not args.file:
                print("--file requis", file=sys.stderr)
                sys.exit(2)
            table = args.table or "parcelles"
            payload["params"]["inputs"] = {
                "file": args.file,
                "table": table,
                "schema": args.schema,
                "srid": args.srid,
            }
            payload["params"]["options"] = {
                "if_exists": args.if_exists,
                "chunksize": args.chunksize,
            }
        elif action == "validate":
            if not args.table:
                print("--table requis", file=sys.stderr)
                sys.exit(2)
            payload["params"]["inputs"] = {"table": args.table}
            payload["params"]["options"] = {"fix_invalid": bool(args.fix_invalid)}
        elif action == "analyze":
            if not args.table:
                print("--table requis", file=sys.stderr)
                sys.exit(2)
            payload["params"]["inputs"] = {"table": args.table}
            payload["params"]["options"] = {"summary": True}
        else:
            print("Actions supportées pour cadastral: import, validate, analyze", file=sys.stderr)
            sys.exit(2)

    else:
        print("Agent inconnu", file=sys.stderr)
        sys.exit(2)

    post(args.url, payload)


if __name__ == "__main__":
    main()
