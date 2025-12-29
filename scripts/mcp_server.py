
import http.server
import socketserver
import json
import importlib
import traceback
import os
import sys

# Ajoute le répertoire du script au sys.path pour permettre les imports relatifs
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
# Assurer que /opt/geoscripts est inclus (montage docker-compose)
opt_dir = "/opt/geoscripts"
if opt_dir not in sys.path and os.path.isdir(opt_dir):
    sys.path.append(opt_dir)
    try:
        # Log minimal en stdout pour diagnostic
        print(f"MCP Server: sys.path updated with {opt_dir}")
        print(f"MCP Server: available files: {sorted(os.listdir(opt_dir))}")
    except Exception:
        pass

PORT = 5001
HOST = "0.0.0.0"

def _resolve_module(tool_name: str):
    """Résoudre le module Python à charger pour un tool_name donné.
    - Essaye 'tool_name' puis un alias '*_adapter' si présent.
    """
    aliases = {
        "cadastral_agent": "cadastral_adapter",
        "domain_agent": "domain_adapter",
        "urbanism_agent": "urbanism_adapter",
        "environmental_agent": "environmental_adapter",
    }
    try:
        m = importlib.import_module(tool_name)
        if hasattr(m, "run_tool"):
            return tool_name
    except Exception:
        pass
    alt = aliases.get(tool_name)
    if alt:
        try:
            m = importlib.import_module(alt)
            if hasattr(m, "run_tool"):
                return alt
        except Exception:
            pass
    return None


import logging
import time

# Configuration du logging structuré JSON
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)

logger = logging.getLogger("mcp_server")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)

def _list_available_tools():
    """Lister les outils disponibles."""
    tools = ["ping"]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for fname in os.listdir(base_dir):
        if not fname.endswith(".py"):
            continue
        mod = fname[:-3]
        if mod.startswith("_") or mod in {"mcp_server"}:
            continue
        try:
            m = importlib.import_module(mod)
            if hasattr(m, "run_tool"):
                tools.append(mod)
        except Exception as e:
            logger.debug(f"Skipping module {mod}: {e}")
    return sorted(set(tools))

class MCPRequestHandler(http.server.BaseHTTPRequestHandler):
    """
    Gère les requêtes HTTP pour le serveur MCP.
    """

    def _send_response(self, status_code, data):
        """Envoie une réponse JSON avec CORS."""
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def _tool_description(self, name: str):
        if name == "ping":
            return {
                "name": "ping",
                "description": "Outil de test qui renvoie 'pong' avec les paramètres.",
                "params": {
                    "source": {"type": "string", "required": False, "description": "Tag d'origine de l'appel"}
                },
                "example": {"tool_name": "ping", "params": {"source": "n8n"}}
            }
        try:
            m = importlib.import_module(name)
            # Priorité: TOOL_SCHEMA > TOOL_DESCRIPTION > __doc__
            schema = getattr(m, "TOOL_SCHEMA", None)
            if isinstance(schema, dict):
                return schema
            desc = getattr(m, "TOOL_DESCRIPTION", None)
            if isinstance(desc, dict):
                return desc
            doc = getattr(m, "__doc__", None)
            return {"name": name, "description": doc or "(pas de description)", "params": "inconnus"}
        except Exception:
            return None

    def _describe_tool(self, name: str):
        return self._tool_description(name)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_HEAD(self):
        """Répondre aux probes HEAD."""
        if self.path == "/healthz":
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        """Gère les requêtes GET pour la santé et l'info des endpoints."""
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/healthz":
            self._send_response(200, {"status": "ok"})
        elif path == "/invoke":
            self._send_response(200, {
                "status": "ready",
                "message": "POST /invoke with {tool_name, params} to run a tool"
            })
        elif path == "/tools":
            # Exposer les noms 'grand public' (agents) si des adaptateurs existent
            tools = set(_list_available_tools())
            # Ajouter alias d'agents si résolus
            for t in ["cadastral_agent","domain_agent","urbanism_agent","environmental_agent"]:
                if _resolve_module(t):
                    tools.add(t)
            self._send_response(200, {"tools": sorted(tools)})
        elif path == "/describe":
            tool = (qs.get("tool") or [None])[0]
            if not tool:
                self._send_response(400, {"error": "Paramètre 'tool' manquant"})
                return
            desc = self._describe_tool(tool)
            if desc is None:
                # Tenter via alias/adaptateur
                mod = _resolve_module(tool)
                if mod:
                    try:
                        m = importlib.import_module(mod)
                        schema = getattr(m, "TOOL_SCHEMA", None)
                        if isinstance(schema, dict):
                            self._send_response(200, schema)
                            return
                    except Exception:
                        pass
                self._send_response(404, {"error": f"Outil inconnu: {tool}"})
            else:
                self._send_response(200, desc)
        else:
            self._send_response(404, {"error": "Endpoint non trouvé"})

    def do_POST(self):
        """Gère les requêtes POST, attendues sur l'endpoint /invoke."""
        if self.path == "/invoke":
            try:
                content_length = int(self.headers.get("Content-Length", "0") or 0)
                post_data = self.rfile.read(content_length) if content_length > 0 else b"{}"
                try:
                    body = json.loads(post_data.decode("utf-8") or "{}")
                except json.JSONDecodeError:
                    self._send_response(400, {"error": "Corps de la requête JSON invalide."})
                    return

                tool_name = body.get("tool_name")
                params = body.get("params", {})

                if not tool_name:
                    self._send_response(400, {"error": "Le paramètre 'tool_name' est manquant."})
                    return

                # --- Logique de dispatch de l'outil ---

                # Outil de test simple intégré
                if tool_name == "ping":
                    self._send_response(200, {"status": "pong", "params": params})
                    return

                # Chargement dynamique des outils depuis les autres scripts
                try:
                    mod_name = _resolve_module(tool_name) or tool_name
                    tool_module = importlib.import_module(mod_name)
                    
                    if hasattr(tool_module, "run_tool"):
                        logger.info(f"Invoking tool '{tool_name}' via '{mod_name}' params={json.dumps(params)}")
                        start_time = time.time()
                        result = tool_module.run_tool(params)
                        duration = time.time() - start_time
                        logger.info(f"Tool '{tool_name}' completed in {duration:.3f}s")
                        self._send_response(200, {"result": result})
                    else:
                        error_msg = f"L'outil '{tool_name}' n'a pas de fonction 'run_tool'."
                        logger.error(error_msg)
                        self._send_response(501, {"error": error_msg})

                except ImportError:
                    error_msg = f"L'outil '{tool_name}' n'a pas été trouvé."
                    logger.warning(error_msg)
                    self._send_response(404, {"error": error_msg})
                except Exception as e:
                    error_msg = f"Erreur lors de l'exécution de l'outil '{tool_name}': {e}"
                    logger.exception(error_msg)
                    self._send_response(500, {"error": error_msg, "trace": traceback.format_exc()})

            except json.JSONDecodeError:
                self._send_response(400, {"error": "Corps de la requête JSON invalide."})
            except Exception as e:
                error_msg = f"Erreur interne du serveur MCP: {e}"
                print(error_msg)
                traceback.print_exc()
                self._send_response(500, {"error": error_msg, "trace": traceback.format_exc()})
        else:
            self._send_response(404, {"error": "Endpoint non trouvé. Utilisez /invoke."})

class ThreadingHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True


def run_server():
    """Lance le serveur MCP."""
    with ThreadingHTTPServer((HOST, PORT), MCPRequestHandler) as httpd:
        print(f"Serveur MCP démarré sur http://{HOST}:{PORT}")
        print("Prêt à invoquer des outils via POST /invoke")
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()
