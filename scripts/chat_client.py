#!/usr/bin/env python3
"""
GeoSpatial AI Chat Client
=========================
Un outil CLI interactif pour dialoguer avec les agents IA de n8n-geospatial.

Usage:
    python chat_client.py [guide|cadastre|domaine] [--url URL]

Requiert: requests
"""

import sys
import json
import time
import requests
import argparse
from datetime import datetime

# Configuration par défaut
DEFAULT_BASE_URL = "http://localhost:5678/webhook"
AGENTS = {
    "guide": {
        "path": "ai-guide",
        "name": "Guide IA",
        "icon": "🧭",
        "description": "Assistant de navigation et diagnostic système"
    },
    "cadastre": {
        "path": "ai-cadastral",
        "name": "Agent Cadastral",
        "icon": "🗺️",
        "description": "Expert en données cadastrales et parcellaires"
    },
    "domaine": {
        "path": "ai-domanial",
        "name": "Agent Domanial",
        "icon": "🏛️",
        "description": "Spécialiste gestion du domaine de l'État"
    },
}

def print_header():
    """Affiche l'en-tête du client."""
    print("\n" + "=" * 60)
    print("  🌍 n8n-geospatial AI Chat Client")
    print("=" * 60)

def print_agent_info(agent_key: str):
    """Affiche les informations sur l'agent sélectionné."""
    agent = AGENTS.get(agent_key, {})
    print(f"\n  {agent.get('icon', '🤖')} Agent: {agent.get('name', agent_key)}")
    print(f"  📝 {agent.get('description', '')}")
    print("-" * 60)

def list_agents():
    """Affiche la liste des agents disponibles."""
    print("\n📋 Agents disponibles:")
    for key, info in AGENTS.items():
        print(f"  • {key:10} {info['icon']} {info['name']}")
        print(f"              {info['description']}")
    print()

def extract_output(data: dict) -> str:
    """Extrait intelligemment la réponse de l'agent."""
    # Priorité des champs de sortie
    for field in ['output', 'text', 'message', 'response', 'content']:
        if field in data:
            value = data[field]
            if isinstance(value, str):
                return value
            elif isinstance(value, (dict, list)):
                return json.dumps(value, indent=2, ensure_ascii=False)
    
    # Si c'est une liste, prendre le premier élément
    if isinstance(data, list) and len(data) > 0:
        return extract_output(data[0]) if isinstance(data[0], dict) else str(data[0])
    
    # Fallback: tout le JSON
    return json.dumps(data, indent=2, ensure_ascii=False)

def chat_loop(agent_key: str, base_url: str, timeout: int = 120):
    """Boucle principale de chat avec un agent."""
    agent = AGENTS.get(agent_key, {"path": agent_key, "name": agent_key, "icon": "🤖"})
    url = f"{base_url}/{agent['path']}"
    
    print_header()
    print_agent_info(agent_key)
    print(f"  🔗 Endpoint: {url}")
    print(f"  ⏱️ Timeout: {timeout}s")
    print("\n  Tapez 'exit', 'quit' ou 'q' pour quitter.")
    print("  Tapez 'help' pour l'aide, 'switch' pour changer d'agent.\n")

    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    while True:
        try:
            user_input = input(f"👤 Vous: ").strip()
            
            # Commandes spéciales
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Au revoir!")
                break
            if not user_input:
                continue
            if user_input.lower() == 'help':
                print("\n📖 Commandes:")
                print("  • exit/quit/q  : Quitter")
                print("  • help         : Cette aide")
                print("  • switch       : Changer d'agent")
                print("  • status       : Tester la connexion\n")
                continue
            if user_input.lower() == 'switch':
                list_agents()
                new_agent = input("Quel agent? ").strip().lower()
                if new_agent in AGENTS:
                    agent_key = new_agent
                    agent = AGENTS[agent_key]
                    url = f"{base_url}/{agent['path']}"
                    print_agent_info(agent_key)
                    print(f"  🔗 Endpoint: {url}\n")
                else:
                    print(f"❌ Agent inconnu: {new_agent}\n")
                continue
            if user_input.lower() == 'status':
                try:
                    resp = session.get(f"{base_url.rsplit('/webhook', 1)[0]}/healthz", timeout=5)
                    print(f"✅ Serveur n8n: OK ({resp.status_code})\n")
                except Exception as e:
                    print(f"❌ Serveur n8n injoignable: {e}\n")
                continue

            # Envoi du message à l'agent
            print(f"\n{agent['icon']} {agent['name']}: ", end="", flush=True)
            print("(réflexion...)", end="\r", flush=True)
            
            start_time = time.time()
            try:
                response = session.post(
                    url, 
                    json={"message": user_input},
                    timeout=timeout
                )
                duration = time.time() - start_time
                
                # Efface le message "réflexion..."
                print(" " * 50, end="\r")
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        output = extract_output(data)
                        
                        # Affichage formaté
                        print(f"{agent['icon']} {agent['name']} ({duration:.1f}s):")
                        print("-" * 40)
                        print(output)
                        print("-" * 40 + "\n")
                        
                    except json.JSONDecodeError:
                        print(f"{agent['icon']} {agent['name']} ({duration:.1f}s):")
                        print(response.text + "\n")
                else:
                    print(f"❌ Erreur HTTP {response.status_code}:")
                    print(response.text[:500] + "\n")

            except requests.exceptions.Timeout:
                print(f"❌ Timeout après {timeout}s. L'agent met trop de temps à répondre.\n")
            except requests.exceptions.ConnectionError:
                print("❌ Erreur de connexion. Vérifiez que n8n tourne sur localhost:5678\n")
            except Exception as e:
                print(f"❌ Erreur inattendue: {e}\n")

        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Interruption, au revoir!")
            break

def main():
    parser = argparse.ArgumentParser(
        description="Client de chat pour les agents IA n8n-geospatial",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python chat_client.py                    # Agent Guide par défaut
  python chat_client.py cadastre           # Agent Cadastral
  python chat_client.py domaine            # Agent Domanial
  python chat_client.py --list             # Liste les agents
"""
    )
    parser.add_argument(
        "agent", 
        nargs="?", 
        default="guide", 
        choices=list(AGENTS.keys()), 
        help="L'agent avec qui dialoguer (défaut: guide)"
    )
    parser.add_argument(
        "--url", 
        default=DEFAULT_BASE_URL, 
        help=f"URL de base des webhooks n8n (défaut: {DEFAULT_BASE_URL})"
    )
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=120, 
        help="Timeout en secondes pour les requêtes (défaut: 120)"
    )
    parser.add_argument(
        "--list", 
        action="store_true", 
        help="Liste les agents disponibles et quitte"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_agents()
        return
    
    chat_loop(args.agent, args.url, args.timeout)

if __name__ == "__main__":
    main()
