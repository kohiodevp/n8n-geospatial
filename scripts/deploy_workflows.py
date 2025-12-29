#!/usr/bin/env python3
"""
Déploiement automatique des workflows n8n
==========================================

Ce script déploie automatiquement tous les workflows disponibles
dans le répertoire /home/node/.n8n/workflows/ vers l'instance n8n.
"""

import os
import json
import requests
import sys
from pathlib import Path
from typing import List, Dict, Any

def login_to_n8n(n8n_url: str, username: str, password: str) -> str:
    """
    Se connecter à n8n et obtenir un token d'authentification
    
    Args:
        n8n_url: URL de l'instance n8n
        username: Nom d'utilisateur
        password: Mot de passe
        
    Returns:
        Token d'authentification ou None en cas d'échec
    """
    try:
        login_data = {
            "email": username,
            "password": password
        }
        
        response = requests.post(
            f"{n8n_url}/rest/login",
            json=login_data
        )
        
        if response.status_code == 200:
            # La réponse peut contenir un token ou les cookies de session
            cookies = response.cookies
            return cookies
        else:
            print(f"Erreur de connexion: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Erreur lors de la connexion: {e}")
        return None

def get_existing_workflows(n8n_url: str, session: requests.Session) -> Dict[str, Any]:
    """
    Récupérer la liste des workflows existants
    
    Args:
        n8n_url: URL de l'instance n8n
        session: Session authentifiée
        
    Returns:
        Dictionnaire des workflows existants
    """
    try:
        response = session.get(f"{n8n_url}/rest/workflows")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Erreur lors de la récupération des workflows existants: {e}")
        return {}

def deploy_workflow(n8n_url: str, session: requests.Session, workflow_path: str) -> bool:
    """
    Déployer un workflow vers l'instance n8n
    
    Args:
        n8n_url: URL de l'instance n8n
        session: Session authentifiée
        workflow_path: Chemin du fichier de workflow
        
    Returns:
        True si le déploiement a réussi, False sinon
    """
    try:
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow_data = json.load(f)
        
        # Vérifier si le workflow existe déjà
        existing_workflows = get_existing_workflows(n8n_url, session)
        workflow_exists = False
        
        if 'data' in existing_workflows:
            for existing_wf in existing_workflows['data']:
                if existing_wf.get('name') == workflow_data.get('name'):
                    workflow_exists = True
                    workflow_id = existing_wf.get('id')
                    # Mettre à jour le workflow existant
                    update_response = session.patch(
                        f"{n8n_url}/rest/workflows/{workflow_id}",
                        json=workflow_data
                    )
                    update_response.raise_for_status()
                    print(f"✅ Workflow mis à jour: {workflow_data['name']}")
                    return True
        
        if not workflow_exists:
            # Créer un nouveau workflow
            response = session.post(
                f"{n8n_url}/rest/workflows",
                json=workflow_data
            )
            response.raise_for_status()
            print(f"✅ Workflow déployé: {workflow_data['name']}")
            return True
            
    except Exception as e:
        print(f"❌ Erreur lors du déploiement du workflow {workflow_path}: {e}")
        return False

def deploy_all_workflows():
    """
    Déployer tous les workflows disponibles
    """
    n8n_url = os.getenv("N8N_URL", "http://localhost:5678").rstrip("/")
    
    # Informations d'authentification
    username = os.getenv('N8N_BASIC_AUTH_USER', 'admin')
    password = os.getenv('N8N_BASIC_AUTH_PASSWORD', 'cadastre2024')
    
    # Se connecter à n8n
    print("🔐 Connexion à l'instance n8n...")
    session = requests.Session()
    
    # Pour n8n avec authentification basique activée, on peut aussi essayer avec les headers
    # Essayer d'abord avec la méthode de connexion standard
    login_cookies = login_to_n8n(n8n_url, username, password)
    if login_cookies:
        session.cookies.update(login_cookies)
        print("✅ Connecté avec succès")
    else:
        print("⚠️ Impossible de se connecter avec les identifiants, tentative avec l'authentification basique...")
        # Si la connexion standard échoue, essayer avec l'authentification basique
        import base64
        credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
        session.headers.update({
            'Authorization': f'Basic {credentials}',
            'Content-Type': 'application/json'
        })
    
    workflows_dir = "/home/node/.n8n/workflows/"
    workflow_files = []
    
    # Trouver tous les fichiers de workflow
    for file in os.listdir(workflows_dir):
        if file.endswith('.json') and not file.startswith('credentials_'):
            workflow_files.append(os.path.join(workflows_dir, file))
    
    print(f"🔍 Trouvé {len(workflow_files)} workflows à déployer")
    
    successful_deployments = 0
    failed_deployments = 0
    
    for workflow_file in workflow_files:
        print(f"📦 Déploiement de: {os.path.basename(workflow_file)}")
        if deploy_workflow(n8n_url, session, workflow_file):
            successful_deployments += 1
        else:
            failed_deployments += 1
    
    print(f"\n📊 Résumé du déploiement:")
    print(f"   • Réussis: {successful_deployments}")
    print(f"   • Échoués: {failed_deployments}")
    print(f"   • Total: {len(workflow_files)}")
    
    if failed_deployments == 0:
        print("🎉 Tous les workflows ont été déployés avec succès!")
        return True
    else:
        print(f"⚠️  {failed_deployments} workflows n'ont pas pu être déployés")
        return failed_deployments == 0

def main():
    """
    Fonction principale
    """
    print("🚀 Déploiement automatique des workflows n8n")
    print("=" * 50)
    
    success = deploy_all_workflows()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()