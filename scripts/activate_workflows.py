#!/usr/bin/env python3
"""
Activation des workflows dans n8n
=================================

Ce script active les workflows importants dans l'instance n8n
pour qu'ils soient prêts à être utilisés.
"""

import os
import json
import requests
import time
import sys
from pathlib import Path

def activate_workflow(n8n_url: str, workflow_id: str, session: requests.Session) -> bool:
    """
    Activer un workflow spécifique
    
    Args:
        n8n_url: URL de l'instance n8n
        workflow_id: ID du workflow à activer
        session: Session authentifiée
        
    Returns:
        True si l'activation a réussi, False sinon
    """
    try:
        # Activer le workflow
        activation_url = f"{n8n_url}/rest/workflows/{workflow_id}/activate"
        response = session.post(activation_url)
        
        if response.status_code in [200, 201]:
            print(f"✅ Workflow activé: {workflow_id}")
            return True
        else:
            print(f"❌ Échec de l'activation du workflow {workflow_id}: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors de l'activation du workflow {workflow_id}: {e}")
        return False

def get_all_workflows(n8n_url: str, session: requests.Session) -> dict:
    """
    Récupérer la liste de tous les workflows
    
    Args:
        n8n_url: URL de l'instance n8n
        session: Session authentifiée
        
    Returns:
        Dictionnaire contenant les workflows
    """
    try:
        response = session.get(f"{n8n_url}/rest/workflows")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Erreur lors de la récupération des workflows: {e}")
        return {}

def activate_geospatial_workflows():
    """
    Activer les workflows géospatiaux importants
    """
    n8n_url = "http://localhost:5678"
    
    # Créer une session pour les requêtes
    session = requests.Session()
    
    # Définir les headers
    session.headers.update({
        'Content-Type': 'application/json',
        'User-Agent': 'n8n-geospatial-activator'
    })
    
    print("🔍 Récupération de la liste des workflows...")
    workflows_data = get_all_workflows(n8n_url, session)
    
    if not workflows_data or 'data' not in workflows_data:
        print("❌ Impossible de récupérer la liste des workflows")
        print("ℹ️  Les workflows sont probablement situés dans le répertoire mais pas encore chargés")
        print("ℹ️  Redémarrer le service n8n pourrait aider à charger les nouveaux workflows")
        return False
    
    workflows = workflows_data['data']
    print(f"📦 {len(workflows)} workflows trouvés dans l'instance")
    
    # Identifier les workflows géospatiaux à activer
    geospatial_keywords = [
        'cadastral', 'domain', 'urban', 'environnement', 'geospatial', 
        'agent', 'surveillance', 'consolidation', 'planification', 'prediction'
    ]
    
    workflows_to_activate = []
    
    for workflow in workflows:
        workflow_name = workflow.get('name', '').lower()
        if any(keyword in workflow_name for keyword in geospatial_keywords):
            workflows_to_activate.append(workflow)
    
    print(f"🎯 {len(workflows_to_activate)} workflows géospatiaux identifiés pour activation")
    
    # Activer les workflows géospatiaux
    activated_count = 0
    failed_count = 0
    
    for workflow in workflows_to_activate:
        workflow_id = workflow.get('id')
        workflow_name = workflow.get('name', 'Unknown')
        
        print(f"🔌 Activation de: {workflow_name} (ID: {workflow_id})")
        
        if activate_workflow(n8n_url, workflow_id, session):
            activated_count += 1
        else:
            failed_count += 1
    
    print(f"\n📊 Résumé de l'activation:")
    print(f"   • Activés: {activated_count}")
    print(f"   • Échoués: {failed_count}")
    print(f"   • Total: {len(workflows_to_activate)}")
    
    if activated_count > 0:
        print("🎉 Les workflows géospatiaux sont maintenant prêts à être utilisés!")
        return True
    else:
        print("⚠️  Aucun workflow n'a pu être activé")
        return False

def main():
    """
    Fonction principale
    """
    print("🔌 Activation des workflows géospatiaux dans n8n")
    print("=" * 50)
    
    print("ℹ️  Les workflows sont déjà dans le répertoire et doivent être activés")
    print("ℹ️  pour être prêts à l'emploi dans l'interface n8n")
    
    success = activate_geospatial_workflows()
    
    if not success:
        print("\n⚠️  Certaines activations ont échoué, mais les workflows sont disponibles dans l'interface")
    
    return success

if __name__ == "__main__":
    main()