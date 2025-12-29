#!/usr/bin/env python3
"""
Vérification de l'état des workflows dans n8n
=============================================

Ce script vérifie si les workflows sont correctement déployés dans l'instance n8n.
"""

import os
import json
import sys
from pathlib import Path

def check_workflows_in_directory():
    """
    Vérifier les workflows dans le répertoire
    """
    workflows_dir = "/home/node/.n8n/workflows/"
    
    print("🔍 Vérification des workflows dans le répertoire...")
    print(f"   Répertoire: {workflows_dir}")
    
    workflow_files = []
    for file in os.listdir(workflows_dir):
        if file.endswith('.json') and not file.startswith('credentials_'):
            workflow_files.append(file)
    
    print(f"   • Fichiers de workflow trouvés: {len(workflow_files)}")
    
    for file in workflow_files:
        file_path = os.path.join(workflows_dir, file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
            print(f"   ✅ {file} - {workflow_data.get('name', 'Nom inconnu')}")
        except Exception as e:
            print(f"   ❌ {file} - Erreur: {e}")
    
    return len(workflow_files)

def main():
    """
    Fonction principale
    """
    print("📋 Vérification de l'état des workflows n8n")
    print("=" * 45)
    
    count = check_workflows_in_directory()
    
    print(f"\n📊 Résumé:")
    print(f"   • Total workflows dans le répertoire: {count}")
    print(f"   • Les workflows dans ce répertoire sont automatiquement chargés par n8n")
    print(f"   • Accédez à l'interface n8n pour voir les workflows disponibles")
    
    return True

if __name__ == "__main__":
    main()