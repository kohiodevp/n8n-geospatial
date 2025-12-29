#!/bin/bash
# cleanup_project.sh - Script de nettoyage du projet n8n-geospatial

echo "Nettoyage du projet n8n-geospatial..."

# Supprimer les fichiers temporaires Python
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyo" -delete

# Supprimer les sauvegardes
find . -type f -name "*~" -delete
find . -type f -name "*.bak" -delete
find . -type f -name "*.tmp" -delete

# Supprimer les fichiers temporaires de l'éditeur
find . -type f -name ".DS_Store" -delete
find . -type f -name "Thumbs.db" -delete

# Supprimer les logs temporaires
find . -type f -name "*.log" -not -path "./logs/*" -delete

echo "Nettoyage terminé!"