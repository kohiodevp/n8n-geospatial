#!/usr/bin/env python3
"""
Configuration du système géospatial
===================================

Ce module fournit les fonctions de configuration pour le système
géospatial IA avec n8n.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional

# Configuration de la journalisation
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_geospatial_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Charger la configuration géospatiale
    
    Args:
        config_path: Chemin vers le fichier de configuration (optionnel)
        
    Returns:
        Dictionnaire de configuration
    """
    if config_path is None:
        config_path = os.getenv('GEOSPATIAL_CONFIG_PATH', '/opt/geoscripts/config.json')
    
    default_config = {
        'crs': 'EPSG:2154',  # Lambert 93 par défaut pour la France
        'buffer_size': 1000,  # 1km par défaut
        'max_features': 10000,
        'processing_chunk_size': 1000,
        'temp_directory': '/tmp/geodata-cache',
        'data_directory': '/geodata',
        'output_directory': '/qgis-output',
        'gdal_options': {
            'GDAL_CACHEMAX': os.getenv('GDAL_CACHEMAX', '1024'),
            'GDAL_NUM_THREADS': os.getenv('GDAL_NUM_THREADS', 'ALL_CPUS'),
            'PROJ_NETWORK': os.getenv('PROJ_NETWORK', 'ON')
        }
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                default_config.update(user_config)
            logger.info(f"Configuration chargée depuis {config_path}")
        except Exception as e:
            logger.warning(f"Impossible de charger la configuration depuis {config_path}: {e}")
    else:
        logger.info(f"Fichier de configuration non trouvé: {config_path}, utilisation des valeurs par défaut")
    
    return default_config


def validate_geospatial_environment() -> bool:
    """
    Valider l'environnement géospatial
    
    Returns:
        True si l'environnement est valide, False sinon
    """
    try:
        # Vérifier les bibliothèques géospatiales
        import geopandas as gpd
        import shapely
        import pyproj
        import rasterio
        
        logger.info("Bibliothèques géospatiales disponibles")
        
        # Vérifier les chemins d'accès
        config = load_geospatial_config()
        
        for path_name, path_value in [
            ('temp_directory', config.get('temp_directory')),
            ('data_directory', config.get('data_directory')),
            ('output_directory', config.get('output_directory'))
        ]:
            if path_value and not os.path.exists(path_value):
                os.makedirs(path_value, exist_ok=True)
                logger.info(f"Créé le répertoire: {path_value}")
        
        logger.info("Environnement géospatial validé avec succès")
        return True
        
    except ImportError as e:
        logger.error(f"Bibliothèque géospatiale manquante: {e}")
        return False
    except Exception as e:
        logger.error(f"Erreur lors de la validation de l'environnement: {e}")
        return False


def setup_geospatial_environment() -> bool:
    """
    Configurer l'environnement géospatial
    
    Returns:
        True si la configuration a réussi, False sinon
    """
    logger.info("Configuration de l'environnement géospatial...")
    
    # Charger la configuration
    config = load_geospatial_config()
    
    # Valider l'environnement
    if not validate_geospatial_environment():
        logger.error("Échec de la validation de l'environnement géospatial")
        return False
    
    # Configurer les variables d'environnement
    gdal_options = config.get('gdal_options', {})
    for key, value in gdal_options.items():
        os.environ[key] = str(value)
    
    logger.info("Environnement géospatial configuré avec succès")
    return True


def main():
    """
    Fonction principale pour la configuration géospatiale
    """
    print("Configuration du système géospatial IA")
    print("=" * 40)
    
    success = setup_geospatial_environment()
    
    if success:
        print("✅ Configuration terminée avec succès")
        config = load_geospatial_config()
        print(f"   • Système de coordonnées: {config['crs']}")
        print(f"   • Répertoire temporaire: {config['temp_directory']}")
        print(f"   • Répertoire de données: {config['data_directory']}")
    else:
        print("❌ Échec de la configuration")
        sys.exit(1)


if __name__ == "__main__":
    main()