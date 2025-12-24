#!/usr/bin/env python3
"""
Wrapper pour utiliser QGIS Processing dans n8n
"""
import os
import sys
import json
from pathlib import Path

# Configuration Qt headless
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['XDG_RUNTIME_DIR'] = '/tmp/runtime-node'

from qgis.core import (
    QgsApplication,
    QgsProcessingFeedback,
    QgsProcessingContext,
    QgsVectorLayer,
    QgsRasterLayer,
    QgsProject,
    QgsCoordinateReferenceSystem,
)
from qgis.analysis import QgsNativeAlgorithms
import processing
from processing.core.Processing import Processing


class QGISProcessor:
    """Gestionnaire QGIS Processing pour n8n"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not QGISProcessor._initialized:
            # Initialiser QGIS
            self.qgs = QgsApplication([], False)
            self.qgs.initQgis()

            # Initialiser Processing
            Processing.initialize()
            self.qgs.processingRegistry().addProvider(QgsNativeAlgorithms())

            self.feedback = QgsProcessingFeedback()
            self.context = QgsProcessingContext()

            QGISProcessor._initialized = True
            print("✅ QGIS Processing initialisé")

    def list_algorithms(self, filter_text: str = None) -> list:
        """Liste tous les algorithmes disponibles"""
        algorithms = []
        for alg in self.qgs.processingRegistry().algorithms():
            if filter_text is None or filter_text.lower() in alg.id().lower():
                algorithms.append({
                    'id': alg.id(),
                    'name': alg.displayName(),
                    'group': alg.group(),
                    'provider': alg.provider().name()
                })
        return algorithms

    def get_algorithm_help(self, algorithm_id: str) -> dict:
        """Retourne l'aide d'un algorithme"""
        alg = self.qgs.processingRegistry().algorithmById(algorithm_id)
        if not alg:
            return {'error': f'Algorithm {algorithm_id} not found'}

        params = []
        for param in alg.parameterDefinitions():
            params.append({
                'name': param.name(),
                'description': param.description(),
                'type': param.type(),
                'optional': not (param.flags() & param.FlagOptional)
            })

        return {
            'id': alg.id(),
            'name': alg.displayName(),
            'description': alg.shortDescription(),
            'parameters': params
        }

    def run(self, algorithm_id: str, parameters: dict, output_path: str = None) -> dict:
        """Exécute un algorithme QGIS Processing"""
        try:
            result = processing.run(
                algorithm_id,
                parameters,
                feedback=self.feedback,
                context=self.context
            )
            return {'success': True, 'result': result}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def buffer(self, input_layer: str, distance: float, output_path: str) -> dict:
        """Crée un buffer autour des géométries"""
        return self.run('native:buffer', {
            'INPUT': input_layer,
            'DISTANCE': distance,
            'SEGMENTS': 16,
            'END_CAP_STYLE': 0,
            'JOIN_STYLE': 0,
            'MITER_LIMIT': 2,
            'DISSOLVE': False,
            'OUTPUT': output_path
        })

    def clip(self, input_layer: str, overlay_layer: str, output_path: str) -> dict:
        """Découpe une couche par une autre"""
        return self.run('native:clip', {
            'INPUT': input_layer,
            'OVERLAY': overlay_layer,
            'OUTPUT': output_path
        })

    def dissolve(self, input_layer: str, field: str, output_path: str) -> dict:
        """Fusionne les géométries par attribut"""
        return self.run('native:dissolve', {
            'INPUT': input_layer,
            'FIELD': [field] if field else [],
            'OUTPUT': output_path
        })

    def reproject(self, input_layer: str, target_crs: str, output_path: str) -> dict:
        """Reprojette une couche"""
        return self.run('native:reprojectlayer', {
            'INPUT': input_layer,
            'TARGET_CRS': target_crs,
            'OUTPUT': output_path
        })

    def raster_clip(self, input_raster: str, mask_layer: str, output_path: str) -> dict:
        """Découpe un raster par un vecteur"""
        return self.run('gdal:cliprasterbymasklayer', {
            'INPUT': input_raster,
            'MASK': mask_layer,
            'NODATA': -9999,
            'CROP_TO_CUTLINE': True,
            'OUTPUT': output_path
        })

    def hillshade(self, input_dem: str, output_path: str, azimuth: float = 315, altitude: float = 45) -> dict:
        """Crée un ombrage à partir d'un MNT"""
        return self.run('gdal:hillshade', {
            'INPUT': input_dem,
            'BAND': 1,
            'AZIMUTH': azimuth,
            'ALTITUDE': altitude,
            'Z_FACTOR': 1,
            'OUTPUT': output_path
        })

    def contour(self, input_dem: str, interval: float, output_path: str) -> dict:
        """Génère des courbes de niveau"""
        return self.run('gdal:contour', {
            'INPUT': input_dem,
            'BAND': 1,
            'INTERVAL': interval,
            'OUTPUT': output_path
        })

    def cleanup(self):
        """Nettoie les ressources QGIS"""
        self.qgs.exitQgis()
        QGISProcessor._initialized = False


# Instance globale
qgis_processor = None

def get_processor() -> QGISProcessor:
    """Retourne l'instance du processeur QGIS"""
    global qgis_processor
    if qgis_processor is None:
        qgis_processor = QGISProcessor()
    return qgis_processor


# Fonctions directes pour n8n
def qgis_buffer(input_path: str, distance: float, output_path: str = None) -> dict:
    """Buffer QGIS - utilisable directement dans n8n"""
    if output_path is None:
        output_path = f"/qgis-output/buffer_{Path(input_path).stem}.gpkg"
    return get_processor().buffer(input_path, distance, output_path)


def qgis_clip(input_path: str, mask_path: str, output_path: str = None) -> dict:
    """Clip QGIS - utilisable directement dans n8n"""
    if output_path is None:
        output_path = f"/qgis-output/clip_{Path(input_path).stem}.gpkg"
    return get_processor().clip(input_path, mask_path, output_path)


def qgis_list_algorithms(filter_text: str = None) -> list:
    """Liste les algorithmes QGIS disponibles"""
    return get_processor().list_algorithms(filter_text)