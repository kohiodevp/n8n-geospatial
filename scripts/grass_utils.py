#!/usr/bin/env python3
"""
Utilitaires GRASS GIS pour n8n
"""
import os
import subprocess
from pathlib import Path


def grass_buffer(input_vector, distance, output_vector):
    """
    Crée un buffer avec GRASS GIS
    """
    try:
        cmd = [
            'v.buffer', f'input={input_vector}', f'distance={distance}', f'output={output_vector}'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return {'success': True, 'output': output_vector, 'message': result.stdout}
    except subprocess.CalledProcessError as e:
        return {'success': False, 'error': str(e), 'stderr': e.stderr}


def grass_slope_aspect(elevation_raster, slope_output, aspect_output):
    """
    Calcule la pente et l'aspect avec GRASS GIS
    """
    try:
        cmd = [
            'r.slope.aspect', f'elevation={elevation_raster}', f'slope={slope_output}', f'aspect={aspect_output}'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return {'success': True, 'slope': slope_output, 'aspect': aspect_output, 'message': result.stdout}
    except subprocess.CalledProcessError as e:
        return {'success': False, 'error': str(e), 'stderr': e.stderr}