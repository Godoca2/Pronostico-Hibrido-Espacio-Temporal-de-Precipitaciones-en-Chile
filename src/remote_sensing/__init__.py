"""
Módulo de Teledetección para procesamiento de datos satelitales
"""

from .satellite_data import SatelliteDataProcessor, MODIS, Sentinel, CHIRPS
from .indices import VegetationIndices, ClimateIndices

__all__ = ['SatelliteDataProcessor', 'MODIS', 'Sentinel', 'CHIRPS', 
           'VegetationIndices', 'ClimateIndices']
