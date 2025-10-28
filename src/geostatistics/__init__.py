"""
Módulo de Geoestadística para interpolación espacial de datos de precipitación
"""

from .kriging import OrdinaryKriging, UniversalKriging, CoKriging
from .variogram import Variogram, VariogramModel

__all__ = ['OrdinaryKriging', 'UniversalKriging', 'CoKriging', 'Variogram', 'VariogramModel']
