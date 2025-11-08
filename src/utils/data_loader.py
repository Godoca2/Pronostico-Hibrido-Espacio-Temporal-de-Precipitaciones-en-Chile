"""Funciones auxiliares para carga de datos (esqueleto)
"""
from typing import Optional


def load_data(path: str, subset: Optional[str] = None):
    """Cargar datos desde `path` y devolver objeto tipo xarray/pandas según corresponda.

    Args:
        path: Ruta al archivo o directorio.
        subset: Nombre de subconjunto si aplica.

    Returns:
        datos cargados (placeholder).
    """
    # TODO: implementar carga real (netCDF, GeoTIFF, CSV...)
    return None
