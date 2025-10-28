"""
Procesador de datos satelitales para precipitación y variables correlacionadas

Este módulo maneja la descarga, preprocesamiento y extracción de features
de datos de teledetección relevantes para pronóstico de precipitación.
"""

import numpy as np
import xarray as xr
from datetime import datetime, timedelta


class SatelliteDataProcessor:
    """
    Clase base para procesamiento de datos satelitales
    
    Proporciona funcionalidades comunes para manejo de datos satelitales
    de diferentes fuentes.
    
    Args:
        roi (tuple): Región de interés (lon_min, lat_min, lon_max, lat_max)
        temporal_range (tuple): Rango temporal (fecha_inicio, fecha_fin)
    """
    
    def __init__(self, roi=None, temporal_range=None):
        self.roi = roi
        self.temporal_range = temporal_range
        self.data = None
        
    def set_roi(self, lon_min, lat_min, lon_max, lat_max):
        """
        Define región de interés
        
        Args:
            lon_min, lat_min, lon_max, lat_max (float): Límites espaciales
        """
        self.roi = (lon_min, lat_min, lon_max, lat_max)
        
    def set_temporal_range(self, start_date, end_date):
        """
        Define rango temporal
        
        Args:
            start_date, end_date (datetime): Fechas inicio y fin
        """
        self.temporal_range = (start_date, end_date)
        
    def clip_to_roi(self, data):
        """
        Recorta datos a región de interés
        
        Args:
            data (xr.Dataset): Dataset con coordenadas lat/lon
        
        Returns:
            xr.Dataset: Datos recortados
        """
        if self.roi is None:
            return data
        
        lon_min, lat_min, lon_max, lat_max = self.roi
        
        # Recortar espacialmente
        clipped = data.sel(
            lon=slice(lon_min, lon_max),
            lat=slice(lat_min, lat_max)
        )
        
        return clipped
    
    def resample_temporal(self, data, frequency='D'):
        """
        Remuestrea datos temporalmente
        
        Args:
            data (xr.Dataset): Dataset con dimensión temporal
            frequency (str): Frecuencia ('D'=diario, 'W'=semanal, 'M'=mensual)
        
        Returns:
            xr.Dataset: Datos remuestreados
        """
        resampled = data.resample(time=frequency).mean()
        return resampled
    
    def interpolate_spatial(self, data, target_resolution):
        """
        Interpola datos a una resolución específica
        
        Args:
            data (xr.Dataset): Dataset original
            target_resolution (float): Resolución objetivo en grados
        
        Returns:
            xr.Dataset: Datos interpolados
        """
        if self.roi is None:
            raise ValueError("Debe definir ROI primero")
        
        lon_min, lat_min, lon_max, lat_max = self.roi
        
        # Crear grilla objetivo
        new_lon = np.arange(lon_min, lon_max, target_resolution)
        new_lat = np.arange(lat_min, lat_max, target_resolution)
        
        # Interpolar
        interpolated = data.interp(lon=new_lon, lat=new_lat, method='linear')
        
        return interpolated
    
    def compute_anomalies(self, data, baseline_period=None):
        """
        Calcula anomalías respecto a climatología
        
        Args:
            data (xr.Dataset): Datos originales
            baseline_period (tuple): Periodo de referencia (start_year, end_year)
        
        Returns:
            xr.Dataset: Anomalías
        """
        if baseline_period is None:
            # Usar todo el periodo disponible
            climatology = data.groupby('time.dayofyear').mean('time')
        else:
            start_year, end_year = baseline_period
            baseline = data.sel(time=slice(f'{start_year}', f'{end_year}'))
            climatology = baseline.groupby('time.dayofyear').mean('time')
        
        anomalies = data.groupby('time.dayofyear') - climatology
        
        return anomalies


class CHIRPS(SatelliteDataProcessor):
    """
    Procesador para datos CHIRPS (Climate Hazards Group InfraRed Precipitation)
    
    CHIRPS combina observaciones satelitales con datos de estaciones terrestres
    para estimar precipitación con alta resolución.
    
    Resolución: 0.05° (~5.5 km)
    Cobertura temporal: 1981-presente
    Actualización: Diaria
    """
    
    def __init__(self, roi=None, temporal_range=None):
        super().__init__(roi, temporal_range)
        self.resolution = 0.05  # grados
        self.dataset_name = 'CHIRPS'
        
    def load_data(self, file_path=None):
        """
        Carga datos CHIRPS desde archivo NetCDF
        
        Args:
            file_path (str): Ruta a archivo NetCDF
        
        Returns:
            xr.Dataset: Datos cargados
        """
        if file_path is None:
            raise ValueError("Debe proporcionar ruta a archivo CHIRPS")
        
        # Cargar datos
        data = xr.open_dataset(file_path)
        
        # Recortar a ROI si está definida
        if self.roi is not None:
            data = self.clip_to_roi(data)
        
        # Filtrar temporalmente
        if self.temporal_range is not None:
            start, end = self.temporal_range
            data = data.sel(time=slice(start, end))
        
        self.data = data
        return data
    
    def extract_timeseries(self, lon, lat):
        """
        Extrae serie temporal en un punto específico
        
        Args:
            lon, lat (float): Coordenadas
        
        Returns:
            np.ndarray: Serie temporal
        """
        if self.data is None:
            raise ValueError("Debe cargar datos primero")
        
        point_data = self.data.sel(lon=lon, lat=lat, method='nearest')
        return point_data['precip'].values


class MODIS(SatelliteDataProcessor):
    """
    Procesador para datos MODIS (Moderate Resolution Imaging Spectroradiometer)
    
    MODIS proporciona datos de temperatura, índices de vegetación, cobertura
    de nubes y otros parámetros útiles para pronóstico de precipitación.
    
    Resolución espacial: 250m - 1km (según producto)
    Resolución temporal: Diaria
    """
    
    def __init__(self, roi=None, temporal_range=None, product='MOD13Q1'):
        super().__init__(roi, temporal_range)
        self.product = product  # MOD13Q1 = NDVI, MOD11A1 = LST, etc.
        
    def load_data(self, file_paths):
        """
        Carga datos MODIS desde archivos HDF
        
        Args:
            file_paths (list): Lista de rutas a archivos MODIS
        
        Returns:
            xr.Dataset: Datos cargados
        """
        # Implementación depende del producto específico
        # Ejemplo simplificado
        datasets = []
        
        for file_path in file_paths:
            try:
                # Los archivos MODIS suelen requerir procesamiento específico
                # Aquí mostramos estructura general
                ds = xr.open_dataset(file_path)
                
                if self.roi is not None:
                    ds = self.clip_to_roi(ds)
                
                datasets.append(ds)
            except Exception as e:
                print(f"Error cargando {file_path}: {e}")
        
        if datasets:
            self.data = xr.concat(datasets, dim='time')
        
        return self.data
    
    def compute_ndvi(self):
        """
        Calcula NDVI (Normalized Difference Vegetation Index)
        
        NDVI = (NIR - Red) / (NIR + Red)
        
        Returns:
            xr.DataArray: NDVI
        """
        if self.data is None:
            raise ValueError("Debe cargar datos primero")
        
        if 'NDVI' in self.data:
            return self.data['NDVI']
        
        # Calcular desde bandas
        nir = self.data['NIR']
        red = self.data['Red']
        
        ndvi = (nir - red) / (nir + red)
        
        return ndvi
    
    def compute_evi(self):
        """
        Calcula EVI (Enhanced Vegetation Index)
        
        EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
        
        Returns:
            xr.DataArray: EVI
        """
        if self.data is None:
            raise ValueError("Debe cargar datos primero")
        
        if 'EVI' in self.data:
            return self.data['EVI']
        
        nir = self.data['NIR']
        red = self.data['Red']
        blue = self.data['Blue']
        
        evi = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1)
        
        return evi


class Sentinel(SatelliteDataProcessor):
    """
    Procesador para datos Sentinel (Sentinel-1, Sentinel-2, Sentinel-3)
    
    Sentinel-1: Radar SAR (útil para nubes, humedad del suelo)
    Sentinel-2: Óptico multiespectral (vegetación, uso de suelo)
    Sentinel-3: Temperatura de superficie, color del océano
    
    Resolución: 10-60m (Sentinel-2), 300m (Sentinel-3)
    """
    
    def __init__(self, roi=None, temporal_range=None, satellite='S2'):
        super().__init__(roi, temporal_range)
        self.satellite = satellite  # 'S1', 'S2', 'S3'
        
    def load_data(self, file_paths):
        """Carga datos Sentinel"""
        datasets = []
        
        for file_path in file_paths:
            try:
                ds = xr.open_dataset(file_path)
                
                if self.roi is not None:
                    ds = self.clip_to_roi(ds)
                
                datasets.append(ds)
            except Exception as e:
                print(f"Error cargando {file_path}: {e}")
        
        if datasets:
            self.data = xr.concat(datasets, dim='time')
        
        return self.data
    
    def compute_moisture_index(self):
        """
        Calcula índice de humedad usando bandas SWIR
        
        NDMI = (NIR - SWIR) / (NIR + SWIR)
        
        Returns:
            xr.DataArray: Índice de humedad
        """
        if self.data is None:
            raise ValueError("Debe cargar datos primero")
        
        nir = self.data['B08']  # NIR en Sentinel-2
        swir = self.data['B11']  # SWIR en Sentinel-2
        
        ndmi = (nir - swir) / (nir + swir)
        
        return ndmi


class MultiSourceIntegrator:
    """
    Integrador de múltiples fuentes de datos satelitales
    
    Combina datos de diferentes sensores y fuentes para crear
    un dataset unificado para pronóstico de precipitación.
    
    Args:
        sources (dict): Diccionario de procesadores de datos
        target_resolution (float): Resolución objetivo
    """
    
    def __init__(self, sources, target_resolution=0.05):
        self.sources = sources
        self.target_resolution = target_resolution
        self.integrated_data = None
        
    def harmonize_spatial(self):
        """
        Armoniza resolución espacial de todas las fuentes
        
        Returns:
            dict: Datos armonizados
        """
        harmonized = {}
        
        for name, processor in self.sources.items():
            if processor.data is not None:
                harmonized[name] = processor.interpolate_spatial(
                    processor.data, 
                    self.target_resolution
                )
        
        return harmonized
    
    def harmonize_temporal(self, frequency='D'):
        """
        Armoniza resolución temporal
        
        Args:
            frequency (str): Frecuencia objetivo
        
        Returns:
            dict: Datos armonizados temporalmente
        """
        harmonized = {}
        
        for name, processor in self.sources.items():
            if processor.data is not None:
                harmonized[name] = processor.resample_temporal(
                    processor.data,
                    frequency
                )
        
        return harmonized
    
    def create_feature_matrix(self, variables):
        """
        Crea matriz de features combinando múltiples fuentes
        
        Args:
            variables (dict): Variables a incluir por fuente
                             {'CHIRPS': ['precip'], 'MODIS': ['NDVI', 'LST']}
        
        Returns:
            xr.Dataset: Dataset integrado
        """
        # Armonizar espacial y temporalmente
        spatial_harmonized = self.harmonize_spatial()
        
        # Combinar variables
        merged_data = {}
        
        for source, vars_list in variables.items():
            if source in spatial_harmonized:
                data = spatial_harmonized[source]
                for var in vars_list:
                    if var in data:
                        merged_data[f'{source}_{var}'] = data[var]
        
        # Crear dataset combinado
        self.integrated_data = xr.Dataset(merged_data)
        
        return self.integrated_data
    
    def extract_spatial_features(self, window_size=3):
        """
        Extrae features espaciales (medias locales, gradientes, etc.)
        
        Args:
            window_size (int): Tamaño de ventana para estadísticas locales
        
        Returns:
            xr.Dataset: Features espaciales
        """
        if self.integrated_data is None:
            raise ValueError("Debe crear matriz de features primero")
        
        features = {}
        
        for var in self.integrated_data.data_vars:
            data = self.integrated_data[var]
            
            # Media móvil espacial
            features[f'{var}_smooth'] = data.rolling(
                lat=window_size, 
                lon=window_size, 
                center=True
            ).mean()
            
            # Gradiente espacial
            if 'lat' in data.dims and 'lon' in data.dims:
                grad_lat = data.diff('lat')
                grad_lon = data.diff('lon')
                
                features[f'{var}_grad_lat'] = grad_lat
                features[f'{var}_grad_lon'] = grad_lon
        
        return xr.Dataset(features)
