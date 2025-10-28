"""
Cálculo de índices de vegetación y climáticos para análisis de precipitación

Estos índices son útiles como variables auxiliares en modelos de predicción.
"""

import numpy as np
import xarray as xr


class VegetationIndices:
    """
    Calculador de índices de vegetación
    
    Los índices de vegetación están correlacionados con precipitación
    y son útiles para co-kriging y como features predictivos.
    """
    
    @staticmethod
    def ndvi(nir, red):
        """
        Normalized Difference Vegetation Index
        
        NDVI = (NIR - Red) / (NIR + Red)
        
        Rango: -1 a 1 (valores altos = vegetación densa)
        
        Args:
            nir (np.ndarray): Banda near-infrared
            red (np.ndarray): Banda red
        
        Returns:
            np.ndarray: NDVI
        """
        numerator = nir - red
        denominator = nir + red
        
        # Evitar división por cero
        denominator = np.where(denominator == 0, np.nan, denominator)
        
        ndvi = numerator / denominator
        
        # Limitar rango
        ndvi = np.clip(ndvi, -1, 1)
        
        return ndvi
    
    @staticmethod
    def evi(nir, red, blue, G=2.5, C1=6.0, C2=7.5, L=1.0):
        """
        Enhanced Vegetation Index
        
        EVI = G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L)
        
        Mejora NDVI en áreas de alta biomasa.
        
        Args:
            nir, red, blue (np.ndarray): Bandas espectrales
            G, C1, C2, L (float): Coeficientes
        
        Returns:
            np.ndarray: EVI
        """
        numerator = G * (nir - red)
        denominator = nir + C1*red - C2*blue + L
        
        denominator = np.where(denominator == 0, np.nan, denominator)
        
        evi = numerator / denominator
        
        return evi
    
    @staticmethod
    def savi(nir, red, L=0.5):
        """
        Soil Adjusted Vegetation Index
        
        SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
        
        Minimiza influencia del suelo.
        
        Args:
            nir, red (np.ndarray): Bandas espectrales
            L (float): Factor de ajuste de suelo (0.5 típico)
        
        Returns:
            np.ndarray: SAVI
        """
        numerator = (nir - red) * (1 + L)
        denominator = nir + red + L
        
        denominator = np.where(denominator == 0, np.nan, denominator)
        
        savi = numerator / denominator
        
        return savi
    
    @staticmethod
    def ndwi(nir, swir):
        """
        Normalized Difference Water Index
        
        NDWI = (NIR - SWIR) / (NIR + SWIR)
        
        Detecta contenido de agua en vegetación.
        
        Args:
            nir (np.ndarray): Near-infrared
            swir (np.ndarray): Short-wave infrared
        
        Returns:
            np.ndarray: NDWI
        """
        numerator = nir - swir
        denominator = nir + swir
        
        denominator = np.where(denominator == 0, np.nan, denominator)
        
        ndwi = numerator / denominator
        
        return ndwi
    
    @staticmethod
    def ndmi(nir, swir):
        """
        Normalized Difference Moisture Index
        
        Similar a NDWI, específico para humedad del suelo/vegetación.
        
        Args:
            nir, swir (np.ndarray): Bandas espectrales
        
        Returns:
            np.ndarray: NDMI
        """
        return VegetationIndices.ndwi(nir, swir)
    
    @staticmethod
    def gndvi(nir, green):
        """
        Green Normalized Difference Vegetation Index
        
        GNDVI = (NIR - Green) / (NIR + Green)
        
        Sensible a clorofila.
        
        Args:
            nir, green (np.ndarray): Bandas espectrales
        
        Returns:
            np.ndarray: GNDVI
        """
        numerator = nir - green
        denominator = nir + green
        
        denominator = np.where(denominator == 0, np.nan, denominator)
        
        gndvi = numerator / denominator
        
        return gndvi


class ClimateIndices:
    """
    Calculador de índices climáticos relevantes para precipitación
    
    Estos índices capturan patrones climáticos de gran escala que
    influyen en la precipitación regional.
    """
    
    @staticmethod
    def spi(precipitation, timescale=30):
        """
        Standardized Precipitation Index
        
        Cuantifica déficit o exceso de precipitación en múltiples escalas.
        
        Args:
            precipitation (np.ndarray): Serie temporal de precipitación
            timescale (int): Escala temporal en días
        
        Returns:
            np.ndarray: SPI
        """
        from scipy import stats
        
        # Calcular acumulados móviles
        cumsum = np.convolve(precipitation, np.ones(timescale), mode='valid')
        
        # Ajustar distribución gamma
        # SPI asume que precipitación sigue distribución gamma
        shape, loc, scale = stats.gamma.fit(cumsum[cumsum > 0])
        
        # Calcular probabilidades
        cdf = stats.gamma.cdf(cumsum, shape, loc, scale)
        
        # Transformar a distribución normal estándar
        spi = stats.norm.ppf(cdf)
        
        # Manejar valores extremos
        spi = np.clip(spi, -3, 3)
        
        return spi
    
    @staticmethod
    def spei(precipitation, pet, timescale=30):
        """
        Standardized Precipitation Evapotranspiration Index
        
        Similar a SPI pero considera evapotranspiración.
        
        Args:
            precipitation (np.ndarray): Precipitación
            pet (np.ndarray): Evapotranspiración potencial
            timescale (int): Escala temporal
        
        Returns:
            np.ndarray: SPEI
        """
        from scipy import stats
        
        # Diferencia precipitación - evapotranspiración
        water_balance = precipitation - pet
        
        # Acumulado móvil
        cumsum = np.convolve(water_balance, np.ones(timescale), mode='valid')
        
        # Ajustar distribución log-logística
        # SPEI típicamente usa log-logística
        # Simplificación: usar normal
        mean = np.mean(cumsum)
        std = np.std(cumsum)
        
        spei = (cumsum - mean) / (std + 1e-10)
        
        spei = np.clip(spei, -3, 3)
        
        return spei
    
    @staticmethod
    def edi(precipitation, evapotranspiration, n_days=365):
        """
        Effective Drought Index
        
        Índice basado en balance hídrico efectivo.
        
        Args:
            precipitation (np.ndarray): Precipitación diaria
            evapotranspiration (np.ndarray): ET diaria
            n_days (int): Ventana temporal
        
        Returns:
            np.ndarray: EDI
        """
        # Balance hídrico
        water_balance = precipitation - evapotranspiration
        
        # Calcular agua disponible efectiva
        edi = np.zeros(len(water_balance))
        
        for i in range(n_days, len(water_balance)):
            window = water_balance[i-n_days:i]
            # EDI es suma ponderada con decaimiento exponencial
            weights = np.exp(-np.arange(n_days) / (n_days/3))
            weights = weights[::-1]
            weights = weights / weights.sum()
            edi[i] = np.sum(window * weights)
        
        # Normalizar
        if np.std(edi) > 0:
            edi = (edi - np.mean(edi)) / np.std(edi)
        
        return edi
    
    @staticmethod
    def oni(sst_nino34, window=3):
        """
        Oceanic Niño Index
        
        Basado en anomalías de temperatura superficial del mar
        en región Niño 3.4 (5°N-5°S, 120°-170°W).
        
        Args:
            sst_nino34 (np.ndarray): SST en región Niño 3.4
            window (int): Ventana de promedio móvil (meses)
        
        Returns:
            np.ndarray: ONI
        """
        # Promedio móvil de 3 meses
        oni = np.convolve(sst_nino34, np.ones(window)/window, mode='valid')
        
        return oni
    
    @staticmethod
    def compute_anomaly(data, climatology=None):
        """
        Calcula anomalías respecto a climatología
        
        Args:
            data (np.ndarray): Serie temporal
            climatology (np.ndarray): Climatología de referencia
        
        Returns:
            np.ndarray: Anomalías
        """
        if climatology is None:
            # Usar media como climatología
            climatology = np.mean(data)
        
        anomaly = data - climatology
        
        return anomaly
    
    @staticmethod
    def pdo_index(sst_north_pacific):
        """
        Pacific Decadal Oscillation Index
        
        Basado en primera EOF de anomalías SST en Pacífico Norte.
        
        Args:
            sst_north_pacific (np.ndarray): SST en Pacífico Norte (time, lat, lon)
        
        Returns:
            np.ndarray: PDO index
        """
        from sklearn.decomposition import PCA
        
        # Reshape para PCA
        n_times = sst_north_pacific.shape[0]
        sst_flat = sst_north_pacific.reshape(n_times, -1)
        
        # Calcular anomalías
        sst_anomaly = sst_flat - sst_flat.mean(axis=0)
        
        # PCA
        pca = PCA(n_components=1)
        pdo = pca.fit_transform(sst_anomaly).flatten()
        
        # Normalizar
        pdo = (pdo - pdo.mean()) / pdo.std()
        
        return pdo


class PrecipitationMetrics:
    """
    Métricas específicas para análisis de precipitación
    """
    
    @staticmethod
    def consecutive_dry_days(precipitation, threshold=1.0):
        """
        Calcula máximo de días secos consecutivos
        
        Args:
            precipitation (np.ndarray): Precipitación diaria
            threshold (float): Umbral para día seco (mm)
        
        Returns:
            int: Días secos consecutivos máximos
        """
        dry_days = precipitation < threshold
        
        max_consecutive = 0
        current_consecutive = 0
        
        for is_dry in dry_days:
            if is_dry:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    @staticmethod
    def consecutive_wet_days(precipitation, threshold=1.0):
        """
        Calcula máximo de días húmedos consecutivos
        
        Args:
            precipitation (np.ndarray): Precipitación diaria
            threshold (float): Umbral para día húmedo (mm)
        
        Returns:
            int: Días húmedos consecutivos máximos
        """
        wet_days = precipitation >= threshold
        
        max_consecutive = 0
        current_consecutive = 0
        
        for is_wet in wet_days:
            if is_wet:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    @staticmethod
    def r95p(precipitation):
        """
        Precipitación total por encima del percentil 95
        
        Args:
            precipitation (np.ndarray): Precipitación diaria
        
        Returns:
            float: Total de precipitación extrema
        """
        p95 = np.percentile(precipitation[precipitation > 0], 95)
        extreme_precip = precipitation[precipitation > p95]
        
        return np.sum(extreme_precip)
    
    @staticmethod
    def r99p(precipitation):
        """
        Precipitación total por encima del percentil 99
        
        Args:
            precipitation (np.ndarray): Precipitación diaria
        
        Returns:
            float: Total de precipitación muy extrema
        """
        p99 = np.percentile(precipitation[precipitation > 0], 99)
        very_extreme_precip = precipitation[precipitation > p99]
        
        return np.sum(very_extreme_precip)
    
    @staticmethod
    def simple_daily_intensity(precipitation, threshold=1.0):
        """
        Intensidad diaria simple (SDII)
        
        Precipitación promedio en días húmedos.
        
        Args:
            precipitation (np.ndarray): Precipitación diaria
            threshold (float): Umbral para día húmedo
        
        Returns:
            float: SDII
        """
        wet_days = precipitation[precipitation >= threshold]
        
        if len(wet_days) > 0:
            return np.mean(wet_days)
        else:
            return 0.0
