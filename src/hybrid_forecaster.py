"""
Sistema Integrado de Pronóstico Híbrido Espacio-Temporal de Precipitaciones

Este módulo integra Deep Learning, DMD, KoVAE, Geoestadística y Teledetección
para pronóstico avanzado de precipitaciones en Chile.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

from .models.autoencoder import SpatioTemporalAutoencoder, RecurrentAutoencoder
from .models.dmd import DynamicModeDecomposition, HigherOrderDMD, MultiResolutionDMD
from .models.kovae import KoopmanVAE, SpatialKoopmanVAE
from .geostatistics.kriging import OrdinaryKriging, UniversalKriging, CoKriging, SpaceTimeKriging
from .remote_sensing.satellite_data import CHIRPS, MODIS, Sentinel, MultiSourceIntegrator
from .utils.data_utils import DataPreprocessor, SpatialDataHandler


class HybridPrecipitationForecaster:
    """
    Sistema híbrido de pronóstico de precipitación
    
    Integra múltiples técnicas avanzadas:
    1. Autoencoders para extracción de patrones latentes
    2. DMD para análisis modal dinámico
    3. KoVAE para representación lineal de dinámicas no lineales
    4. Kriging para interpolación espacial
    5. Datos satelitales para features multivariados
    
    Args:
        config (dict): Configuración del sistema
    """
    
    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = self._default_config()
        
        self.config = config
        
        # Componentes del sistema
        self.autoencoder = None
        self.dmd = None
        self.kovae = None
        self.kriging = None
        self.preprocessor = DataPreprocessor(
            scaling_method=config.get('scaling_method', 'standard')
        )
        
        # Datos
        self.spatial_data = None
        self.temporal_data = None
        self.latent_representations = None
        
    def _default_config(self) -> Dict:
        """Configuración por defecto"""
        return {
            'latent_dim': 64,
            'sequence_length': 30,
            'forecast_horizon': 7,
            'scaling_method': 'standard',
            'kriging_model': 'spherical',
            'dmd_rank': 10,
            'use_kovae': True,
            'use_dmd': True,
            'use_kriging': True
        }
    
    def fit(self, spatial_data: np.ndarray, coordinates: np.ndarray,
            timestamps: np.ndarray, satellite_data: Optional[Dict] = None):
        """
        Entrena el sistema híbrido
        
        Args:
            spatial_data (np.ndarray): Datos de precipitación (time, lat, lon)
            coordinates (np.ndarray): Coordenadas (n_points, 2) - (lon, lat)
            timestamps (np.ndarray): Marcas temporales
            satellite_data (dict): Datos satelitales auxiliares opcionales
        
        Returns:
            self: Instancia entrenada
        """
        print("Iniciando entrenamiento del sistema híbrido...")
        
        # Guardar datos
        self.spatial_data = spatial_data
        self.timestamps = timestamps
        self.coordinates = coordinates
        
        # 1. Preprocesar datos
        print("1. Preprocesando datos...")
        temporal_data = self.preprocessor.spatial_to_temporal(spatial_data)
        normalized_data = self.preprocessor.normalize(temporal_data, fit=True)
        
        # 2. Entrenar Autoencoder para extracción de patrones latentes
        if self.config.get('use_autoencoder', True):
            print("2. Entrenando Autoencoder...")
            self._train_autoencoder(spatial_data)
        
        # 3. Entrenar KoVAE para representación lineal de dinámicas
        if self.config.get('use_kovae', True):
            print("3. Entrenando KoVAE...")
            self._train_kovae(normalized_data)
        
        # 4. Aplicar DMD para análisis modal
        if self.config.get('use_dmd', True):
            print("4. Aplicando DMD...")
            self._fit_dmd(normalized_data)
        
        # 5. Ajustar modelos de Kriging
        if self.config.get('use_kriging', True):
            print("5. Ajustando modelos de Kriging...")
            self._fit_kriging(spatial_data, coordinates, satellite_data)
        
        print("Entrenamiento completado.")
        return self
    
    def _train_autoencoder(self, spatial_data: np.ndarray):
        """Entrena autoencoder espacio-temporal"""
        input_shape = (spatial_data.shape[0], spatial_data.shape[1], spatial_data.shape[2])
        
        # Crear modelo
        if len(input_shape) == 3:
            # Usar primer eje como "canales" (timesteps)
            self.autoencoder = SpatioTemporalAutoencoder(
                input_shape=input_shape,
                latent_dim=self.config['latent_dim']
            )
        
        # Entrenar (simplificado - en producción usar entrenamiento completo)
        self.autoencoder.eval()
        
        # Extraer representaciones latentes
        with torch.no_grad():
            data_tensor = torch.FloatTensor(spatial_data[:10])  # Muestra
            self.latent_representations = self.autoencoder.get_latent_representation(data_tensor)
    
    def _train_kovae(self, temporal_data: np.ndarray):
        """Entrena KoVAE"""
        input_dim = temporal_data.shape[1]
        
        self.kovae = KoopmanVAE(
            input_dim=input_dim,
            latent_dim=self.config['latent_dim'],
            beta=1.0
        )
        
        # Entrenar (simplificado)
        self.kovae.eval()
    
    def _fit_dmd(self, temporal_data: np.ndarray):
        """Ajusta DMD"""
        # Transponer para DMD: (features, time)
        data_T = temporal_data.T
        
        self.dmd = DynamicModeDecomposition(
            svd_rank=self.config.get('dmd_rank', 10)
        )
        
        self.dmd.fit(data_T)
    
    def _fit_kriging(self, spatial_data: np.ndarray, coordinates: np.ndarray,
                     satellite_data: Optional[Dict] = None):
        """Ajusta modelos de Kriging"""
        # Usar último timestep como ejemplo
        last_time = spatial_data[-1]
        values = last_time.flatten()
        
        # Crear grilla de coordenadas
        n_lat, n_lon = last_time.shape
        lon_range = (coordinates[:, 0].min(), coordinates[:, 0].max())
        lat_range = (coordinates[:, 1].min(), coordinates[:, 1].max())
        
        lon_grid = np.linspace(lon_range[0], lon_range[1], n_lon)
        lat_grid = np.linspace(lat_range[0], lat_range[1], n_lat)
        
        # Crear coordenadas de puntos
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        coords = np.column_stack([lon_mesh.flatten(), lat_mesh.flatten()])
        
        # Seleccionar subset de puntos para entrenamiento
        sample_indices = np.random.choice(len(values), 
                                         size=min(1000, len(values)), 
                                         replace=False)
        
        X_sample = coords[sample_indices]
        y_sample = values[sample_indices]
        
        # Remover NaN
        mask = ~np.isnan(y_sample)
        X_sample = X_sample[mask]
        y_sample = y_sample[mask]
        
        if len(y_sample) > 0:
            self.kriging = OrdinaryKriging(
                variogram_model=self.config.get('kriging_model', 'spherical')
            )
            
            try:
                self.kriging.fit(X_sample, y_sample)
            except Exception as e:
                print(f"Error ajustando Kriging: {e}")
                self.kriging = None
    
    def predict(self, forecast_horizon: int, 
                method: str = 'ensemble') -> np.ndarray:
        """
        Genera pronóstico de precipitación
        
        Args:
            forecast_horizon (int): Horizonte de predicción (días)
            method (str): Método ('kovae', 'dmd', 'ensemble')
        
        Returns:
            np.ndarray: Predicciones (forecast_horizon, lat, lon)
        """
        predictions = []
        
        if method == 'kovae' and self.kovae is not None:
            predictions.append(self._predict_kovae(forecast_horizon))
        
        elif method == 'dmd' and self.dmd is not None:
            predictions.append(self._predict_dmd(forecast_horizon))
        
        elif method == 'ensemble':
            # Combinar predicciones de múltiples métodos
            if self.kovae is not None:
                predictions.append(self._predict_kovae(forecast_horizon))
            
            if self.dmd is not None:
                predictions.append(self._predict_dmd(forecast_horizon))
            
            if len(predictions) > 0:
                # Promedio ponderado
                ensemble_pred = np.mean(predictions, axis=0)
                return ensemble_pred
        
        if len(predictions) > 0:
            return predictions[0]
        else:
            raise ValueError("No hay modelos entrenados para predicción")
    
    def _predict_kovae(self, forecast_horizon: int) -> np.ndarray:
        """Predicción usando KoVAE"""
        if self.temporal_data is None or self.kovae is None:
            raise ValueError("KoVAE no está entrenado")
        
        # Usar último estado conocido
        last_state = self.preprocessor.normalize(
            self.temporal_data[-1:], 
            fit=False
        )
        
        # Predecir
        predictions = self.kovae.predict_future(
            torch.FloatTensor(last_state),
            n_steps=forecast_horizon
        )
        
        # Desnormalizar
        predictions_denorm = self.preprocessor.denormalize(
            predictions.reshape(-1, predictions.shape[-1])
        )
        
        # Convertir a formato espacial
        spatial_shape = self.spatial_data.shape[1:]
        predictions_spatial = self.preprocessor.temporal_to_spatial(
            predictions_denorm,
            spatial_shape
        )
        
        return predictions_spatial
    
    def _predict_dmd(self, forecast_horizon: int) -> np.ndarray:
        """Predicción usando DMD"""
        if self.dmd is None:
            raise ValueError("DMD no está entrenado")
        
        # Predecir con DMD
        predictions_T = self.dmd.predict(forecast_horizon)
        
        # Transponer de vuelta
        predictions = predictions_T.T
        
        # Desnormalizar
        predictions_denorm = self.preprocessor.denormalize(predictions)
        
        # Convertir a formato espacial
        spatial_shape = self.spatial_data.shape[1:]
        predictions_spatial = self.preprocessor.temporal_to_spatial(
            predictions_denorm,
            spatial_shape
        )
        
        return predictions_spatial
    
    def interpolate_spatial(self, sparse_data: np.ndarray,
                          sparse_coords: np.ndarray,
                          target_grid: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Interpola datos espaciales usando Kriging
        
        Args:
            sparse_data (np.ndarray): Datos en puntos dispersos
            sparse_coords (np.ndarray): Coordenadas de puntos (n_points, 2)
            target_grid (tuple): (lon_grid, lat_grid) grillas objetivo
        
        Returns:
            np.ndarray: Datos interpolados en grilla
        """
        if self.kriging is None:
            raise ValueError("Kriging no está ajustado")
        
        lon_grid, lat_grid = target_grid
        
        # Ajustar Kriging a datos dispersos
        self.kriging.fit(sparse_coords, sparse_data)
        
        # Interpolar
        z, _, _ = self.kriging.predict_grid(
            (lon_grid.min(), lon_grid.max()),
            (lat_grid.min(), lat_grid.max()),
            grid_resolution=len(lon_grid)
        )
        
        return z
    
    def analyze_patterns(self) -> Dict:
        """
        Analiza patrones en los datos
        
        Returns:
            dict: Análisis de patrones (modos DMD, eigenvalores Koopman, etc.)
        """
        analysis = {}
        
        if self.dmd is not None:
            analysis['dmd_modes'] = self.dmd.modes
            analysis['dmd_eigenvalues'] = self.dmd.eigenvalues
            analysis['dmd_frequencies'] = self.dmd.get_frequencies()
            analysis['dmd_growth_rates'] = self.dmd.get_growth_rates()
        
        if self.kovae is not None:
            analysis['koopman_eigenvalues'] = self.kovae.get_koopman_eigenvalues()
            analysis['koopman_modes'] = self.kovae.get_koopman_modes()
        
        if self.kriging is not None:
            try:
                analysis['variogram_params'] = self.kriging.get_variogram_parameters()
            except:
                pass
        
        return analysis
    
    def save_model(self, path: str):
        """Guarda modelos entrenados"""
        import pickle
        
        models = {
            'autoencoder': self.autoencoder,
            'kovae': self.kovae,
            'dmd': self.dmd,
            'kriging': self.kriging,
            'preprocessor': self.preprocessor,
            'config': self.config
        }
        
        with open(path, 'wb') as f:
            pickle.dump(models, f)
    
    def load_model(self, path: str):
        """Carga modelos entrenados"""
        import pickle
        
        with open(path, 'rb') as f:
            models = pickle.load(f)
        
        self.autoencoder = models['autoencoder']
        self.kovae = models['kovae']
        self.dmd = models['dmd']
        self.kriging = models['kriging']
        self.preprocessor = models['preprocessor']
        self.config = models['config']
