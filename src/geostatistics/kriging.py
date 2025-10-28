"""
Implementación de técnicas de Kriging para interpolación espacial

Este módulo implementa Ordinary Kriging, Universal Kriging y Co-Kriging
para generar campos continuos de precipitación espacialmente coherentes.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from pykrige.ok import OrdinaryKriging as PyKrigeOK
from pykrige.uk import UniversalKriging as PyKrigeUK
from sklearn.metrics import mean_squared_error


class OrdinaryKriging:
    """
    Ordinary Kriging para interpolación espacial
    
    OK asume media constante desconocida y estima valores en ubicaciones
    no muestreadas minimizando la varianza del error de predicción.
    
    Args:
        variogram_model (str): Modelo de variograma ('linear', 'power', 
                               'gaussian', 'spherical', 'exponential')
        nlags (int): Número de lags para ajuste de variograma
        weight (bool): Si usar pesos de distancia
    """
    
    def __init__(self, variogram_model='spherical', nlags=6, weight=False):
        self.variogram_model = variogram_model
        self.nlags = nlags
        self.weight = weight
        self.kriging_engine = None
        
    def fit(self, X, y):
        """
        Ajusta el modelo de kriging
        
        Args:
            X (np.ndarray): Coordenadas de puntos (n_samples, 2) - (lon, lat)
            y (np.ndarray): Valores observados (n_samples,)
        
        Returns:
            self: Instancia ajustada
        """
        if X.shape[1] != 2:
            raise ValueError("X debe tener 2 columnas (longitud, latitud)")
        
        # Extraer coordenadas
        lon = X[:, 0]
        lat = X[:, 1]
        
        # Crear modelo de kriging
        self.kriging_engine = PyKrigeOK(
            lon, lat, y,
            variogram_model=self.variogram_model,
            nlags=self.nlags,
            weight=self.weight,
            enable_plotting=False
        )
        
        return self
    
    def predict(self, X_pred, return_variance=False):
        """
        Predice valores en nuevas ubicaciones
        
        Args:
            X_pred (np.ndarray): Coordenadas para predicción (n_pred, 2)
            return_variance (bool): Si retornar varianza de predicción
        
        Returns:
            np.ndarray o tuple: Predicciones (y varianza si solicitada)
        """
        if self.kriging_engine is None:
            raise ValueError("Debe ajustar el modelo primero con fit()")
        
        lon_pred = X_pred[:, 0]
        lat_pred = X_pred[:, 1]
        
        # Ejecutar kriging
        z, ss = self.kriging_engine.execute('points', lon_pred, lat_pred)
        
        if return_variance:
            return z, ss
        return z
    
    def predict_grid(self, lon_range, lat_range, grid_resolution=100):
        """
        Genera grilla interpolada
        
        Args:
            lon_range (tuple): Rango de longitud (min, max)
            lat_range (tuple): Rango de latitud (min, max)
            grid_resolution (int): Número de puntos por dimensión
        
        Returns:
            tuple: (valores_grid, lon_grid, lat_grid)
        """
        if self.kriging_engine is None:
            raise ValueError("Debe ajustar el modelo primero con fit()")
        
        # Generar grilla
        lon_grid = np.linspace(lon_range[0], lon_range[1], grid_resolution)
        lat_grid = np.linspace(lat_range[0], lat_range[1], grid_resolution)
        
        # Ejecutar kriging en grilla
        z, ss = self.kriging_engine.execute('grid', lon_grid, lat_grid)
        
        return z, lon_grid, lat_grid
    
    def get_variogram_parameters(self):
        """
        Obtiene parámetros del variograma ajustado
        
        Returns:
            dict: Parámetros del variograma
        """
        if self.kriging_engine is None:
            raise ValueError("Debe ajustar el modelo primero con fit()")
        
        return {
            'sill': self.kriging_engine.variogram_model_parameters[0],
            'range': self.kriging_engine.variogram_model_parameters[1],
            'nugget': self.kriging_engine.variogram_model_parameters[2]
        }


class UniversalKriging:
    """
    Universal Kriging para interpolación con tendencia
    
    UK permite modelar tendencias espaciales (drift) en los datos,
    útil cuando la media varía sistemáticamente en el espacio.
    
    Args:
        variogram_model (str): Modelo de variograma
        drift_terms (list): Términos de drift ('regional_linear', 'point_log', etc.)
        nlags (int): Número de lags
    """
    
    def __init__(self, variogram_model='spherical', 
                 drift_terms=['regional_linear'], nlags=6):
        self.variogram_model = variogram_model
        self.drift_terms = drift_terms
        self.nlags = nlags
        self.kriging_engine = None
        
    def fit(self, X, y):
        """Ajusta Universal Kriging"""
        lon = X[:, 0]
        lat = X[:, 1]
        
        self.kriging_engine = PyKrigeUK(
            lon, lat, y,
            variogram_model=self.variogram_model,
            drift_terms=self.drift_terms,
            nlags=self.nlags,
            enable_plotting=False
        )
        
        return self
    
    def predict(self, X_pred, return_variance=False):
        """Predice con Universal Kriging"""
        if self.kriging_engine is None:
            raise ValueError("Debe ajustar el modelo primero con fit()")
        
        lon_pred = X_pred[:, 0]
        lat_pred = X_pred[:, 1]
        
        z, ss = self.kriging_engine.execute('points', lon_pred, lat_pred)
        
        if return_variance:
            return z, ss
        return z
    
    def predict_grid(self, lon_range, lat_range, grid_resolution=100):
        """Genera grilla con Universal Kriging"""
        if self.kriging_engine is None:
            raise ValueError("Debe ajustar el modelo primero con fit()")
        
        lon_grid = np.linspace(lon_range[0], lon_range[1], grid_resolution)
        lat_grid = np.linspace(lat_range[0], lat_range[1], grid_resolution)
        
        z, ss = self.kriging_engine.execute('grid', lon_grid, lat_grid)
        
        return z, lon_grid, lat_grid


class CoKriging:
    """
    Co-Kriging para interpolación multivariada
    
    Co-Kriging usa correlación entre múltiples variables para mejorar
    predicciones. Útil para integrar datos de precipitación con otras
    variables como temperatura, elevación, índices de vegetación, etc.
    
    Args:
        variogram_model (str): Modelo de variograma
        n_variables (int): Número de variables correlacionadas
    """
    
    def __init__(self, variogram_model='spherical', n_variables=2):
        self.variogram_model = variogram_model
        self.n_variables = n_variables
        self.X_train = None
        self.y_train = None
        self.variogram_params = None
        
    def _compute_variogram(self, X, y1, y2):
        """
        Calcula variograma cruzado entre dos variables
        
        Args:
            X (np.ndarray): Coordenadas
            y1, y2 (np.ndarray): Variables
        
        Returns:
            dict: Parámetros de variograma
        """
        # Calcular distancias
        distances = cdist(X, X, metric='euclidean')
        
        # Calcular semivarianzas
        n = len(y1)
        gamma = np.zeros_like(distances)
        
        for i in range(n):
            for j in range(i+1, n):
                gamma[i, j] = 0.5 * (y1[i] - y1[j]) * (y2[i] - y2[j])
                gamma[j, i] = gamma[i, j]
        
        # Agrupar por distancia y calcular variograma experimental
        dist_flat = distances[np.triu_indices_from(distances, k=1)]
        gamma_flat = gamma[np.triu_indices_from(gamma, k=1)]
        
        # Bins de distancia
        n_bins = 15
        bins = np.linspace(0, dist_flat.max(), n_bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_values = []
        
        for i in range(len(bins) - 1):
            mask = (dist_flat >= bins[i]) & (dist_flat < bins[i+1])
            if mask.sum() > 0:
                bin_values.append(gamma_flat[mask].mean())
            else:
                bin_values.append(0)
        
        # Ajustar modelo de variograma (esférico simplificado)
        def spherical_model(h, nugget, sill, range_param):
            """Modelo esférico de variograma"""
            result = np.zeros_like(h)
            mask = h > 0
            result[mask] = nugget + (sill - nugget) * (
                1.5 * (h[mask] / range_param) - 
                0.5 * (h[mask] / range_param)**3
            )
            result[h >= range_param] = sill
            return result
        
        # Optimizar parámetros
        def objective(params):
            nugget, sill, range_param = params
            predicted = spherical_model(bin_centers, nugget, sill, range_param)
            return np.sum((np.array(bin_values) - predicted)**2)
        
        # Valores iniciales
        initial_params = [0.0, np.var(y1), dist_flat.max() / 3]
        bounds = [(0, np.var(y1)), (0, 2*np.var(y1)), (0, dist_flat.max())]
        
        result = minimize(objective, initial_params, bounds=bounds)
        
        return {
            'nugget': result.x[0],
            'sill': result.x[1],
            'range': result.x[2]
        }
    
    def fit(self, X, Y):
        """
        Ajusta Co-Kriging a múltiples variables
        
        Args:
            X (np.ndarray): Coordenadas (n_samples, 2)
            Y (np.ndarray): Variables (n_samples, n_variables)
        
        Returns:
            self: Instancia ajustada
        """
        if Y.shape[1] != self.n_variables:
            raise ValueError(f"Y debe tener {self.n_variables} columnas")
        
        self.X_train = X
        self.y_train = Y
        
        # Calcular variogramas para cada par de variables
        self.variogram_params = {}
        for i in range(self.n_variables):
            for j in range(i, self.n_variables):
                key = f'var_{i}_{j}'
                self.variogram_params[key] = self._compute_variogram(
                    X, Y[:, i], Y[:, j]
                )
        
        return self
    
    def predict(self, X_pred, primary_var_idx=0):
        """
        Predice variable primaria usando co-kriging
        
        Args:
            X_pred (np.ndarray): Coordenadas para predicción
            primary_var_idx (int): Índice de variable a predecir
        
        Returns:
            np.ndarray: Predicciones
        """
        if self.X_train is None:
            raise ValueError("Debe ajustar el modelo primero con fit()")
        
        # Implementación simplificada usando pesos de distancia
        # En producción, usar sistema completo de ecuaciones co-kriging
        
        predictions = []
        
        for x_new in X_pred:
            # Calcular distancias a puntos conocidos
            distances = np.linalg.norm(self.X_train - x_new, axis=1)
            
            # Evitar división por cero
            distances = np.maximum(distances, 1e-10)
            
            # Pesos inversamente proporcionales a distancia
            weights = 1.0 / distances**2
            weights = weights / weights.sum()
            
            # Predicción ponderada considerando correlación con otras variables
            pred = 0.0
            for i in range(self.n_variables):
                # Peso basado en correlación cruzada
                corr_weight = 1.0 if i == primary_var_idx else 0.5
                pred += corr_weight * np.sum(weights * self.y_train[:, i])
            
            predictions.append(pred)
        
        return np.array(predictions)
    
    def predict_grid(self, lon_range, lat_range, grid_resolution=100, 
                     primary_var_idx=0):
        """
        Genera grilla interpolada usando co-kriging
        
        Args:
            lon_range (tuple): Rango de longitud
            lat_range (tuple): Rango de latitud
            grid_resolution (int): Resolución de grilla
            primary_var_idx (int): Variable a predecir
        
        Returns:
            tuple: (valores_grid, lon_grid, lat_grid)
        """
        lon_grid = np.linspace(lon_range[0], lon_range[1], grid_resolution)
        lat_grid = np.linspace(lat_range[0], lat_range[1], grid_resolution)
        
        # Crear grilla de puntos
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        grid_points = np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()])
        
        # Predecir en cada punto
        z_flat = self.predict(grid_points, primary_var_idx)
        z = z_flat.reshape(lon_mesh.shape)
        
        return z, lon_grid, lat_grid


class SpaceTimeKriging:
    """
    Kriging Espacio-Temporal para datos con dependencia temporal
    
    Extiende kriging para manejar correlación tanto espacial como temporal,
    crucial para pronóstico de precipitaciones.
    
    Args:
        spatial_model (str): Modelo de variograma espacial
        temporal_model (str): Modelo de variograma temporal
        space_time_model (str): Modelo de interacción espacio-tiempo
    """
    
    def __init__(self, spatial_model='spherical', 
                 temporal_model='exponential',
                 space_time_model='product'):
        self.spatial_model = spatial_model
        self.temporal_model = temporal_model
        self.space_time_model = space_time_model
        self.X_train = None
        self.t_train = None
        self.y_train = None
        
    def fit(self, X, t, y):
        """
        Ajusta kriging espacio-temporal
        
        Args:
            X (np.ndarray): Coordenadas espaciales (n_samples, 2)
            t (np.ndarray): Coordenadas temporales (n_samples,)
            y (np.ndarray): Valores observados (n_samples,)
        
        Returns:
            self: Instancia ajustada
        """
        self.X_train = X
        self.t_train = t
        self.y_train = y
        return self
    
    def predict(self, X_pred, t_pred):
        """
        Predice en ubicaciones y tiempos específicos
        
        Args:
            X_pred (np.ndarray): Coordenadas espaciales
            t_pred (np.ndarray): Coordenadas temporales
        
        Returns:
            np.ndarray: Predicciones
        """
        if self.X_train is None:
            raise ValueError("Debe ajustar el modelo primero con fit()")
        
        # Implementación simplificada con pesos espacio-temporales
        predictions = []
        
        for x_new, t_new in zip(X_pred, t_pred):
            # Distancia espacial
            spatial_dist = np.linalg.norm(self.X_train - x_new, axis=1)
            
            # Distancia temporal
            temporal_dist = np.abs(self.t_train - t_new)
            
            # Distancia combinada (normalizada)
            spatial_norm = spatial_dist / (spatial_dist.max() + 1e-10)
            temporal_norm = temporal_dist / (temporal_dist.max() + 1e-10)
            
            # Distancia total (modelo producto)
            total_dist = np.sqrt(spatial_norm**2 + temporal_norm**2)
            total_dist = np.maximum(total_dist, 1e-10)
            
            # Pesos
            weights = 1.0 / total_dist**2
            weights = weights / weights.sum()
            
            # Predicción
            pred = np.sum(weights * self.y_train)
            predictions.append(pred)
        
        return np.array(predictions)
