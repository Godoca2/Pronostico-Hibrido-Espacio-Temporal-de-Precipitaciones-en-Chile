"""
Modelos de Variograma para análisis geoestadístico

El variograma cuantifica la correlación espacial y es fundamental
para técnicas de kriging.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform


class Variogram:
    """
    Calculador de variogramas experimentales y ajuste de modelos teóricos
    
    El variograma describe cómo la variabilidad entre pares de puntos
    cambia con la distancia de separación.
    
    Args:
        coordinates (np.ndarray): Coordenadas de puntos (n_points, 2)
        values (np.ndarray): Valores en cada punto (n_points,)
        n_lags (int): Número de intervalos de distancia
        maxlag (float): Distancia máxima a considerar (None = auto)
    """
    
    def __init__(self, coordinates, values, n_lags=15, maxlag=None):
        self.coordinates = coordinates
        self.values = values
        self.n_lags = n_lags
        
        # Calcular distancias entre todos los pares
        self.distances = squareform(pdist(coordinates))
        
        # Determinar maxlag si no se especifica
        if maxlag is None:
            self.maxlag = self.distances.max() / 2
        else:
            self.maxlag = maxlag
        
        # Calcular variograma experimental
        self.lags, self.semivariance = self._compute_experimental()
        
    def _compute_experimental(self):
        """
        Calcula variograma experimental
        
        Returns:
            tuple: (lags, semivarianzas)
        """
        # Crear bins de distancia
        lag_edges = np.linspace(0, self.maxlag, self.n_lags + 1)
        lag_centers = (lag_edges[:-1] + lag_edges[1:]) / 2
        
        # Calcular semivarianzas para cada lag
        semivariances = []
        
        for i in range(self.n_lags):
            # Encontrar pares en este rango de distancia
            mask = (self.distances >= lag_edges[i]) & \
                   (self.distances < lag_edges[i+1])
            
            if mask.sum() > 0:
                # Extraer valores de los pares
                n_points = len(self.values)
                pairs_diff_squared = []
                
                for j in range(n_points):
                    for k in range(j+1, n_points):
                        if mask[j, k]:
                            diff = (self.values[j] - self.values[k])**2
                            pairs_diff_squared.append(diff)
                
                # Semivarianza = promedio de (diferencia)^2 / 2
                if len(pairs_diff_squared) > 0:
                    semivar = np.mean(pairs_diff_squared) / 2
                else:
                    semivar = 0.0
            else:
                semivar = 0.0
            
            semivariances.append(semivar)
        
        return lag_centers, np.array(semivariances)
    
    def fit_model(self, model='spherical'):
        """
        Ajusta un modelo teórico al variograma experimental
        
        Args:
            model (str): Tipo de modelo ('spherical', 'exponential', 
                        'gaussian', 'linear', 'power')
        
        Returns:
            dict: Parámetros del modelo ajustado
        """
        model_func = VariogramModel.get_model(model)
        
        # Estimación inicial de parámetros
        sill_init = self.semivariance.max()
        range_init = self.lags[np.argmax(self.semivariance > 0.95 * sill_init)] \
                     if any(self.semivariance > 0.95 * sill_init) else self.maxlag / 3
        nugget_init = self.semivariance[0] if len(self.semivariance) > 0 else 0
        
        # Asegurar valores positivos
        sill_init = max(sill_init, 1e-6)
        range_init = max(range_init, 1e-6)
        nugget_init = max(nugget_init, 0)
        
        try:
            if model in ['spherical', 'exponential', 'gaussian']:
                # Modelos con nugget, sill, range
                params, _ = curve_fit(
                    lambda h, n, s, r: model_func(h, n, s, r),
                    self.lags,
                    self.semivariance,
                    p0=[nugget_init, sill_init, range_init],
                    bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
                    maxfev=10000
                )
                return {
                    'model': model,
                    'nugget': params[0],
                    'sill': params[1],
                    'range': params[2]
                }
            elif model == 'linear':
                # Modelo lineal: nugget + slope * h
                params, _ = curve_fit(
                    lambda h, n, s: model_func(h, n, s),
                    self.lags,
                    self.semivariance,
                    p0=[nugget_init, sill_init / self.maxlag],
                    bounds=([0, 0], [np.inf, np.inf]),
                    maxfev=10000
                )
                return {
                    'model': model,
                    'nugget': params[0],
                    'slope': params[1]
                }
            elif model == 'power':
                # Modelo potencial: scale * h^exponent
                params, _ = curve_fit(
                    lambda h, s, e: model_func(h, s, e),
                    self.lags[self.lags > 0],  # Evitar h=0 para power
                    self.semivariance[self.lags > 0],
                    p0=[sill_init, 1.0],
                    bounds=([0, 0], [np.inf, 2]),
                    maxfev=10000
                )
                return {
                    'model': model,
                    'scale': params[0],
                    'exponent': params[1]
                }
        except Exception as e:
            print(f"Error ajustando modelo {model}: {e}")
            # Retornar parámetros iniciales en caso de error
            if model in ['spherical', 'exponential', 'gaussian']:
                return {
                    'model': model,
                    'nugget': nugget_init,
                    'sill': sill_init,
                    'range': range_init
                }
            elif model == 'linear':
                return {
                    'model': model,
                    'nugget': nugget_init,
                    'slope': sill_init / self.maxlag
                }
            else:
                return {
                    'model': model,
                    'scale': sill_init,
                    'exponent': 1.0
                }


class VariogramModel:
    """
    Modelos teóricos de variograma
    
    Implementa funciones matemáticas para variogramas teóricos
    comúnmente usados en geoestadística.
    """
    
    @staticmethod
    def spherical(h, nugget, sill, range_param):
        """
        Modelo esférico de variograma
        
        Args:
            h (float or np.ndarray): Distancia de separación
            nugget (float): Efecto pepita (discontinuidad en origen)
            sill (float): Meseta (varianza total)
            range_param (float): Alcance (distancia donde se alcanza sill)
        
        Returns:
            float or np.ndarray: Semivarianza
        """
        h = np.asarray(h)
        gamma = np.zeros_like(h, dtype=float)
        
        # Para h = 0
        gamma[h == 0] = 0
        
        # Para 0 < h < range
        mask1 = (h > 0) & (h < range_param)
        gamma[mask1] = nugget + (sill - nugget) * (
            1.5 * (h[mask1] / range_param) - 
            0.5 * (h[mask1] / range_param)**3
        )
        
        # Para h >= range
        mask2 = h >= range_param
        gamma[mask2] = sill
        
        return gamma
    
    @staticmethod
    def exponential(h, nugget, sill, range_param):
        """
        Modelo exponencial de variograma
        
        Alcanza asintóticamente el sill.
        """
        h = np.asarray(h)
        gamma = np.zeros_like(h, dtype=float)
        
        gamma[h == 0] = 0
        gamma[h > 0] = nugget + (sill - nugget) * (
            1 - np.exp(-h[h > 0] / range_param)
        )
        
        return gamma
    
    @staticmethod
    def gaussian(h, nugget, sill, range_param):
        """
        Modelo Gaussiano de variograma
        
        Muy suave en el origen, alcanza sill asintóticamente.
        """
        h = np.asarray(h)
        gamma = np.zeros_like(h, dtype=float)
        
        gamma[h == 0] = 0
        gamma[h > 0] = nugget + (sill - nugget) * (
            1 - np.exp(-(h[h > 0] / range_param)**2)
        )
        
        return gamma
    
    @staticmethod
    def linear(h, nugget, slope):
        """
        Modelo lineal de variograma
        
        Crece linealmente sin límite.
        """
        h = np.asarray(h)
        return nugget + slope * h
    
    @staticmethod
    def power(h, scale, exponent):
        """
        Modelo potencial de variograma
        
        Forma: scale * h^exponent
        """
        h = np.asarray(h)
        gamma = np.zeros_like(h, dtype=float)
        gamma[h > 0] = scale * (h[h > 0] ** exponent)
        return gamma
    
    @staticmethod
    def get_model(model_name):
        """
        Obtiene función de modelo por nombre
        
        Args:
            model_name (str): Nombre del modelo
        
        Returns:
            callable: Función del modelo
        """
        models = {
            'spherical': VariogramModel.spherical,
            'exponential': VariogramModel.exponential,
            'gaussian': VariogramModel.gaussian,
            'linear': VariogramModel.linear,
            'power': VariogramModel.power
        }
        
        if model_name not in models:
            raise ValueError(f"Modelo {model_name} no reconocido. "
                           f"Opciones: {list(models.keys())}")
        
        return models[model_name]


class DirectionalVariogram(Variogram):
    """
    Variograma direccional para detectar anisotropía
    
    Calcula variogramas en diferentes direcciones para identificar
    si la correlación espacial varía con la dirección.
    
    Args:
        coordinates (np.ndarray): Coordenadas
        values (np.ndarray): Valores
        directions (list): Lista de ángulos en grados
        tolerance (float): Tolerancia angular en grados
        n_lags (int): Número de lags
    """
    
    def __init__(self, coordinates, values, directions=[0, 45, 90, 135],
                 tolerance=22.5, n_lags=15, maxlag=None):
        self.directions = directions
        self.tolerance = tolerance
        
        super().__init__(coordinates, values, n_lags, maxlag)
        
        # Calcular variogramas direccionales
        self.directional_variograms = {}
        for direction in directions:
            lags, semivar = self._compute_directional(direction, tolerance)
            self.directional_variograms[direction] = (lags, semivar)
    
    def _compute_directional(self, direction, tolerance):
        """
        Calcula variograma en una dirección específica
        
        Args:
            direction (float): Ángulo en grados (0 = Este, 90 = Norte)
            tolerance (float): Tolerancia angular
        
        Returns:
            tuple: (lags, semivarianzas)
        """
        # Convertir a radianes
        dir_rad = np.radians(direction)
        tol_rad = np.radians(tolerance)
        
        # Calcular ángulos entre todos los pares
        n_points = len(self.coordinates)
        angles = np.zeros((n_points, n_points))
        
        for i in range(n_points):
            for j in range(i+1, n_points):
                dx = self.coordinates[j, 0] - self.coordinates[i, 0]
                dy = self.coordinates[j, 1] - self.coordinates[i, 1]
                angle = np.arctan2(dy, dx)
                angles[i, j] = angle
                angles[j, i] = angle
        
        # Máscara direccional
        angle_diff = np.abs(angles - dir_rad)
        angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
        directional_mask = angle_diff <= tol_rad
        
        # Crear bins de distancia
        lag_edges = np.linspace(0, self.maxlag, self.n_lags + 1)
        lag_centers = (lag_edges[:-1] + lag_edges[1:]) / 2
        
        # Calcular semivarianzas
        semivariances = []
        
        for i in range(self.n_lags):
            distance_mask = (self.distances >= lag_edges[i]) & \
                          (self.distances < lag_edges[i+1])
            mask = distance_mask & directional_mask
            
            if mask.sum() > 0:
                pairs_diff_squared = []
                for j in range(n_points):
                    for k in range(j+1, n_points):
                        if mask[j, k]:
                            diff = (self.values[j] - self.values[k])**2
                            pairs_diff_squared.append(diff)
                
                if len(pairs_diff_squared) > 0:
                    semivar = np.mean(pairs_diff_squared) / 2
                else:
                    semivar = 0.0
            else:
                semivar = 0.0
            
            semivariances.append(semivar)
        
        return lag_centers, np.array(semivariances)
