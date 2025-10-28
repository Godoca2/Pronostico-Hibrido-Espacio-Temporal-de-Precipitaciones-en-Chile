"""
Utilidades para carga y preprocesamiento de datos
"""

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


class DataLoader:
    """
    Cargador de datos para pronóstico de precipitación
    
    Maneja diferentes formatos y fuentes de datos.
    """
    
    @staticmethod
    def load_csv(file_path, date_column='date', value_column='precipitation'):
        """
        Carga datos desde CSV
        
        Args:
            file_path (str): Ruta al archivo
            date_column (str): Nombre de columna de fecha
            value_column (str): Nombre de columna de valores
        
        Returns:
            pd.DataFrame: Datos cargados
        """
        df = pd.read_csv(file_path, parse_dates=[date_column])
        df = df.set_index(date_column)
        return df
    
    @staticmethod
    def load_netcdf(file_path, variable='precip'):
        """
        Carga datos desde NetCDF
        
        Args:
            file_path (str): Ruta al archivo
            variable (str): Variable a cargar
        
        Returns:
            xr.DataArray: Datos cargados
        """
        ds = xr.open_dataset(file_path)
        return ds[variable]
    
    @staticmethod
    def load_multiple_stations(file_paths, station_names=None):
        """
        Carga datos de múltiples estaciones
        
        Args:
            file_paths (list): Lista de rutas
            station_names (list): Nombres de estaciones
        
        Returns:
            pd.DataFrame: DataFrame con columna por estación
        """
        if station_names is None:
            station_names = [f'station_{i}' for i in range(len(file_paths))]
        
        dfs = []
        for path, name in zip(file_paths, station_names):
            df = pd.read_csv(path, parse_dates=['date'])
            df = df.set_index('date')
            df = df.rename(columns={'value': name})
            dfs.append(df)
        
        combined = pd.concat(dfs, axis=1)
        return combined


class DataPreprocessor:
    """
    Preprocesador de datos para modelos de predicción
    """
    
    def __init__(self, scaling_method='standard'):
        """
        Args:
            scaling_method (str): 'standard', 'minmax', o None
        """
        self.scaling_method = scaling_method
        self.scaler = None
        
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
    
    def handle_missing_values(self, data, method='interpolate'):
        """
        Maneja valores faltantes
        
        Args:
            data (np.ndarray o pd.DataFrame): Datos con valores faltantes
            method (str): 'interpolate', 'forward_fill', 'backward_fill', 'mean'
        
        Returns:
            Datos sin valores faltantes
        """
        if isinstance(data, pd.DataFrame):
            if method == 'interpolate':
                return data.interpolate(method='linear')
            elif method == 'forward_fill':
                return data.fillna(method='ffill')
            elif method == 'backward_fill':
                return data.fillna(method='bfill')
            elif method == 'mean':
                return data.fillna(data.mean())
        else:
            # NumPy array
            if method == 'mean':
                col_mean = np.nanmean(data, axis=0)
                inds = np.where(np.isnan(data))
                data[inds] = np.take(col_mean, inds[1])
            
            return data
    
    def create_sequences(self, data, sequence_length, forecast_horizon=1, stride=1):
        """
        Crea secuencias para entrenamiento de modelos temporales
        
        Args:
            data (np.ndarray): Datos (n_samples, n_features)
            sequence_length (int): Longitud de secuencia de entrada
            forecast_horizon (int): Horizonte de predicción
            stride (int): Paso entre secuencias
        
        Returns:
            tuple: (X, y) secuencias de entrada y salida
        """
        X, y = [], []
        
        for i in range(0, len(data) - sequence_length - forecast_horizon + 1, stride):
            X.append(data[i:i+sequence_length])
            y.append(data[i+sequence_length:i+sequence_length+forecast_horizon])
        
        return np.array(X), np.array(y)
    
    def normalize(self, data, fit=True):
        """
        Normaliza datos
        
        Args:
            data (np.ndarray): Datos a normalizar
            fit (bool): Si ajustar el scaler
        
        Returns:
            np.ndarray: Datos normalizados
        """
        if self.scaler is None:
            return data
        
        if fit:
            normalized = self.scaler.fit_transform(data)
        else:
            normalized = self.scaler.transform(data)
        
        return normalized
    
    def denormalize(self, data):
        """
        Desnormaliza datos
        
        Args:
            data (np.ndarray): Datos normalizados
        
        Returns:
            np.ndarray: Datos en escala original
        """
        if self.scaler is None:
            return data
        
        return self.scaler.inverse_transform(data)
    
    def spatial_to_temporal(self, spatial_data):
        """
        Convierte datos espaciales a formato temporal
        
        Args:
            spatial_data (np.ndarray): Datos (time, lat, lon)
        
        Returns:
            np.ndarray: Datos (time, features)
        """
        n_times = spatial_data.shape[0]
        flattened = spatial_data.reshape(n_times, -1)
        return flattened
    
    def temporal_to_spatial(self, temporal_data, spatial_shape):
        """
        Convierte datos temporales de vuelta a espaciales
        
        Args:
            temporal_data (np.ndarray): Datos (time, features)
            spatial_shape (tuple): Forma espacial objetivo (lat, lon)
        
        Returns:
            np.ndarray: Datos (time, lat, lon)
        """
        n_times = temporal_data.shape[0]
        spatial = temporal_data.reshape(n_times, *spatial_shape)
        return spatial
    
    def add_temporal_features(self, df, date_column=None):
        """
        Agrega features temporales (día del año, mes, etc.)
        
        Args:
            df (pd.DataFrame): DataFrame con índice de fecha
            date_column (str): Columna de fecha (None si es índice)
        
        Returns:
            pd.DataFrame: DataFrame con features adicionales
        """
        if date_column is not None:
            dates = df[date_column]
        else:
            dates = df.index
        
        df['year'] = dates.year
        df['month'] = dates.month
        df['day'] = dates.day
        df['dayofyear'] = dates.dayofyear
        df['week'] = dates.isocalendar().week
        df['season'] = dates.month % 12 // 3 + 1
        
        # Features cíclicas
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
        
        return df
    
    def train_test_split_temporal(self, X, y, test_size=0.2, validation_size=0.1):
        """
        Divide datos respetando orden temporal
        
        Args:
            X, y (np.ndarray): Datos de entrada y salida
            test_size (float): Proporción de test
            validation_size (float): Proporción de validación
        
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        n_samples = len(X)
        
        # Índices de división
        test_start = int(n_samples * (1 - test_size))
        val_start = int(test_start * (1 - validation_size))
        
        # División
        X_train = X[:val_start]
        y_train = y[:val_start]
        
        X_val = X[val_start:test_start]
        y_val = y[val_start:test_start]
        
        X_test = X[test_start:]
        y_test = y[test_start:]
        
        return X_train, X_val, X_test, y_train, y_val, y_test


class SpatialDataHandler:
    """
    Maneja operaciones espaciales en datos de precipitación
    """
    
    @staticmethod
    def extract_point_timeseries(spatial_data, lon, lat, lon_coords, lat_coords):
        """
        Extrae serie temporal en un punto específico
        
        Args:
            spatial_data (np.ndarray): Datos (time, lat, lon)
            lon, lat (float): Coordenadas del punto
            lon_coords, lat_coords (np.ndarray): Coordenadas de la grilla
        
        Returns:
            np.ndarray: Serie temporal en el punto
        """
        # Encontrar índice más cercano
        lon_idx = np.argmin(np.abs(lon_coords - lon))
        lat_idx = np.argmin(np.abs(lat_coords - lat))
        
        timeseries = spatial_data[:, lat_idx, lon_idx]
        
        return timeseries
    
    @staticmethod
    def spatial_average(spatial_data, weights=None):
        """
        Calcula promedio espacial
        
        Args:
            spatial_data (np.ndarray): Datos (time, lat, lon)
            weights (np.ndarray): Pesos espaciales opcionales
        
        Returns:
            np.ndarray: Serie temporal promediada
        """
        if weights is None:
            # Promedio simple
            avg = np.mean(spatial_data, axis=(1, 2))
        else:
            # Promedio ponderado
            avg = np.average(spatial_data, axis=(1, 2), weights=weights)
        
        return avg
    
    @staticmethod
    def compute_spatial_correlation(data1, data2):
        """
        Calcula correlación espacial entre dos campos
        
        Args:
            data1, data2 (np.ndarray): Campos espaciales (lat, lon)
        
        Returns:
            float: Correlación espacial
        """
        flat1 = data1.flatten()
        flat2 = data2.flatten()
        
        # Remover NaN
        mask = ~(np.isnan(flat1) | np.isnan(flat2))
        
        if mask.sum() > 0:
            corr = np.corrcoef(flat1[mask], flat2[mask])[0, 1]
        else:
            corr = np.nan
        
        return corr
