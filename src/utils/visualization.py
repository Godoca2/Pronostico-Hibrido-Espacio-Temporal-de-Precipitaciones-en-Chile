"""
Utilidades de visualización para datos de precipitación
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Visualizer:
    """
    Herramientas de visualización para análisis de precipitación
    """
    
    def __init__(self, style='seaborn-v0_8-darkgrid'):
        """
        Args:
            style (str): Estilo de matplotlib
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        self.cmap_precip = 'YlGnBu'
        self.cmap_anomaly = 'RdBu_r'
    
    def plot_timeseries(self, dates, values, title='Precipitación', 
                       ylabel='Precipitación (mm)', figsize=(12, 4)):
        """
        Grafica serie temporal
        
        Args:
            dates (array-like): Fechas
            values (array-like): Valores
            title (str): Título
            ylabel (str): Etiqueta eje y
            figsize (tuple): Tamaño de figura
        
        Returns:
            matplotlib.figure.Figure: Figura
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(dates, values, linewidth=1.5)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Fecha', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_spatial_field(self, data, lon, lat, title='Campo Espacial',
                          cmap=None, vmin=None, vmax=None, figsize=(10, 8)):
        """
        Grafica campo espacial
        
        Args:
            data (np.ndarray): Datos espaciales (lat, lon)
            lon, lat (np.ndarray): Coordenadas
            title (str): Título
            cmap (str): Mapa de colores
            vmin, vmax (float): Límites de escala
            figsize (tuple): Tamaño de figura
        
        Returns:
            matplotlib.figure.Figure: Figura
        """
        if cmap is None:
            cmap = self.cmap_precip
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Crear mesh
        lon_mesh, lat_mesh = np.meshgrid(lon, lat)
        
        # Plot
        im = ax.pcolormesh(lon_mesh, lat_mesh, data, cmap=cmap, 
                          vmin=vmin, vmax=vmax, shading='auto')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitud', fontsize=12)
        ax.set_ylabel('Latitud', fontsize=12)
        
        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Precipitación (mm)', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def plot_variogram(self, lags, semivariance, model_func=None, 
                      model_params=None, title='Variograma'):
        """
        Grafica variograma experimental y modelo ajustado
        
        Args:
            lags (np.ndarray): Distancias
            semivariance (np.ndarray): Semivarianzas
            model_func (callable): Función del modelo
            model_params (dict): Parámetros del modelo
            title (str): Título
        
        Returns:
            matplotlib.figure.Figure: Figura
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Variograma experimental
        ax.scatter(lags, semivariance, c='blue', s=50, 
                  label='Experimental', alpha=0.7, edgecolors='black')
        
        # Modelo ajustado
        if model_func is not None and model_params is not None:
            h_model = np.linspace(0, lags.max(), 100)
            if 'nugget' in model_params:
                gamma_model = model_func(h_model, model_params['nugget'],
                                        model_params['sill'], 
                                        model_params['range'])
            else:
                gamma_model = model_func(h_model, **model_params)
            
            ax.plot(h_model, gamma_model, 'r-', linewidth=2, 
                   label='Modelo Ajustado')
            
            # Líneas de referencia
            if 'sill' in model_params:
                ax.axhline(y=model_params['sill'], color='gray', 
                          linestyle='--', alpha=0.5, label='Sill')
            if 'range' in model_params:
                ax.axvline(x=model_params['range'], color='gray',
                          linestyle='--', alpha=0.5, label='Range')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Distancia', fontsize=12)
        ax.set_ylabel('Semivarianza', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_dmd_modes(self, modes, n_modes=6, spatial_shape=None):
        """
        Visualiza modos DMD
        
        Args:
            modes (np.ndarray): Modos DMD
            n_modes (int): Número de modos a visualizar
            spatial_shape (tuple): Forma espacial si aplica
        
        Returns:
            matplotlib.figure.Figure: Figura
        """
        n_modes = min(n_modes, modes.shape[1])
        
        if spatial_shape is not None:
            # Modos espaciales
            n_rows = int(np.ceil(n_modes / 3))
            fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
            axes = axes.flatten()
            
            for i in range(n_modes):
                mode_spatial = np.real(modes[:, i].reshape(spatial_shape))
                im = axes[i].imshow(mode_spatial, cmap='RdBu_r')
                axes[i].set_title(f'Modo {i+1}')
                axes[i].axis('off')
                plt.colorbar(im, ax=axes[i])
            
            # Ocultar ejes extras
            for i in range(n_modes, len(axes)):
                axes[i].axis('off')
        else:
            # Modos temporales
            fig, ax = plt.subplots(figsize=(12, 6))
            for i in range(n_modes):
                ax.plot(np.real(modes[:, i]), label=f'Modo {i+1}')
            
            ax.set_title('Modos DMD', fontsize=14, fontweight='bold')
            ax.set_xlabel('Índice', fontsize=12)
            ax.set_ylabel('Amplitud', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_prediction_comparison(self, dates, actual, predicted, 
                                   title='Predicción vs Real'):
        """
        Compara predicciones con valores reales
        
        Args:
            dates (array-like): Fechas
            actual (array-like): Valores reales
            predicted (array-like): Predicciones
            title (str): Título
        
        Returns:
            matplotlib.figure.Figure: Figura
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Serie temporal
        ax1.plot(dates, actual, 'b-', label='Real', linewidth=1.5, alpha=0.7)
        ax1.plot(dates, predicted, 'r--', label='Predicción', linewidth=1.5)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_ylabel('Precipitación (mm)', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot
        ax2.scatter(actual, predicted, alpha=0.5, edgecolors='black')
        
        # Línea 1:1
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', 
                linewidth=2, label='1:1')
        
        ax2.set_xlabel('Real (mm)', fontsize=12)
        ax2.set_ylabel('Predicción (mm)', fontsize=12)
        ax2.set_title('Correlación', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Calcular métricas
        from sklearn.metrics import mean_squared_error, r2_score
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)
        
        ax2.text(0.05, 0.95, f'RMSE: {rmse:.2f}\nR²: {r2:.3f}',
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def plot_spatial_prediction_error(self, error, lon, lat, 
                                      title='Error de Predicción'):
        """
        Visualiza error espacial de predicción
        
        Args:
            error (np.ndarray): Campo de error (lat, lon)
            lon, lat (np.ndarray): Coordenadas
            title (str): Título
        
        Returns:
            matplotlib.figure.Figure: Figura
        """
        fig = self.plot_spatial_field(error, lon, lat, title=title,
                                      cmap='RdBu_r', 
                                      vmin=-np.abs(error).max(),
                                      vmax=np.abs(error).max())
        
        return fig
    
    def plot_koopman_eigenvalues(self, eigenvalues, title='Eigenvalores de Koopman'):
        """
        Visualiza eigenvalores del operador de Koopman
        
        Args:
            eigenvalues (np.ndarray): Eigenvalores complejos
            title (str): Título
        
        Returns:
            matplotlib.figure.Figure: Figura
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Círculo unitario
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1, alpha=0.3)
        
        # Eigenvalores
        ax.scatter(eigenvalues.real, eigenvalues.imag, s=100, 
                  c='red', edgecolors='black', alpha=0.7, zorder=3)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Parte Real', fontsize=12)
        ax.set_ylabel('Parte Imaginaria', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        
        plt.tight_layout()
        return fig
