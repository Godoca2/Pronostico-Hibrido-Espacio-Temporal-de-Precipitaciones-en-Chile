"""
Descomposición Modal Dinámica (DMD) para análisis y predicción temporal

Este módulo implementa DMD y sus variantes para extraer modos dinámicos
y realizar predicciones de la evolución temporal de patrones de precipitación.
"""

import numpy as np
from scipy import linalg
from pydmd import DMD as PyDMD
from pydmd import HODMD, MrDMD, CDMD


class DynamicModeDecomposition:
    """
    Descomposición Modal Dinámica para series temporales espaciales
    
    DMD descompone datos espacio-temporales en modos dinámicos coherentes,
    cada uno asociado con una frecuencia y tasa de crecimiento/decaimiento.
    
    Args:
        svd_rank (int): Rango para truncamiento SVD (0 para óptimo automático)
        tlsq_rank (int): Rango para mínimos cuadrados totales
        exact (bool): Si usar DMD exacto
        opt (bool): Si usar DMD óptimo
    """
    
    def __init__(self, svd_rank=0, tlsq_rank=0, exact=True, opt=False):
        self.svd_rank = svd_rank
        self.tlsq_rank = tlsq_rank
        self.exact = exact
        self.opt = opt
        
        # Atributos calculados
        self.modes = None
        self.dynamics = None
        self.eigenvalues = None
        self.amplitudes = None
        self.timesteps = None
        
    def fit(self, X, time=None):
        """
        Ajusta DMD a los datos
        
        Args:
            X (np.ndarray): Datos de forma (n_features, n_timesteps)
            time (np.ndarray): Vector de tiempo opcional
        
        Returns:
            self: Instancia ajustada
        """
        if X.ndim != 2:
            raise ValueError("X debe ser 2D: (n_features, n_timesteps)")
        
        n_features, n_timesteps = X.shape
        self.timesteps = time if time is not None else np.arange(n_timesteps)
        
        # Matrices de snapshots
        X1 = X[:, :-1]
        X2 = X[:, 1:]
        
        # SVD de X1
        U, s, Vh = np.linalg.svd(X1, full_matrices=False)
        
        # Determinar rango
        if self.svd_rank == 0:
            # Criterio de energía: mantener 99% de la energía
            cumsum_energy = np.cumsum(s**2) / np.sum(s**2)
            rank = np.searchsorted(cumsum_energy, 0.99) + 1
        else:
            rank = min(self.svd_rank, len(s))
        
        # Truncar SVD
        U_r = U[:, :rank]
        s_r = s[:rank]
        V_r = Vh[:rank, :].conj().T
        
        # Matriz A reducida
        A_tilde = U_r.conj().T @ X2 @ V_r @ np.diag(1/s_r)
        
        # Eigendescomposición
        eigenvalues, eigenvectors = np.linalg.eig(A_tilde)
        
        # Calcular modos DMD
        if self.exact:
            # DMD exacto
            self.modes = X2 @ V_r @ np.diag(1/s_r) @ eigenvectors
        else:
            # DMD proyectado
            self.modes = U_r @ eigenvectors
        
        self.eigenvalues = eigenvalues
        
        # Calcular amplitudes
        self.amplitudes = np.linalg.lstsq(self.modes, X[:, 0], rcond=None)[0]
        
        # Calcular dinámica
        time_dynamics = np.zeros((rank, len(self.timesteps)), dtype=complex)
        for i, lam in enumerate(eigenvalues):
            time_dynamics[i, :] = self.amplitudes[i] * (lam ** self.timesteps)
        
        self.dynamics = time_dynamics
        
        return self
    
    def predict(self, n_steps):
        """
        Predice evolución futura usando modos DMD
        
        Args:
            n_steps (int): Número de pasos a predecir
        
        Returns:
            np.ndarray: Predicciones de forma (n_features, n_steps)
        """
        if self.modes is None:
            raise ValueError("Debe ajustar el modelo primero con fit()")
        
        # Tiempo futuro
        last_time = self.timesteps[-1]
        dt = self.timesteps[1] - self.timesteps[0]
        future_times = last_time + dt * (np.arange(1, n_steps + 1))
        
        # Calcular dinámica futura
        future_dynamics = np.zeros((len(self.eigenvalues), n_steps), dtype=complex)
        for i, lam in enumerate(self.eigenvalues):
            future_dynamics[i, :] = self.amplitudes[i] * (lam ** future_times)
        
        # Reconstruir predicción
        prediction = np.real(self.modes @ future_dynamics)
        
        return prediction
    
    def reconstruct(self):
        """
        Reconstruye los datos originales usando modos DMD
        
        Returns:
            np.ndarray: Datos reconstruidos
        """
        if self.modes is None:
            raise ValueError("Debe ajustar el modelo primero con fit()")
        
        return np.real(self.modes @ self.dynamics)
    
    def get_frequencies(self, dt=1.0):
        """
        Obtiene frecuencias asociadas a cada modo
        
        Args:
            dt (float): Paso de tiempo
        
        Returns:
            np.ndarray: Frecuencias
        """
        if self.eigenvalues is None:
            raise ValueError("Debe ajustar el modelo primero con fit()")
        
        omega = np.log(self.eigenvalues) / dt
        frequencies = np.imag(omega) / (2 * np.pi)
        
        return frequencies
    
    def get_growth_rates(self, dt=1.0):
        """
        Obtiene tasas de crecimiento/decaimiento de cada modo
        
        Args:
            dt (float): Paso de tiempo
        
        Returns:
            np.ndarray: Tasas de crecimiento
        """
        if self.eigenvalues is None:
            raise ValueError("Debe ajustar el modelo primero con fit()")
        
        omega = np.log(self.eigenvalues) / dt
        growth_rates = np.real(omega)
        
        return growth_rates


class HigherOrderDMD:
    """
    DMD de Orden Superior (HODMD) para capturar dinámicas de múltiples escalas
    
    HODMD extiende DMD estándar usando ventanas temporales deslizantes
    para capturar mejor dinámicas complejas.
    
    Args:
        svd_rank (int): Rango SVD
        d (int): Número de delays (orden)
    """
    
    def __init__(self, svd_rank=0, d=10):
        self.svd_rank = svd_rank
        self.d = d
        self.hodmd = HODMD(svd_rank=svd_rank, d=d)
        
    def fit(self, X, time=None):
        """Ajusta HODMD a los datos"""
        self.hodmd.fit(X)
        return self
    
    def predict(self, n_steps):
        """Predice usando HODMD"""
        self.hodmd.dmd_time['tend'] = self.hodmd.dmd_time['tend'] + n_steps
        prediction = self.hodmd.reconstructed_data[:, -n_steps:]
        return prediction
    
    @property
    def modes(self):
        """Modos HODMD"""
        return self.hodmd.modes
    
    @property
    def eigenvalues(self):
        """Eigenvalores HODMD"""
        return self.hodmd.eigs


class MultiResolutionDMD:
    """
    DMD Multi-Resolución (mrDMD) para capturar dinámicas en múltiples escalas
    
    mrDMD aplica DMD recursivamente en diferentes escalas temporales,
    capturando tanto dinámicas rápidas como lentas.
    
    Args:
        svd_rank (int): Rango SVD
        max_level (int): Nivel máximo de recursión
        max_cycles (int): Ciclos máximos por nivel
    """
    
    def __init__(self, svd_rank=0, max_level=5, max_cycles=2):
        self.svd_rank = svd_rank
        self.max_level = max_level
        self.max_cycles = max_cycles
        self.mrdmd = MrDMD(svd_rank=svd_rank, max_level=max_level, 
                           max_cycles=max_cycles)
        
    def fit(self, X, time=None):
        """Ajusta mrDMD a los datos"""
        self.mrdmd.fit(X)
        return self
    
    def predict(self, n_steps):
        """Predice usando mrDMD"""
        # mrDMD requiere implementación personalizada para predicción
        # Usar DMD de nivel superior
        self.mrdmd.dmd_time['tend'] = self.mrdmd.dmd_time['tend'] + n_steps
        return self.mrdmd.reconstructed_data
    
    @property
    def modes(self):
        """Modos mrDMD en todas las escalas"""
        return self.mrdmd.modes
    
    @property
    def eigenvalues(self):
        """Eigenvalores mrDMD"""
        return self.mrdmd.eigs


class CompressedDMD:
    """
    DMD Comprimido (CDMD) para datos de alta dimensión
    
    CDMD usa compresión aleatoria para escalar DMD a datasets muy grandes.
    
    Args:
        svd_rank (int): Rango SVD
        compression_ratio (float): Ratio de compresión (0, 1)
    """
    
    def __init__(self, svd_rank=0, compression_ratio=0.5):
        self.svd_rank = svd_rank
        self.compression_ratio = compression_ratio
        self.cdmd = CDMD(svd_rank=svd_rank, compression_matrix=None)
        
    def fit(self, X, time=None):
        """Ajusta CDMD a los datos"""
        # Crear matriz de compresión aleatoria
        n_features = X.shape[0]
        compressed_dim = int(n_features * self.compression_ratio)
        compression_matrix = np.random.randn(compressed_dim, n_features)
        compression_matrix /= np.linalg.norm(compression_matrix, axis=1, keepdims=True)
        
        self.cdmd.compression_matrix = compression_matrix
        self.cdmd.fit(X)
        return self
    
    def predict(self, n_steps):
        """Predice usando CDMD"""
        self.cdmd.dmd_time['tend'] = self.cdmd.dmd_time['tend'] + n_steps
        return self.cdmd.reconstructed_data
    
    @property
    def modes(self):
        """Modos CDMD"""
        return self.cdmd.modes
    
    @property
    def eigenvalues(self):
        """Eigenvalores CDMD"""
        return self.cdmd.eigs
