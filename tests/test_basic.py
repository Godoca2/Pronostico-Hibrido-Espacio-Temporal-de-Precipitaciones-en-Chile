"""
Tests básicos para el sistema de pronóstico
"""

import numpy as np
import pytest
import sys
sys.path.append('..')

from src.models.autoencoder import SpatioTemporalAutoencoder
from src.models.dmd import DynamicModeDecomposition
from src.models.kovae import KoopmanVAE
from src.geostatistics.kriging import OrdinaryKriging
from src.geostatistics.variogram import Variogram, VariogramModel


class TestAutoencoder:
    """Tests para Autoencoder"""
    
    def test_initialization(self):
        """Test inicialización del autoencoder"""
        input_shape = (10, 32, 32)
        model = SpatioTemporalAutoencoder(input_shape, latent_dim=64)
        assert model.latent_dim == 64
        assert model.input_shape == input_shape
    
    def test_forward_pass(self):
        """Test forward pass"""
        import torch
        
        input_shape = (5, 16, 16)
        model = SpatioTemporalAutoencoder(input_shape, latent_dim=32)
        
        # Crear input dummy
        x = torch.randn(2, *input_shape)
        
        # Forward pass
        reconstruction, z = model(x)
        
        assert reconstruction.shape == x.shape
        assert z.shape == (2, 32)


class TestDMD:
    """Tests para DMD"""
    
    def test_fit_predict(self):
        """Test ajuste y predicción"""
        # Datos sintéticos
        t = np.linspace(0, 10, 100)
        X = np.array([
            np.sin(2 * np.pi * t),
            np.cos(2 * np.pi * t)
        ])
        
        dmd = DynamicModeDecomposition(svd_rank=2)
        dmd.fit(X)
        
        # Predicción
        predictions = dmd.predict(10)
        
        assert predictions.shape == (2, 10)
        assert dmd.modes is not None
        assert dmd.eigenvalues is not None
    
    def test_reconstruction(self):
        """Test reconstrucción"""
        t = np.linspace(0, 10, 50)
        X = np.array([np.sin(t), np.cos(t)])
        
        dmd = DynamicModeDecomposition(svd_rank=2)
        dmd.fit(X)
        
        reconstruction = dmd.reconstruct()
        
        assert reconstruction.shape == X.shape


class TestKoVAE:
    """Tests para KoVAE"""
    
    def test_initialization(self):
        """Test inicialización"""
        model = KoopmanVAE(input_dim=100, latent_dim=32)
        assert model.latent_dim == 32
        assert model.input_dim == 100
    
    def test_forward_pass(self):
        """Test forward pass"""
        import torch
        
        model = KoopmanVAE(input_dim=50, latent_dim=16)
        x = torch.randn(4, 50)
        
        output = model(x, predict_steps=0)
        
        assert 'reconstruction' in output
        assert 'mu' in output
        assert 'logvar' in output
        assert output['reconstruction'].shape == x.shape


class TestKriging:
    """Tests para Kriging"""
    
    def test_ordinary_kriging(self):
        """Test Ordinary Kriging"""
        # Datos sintéticos
        np.random.seed(42)
        n_points = 50
        X = np.random.rand(n_points, 2) * 10
        y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + np.random.randn(n_points) * 0.1
        
        # Ajustar kriging
        ok = OrdinaryKriging(variogram_model='spherical')
        ok.fit(X, y)
        
        # Predicción en nuevos puntos
        X_new = np.array([[5.0, 5.0], [2.0, 8.0]])
        predictions = ok.predict(X_new)
        
        assert len(predictions) == 2
        assert ok.kriging_engine is not None


class TestVariogram:
    """Tests para Variogram"""
    
    def test_variogram_computation(self):
        """Test cálculo de variograma"""
        np.random.seed(42)
        n_points = 30
        coordinates = np.random.rand(n_points, 2) * 10
        values = np.random.randn(n_points)
        
        # Calcular variograma
        vario = Variogram(coordinates, values, n_lags=10)
        
        assert vario.lags is not None
        assert vario.semivariance is not None
        assert len(vario.lags) == 10
    
    def test_model_fitting(self):
        """Test ajuste de modelo"""
        np.random.seed(42)
        coordinates = np.random.rand(30, 2) * 10
        values = np.random.randn(30)
        
        vario = Variogram(coordinates, values, n_lags=10)
        params = vario.fit_model('spherical')
        
        assert 'nugget' in params
        assert 'sill' in params
        assert 'range' in params


class TestVariogramModels:
    """Tests para modelos de variograma"""
    
    def test_spherical_model(self):
        """Test modelo esférico"""
        h = np.array([0, 1, 2, 3, 4, 5])
        gamma = VariogramModel.spherical(h, nugget=0.1, sill=1.0, range_param=3.0)
        
        assert gamma[0] == 0  # En h=0
        assert gamma[-1] == 1.0  # En h >= range, gamma = sill
        assert len(gamma) == len(h)
    
    def test_exponential_model(self):
        """Test modelo exponencial"""
        h = np.array([0, 1, 2, 3])
        gamma = VariogramModel.exponential(h, nugget=0.0, sill=1.0, range_param=1.0)
        
        assert gamma[0] == 0
        assert all(gamma >= 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
