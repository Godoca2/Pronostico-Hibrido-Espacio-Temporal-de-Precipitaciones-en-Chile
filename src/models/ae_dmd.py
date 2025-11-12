"""(AE minimal + DMD sintético) (esqueleto)
Esto es un esqueleto para validar el flujo. Luego lo reemplazas por tu AE real (TF/Keras) 
y DMD con PyDMD sobre tus datos.
"""

import numpy as np
from pydmd import DMD

# AE de juguete (no entrena nada real: solo reduce dimensión con PCA-like)
from sklearn.decomposition import PCA

class SimpleAEdmd:
    def __init__(self, latent_dim=10):
        self.latent_dim = latent_dim
        self.encoder = PCA(n_components=latent_dim)
        self.decoder = None  # en PCA, inverse_transform sirve de "decoder"
        self.dmd = None

    def fit(self, X_time_series):
        """
        X_time_series: np.array shape (T, N_features)
        """
        Z = self.encoder.fit_transform(X_time_series)
        # DMD en espacio latente (cada columna es un "snapshot")
        snapshots = Z.T
        self.dmd = DMD(svd_rank=min(self.latent_dim, 20))
        self.dmd.fit(snapshots)
        # decoder implícito es el inverse_transform de PCA
        self.decoder = self.encoder
        return self

    def forecast(self, steps=10):
        # predicción en latente y decodificación
        Z_future = self.dmd.reconstructed_data.real.T[-steps:]
        X_future = self.decoder.inverse_transform(Z_future)
        return X_future

