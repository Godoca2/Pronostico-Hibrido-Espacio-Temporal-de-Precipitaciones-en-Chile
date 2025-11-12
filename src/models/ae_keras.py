"""
ae_keras.py
------------
Autoencoder mínimo en Keras para reducción de dimensionalidad
de series temporales climáticas (ej. precipitación).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def build_autoencoder(input_dim: int, latent_dim: int = 32) -> models.Model:
    """
    Crea un autoencoder denso simple (puede reemplazarse luego por CNN/LSTM).
    
    Parámetros
    ----------
    input_dim : int
        Dimensión de entrada (número de características)
    latent_dim : int
        Dimensión del espacio latente
    """
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(128, activation="relu")(input_layer)
    encoded = layers.Dense(latent_dim, activation="relu")(encoded)

    decoded = layers.Dense(128, activation="relu")(encoded)
    decoded = layers.Dense(input_dim, activation="linear")(decoded)

    autoencoder = models.Model(input_layer, decoded)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


def train_autoencoder(X_train: np.ndarray, latent_dim: int = 32, epochs: int = 20, batch_size: int = 32):
    """
    Entrena el autoencoder y retorna el modelo entrenado.
    """
    input_dim = X_train.shape[1]
    model = build_autoencoder(input_dim, latent_dim)
    print(f"Entrenando Autoencoder: input_dim={input_dim}, latent_dim={latent_dim}")

    history = model.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )
    return model, history


def encode(model: models.Model, X: np.ndarray) -> np.ndarray:
    """
    Retorna representaciones latentes (codificadas).
    """
    encoder = models.Model(model.input, model.layers[2].output)
    return encoder.predict(X)


def decode(model: models.Model, Z: np.ndarray) -> np.ndarray:
    """
    Reconstruye datos a partir del espacio latente.
    """
    decoder_input = layers.Input(shape=(Z.shape[1],))
    x = model.layers[-2](decoder_input)
    output = model.layers[-1](x)
    decoder = models.Model(decoder_input, output)
    return decoder.predict(Z)


if __name__ == "__main__":
    # Ejemplo rápido
    X = np.random.rand(500, 50)
    model, hist = train_autoencoder(X, latent_dim=10, epochs=5)
    model.save("data/models/ae_minimal.h5")
    print("✅ Entrenamiento finalizado y modelo guardado.")
