"""
Autoencoder Espacio-Temporal para extracción de patrones latentes en datos de precipitación

Este módulo implementa un autoencoder profundo diseñado para capturar patrones
espacio-temporales en datos de precipitación.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpatioTemporalAutoencoder(nn.Module):
    """
    Autoencoder Espacio-Temporal para datos de precipitación
    
    Arquitectura diseñada para:
    - Extraer patrones latentes de series temporales espaciales
    - Reducción de dimensionalidad preservando información espacial
    - Reconstrucción de campos de precipitación
    
    Args:
        input_shape (tuple): Forma de entrada (time_steps, height, width)
        latent_dim (int): Dimensión del espacio latente
        conv_filters (list): Lista de filtros para capas convolucionales
        use_batch_norm (bool): Si usar normalización por lotes
    """
    
    def __init__(self, input_shape, latent_dim=64, 
                 conv_filters=[32, 64, 128], use_batch_norm=True):
        super(SpatioTemporalAutoencoder, self).__init__()
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.conv_filters = conv_filters
        self.use_batch_norm = use_batch_norm
        
        # Encoder
        encoder_layers = []
        in_channels = input_shape[0]  # time_steps como canales
        
        for filters in conv_filters:
            encoder_layers.append(
                nn.Conv2d(in_channels, filters, kernel_size=3, stride=2, padding=1)
            )
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm2d(filters))
            encoder_layers.append(nn.ReLU(inplace=True))
            in_channels = filters
        
        self.encoder_conv = nn.Sequential(*encoder_layers)
        
        # Calcular dimensión después de convoluciones
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_shape[0], input_shape[1], input_shape[2])
            conv_output = self.encoder_conv(dummy_input)
            self.conv_output_shape = conv_output.shape[1:]
            flattened_size = conv_output.view(1, -1).shape[1]
        
        # Capa densa para espacio latente
        self.encoder_fc = nn.Linear(flattened_size, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, flattened_size)
        
        decoder_layers = []
        in_channels = conv_filters[-1]
        
        for i in range(len(conv_filters) - 1, 0, -1):
            decoder_layers.append(
                nn.ConvTranspose2d(in_channels, conv_filters[i-1], 
                                   kernel_size=3, stride=2, padding=1, output_padding=1)
            )
            if use_batch_norm:
                decoder_layers.append(nn.BatchNorm2d(conv_filters[i-1]))
            decoder_layers.append(nn.ReLU(inplace=True))
            in_channels = conv_filters[i-1]
        
        # Última capa para reconstruir la entrada original
        decoder_layers.append(
            nn.ConvTranspose2d(conv_filters[0], input_shape[0], 
                               kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        decoder_layers.append(nn.Sigmoid())  # Para normalizar salida
        
        self.decoder_conv = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """
        Codifica la entrada al espacio latente
        
        Args:
            x (torch.Tensor): Tensor de entrada (batch, time_steps, height, width)
        
        Returns:
            torch.Tensor: Representación latente (batch, latent_dim)
        """
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        z = self.encoder_fc(x)
        return z
    
    def decode(self, z):
        """
        Decodifica del espacio latente a la forma original
        
        Args:
            z (torch.Tensor): Representación latente (batch, latent_dim)
        
        Returns:
            torch.Tensor: Reconstrucción (batch, time_steps, height, width)
        """
        x = self.decoder_fc(z)
        x = x.view(-1, self.conv_output_shape[0], 
                   self.conv_output_shape[1], self.conv_output_shape[2])
        x = self.decoder_conv(x)
        return x
    
    def forward(self, x):
        """
        Forward pass completo: codificación y decodificación
        
        Args:
            x (torch.Tensor): Tensor de entrada
        
        Returns:
            tuple: (reconstrucción, representación_latente)
        """
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z
    
    def get_latent_representation(self, x):
        """
        Obtiene solo la representación latente
        
        Args:
            x (torch.Tensor): Tensor de entrada
        
        Returns:
            np.ndarray: Representación latente como numpy array
        """
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            z = self.encode(x)
        return z.cpu().numpy()


class RecurrentAutoencoder(nn.Module):
    """
    Autoencoder Recurrente para capturar dependencias temporales
    
    Usa LSTM para modelar secuencias temporales de datos espaciales
    
    Args:
        input_shape (tuple): Forma de entrada (seq_len, height, width)
        latent_dim (int): Dimensión del espacio latente
        hidden_size (int): Tamaño de estados ocultos LSTM
        num_layers (int): Número de capas LSTM
    """
    
    def __init__(self, input_shape, latent_dim=64, hidden_size=128, num_layers=2):
        super(RecurrentAutoencoder, self).__init__()
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Convolución espacial para cada timestep
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Calcular tamaño después de convoluciones
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_shape[1], input_shape[2])
            spatial_out = self.spatial_encoder(dummy)
            self.spatial_out_size = spatial_out.view(1, -1).shape[1]
        
        # LSTM Encoder
        self.lstm_encoder = nn.LSTM(self.spatial_out_size, hidden_size, 
                                     num_layers, batch_first=True)
        self.fc_latent = nn.Linear(hidden_size, latent_dim)
        
        # LSTM Decoder
        self.fc_decoder = nn.Linear(latent_dim, hidden_size)
        self.lstm_decoder = nn.LSTM(hidden_size, self.spatial_out_size, 
                                     num_layers, batch_first=True)
        
        # Deconvolución espacial
        self.spatial_decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Codifica secuencia temporal al espacio latente"""
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Procesar espacialmente cada timestep
        spatial_features = []
        for t in range(seq_len):
            feat = self.spatial_encoder(x[:, t:t+1, :, :])
            spatial_features.append(feat.view(batch_size, -1))
        
        spatial_features = torch.stack(spatial_features, dim=1)
        
        # Procesar temporalmente con LSTM
        _, (hidden, _) = self.lstm_encoder(spatial_features)
        z = self.fc_latent(hidden[-1])
        
        return z
    
    def decode(self, z, seq_len):
        """Decodifica del espacio latente a secuencia temporal"""
        batch_size = z.shape[0]
        
        # Expandir latente para secuencia
        h = self.fc_decoder(z)
        h = h.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Decodificar temporalmente
        spatial_features, _ = self.lstm_decoder(h)
        
        # Decodificar espacialmente cada timestep
        output_seq = []
        for t in range(seq_len):
            feat = spatial_features[:, t, :].view(batch_size, 64, 
                                                   self.input_shape[1]//4, 
                                                   self.input_shape[2]//4)
            frame = self.spatial_decoder(feat)
            output_seq.append(frame)
        
        return torch.stack(output_seq, dim=1).squeeze(2)
    
    def forward(self, x):
        """Forward pass completo"""
        z = self.encode(x)
        reconstruction = self.decode(z, x.shape[1])
        return reconstruction, z
