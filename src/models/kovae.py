"""
Koopman Variational Autoencoder (KoVAE) para representación lineal de dinámicas no lineales

Este módulo implementa KoVAE, que combina autoencoders variacionales con la teoría
del operador de Koopman para representar dinámicas no lineales de forma lineal en
el espacio latente, mejorando la capacidad predictiva y probabilística.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class KoopmanVAE(nn.Module):
    """
    Koopman Variational Autoencoder para dinámicas no lineales
    
    KoVAE aprende un espacio latente donde las dinámicas no lineales se vuelven
    lineales, permitiendo predicciones de largo plazo más precisas.
    
    El operador de Koopman permite representar sistemas dinámicos no lineales
    como operadores lineales en un espacio de funciones de dimensión infinita.
    
    Args:
        input_dim (int): Dimensión de entrada
        latent_dim (int): Dimensión del espacio latente
        hidden_dims (list): Dimensiones de capas ocultas
        beta (float): Peso del término KL en la pérdida
        koopman_steps (int): Pasos para regularización Koopman
    """
    
    def __init__(self, input_dim, latent_dim=32, hidden_dims=[256, 128, 64],
                 beta=1.0, koopman_steps=1):
        super(KoopmanVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.beta = beta
        self.koopman_steps = koopman_steps
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim)
            ])
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Parámetros distribución latente
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Operador de Koopman (matriz de transición lineal en espacio latente)
        self.koopman_operator = nn.Linear(latent_dim, latent_dim, bias=False)
        
        # Inicializar operador Koopman cerca de identidad para estabilidad
        nn.init.eye_(self.koopman_operator.weight)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim)
            ])
            prev_dim = h_dim
        
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """
        Codifica entrada a parámetros de distribución latente
        
        Args:
            x (torch.Tensor): Entrada (batch, input_dim)
        
        Returns:
            tuple: (mu, logvar) parámetros de distribución Gaussiana
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Truco de reparametrización para muestreo
        
        Args:
            mu (torch.Tensor): Media
            logvar (torch.Tensor): Log-varianza
        
        Returns:
            torch.Tensor: Muestra latente
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """
        Decodifica representación latente
        
        Args:
            z (torch.Tensor): Vector latente
        
        Returns:
            torch.Tensor: Reconstrucción
        """
        return self.decoder(z)
    
    def koopman_forward(self, z, steps=1):
        """
        Aplica operador de Koopman para evolucionar el estado latente
        
        Args:
            z (torch.Tensor): Estado latente inicial
            steps (int): Número de pasos a evolucionar
        
        Returns:
            torch.Tensor: Estado latente evolucionado
        """
        z_evolved = z
        for _ in range(steps):
            z_evolved = self.koopman_operator(z_evolved)
        return z_evolved
    
    def forward(self, x, predict_steps=0):
        """
        Forward pass del KoVAE
        
        Args:
            x (torch.Tensor): Entrada
            predict_steps (int): Pasos futuros a predecir
        
        Returns:
            dict: Diccionario con reconstrucción, predicción y parámetros
        """
        # Codificar
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        # Reconstruir
        recon = self.decode(z)
        
        # Predicción futura con operador Koopman
        predictions = []
        if predict_steps > 0:
            z_current = z
            for step in range(1, predict_steps + 1):
                z_current = self.koopman_forward(z_current, steps=1)
                pred = self.decode(z_current)
                predictions.append(pred)
        
        return {
            'reconstruction': recon,
            'predictions': predictions,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
    
    def loss_function(self, x, x_next, output, output_next):
        """
        Función de pérdida para KoVAE
        
        Combina:
        - Pérdida de reconstrucción
        - Divergencia KL
        - Consistencia Koopman (predicción de siguiente paso)
        
        Args:
            x (torch.Tensor): Entrada actual
            x_next (torch.Tensor): Entrada siguiente paso
            output (dict): Salida del forward pass para x
            output_next (dict): Salida del forward pass para x_next
        
        Returns:
            dict: Diccionario con pérdidas
        """
        # Reconstrucción
        recon_loss = F.mse_loss(output['reconstruction'], x, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(
            1 + output['logvar'] - output['mu'].pow(2) - output['logvar'].exp()
        )
        
        # Consistencia Koopman
        # El operador debe predecir correctamente el siguiente estado latente
        z_next_pred = self.koopman_forward(output['z'], steps=1)
        z_next_true = output_next['mu']  # Usar media como target
        koopman_loss = F.mse_loss(z_next_pred, z_next_true, reduction='mean')
        
        # Pérdida total
        total_loss = recon_loss + self.beta * kl_loss + koopman_loss
        
        return {
            'loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss,
            'koopman_loss': koopman_loss
        }
    
    def predict_future(self, x, n_steps):
        """
        Predice evolución futura usando operador Koopman
        
        Args:
            x (torch.Tensor): Estado inicial
            n_steps (int): Número de pasos a predecir
        
        Returns:
            np.ndarray: Predicciones futuras
        """
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            
            # Codificar estado inicial
            mu, _ = self.encode(x)
            
            # Evolucionar con Koopman
            predictions = []
            z_current = mu
            for _ in range(n_steps):
                z_current = self.koopman_forward(z_current, steps=1)
                pred = self.decode(z_current)
                predictions.append(pred.cpu().numpy())
        
        return np.array(predictions)
    
    def get_koopman_eigenvalues(self):
        """
        Obtiene eigenvalores del operador de Koopman
        
        Los eigenvalores revelan las frecuencias y modos de la dinámica.
        
        Returns:
            np.ndarray: Eigenvalores del operador Koopman
        """
        K = self.koopman_operator.weight.detach().cpu().numpy()
        eigenvalues = np.linalg.eigvals(K)
        return eigenvalues
    
    def get_koopman_modes(self):
        """
        Obtiene eigenvectores (modos) del operador de Koopman
        
        Returns:
            np.ndarray: Modos de Koopman
        """
        K = self.koopman_operator.weight.detach().cpu().numpy()
        eigenvalues, eigenvectors = np.linalg.eig(K)
        return eigenvectors


class SpatialKoopmanVAE(nn.Module):
    """
    KoVAE adaptado para datos espaciales (imágenes)
    
    Usa convoluciones para procesar datos espaciales y aplica
    el operador de Koopman en el espacio latente.
    
    Args:
        input_shape (tuple): Forma de entrada (channels, height, width)
        latent_dim (int): Dimensión del espacio latente
        conv_dims (list): Filtros para capas convolucionales
        beta (float): Peso del término KL
    """
    
    def __init__(self, input_shape, latent_dim=64, 
                 conv_dims=[32, 64, 128], beta=1.0):
        super(SpatialKoopmanVAE, self).__init__()
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.conv_dims = conv_dims
        self.beta = beta
        
        # Encoder convolucional
        encoder_layers = []
        in_channels = input_shape[0]
        
        for out_channels in conv_dims:
            encoder_layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ])
            in_channels = out_channels
        
        self.encoder_conv = nn.Sequential(*encoder_layers)
        
        # Calcular tamaño después de convoluciones
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            conv_out = self.encoder_conv(dummy)
            self.conv_out_shape = conv_out.shape[1:]
            self.flatten_dim = np.prod(self.conv_out_shape)
        
        # Parámetros latentes
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # Operador Koopman
        self.koopman_operator = nn.Linear(latent_dim, latent_dim, bias=False)
        nn.init.eye_(self.koopman_operator.weight)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        
        decoder_layers = []
        in_channels = conv_dims[-1]
        
        for i in range(len(conv_dims) - 1, 0, -1):
            decoder_layers.extend([
                nn.ConvTranspose2d(in_channels, conv_dims[i-1], 3, 
                                   stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(conv_dims[i-1]),
                nn.ReLU()
            ])
            in_channels = conv_dims[i-1]
        
        decoder_layers.append(
            nn.ConvTranspose2d(conv_dims[0], input_shape[0], 3, 
                               stride=2, padding=1, output_padding=1)
        )
        decoder_layers.append(nn.Sigmoid())
        
        self.decoder_conv = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Codifica entrada espacial"""
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparametrización"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decodifica a espacio original"""
        h = self.fc_decode(z)
        h = h.view(-1, *self.conv_out_shape)
        return self.decoder_conv(h)
    
    def koopman_forward(self, z, steps=1):
        """Aplica operador Koopman"""
        z_evolved = z
        for _ in range(steps):
            z_evolved = self.koopman_operator(z_evolved)
        return z_evolved
    
    def forward(self, x, predict_steps=0):
        """Forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        
        predictions = []
        if predict_steps > 0:
            z_current = z
            for _ in range(predict_steps):
                z_current = self.koopman_forward(z_current, 1)
                pred = self.decode(z_current)
                predictions.append(pred)
        
        return {
            'reconstruction': recon,
            'predictions': predictions,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
    
    def loss_function(self, x, x_next, output, output_next):
        """Función de pérdida espacial"""
        recon_loss = F.mse_loss(output['reconstruction'], x)
        
        kl_loss = -0.5 * torch.mean(
            1 + output['logvar'] - output['mu'].pow(2) - output['logvar'].exp()
        )
        
        z_next_pred = self.koopman_forward(output['z'], 1)
        koopman_loss = F.mse_loss(z_next_pred, output_next['mu'])
        
        total_loss = recon_loss + self.beta * kl_loss + koopman_loss
        
        return {
            'loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss,
            'koopman_loss': koopman_loss
        }
