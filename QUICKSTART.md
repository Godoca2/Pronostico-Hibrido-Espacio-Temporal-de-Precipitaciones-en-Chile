# Guía de Inicio Rápido

Esta guía te ayudará a comenzar a usar el Sistema Híbrido de Pronóstico de Precipitación para Chile en pocos minutos.

## Instalación Rápida

```bash
# 1. Clonar el repositorio
git clone https://github.com/Godoca2/Pronostico-Hibrido-Espacio-Temporal-de-Precipitaciones-en-Chile.git
cd Pronostico-Hibrido-Espacio-Temporal-de-Precipitaciones-en-Chile

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Instalar el paquete
pip install -e .
```

## Uso Básico - 5 Minutos

### 1. Ejemplo Mínimo

```python
import numpy as np
from src.hybrid_forecaster import HybridPrecipitationForecaster

# Datos sintéticos (reemplazar con datos reales)
n_time, n_lat, n_lon = 100, 50, 40
spatial_data = np.random.randn(n_time, n_lat, n_lon) * 10 + 50

# Coordenadas (Chile central aproximado)
lon = np.linspace(-75, -70, n_lon)
lat = np.linspace(-35, -30, n_lat)
lon_mesh, lat_mesh = np.meshgrid(lon, lat)
coordinates = np.column_stack([lon_mesh.flatten(), lat_mesh.flatten()])

# Timestamps
from datetime import datetime, timedelta
start = datetime(2023, 1, 1)
timestamps = np.array([start + timedelta(days=i) for i in range(n_time)])

# Configurar y entrenar
forecaster = HybridPrecipitationForecaster()
forecaster.fit(spatial_data, coordinates, timestamps)

# Predecir próximos 7 días
predictions = forecaster.predict(forecast_horizon=7, method='ensemble')

print(f"Predicciones: {predictions.shape}")
print(f"Media pronosticada: {predictions.mean():.2f} mm")
```

### 2. Ejemplo con Visualización

```python
from src.utils.visualization import Visualizer

viz = Visualizer()

# Serie temporal en un punto
from src.utils.data_utils import SpatialDataHandler

timeseries = SpatialDataHandler.extract_point_timeseries(
    spatial_data, lon=-72.5, lat=-32.5, 
    lon_coords=lon, lat_coords=lat
)

fig = viz.plot_timeseries(
    timestamps, timeseries,
    title='Precipitación en Santiago'
)
fig.savefig('precipitacion_santiago.png')

# Campo espacial
fig = viz.plot_spatial_field(
    predictions.mean(axis=0), lon, lat,
    title='Pronóstico Promedio'
)
fig.savefig('pronostico_espacial.png')
```

### 3. Análisis de Patrones

```python
# Analizar patrones descubiertos
analysis = forecaster.analyze_patterns()

# Modos DMD
if 'dmd_modes' in analysis:
    print(f"Modos DMD encontrados: {analysis['dmd_modes'].shape[1]}")
    print(f"Frecuencias dominantes: {analysis['dmd_frequencies'][:5]}")

# Eigenvalores Koopman
if 'koopman_eigenvalues' in analysis:
    eigenvals = analysis['koopman_eigenvalues']
    print(f"Eigenvalores Koopman (magnitud): {np.abs(eigenvals)[:5]}")
    
    # Visualizar
    fig = viz.plot_koopman_eigenvalues(eigenvals)
    fig.savefig('koopman_eigenvalues.png')
```

### 4. Kriging para Interpolación

```python
from src.geostatistics.kriging import OrdinaryKriging

# Datos dispersos (ej: estaciones meteorológicas)
station_coords = np.random.rand(20, 2) * 5 - 2.5  # Cerca de Santiago
station_values = np.random.randn(20) * 10 + 50

# Ajustar kriging
ok = OrdinaryKriging(variogram_model='spherical')
ok.fit(station_coords, station_values)

# Interpolar en grilla
z, lon_grid, lat_grid = ok.predict_grid(
    lon_range=(-75, -70),
    lat_range=(-35, -30),
    grid_resolution=100
)

# Visualizar
fig = viz.plot_spatial_field(z, lon_grid, lat_grid,
                            title='Interpolación Kriging')
fig.savefig('kriging_interpolation.png')
```

### 5. Calcular Índices Climáticos

```python
from src.remote_sensing.indices import ClimateIndices, PrecipitationMetrics

# Serie temporal de precipitación
precip_series = timeseries

# SPI (Standardized Precipitation Index)
spi = ClimateIndices.spi(precip_series, timescale=30)

# Días secos consecutivos
max_dry = PrecipitationMetrics.consecutive_dry_days(precip_series, threshold=1.0)

print(f"SPI actual: {spi[-1]:.2f}")
print(f"Máximo días secos consecutivos: {max_dry}")
```

## Ejecutar Ejemplo Completo

```bash
cd examples
python ejemplo_basico.py
```

Este script genera:
- `ejemplo_serie_temporal.png`: Serie temporal histórica
- `ejemplo_campo_espacial.png`: Pronóstico espacial promedio
- `ejemplo_koopman_eigenvalues.png`: Eigenvalores del operador Koopman
- `modelo_hibrido.pkl`: Modelo entrenado guardado

## Estructura de Datos

### Formato de Entrada

**Datos espaciales:**
```python
# Forma: (n_timesteps, n_lat, n_lon)
spatial_data = np.array(...)  # Ej: (365, 50, 40) para 1 año

# Coordenadas
coordinates = np.array([...])  # Forma: (n_points, 2) - (lon, lat)

# Timestamps
timestamps = np.array([...])  # Array de datetime
```

### Formato de Salida

**Predicciones:**
```python
predictions = forecaster.predict(forecast_horizon=7)
# Forma: (7, 50, 40) - 7 días de pronóstico en grilla 50x40
```

## Configuración Avanzada

```python
config = {
    # Autoencoder
    'latent_dim': 64,              # Dimensión del espacio latente
    'conv_filters': [32, 64, 128], # Filtros convolucionales
    
    # Secuencias temporales
    'sequence_length': 30,         # Días de historia
    'forecast_horizon': 7,         # Días a predecir
    
    # Preprocesamiento
    'scaling_method': 'standard',  # 'standard', 'minmax', o None
    
    # Kriging
    'kriging_model': 'spherical',  # 'spherical', 'exponential', 'gaussian'
    
    # DMD
    'dmd_rank': 10,               # Rango SVD para DMD
    
    # Activar/desactivar componentes
    'use_kovae': True,
    'use_dmd': True,
    'use_kriging': True,
    'use_autoencoder': True
}

forecaster = HybridPrecipitationForecaster(config)
```

## Consejos de Rendimiento

1. **Datos grandes**: Usa `CompressedDMD` para datasets masivos
2. **GPU**: PyTorch detectará automáticamente GPU si está disponible
3. **Memoria**: Procesa datos en lotes si tienes limitaciones de RAM
4. **Paralelización**: Kriging puede paralelizarse por puntos

## Solución de Problemas

### Error: "No module named 'torch'"
```bash
pip install torch torchvision
```

### Error: "GDAL not found"
```bash
# Ubuntu/Debian
sudo apt-get install gdal-bin libgdal-dev

# macOS
brew install gdal

# Luego
pip install gdal
```

### Datos Faltantes
```python
from src.utils.data_utils import DataPreprocessor

preprocessor = DataPreprocessor()
clean_data = preprocessor.handle_missing_values(
    data, method='interpolate'
)
```

## Siguientes Pasos

1. **Cargar datos reales**: Reemplaza datos sintéticos con CHIRPS, estaciones, etc.
2. **Validación**: Usa validación cruzada temporal
3. **Optimización**: Ajusta hiperparámetros con grid search
4. **Visualización**: Explora módulo de visualización completo
5. **Integración**: Combina múltiples fuentes de datos satelitales

## Recursos Adicionales

- **README.md**: Documentación completa del proyecto
- **IMPLEMENTATION.md**: Detalles técnicos de implementación
- **examples/**: Más ejemplos de uso
- **tests/**: Tests unitarios para referencia

## Soporte

Para preguntas o problemas:
1. Revisa la documentación en README.md
2. Consulta los ejemplos en `examples/`
3. Abre un issue en GitHub

¡Buena suerte con tus pronósticos de precipitación! 🌧️
