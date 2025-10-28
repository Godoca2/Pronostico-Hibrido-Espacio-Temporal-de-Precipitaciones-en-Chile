# Pronóstico Híbrido Espacio-Temporal de Precipitaciones en Chile

Sistema avanzado de pronóstico de precipitaciones que integra **Deep Learning**, **Geoestadística** y **Teledetección** para predicciones espaciotemporales precisas.

## 🌟 Características Principales

Este sistema implementa técnicas de vanguardia para el pronóstico de precipitaciones:

### 1. **Deep Learning con Autoencoders**
- Extracción de patrones latentes espacio-temporales
- Arquitecturas convolucionales para datos espaciales
- Autoencoders recurrentes (LSTM) para dependencias temporales

### 2. **Descomposición Modal Dinámica (DMD)**
- Análisis de modos dinámicos coherentes
- Identificación de frecuencias y tasas de crecimiento/decaimiento
- Variantes: DMD estándar, HODMD (orden superior), mrDMD (multi-resolución)

### 3. **Operador de Koopman con KoVAE**
- Representación lineal de dinámicas no lineales en espacio latente
- Variational Autoencoder con operador de Koopman
- Mejora capacidad predictiva y probabilística para largo plazo

### 4. **Geoestadística**
- **Ordinary Kriging**: Interpolación espacial con media constante
- **Universal Kriging**: Modelado de tendencias espaciales
- **Co-Kriging**: Integración multivariada con variables auxiliares
- **Space-Time Kriging**: Correlación espacio-temporal
- Modelos de variograma: esférico, exponencial, gaussiano, lineal, potencial

### 5. **Teledetección**
- Procesamiento de datos CHIRPS (precipitación satelital)
- Integración de datos MODIS (vegetación, temperatura)
- Datos Sentinel (radar SAR, multispectral)
- Índices: NDVI, EVI, NDWI, NDMI
- Índices climáticos: SPI, SPEI, ONI, PDO

## 🚀 Instalación

### Requisitos Previos
- Python 3.8 o superior
- pip

### Instalación Estándar

```bash
# Clonar repositorio
git clone https://github.com/Godoca2/Pronostico-Hibrido-Espacio-Temporal-de-Precipitaciones-en-Chile.git
cd Pronostico-Hibrido-Espacio-Temporal-de-Precipitaciones-en-Chile

# Instalar dependencias
pip install -r requirements.txt

# Instalar paquete
pip install -e .
```

### Instalación para Desarrollo

```bash
pip install -e ".[dev]"
```

## 📖 Uso Básico

### Ejemplo Rápido

```python
from src.hybrid_forecaster import HybridPrecipitationForecaster
import numpy as np

# Configurar sistema
config = {
    'latent_dim': 64,
    'forecast_horizon': 7,
    'use_kovae': True,
    'use_dmd': True,
    'use_kriging': True
}

forecaster = HybridPrecipitationForecaster(config)

# Entrenar con datos espaciotemporales
# spatial_data: (n_timesteps, n_lat, n_lon)
# coordinates: (n_points, 2) - (lon, lat)
# timestamps: array de fechas
forecaster.fit(spatial_data, coordinates, timestamps)

# Generar pronóstico
predictions = forecaster.predict(forecast_horizon=7, method='ensemble')

# Analizar patrones
analysis = forecaster.analyze_patterns()
print("Modos DMD:", analysis['dmd_modes'].shape)
print("Eigenvalores Koopman:", analysis['koopman_eigenvalues'])
```

### Ejemplo Completo

Ver `examples/ejemplo_basico.py` para un ejemplo completo con:
- Generación de datos sintéticos
- Entrenamiento del sistema
- Análisis de patrones
- Visualización de resultados

```bash
cd examples
python ejemplo_basico.py
```

## 🏗️ Estructura del Proyecto

```
├── src/
│   ├── models/
│   │   ├── autoencoder.py      # Autoencoders espacio-temporales
│   │   ├── dmd.py              # Dynamic Mode Decomposition
│   │   └── kovae.py            # Koopman Variational Autoencoder
│   ├── geostatistics/
│   │   ├── kriging.py          # Kriging y variantes
│   │   └── variogram.py        # Modelos de variograma
│   ├── remote_sensing/
│   │   ├── satellite_data.py   # Procesadores de datos satelitales
│   │   └── indices.py          # Índices de vegetación y climáticos
│   ├── utils/
│   │   ├── data_utils.py       # Utilidades de datos
│   │   └── visualization.py    # Visualización
│   └── hybrid_forecaster.py    # Sistema integrado
├── examples/
│   └── ejemplo_basico.py       # Ejemplo de uso
├── tests/
├── data/
├── requirements.txt
├── setup.py
└── README.md
```

## 🔬 Metodología

### Flujo de Trabajo

1. **Preprocesamiento**
   - Normalización de datos
   - Manejo de valores faltantes
   - Extracción de features temporales

2. **Extracción de Patrones**
   - Autoencoders: Reducción de dimensionalidad
   - DMD: Identificación de modos dinámicos
   - KoVAE: Aprendizaje de dinámicas lineales

3. **Interpolación Espacial**
   - Kriging: Generación de campos continuos
   - Co-Kriging: Integración de variables auxiliares

4. **Predicción**
   - Ensemble de métodos (DMD, KoVAE)
   - Refinamiento espacial con kriging
   - Cuantificación de incertidumbre

## 📊 Fuentes de Datos

### Datos de Precipitación
- **CHIRPS**: Climate Hazards Group InfraRed Precipitation with Station data
  - Resolución: 0.05° (~5.5 km)
  - Cobertura: 1981-presente
  
### Datos Auxiliares
- **MODIS**: Vegetación (NDVI, EVI), Temperatura de Superficie
- **Sentinel-1**: Radar SAR (humedad del suelo)
- **Sentinel-2**: Multiespectral (vegetación, humedad)
- **Estaciones meteorológicas**: Validación y calibración

## 🧪 Validación

El sistema incluye múltiples métricas de evaluación:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)
- Skill Score espacial
- Validación cruzada temporal

## 📝 Publicaciones y Referencias

### Fundamentos Teóricos

**Dynamic Mode Decomposition:**
- Schmid, P. J. (2010). Dynamic mode decomposition of numerical and experimental data. *Journal of Fluid Mechanics*, 656, 5-28.
- Kutz, J. N., et al. (2016). *Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems*. SIAM.

**Koopman Operator:**
- Lusch, B., et al. (2018). Deep learning for universal linear embeddings of nonlinear dynamics. *Nature Communications*, 9(1), 4950.
- Champion, K., et al. (2019). Data-driven discovery of coordinates and governing equations. *PNAS*, 116(45), 22445-22451.

**Geoestadística:**
- Cressie, N. (1993). *Statistics for Spatial Data*. Wiley.
- Goovaerts, P. (1997). *Geostatistics for Natural Resources Evaluation*. Oxford University Press.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Cree una rama para su feature (`git checkout -b feature/AmazingFeature`)
3. Commit sus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abra un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 👥 Autores

- **Godoca2** - *Desarrollo inicial*

## 🙏 Agradecimientos

- Climate Hazards Group (CHIRPS)
- NASA Earth Observing System (MODIS)
- European Space Agency (Sentinel)
- Comunidad científica de geoestadística y machine learning

## 📧 Contacto

Para preguntas, sugerencias o colaboraciones, por favor abra un issue en GitHub.

---

**Nota**: Este es un proyecto de investigación en desarrollo. Los resultados deben ser validados antes de uso operacional.
