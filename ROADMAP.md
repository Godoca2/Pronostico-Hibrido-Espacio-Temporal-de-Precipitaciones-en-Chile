# Roadmap del Proyecto - Pronóstico Híbrido de Precipitaciones

## Estado Actual: Fase 2 Completada ✅ → Iniciando Fase 3

### Completado

- [x] Estructura de proyecto creada
- [x] Entorno Conda configurado (Python 3.10.13, TensorFlow 2.10.0 GPU)
- [x] MLflow integrado (tracking deshabilitado temporalmente por conflictos protobuf)
- [x] Repositorio Git conectado a GitHub
- [x] Documentación base (README, MLflow.md) actualizada
- [x] Pipeline ERA5 completo (download, merge, processing)
- [x] GPU habilitada (NVIDIA RTX A4000, CUDA 11.2, cuDNN 8.1)

---

## ✅ Fase 1: Preparación y Exploración de Datos (Completada)

### 1.1 EDA Espacio-Temporal ✅

- [x] Ejecutar notebook `01_EDA.ipynb` completo
- [x] Análisis espacio-temporal en `01A_Eda_spatiotemporal.ipynb`
- [x] Generar mapas y visualizaciones por macrozona (Norte/Centro/Sur)
- [x] Identificar patrones estacionales: Jun-Ago (invierno) pico, Dic-Feb mínimo
- [x] Estadísticas por región: Norte (0.27 mm/día), Centro (3.49 mm/día), Sur (3.70 mm/día)
- [x] Exportar series: `era5_precipitation_chile_full.nc`
- [x] Documentar hallazgos clave (10 visualizaciones guardadas)

### 1.2 Procesamiento de Datos ERA5 ✅

- [x] Pipeline automatizado ERA5:
  - `download_era5.py`: Descarga desde CDS Copernicus
  - `merge_era5.py`: Combinación de archivos mensuales
  - `merge_era5_advanced.py`: Validación y limpieza avanzada
- [x] Dataset ERA5 2020: 366 días, resolución 0.25° (157×41 grid)
- [x] Región Chile: -56° a -17.5° lat, -76° a -66° lon
- [x] Conversión horaria → diaria (agregación mm/día)
- [x] Validación completa: sin NaNs, outliers detectados y documentados

### 1.3 Geoestadística ✅

- [x] Notebook `02_DL_DMD_Forecast.ipynb` completo
- [x] Variogramas experimentales (Jun 2020): Spherical model
  - Range: 8.23° (~913 km)
  - Sill: 23.45 (varianza total)
  - Nugget: 0.0 (datos limpios, sin ruido sub-grid)
- [x] Kriging ordinario con validación cruzada
- [x] Varianza kriging para pesos espaciales en loss function
- [x] Mallas interpoladas visualizadas

**Entregables Fase 1:** ✅

- ✅ Notebooks EDA completos con 15+ visualizaciones
- ✅ Dataset ERA5 procesado (366 días × 157×41 grid)
- ✅ Pipeline descarga automática documentado
- ✅ Análisis geoestadístico con variogramas

---

## ✅ Fase 2: Implementación AE+DMD (Completada)

### 2.1 Autoencoder + DMD ✅

- [x] Notebook `03_AE_DMD_Training.ipynb` completo
- [x] Arquitectura encoder-decoder Dilated CNN:
  - Receptive field ~40 celdas (cumple range 8.23°)
  - Dilations [1,2,4,8] para capturar correlación espacial
  - Latent dim: 64 (compresión 100x)
  - Regularización L2=0.0001 (nugget≈0)
- [x] Loss function ponderada por varianza kriging
- [x] Entrenamiento con GPU (~69 segundos, 100 épocas)
  - Train loss: 0.013
  - Val loss: 0.035
  - Early stopping en época óptima
- [x] DMD sobre espacio latente:
  - 42 modos dinámicos (SVD rank 0.99)
  - 100% modos estables (|λ| < 1)
  - Frecuencias dominantes: 2-2.5 días/ciclo

### 2.2 Forecasting Multi-Step ✅

- [x] Predicciones 1, 3, 7 días adelante
- [x] Métricas en escala real (mm/día):
  - **1 día**: MAE 1.691, RMSE 4.073
  - **3 días**: MAE 1.751, RMSE 4.213
  - **7 días**: MAE 1.777, RMSE 4.234
- [x] Desnormalización correcta usando scaler
- [x] Validación temporal (train 70%, val 15%, test 15%)

### 2.3 Baselines y Comparación ✅

- [x] Baseline Persistence (último día observado)
- [x] Baseline Climatología (media por día del año)
- [x] **Resultados comparativos (horizonte 1 día)**:
  - AE+DMD: MAE 1.691 mm/día
  - Persistence: MAE 1.898 mm/día (+10.9% mejora ✅)
  - Climatología: MAE 2.024 mm/día (+16.5% mejora ✅)
- [x] AE+DMD supera ambos baselines en todos los horizontes

### 2.4 Análisis Espacial ✅

- [x] Evaluación por macrozona (horizonte 1 día):
  - **Norte**: MAE 3.283 mm/día, RMSE 6.023
  - **Centro**: MAE 1.253 mm/día, RMSE 3.152
  - **Sur**: MAE 0.679 mm/día, RMSE 2.268
- [x] Mapas espaciales: predicción, ground truth, error
- [x] Mayor error en Norte (mayor precipitación media)

### 2.5 Visualizaciones y Documentación ✅

- [x] 15+ figuras generadas y guardadas
- [x] Curvas de aprendizaje
- [x] Ejemplos de reconstrucción
- [x] Eigenvalues DMD y frecuencias
- [x] Mapas de error espacial
- [x] Tabla comparativa de métodos

**Entregables Fase 2:** ✅

- ✅ Modelo AE+DMD funcionando end-to-end
- ✅ Forecasting multi-step validado
- ✅ Superioridad vs baselines demostrada
- ✅ Análisis espacial completo
- ✅ Notebook completo con resultados reproducibles
- ✅ Resultados guardados en pickle (`forecast_results_2020.pkl`)

---

## 🔄 Fase 3: Optimización y Análisis Avanzado (En Progreso)

### 3.0 Métricas Avanzadas ✅

- [x] Implementar `src/utils/metrics.py` completo:
  - NSE (Nash-Sutcliffe Efficiency)
  - Skill Score vs persistence
  - Skill Score vs climatología
  - Métricas por tipo de evento (seco/normal/extremo)
  - Análisis de residuos (percentiles, skewness, kurtosis)
- [x] Notebook `04_Advanced_Metrics.ipynb` creado y ejecutado
- [x] Análisis comparativo con datos reales:
  - **Rankings por horizonte**: AE+DMD 🥇 en todos (1d, 3d, 7d)
  - Persistence 🥈, Climatology 🥉
  - Mejoras relativas: +10.9% vs Persistence, +16.5% vs Climatología (1 día)
- [x] Visualizaciones comparativas exportadas
- [x] Tabla resumen guardada: `metrics_summary.csv`
- [x] Sistema de carga/guardado de resultados implementado

### 3.1 Experimentos con Hiperparámetros 🔄

- [ ] Variar `latent_dim`: [32, 64, 128, 256]
- [ ] Variar SVD rank DMD: [0.9, 0.95, 0.99, 1.0]
- [ ] Experimentos con arquitecturas:
  - LSTM encoder vs CNN encoder
  - Diferentes dilations [1,2,4,8] vs [1,3,9,27]
  - Skip connections (U-Net style)
- [ ] Registrar >= 20 experimentos MLflow
- [ ] Análisis de sensibilidad con pandas/seaborn
- [ ] Identificar configuración óptima

### 3.2 Validación Temporal Extendida

- [ ] Validar en múltiples años (2019-2023)
- [ ] Análisis estacional (DJF, MAM, JJA, SON)
- [ ] Eventos extremos: Niño/Niña, sequías, sistemas frontales
- [ ] Skill scores por estación del año

### 3.3 Interpretabilidad DMD 🔄

- [ ] Análisis de modos dominantes (top 5-10 modos)
- [ ] Visualizar modos en espacio físico (decodificar con decoder)
- [ ] Correlación modos DMD con patrones meteorológicos conocidos
- [ ] Frecuencias dominantes vs ciclos sinópticos (2-7 días)
- [ ] Estabilidad de modos (análisis de |λ|)

**Entregables Fase 3:**

- ✅ Métricas avanzadas implementadas y validadas
- 🔄 >= 20 experimentos MLflow documentados (en progreso)
- ⏳ Notebook de análisis de hiperparámetros
- ⏳ Análisis de interpretabilidad DMD

---

## Fase 4: Integración Geoespacial y Casos de Estudio (Futuro)

### 4.1 Pronóstico Espacialmente Explícito

- [ ] Extender para pronóstico multi-point simultáneo
- [ ] Generar mapas de pronóstico 1-7 días
- [ ] Validación espacial por cuenca hidrográfica
- [ ] Análisis de propagación espacial de errores

### 4.2 Datos Multifuente (Opcional)

- [ ] Integrar CHIRPS (precipitación satelital)
- [ ] Integrar MODIS (NDVI, LST)
- [ ] Co-Kriging precipitation + covariables
- [ ] Fusión de múltiples fuentes

### 4.3 Casos de Estudio Aplicados

- [ ] Validación en cuencas prioritarias:
  - Cuenca Río Maipo (Centro)
  - Cuenca Río Biobío (Sur)
  - Cuenca Río Loa (Norte)
- [ ] Análisis eventos extremos históricos:
  - Sequía megasequía 2010-2022
  - Sistemas frontales invierno 2023
  - Bloques de altas presiones
- [ ] Pronóstico agregado mensual/estacional

**Entregables Fase 4:**

- Pipeline espacio-temporal completo
- Mapas interactivos (Folium/Plotly)
- Reporte de casos de estudio (12-15 páginas)
- Validación en cuencas reales

---

## Fase 5: Documentación y Difusión Científica (Futuro)

### 5.1 Model Registry y Producción

- [ ] Resolver conflictos MLflow (protobuf/pyarrow)
- [ ] Registrar modelo final en MLflow Registry
- [ ] Marcar mejor configuración como "Production"
- [ ] Documentar versión y performance

### 5.2 Paper Científico

- [ ] Redactar paper formato IEEE/Springer:
  - Abstract
  - Introduction (estado del arte)
  - Methodology (AE+DMD con geoestadística)
  - Results (comparación baselines, análisis espacial)
  - Discussion (interpretación, limitaciones)
  - Conclusions
- [ ] Figuras de calidad publicación
- [ ] Referencias bibliográficas (Zotero)

### 5.3 Presentación Defensa Capstone

- [ ] Slides presentación (20-30 min)
- [ ] Demo en vivo del modelo
- [ ] Video explicativo (5-10 min)
- [ ] Poster científico (opcional)

### 5.4 Código y Reproducibilidad

- [ ] README completo con instrucciones
- [ ] Notebooks ejecutables con datos ejemplo
- [ ] Requirements.txt/environment.yml actualizados
- [ ] Licencia MIT/Apache
- [ ] Documentación API (Sphinx/mkdocs)

**Entregables Fase 5:**

- Paper científico draft completo
- Presentación defensa preparada
- Repositorio GitHub público
- Documentación técnica completa

---

## 📊 Resumen de Progreso Global

| Fase | Estado | Completitud | Hitos Clave |
|------|--------|-------------|-------------|
| Fase 1: EDA y Datos | ✅ Completada | 100% | Pipeline ERA5, geoestadística, visualizaciones |
| Fase 2: AE+DMD Base | ✅ Completada | 100% | Modelo entrenado, forecasting, baselines |
| Fase 3: Optimización | 🔄 En Progreso | 25% | Métricas avanzadas ✅, experimentos iniciados |
| Fase 4: Geoespacial | ⏳ Pendiente | 0% | Casos de estudio, cuencas |
| Fase 5: Documentación | ⏳ Pendiente | 0% | Paper, presentación |

## Progreso Total

**45% completado (2/5 fases completas + Fase 3 al 25%)**

---

## 🎯 Próximos Pasos Inmediatos

### Esta Semana (Semana 3)

1. ✅ Actualizar ROADMAP con Fase 2 completa
2. ✅ Implementar `src/utils/metrics.py` con NSE y Skill Score
3. ✅ Notebook 04_Advanced_Metrics.ipynb completo
4. 🔄 Experimentos con diferentes `latent_dim` (32, 128, 256) - SIGUIENTE
5. [ ] Análisis de sensibilidad SVD rank DMD
6. [ ] Visualizar modos DMD en espacio físico

### Próxima Semana (Semana 4)

1. [ ] Validar en años 2019-2021 (datos adicionales)
2. [ ] Análisis estacional (verano vs invierno)
3. [ ] Identificar eventos extremos para validación
4. [ ] Dashboard Streamlit básico (opcional)
5. [ ] Comenzar draft introducción paper

---

## ✅ Criterios de Éxito del Proyecto

### ✅ Mínimo Viable - ALCANZADO

1. ✅ Pipeline completo datos → modelo → predicción
2. ✅ Comparación AE+DMD vs baselines (10-17% mejora)
3. ✅ Validación científica con métricas estándar (MAE, RMSE)
4. ✅ Documentación técnica clara (notebooks + README)

### 🎯 Objetivo Distinción - EN PROGRESO

1. ✅ Todo lo anterior
2. ⏳ Experimentos MLflow > 20 runs (actualmente: 2)
3. ⏳ Integración geoestadística avanzada (kriging completado parcialmente)
4. ⏳ Casos de estudio aplicados (pendiente)
5. ⏳ Paper científico draft (pendiente)

### 🏆 Excelencia - ASPIRACIONAL

1. Todo lo anterior
2. Resultados superiores a estado del arte
3. Contribución metodológica original (DMD + kriging weights)
4. API/Dashboard funcional
5. Paper enviado a conferencia/journal

---

## 📅 Cronograma Actualizado (10 semanas totales)

| Semana | Fase | Hitos Clave | Estado |
|--------|------|-------------|--------|
| 1-2 | Fase 1 | EDA completo, datos procesados | ✅ Completado |
| 3-4 | Fase 2 | AE+DMD funcionando, forecasting, baselines | ✅ Completado |
| 5-6 | Fase 3 | Experimentos, métricas avanzadas | 🔄 Actual |
| 7-8 | Fase 4 | Geoespacial, casos estudio | ⏳ Planificado |
| 9-10 | Fase 5 | Documentación, defensa | ⏳ Planificado |

**Semana Actual: 3** (iniciando Fase 3)

---

## Stack Tecnológico Confirmado

- **Datos**: xarray, netCDF4, pandas, geopandas
- **Geoestadística**: PyKrige, scikit-gstat, cartopy
- **ML/DL**: TensorFlow 2.10.0 (GPU), PyDMD, scikit-learn
- **GPU**: NVIDIA RTX A4000, CUDA 11.2, cuDNN 8.1
- **Experimentación**: MLflow (pendiente resolver conflictos)
- **Visualización**: matplotlib, seaborn, plotly, folium
- **Producción**: FastAPI (opcional), Streamlit (opcional)
- **Infraestructura**: Conda, Git, GitHub

---

## Consejos Prácticos

1. **Commitea frecuentemente**: Cada avance importante al repo ✅
2. **Usa MLflow desde el día 1**: Rastrea TODO (pendiente resolver)
3. **Valida incremental**: No esperes al final para validar ✅
4. **Documenta mientras avanzas**: README, notebooks con markdown ✅
5. **Pide feedback temprano**: Mostrar avances a tutor/equipo cada 2 semanas
6. **No optimices prematuramente**: Primero que funcione, luego optimiza ✅

---

## Referencias Técnicas Clave

1. **PyDMD**: Paper adjunto en `/doc/`
2. **Geoestadística**: Cressie & Wikle (2011) - Statistical Analysis of Spatio-Temporal Data
3. **ERA5**: Hersbach et al. (2020) - The ERA5 global reanalysis
4. **MLflow**: Documentación oficial - https://mlflow.org/docs/latest/
5. **TensorFlow**: https://www.tensorflow.org/api_docs/python/tf

---

**Última actualización**: 19 nov 2025  
**Responsable**: César Godoy Delaigue  
**Versión**: 3.0
