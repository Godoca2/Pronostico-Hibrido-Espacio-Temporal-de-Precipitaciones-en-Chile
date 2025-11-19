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

---

## 🔄 Fase 3: Optimización y Análisis Avanzado (En Progreso)

## Fase 3: Modelo Avanzado KoVAE (Semanas 5-6)

### 3.1 Implementación KoVAE
- [ ] Estudiar paper original KoVAE (operador de Koopman)
- [ ] Implementar `kovae.py`:
  - VAE con representación Koopman
  - Loss function específica
  - Sampleo estocástico para incertidumbre
- [ ] Entrenar con mismos datos que AE+DMD
- [ ] Logging en MLflow con experimento separado

### 3.2 Comparación Exhaustiva
- [ ] Experimento A vs B:
  - AE+DMD (determinístico)
  - KoVAE (probabilístico)
- [ ] Métricas adicionales:
  - CRPS (Continuous Ranked Probability Score)
  - Interval coverage (bandas de confianza)
  - Calibración probabilística
- [ ] Análisis de ventajas/desventajas

### 3.3 Optimización Bayesiana (Opcional)
- [ ] Usar Optuna/Hyperopt para búsqueda de hiperparámetros
- [ ] Integrar con MLflow
- [ ] Identificar configuración óptima

**Entregables Fase 3:**
- KoVAE implementado y validado
- Comparación científica rigurosa AE+DMD vs KoVAE
- Visualizaciones de incertidumbre

---

## Fase 4: Integración Geoespacial (Semanas 7-8)

### 4.1 Pronóstico Espacialmente Explícito
- [ ] Extender modelos para output multipoint
- [ ] Generar mapas de pronóstico 1-7 días
- [ ] Validación espacial (por cuenca hidrográfica)

### 4.2 Co-Kriging con Covariables
- [ ] Integrar MODIS como covariable secundaria
- [ ] Co-Kriging precipitation + NDVI/LST
- [ ] Comparar vs Kriging simple

### 4.3 Casos de Estudio Aplicados
- [ ] Validar en 3-5 cuencas prioritarias
- [ ] Análisis de eventos extremos (sequías 2019-2022)
- [ ] Pronóstico estacional (agregado mensual)

**Entregables Fase 4:**
- Pipeline completo espacio-temporal
- Mapas interactivos (Folium/Plotly)
- Reporte de casos de estudio (10-12 páginas)

---

## Fase 5: Producción y Despliegue (Semanas 9-10)

### 5.1 Model Registry y Versionado
- [ ] Registrar modelo final en MLflow Registry
- [ ] Marcar como "Production"
- [ ] Documentar versión y performance

### 5.2 API de Pronóstico (Opcional)
- [ ] Crear `serve_model.py` con FastAPI
- [ ] Endpoint `/predict` para scoring
- [ ] Dockerizar aplicación

### 5.3 Dashboard de Monitoreo
- [ ] Streamlit app para visualización
- [ ] Input: fecha, región
- [ ] Output: pronóstico + incertidumbre + mapa

### 5.4 Documentación Final
- [ ] Paper científico (formato IEEE/Springer)
- [ ] Presentación para defensa Capstone
- [ ] README completo con instrucciones de uso
- [ ] Video demo (5-10 min)

**Entregables Fase 5:**
- Sistema en producción (local o cloud)
- Documentación científica completa
- Presentación final

---

##  Criterios de Éxito del Proyecto

### Mínimo Viable (Aprobación)
1. Pipeline completo datos → modelo → predicción
2. Comparación AE+DMD vs benchmark
3. Validación científica con métricas estándar
4. Documentación técnica clara

### Objetivo (Distinción)
1. Todo lo anterior +
2. KoVAE implementado y comparado
3. Integración geoespacial (kriging/mapas)
4. Experimentos MLflow > 20 runs
5. Casos de estudio aplicados
6. Paper científico draft

### Excelencia (Publicación)
1. Todo lo anterior +
2. Resultados superiores a estado del arte
3. Contribución metodológica original
4. API/Dashboard funcional
5. Paper enviado a conferencia/journal

---

## Cronograma Sugerido (10 semanas)

| Semana | Fase | Hitos Clave |
|--------|------|-------------|
| 1-2 | Fase 1 | EDA completo, datos procesados |
| 3-4 | Fase 2 | AE+DMD funcionando, 10+ experimentos |
| 5-6 | Fase 3 | KoVAE implementado, comparación |
| 7-8 | Fase 4 | Integración geoespacial, casos estudio |
| 9-10 | Fase 5 | Producción, documentación, defensa |

---

## Stack Tecnológico Confirmado

- **Datos**: xarray, netCDF4, pandas, geopandas
- **Geoestadística**: PyKrige, scikit-gstat, cartopy
- **ML/DL**: TensorFlow/Keras, PyDMD, scikit-learn
- **Experimentación**: MLflow, Optuna (opcional)
- **Visualización**: matplotlib, seaborn, plotly, folium
- **Producción**: FastAPI (opcional), Streamlit (opcional)
- **Infraestructura**: Conda, Git, GitHub

---

##  Consejos Prácticos

1. **Commitea frecuentemente**: Cada avance importante al repo
2. **Usa MLflow desde el día 1**: Rastrea TODO (hasta experimentos fallidos)
3. **Valida incremental**: No esperes al final para validar
4. **Documenta mientras avanzas**: README, notebooks con markdown
5. **Pide feedback temprano**: Mostrar avances a tutor/equipo cada 2 semanas
6. **No optimices prematuramente**: Primero que funcione, luego optimiza

---

## Referencias Técnicas Clave

1. **PyDMD**: Paper adjunto en `/doc/`
2. **KoVAE**: Buscar papers recientes sobre Koopman VAE
3. **Geoestadística**: Cressie & Wikle (2011) - Statistical Analysis of Spatio-Temporal Data
4. **MLflow**: Documentación oficial - https://mlflow.org/docs/latest/

---

**Última actualización**: 12 nov 2025  
**Responsable**: César Godoy Delaigue  
**Versión**: 1.0
