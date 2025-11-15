# Roadmap del Proyecto - Pronóstico Híbrido de Precipitaciones

## Estado Actual: Fase de Inicialización ✅

### Completado
- [x] Estructura de proyecto creada
- [x] Entorno Conda configurado
- [x] MLflow integrado
- [x] Repositorio Git conectado a GitHub
- [x] Documentación base (README, MLflow.md)
- [x] Esqueletos de código (data_prep, ae_dmd, train_ae_dmd)
- [x] Notebook EDA iniciado

---

## Fase 1: Preparación y Exploración de Datos (Semanas 1-2)

### 1.1 Completar EDA Espacio-Temporal
- [ ] Ejecutar notebook `01A_Eda_spatiotemporal.ipynb` completo
- [ ] Generar todos los mapas y visualizaciones
- [ ] Identificar patrones estacionales y tendencias por macrozona
- [ ] Exportar series limpias: `era5_daily_national_mean.csv`
- [ ] Documentar hallazgos clave en el notebook

### 1.2 Procesamiento de Datos Multifuente
- [ ] Integrar datos CHIRPS (precipitación satelital)
- [ ] Integrar datos MODIS (índices de vegetación/temperatura)
- [ ] Implementar `data_prep.py` completo:
  - Función de lectura NetCDF robusta
  - Subset por región Chile (-56° a -17° lat)
  - Agregación temporal (horaria → diaria)
  - Validación de calidad de datos (NaNs, outliers)
- [ ] Crear dataset unificado: `data/processed/multivariate_daily.parquet`

### 1.3 Geoestadística (Opcional pero recomendado)
- [ ] Calcular variogramas experimentales por región
- [ ] Implementar Kriging/Co-Kriging con PyKrige
- [ ] Generar mallas interpoladas de alta resolución
- [ ] Validar interpolación con cross-validation

**Entregables Fase 1:**
- Notebook EDA completo con visualizaciones
- Dataset procesado listo para modelado
- Reporte breve de análisis exploratorio (2-3 páginas)

---

## Fase 2: Implementación de Modelos Base (Semanas 3-4)

### 2.1 Autoencoder + DMD (Baseline)
- [ ] Implementar `ae_keras.py` completo:
  - Arquitectura encoder-decoder (LSTM/Conv1D)
  - Entrenamiento con datos reales
  - Validación del espacio latente
- [ ] Integrar PyDMD en `ae_dmd.py`:
  - DMD sobre representaciones latentes
  - Pronóstico multi-step
  - Validación temporal (train/val/test)
- [ ] Experimentos MLflow:
  - Variar `latent_dim`: [50, 100, 150]
  - Variar arquitecturas (LSTM vs Conv1D)
  - Comparar SVD ranks en DMD

### 2.2 Métricas de Evaluación
- [ ] Implementar `utils/metrics.py`:
  - MAE, RMSE, R² (globales y por estación)
  - Nash-Sutcliffe Efficiency (NSE)
  - Skill Score vs modelo persistente
- [ ] Visualizaciones de resultados:
  - Series temporales observadas vs predichas
  - Mapas de error espacial
  - Distribuciones de residuos

### 2.3 Primera Validación Científica
- [ ] Comparar con benchmark simple (media móvil, ARIMA)
- [ ] Validar en eventos extremos (años Niño/Niña)
- [ ] Análisis de sensibilidad a hiperparámetros

**Entregables Fase 2:**
- Modelo AE+DMD funcionando end-to-end
- Dashboard MLflow con >= 10 experimentos
- Informe técnico preliminar (5-7 páginas)

---

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
