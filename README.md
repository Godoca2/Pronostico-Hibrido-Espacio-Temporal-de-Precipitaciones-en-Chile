# Pronóstico Híbrido Espacio-Temporal de Precipitaciones en Chile: Integrando Aprendizaje Profundo, Geoestadística y Teledetección

Chile presenta una fuerte variabilidad espacio-temporal de precipitaciones, lo que impacta la gestión hídrica, la agricultura y la planificación territorial. Los modelos numéricos tradicionales tienen dificultades para representar las correlaciones espaciales y las dependencias no lineales que caracterizan el clima chileno.

Este proyecto propone un modelo híbrido de pronóstico espacio-temporal de precipitaciones, integrando tres pilares metodológicos:

1. **Aprendizaje profundo** mediante Autoencoders y **Descomposición Modal Dinámica (DMD)** para extraer patrones latentes y predecir su evolución temporal.

3. **El operador de Koopman**, incorporado mediante el enfoque **KoVAE**, que permite representar dinámicas no lineales de forma lineal en el espacio latente, mejorando la capacidad predictiva y probabilística.

4. **Geoestadística y teledetección**, empleando técnicas de kriging y co-kriging junto con datos satelitales (CHIRPS, GPM y MODIS) para generar mallas continuas y coherentes espacialmente.

# Pregunta de investigación:

¿Puede la integración de aprendizaje profundo, geoestadística y teledetección mejorar la precisión y coherencia espacial del pronóstico de precipitaciones en Chile respecto al AE + DMD tradicional?

# Hipótesis:

La combinación del operador de Koopman con Autoencoders, junto a la interpolación geoestadística de alta resolución y datos satélite, permitirá modelar mejor las correlaciones espacio-temporales y reducir el error de predicción a nivel local y regional.

**Impacto potencial:**

Los resultados apoyarán la planificación hídrica y la gestión del riesgo climático, entregando mapas predictivos de precipitación para Chile. Este proyecto pretende validará la aplicación práctica del modelo en cuencas hidrográficas prioritarias en zonas de sequias.

-----------

# 2. Revisión de literatura / Estado del arte

La predicción de variables climáticas ha evolucionado desde métodos estadísticos lineales (ARIMA, SARIMA, VAR, PROPHET) hacia modelos de Deep Learning y enfoques híbridos, capaces de capturar relaciones no lineales y multiescalares.

**Trabajos previos UDD – Herrera (2023-2024):**

Marchant & Silva (2024) demostraron la eficacia del enfoque Autoencoder + DMD para pronosticar precipitaciones locales, obteniendo mejoras de precisión superiores al 80 % respecto al modelo DeepAR, con costos computacionales bajos.

Pérez & Zavala (2023) aplicaron EOFs + Deep Learning a datos ERA5, destacando la utilidad de la reducción de dimensionalidad mediante SVD para representar patrones climáticos dominantes.

**Literatura internacional:**

Amato et al. (2020) propusieron un marco de predicción espaciotemporal basado en Deep Learning aplicado a variables ambientales.

Lusch et al. (2018) y Kutz et al. (2016) desarrollaron la DMD como técnica data-driven para sistemas dinámicos complejos.

Lam et al. (2023) y Wong (2023) evidenciaron el potencial del AI aplicado a la predicción meteorológica global (GraphCast, DeepMind Weather).

Cressie & Wikle (2011) fundamentaron la geoestadística espaciotemporal como marco probabilístico para modelar dependencias espaciales.

