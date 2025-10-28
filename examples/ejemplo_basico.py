"""
Ejemplo de uso del sistema de pronóstico híbrido de precipitación

Este script demuestra cómo usar el sistema completo para pronóstico
de precipitaciones en Chile.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Importar componentes del sistema
import sys
sys.path.append('..')

from src.hybrid_forecaster import HybridPrecipitationForecaster
from src.utils.visualization import Visualizer


def generate_synthetic_data():
    """
    Genera datos sintéticos para demostración
    
    En producción, estos datos vendrían de fuentes reales (CHIRPS, estaciones, etc.)
    """
    # Configuración espacial (Chile central aproximado)
    n_lat, n_lon = 50, 40
    n_time = 365  # 1 año de datos
    
    lon_range = (-75, -70)  # Longitud
    lat_range = (-35, -30)  # Latitud
    
    lon = np.linspace(lon_range[0], lon_range[1], n_lon)
    lat = np.linspace(lat_range[0], lat_range[1], n_lat)
    
    # Generar coordenadas
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    coordinates = np.column_stack([lon_mesh.flatten(), lat_mesh.flatten()])
    
    # Generar timestamps
    start_date = datetime(2023, 1, 1)
    timestamps = np.array([start_date + timedelta(days=i) for i in range(n_time)])
    
    # Generar datos sintéticos de precipitación con patrones espaciotemporales
    spatial_data = np.zeros((n_time, n_lat, n_lon))
    
    for t in range(n_time):
        # Patrón estacional
        seasonal = 50 * np.sin(2 * np.pi * t / 365) + 50
        
        # Patrón espacial (gradiente de precipitación)
        spatial_pattern = 100 * np.exp(-((lon_mesh + 72.5)**2 + (lat_mesh + 32.5)**2) / 10)
        
        # Variabilidad aleatoria
        noise = np.random.randn(n_lat, n_lon) * 10
        
        spatial_data[t] = seasonal * spatial_pattern / 100 + noise
        spatial_data[t] = np.maximum(spatial_data[t], 0)  # No precipitación negativa
    
    return spatial_data, coordinates, timestamps, lon, lat


def main():
    """Función principal del ejemplo"""
    
    print("=" * 70)
    print("Sistema Híbrido de Pronóstico de Precipitación para Chile")
    print("=" * 70)
    print()
    
    # 1. Generar datos sintéticos
    print("1. Generando datos sintéticos...")
    spatial_data, coordinates, timestamps, lon, lat = generate_synthetic_data()
    print(f"   Datos: {spatial_data.shape[0]} timesteps, {spatial_data.shape[1]}x{spatial_data.shape[2]} grilla")
    print()
    
    # 2. Configurar sistema
    print("2. Configurando sistema híbrido...")
    config = {
        'latent_dim': 32,
        'sequence_length': 30,
        'forecast_horizon': 7,
        'scaling_method': 'standard',
        'kriging_model': 'spherical',
        'dmd_rank': 10,
        'use_kovae': True,
        'use_dmd': True,
        'use_kriging': True
    }
    
    forecaster = HybridPrecipitationForecaster(config)
    print("   ✓ Sistema configurado")
    print()
    
    # 3. Entrenar sistema
    print("3. Entrenando sistema...")
    print("   (En datos reales, esto tomaría más tiempo)")
    
    # Usar subset de datos para entrenamiento rápido
    train_data = spatial_data[:300]
    train_timestamps = timestamps[:300]
    
    forecaster.fit(
        spatial_data=train_data,
        coordinates=coordinates,
        timestamps=train_timestamps
    )
    print("   ✓ Sistema entrenado")
    print()
    
    # 4. Analizar patrones
    print("4. Analizando patrones en los datos...")
    analysis = forecaster.analyze_patterns()
    
    if 'dmd_eigenvalues' in analysis:
        print(f"   - Modos DMD: {len(analysis['dmd_eigenvalues'])}")
        print(f"   - Frecuencias dominantes: {analysis['dmd_frequencies'][:3]}")
    
    if 'koopman_eigenvalues' in analysis:
        print(f"   - Eigenvalores Koopman (magnitud): {np.abs(analysis['koopman_eigenvalues'][:5])}")
    
    print()
    
    # 5. Generar pronóstico
    print("5. Generando pronóstico para próximos 7 días...")
    
    try:
        forecast_horizon = 7
        predictions = forecaster.predict(
            forecast_horizon=forecast_horizon,
            method='ensemble'
        )
        
        print(f"   ✓ Pronóstico generado: {predictions.shape}")
        print()
        
    except Exception as e:
        print(f"   ⚠ Error en predicción: {e}")
        print("   Usando predicción simplificada...")
        
        # Fallback: usar media de últimos días
        predictions = np.mean(train_data[-7:], axis=0, keepdims=True)
        predictions = np.repeat(predictions, 7, axis=0)
    
    # 6. Visualizar resultados
    print("6. Visualizando resultados...")
    viz = Visualizer()
    
    # Datos históricos recientes
    recent_data = train_data[-30:]
    recent_dates = train_timestamps[-30:]
    
    # Serie temporal en un punto
    point_lon, point_lat = -72.5, -32.5
    from src.utils.data_utils import SpatialDataHandler
    
    ts_historical = SpatialDataHandler.extract_point_timeseries(
        recent_data, point_lon, point_lat, lon, lat
    )
    
    # Gráfico de serie temporal
    fig1 = viz.plot_timeseries(
        recent_dates,
        ts_historical,
        title=f'Precipitación Histórica en ({point_lon}°, {point_lat}°)'
    )
    plt.savefig('ejemplo_serie_temporal.png', dpi=150, bbox_inches='tight')
    print("   ✓ Serie temporal guardada: ejemplo_serie_temporal.png")
    
    # Campo espacial (promedio del pronóstico)
    forecast_mean = np.mean(predictions, axis=0)
    
    fig2 = viz.plot_spatial_field(
        forecast_mean,
        lon, lat,
        title='Pronóstico Promedio (7 días)'
    )
    plt.savefig('ejemplo_campo_espacial.png', dpi=150, bbox_inches='tight')
    print("   ✓ Campo espacial guardado: ejemplo_campo_espacial.png")
    
    # Eigenvalores de Koopman si están disponibles
    if 'koopman_eigenvalues' in analysis:
        fig3 = viz.plot_koopman_eigenvalues(
            analysis['koopman_eigenvalues'],
            title='Eigenvalores del Operador de Koopman'
        )
        plt.savefig('ejemplo_koopman_eigenvalues.png', dpi=150, bbox_inches='tight')
        print("   ✓ Eigenvalores Koopman guardados: ejemplo_koopman_eigenvalues.png")
    
    print()
    
    # 7. Guardar modelo
    print("7. Guardando modelo entrenado...")
    forecaster.save_model('modelo_hibrido.pkl')
    print("   ✓ Modelo guardado: modelo_hibrido.pkl")
    print()
    
    # Resumen final
    print("=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"✓ Sistema entrenado con {train_data.shape[0]} días de datos")
    print(f"✓ Pronóstico generado para {forecast_horizon} días")
    print(f"✓ Precipitación promedio pronosticada: {forecast_mean.mean():.2f} mm")
    print(f"✓ Rango de pronóstico: {forecast_mean.min():.2f} - {forecast_mean.max():.2f} mm")
    print()
    print("El sistema integra:")
    print("  • Autoencoders para extracción de patrones latentes")
    print("  • DMD para análisis de modos dinámicos")
    print("  • KoVAE para representación lineal de dinámicas no lineales")
    print("  • Kriging para interpolación espacial coherente")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
