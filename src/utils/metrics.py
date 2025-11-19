"""metrics.py
=================
Conjunto de métricas para evaluación de pronósticos hidrológico/climáticos.
Incluye métricas genéricas (MAE, RMSE, R²) y de dominio (NSE, Skill Score,
PBIAS). Todas las funciones aceptan arrays o listas y aplastan la entrada
para evaluación global sobre múltiples estaciones/tiempos.

Notas de uso
------------
* NSE y Skill Score permiten contextualizar el desempeño frente a baseline
    (media y persistencia respectivamente).
* PBIAS reporta sesgo porcentual; se recomienda complementarlo con MAE/RMSE.
* Las métricas añaden `1e-10` en denominadores para estabilidad numérica.

Autor: César Godoy Delaigue (Magíster Data Science UDD - 2025)
"""
import numpy as np
from typing import Union


def mse(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """Mean Squared Error (MSE).

    Minimiza grandes errores cuadráticamente; sensible a outliers.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return float(np.mean((y_true - y_pred)**2))


def rmse(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """Root Mean Squared Error (RMSE).

    Raíz del MSE para mantener unidades originales.
    """
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """Mean Absolute Error (MAE).

    Penaliza todos los errores linealmente; más robusto a outliers que RMSE.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """Coeficiente de determinación R².

    Proporción de varianza explicada por el modelo (1 = perfecto).
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return float(1 - (ss_res / (ss_tot + 1e-10)))


def nse(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """Nash-Sutcliffe Efficiency (NSE).

    NSE = 1 (perfecto), 0 (igual a usar la media), <0 (peor que la media).
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    numerator = np.sum((y_true - y_pred)**2)
    denominator = np.sum((y_true - np.mean(y_true))**2)
    return float(1 - (numerator / (denominator + 1e-10)))


def pbias(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """Percent Bias (PBIAS).

    PBIAS > 0 indica subestimación; < 0 sobreestimación.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return float(100 * np.sum(y_true - y_pred) / (np.sum(y_true) + 1e-10))


def skill_score_persistence(y_true: Union[np.ndarray, list], 
                           y_pred: Union[np.ndarray, list]) -> float:
    """Skill Score contra baseline de persistencia.

    SS = 1 - MSE_modelo / MSE_persistencia.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Modelo persistente: y_pred_naive[t] = y_true[t-1]
    y_persist = np.roll(y_true, 1)
    y_persist[0] = y_true[0]  # primer valor no tiene histórico
    
    mse_model = mse(y_true, y_pred)
    mse_persist = mse(y_true, y_persist)
    
    return float(1 - (mse_model / (mse_persist + 1e-10)))


def evaluate_all(y_true: Union[np.ndarray, list], 
                y_pred: Union[np.ndarray, list]) -> dict:
    """Calcula métricas agregadas y retorna dict.

    Returns
    -------
    dict
        Keys: MAE, RMSE, R2, NSE, PBIAS, SkillScore.
    """
    return {
        'MAE': mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'NSE': nse(y_true, y_pred),
        'PBIAS': pbias(y_true, y_pred),
        'SkillScore': skill_score_persistence(y_true, y_pred)
    }
