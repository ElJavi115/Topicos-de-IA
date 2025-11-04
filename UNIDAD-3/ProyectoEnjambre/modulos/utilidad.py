
import numpy as np

"""
Calcula la puntuación tipificada (Z-score) de un conjunto de valores numéricos.

El Z-score indica cuántas desviaciones estándar se encuentra cada valor respecto a la media del conjunto. 
Se usa para normalizar variables con diferentes escalas (como elevación, salinidad o temperatura) dentro del cálculo de utilidad.

Parámetros:
- x : arreglo o lista numérica.

Retorna:
- Arreglo de Z-scores con la misma forma que x.
"""


def zscore(x):
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x) + 1e-9
    return (x - mu)/sd


"""
Devuelve los rangos de humedad óptima para distintos tipos de cultivo.

Estos valores sirven como referencia para calcular la desviación de humedad de cada punto del terreno respecto 
a la humedad ideal de su cultivo asociado.

Retorna:
- Diccionario con pares {nombre_cultivo: (humedad_mínima, humedad_máxima)}.
"""

def rangos_humedad_objetivo():
    return {
        'maíz': (30.0, 35.0),
        'maiz': (30.0, 35.0),
        'chile': (25.0, 30.0),
        'tomate': (20.0, 25.0)
    }


"""
Calcula un índice de utilidad para cada punto del terreno, basado en la humedad, salinidad, elevación y temperatura del lugar.

Este valor representa qué tan favorables son las condiciones ambientales para la colocación de sensores de riego o monitoreo.

Parámetros:
- humedad : lista o arreglo con valores de humedad del suelo (%).
- cultivo : lista con el nombre del cultivo presente en cada punto.
- elevacion : lista con alturas sobre el nivel del mar (m).
- salinidad : lista con niveles de salinidad del suelo.
- temperatura : lista con temperatura ambiental (°C).

Retorna:
- Arreglo NumPy con los valores de utilidad normalizados (0 = condiciones menos favorables, 1 = más favorables).
"""


def construir_utilidad(humedad, cultivo, elevacion, salinidad, temperatura):
    hum = np.array(humedad, dtype=float)
    sal = np.array(salinidad, dtype=float)
    ele = np.array(elevacion, dtype=float)
    temp = np.array(temperatura, dtype=float)
    cultivos = [str(c).strip().lower() for c in cultivo]

    rangos = rangos_humedad_objetivo()
    hum_media_global = float(np.nanmean(hum))

    hum_ideal = []
    for c in cultivos:
        if c in rangos:
            hum_ideal.append((rangos[c][0] + rangos[c][1]) / 2.0)
        else:
            hum_ideal.append(hum_media_global)
    hum_ideal = np.array(hum_ideal, dtype=float)

    desvio_hum = np.abs(hum - hum_ideal)

    utilidad = (
        1.0 * desvio_hum +
        0.6 * zscore(sal) +
        0.3 * zscore(ele) +
        0.2 * zscore(temp)
    )

    utilidad = (utilidad - utilidad.min()) / (utilidad.max() - utilidad.min() + 1e-12)
    return utilidad
