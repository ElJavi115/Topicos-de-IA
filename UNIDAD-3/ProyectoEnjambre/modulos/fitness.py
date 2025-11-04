
import numpy as np
from geodesia import haversine_km
"""
Calcula el valor de fitness que el algoritmo PSO debe minimizar.

Evalúa qué tan buena es una configuración de sensores tomando en cuenta las zonas con mayor utilidad, 
la dispersión de los sensores y la distancia mínima entre ellos.

El valor que devuelve es negativo porque el PSO de PySwarm busca minimizarlo.

Parámetros:
- x : arreglo con las coordenadas 
- latitudes, longitudes : listas con las coordenadas del terreno.
- utilidades : lista de utilidades de cada punto (0 a 1).
- K : número de sensores.
- r_km : radio de cobertura en km.
- alfa : peso de la repulsión.
- beta : peso de la penalización por cercanía.
- dmin_km : distancia mínima deseada entre sensores (km).

Retorna:
- valor numérico negativo del ajuste total (-J). Porque queremos minimizar.
"""

def fitness(x, latitudes, longitudes, utilidades, K, r_km, alfa, beta, dmin_km):
    lat_s = x[0::2]
    lon_s = x[1::2]

    lat_pts = np.asarray(latitudes, dtype=float)
    lon_pts = np.asarray(longitudes, dtype=float)
    Kint = int(K)

    lat_mat = np.repeat(lat_pts[:, None], Kint, axis=1)
    lon_mat = np.repeat(lon_pts[:, None], Kint, axis=1)
    dist = haversine_km(lat_mat, lon_mat, lat_s[None, :], lon_s[None, :])

    cobertura_total = 1.0 - np.prod(np.exp(- (dist / r_km)**2 ), axis=1)
    J_cobertura = float(np.sum(utilidades * cobertura_total))

    repulsion = 0.0
    penalizacion = 0.0
    if Kint > 1:
        for k in range(Kint):
            for l in range(k+1, Kint):
                dk = float(haversine_km(lat_s[k], lon_s[k], lat_s[l], lon_s[l]))
                repulsion += 1.0 / (dk**2 + 1e-12)
                if dk < dmin_km:
                    penalizacion += (dmin_km - dk)**2
    J_penal = alfa*repulsion + beta*penalizacion
    J = J_cobertura - J_penal
    return -J
