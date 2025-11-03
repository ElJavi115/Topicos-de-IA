
import numpy as np
from geodesia import haversine_km

# Devuelve costo a minimizar (-J).
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
