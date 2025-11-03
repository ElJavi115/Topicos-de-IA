
import numpy as np

def zscore(x):
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x) + 1e-9
    return (x - mu)/sd

def rangos_humedad_objetivo():
    return {
        'maíz': (30.0, 35.0),
        'maiz': (30.0, 35.0),
        'chile': (25.0, 30.0),
        'tomate': (20.0, 25.0)
    }

# Construye u_i en [0,1] con desviación de humedad + sal, elev, temp
def construir_utilidad(humedad, cultivo, elevacion, salinidad, temperatura):
    H = np.array(humedad, dtype=float)
    S = np.array(salinidad, dtype=float)
    E = np.array(elevacion, dtype=float)
    T = np.array(temperatura, dtype=float)
    cult = [str(c).strip().lower() for c in cultivo]

    rangos = rangos_humedad_objetivo()
    H_prom = float(np.nanmean(H))
    objetivo = []
    for c in cult:
        if c in rangos:
            objetivo.append( (rangos[c][0] + rangos[c][1]) / 2.0 )
        else:
            objetivo.append(H_prom)
    objetivo = np.array(objetivo, dtype=float)

    dev = np.abs(H - objetivo)
    u = 1.0*dev + 0.6*zscore(S) + 0.3*zscore(E) + 0.2*zscore(T)
    u = (u - u.min())/(u.max() - u.min() + 1e-12)
    return u
