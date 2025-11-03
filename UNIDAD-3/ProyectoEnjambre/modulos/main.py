
# PSO simple (propio) que registra el movimiento del mejor (gbest) en cada iteración
# y genera una gráfica de las trayectorias de los sensores (lat, lon).
# Requiere matplotlib instalado.

import numpy as np
import matplotlib.pyplot as plt
import os
from datos import cargar_csv
from utilidad import construir_utilidad
from fitness import fitness

def pso_simple(objetivo, lb, ub,
               particulas=60, iteraciones=200,
               w=0.8, c1=1.6, c2=1.6,
               semilla=None,
               vel_escala_ini=0.2,   
               jitter=0.01,         
               respawn_cada=40,      
               frac_respawn=0.15):   
    import numpy as np
    rng = np.random.default_rng(semilla) if semilla is not None else np.random.default_rng()
    dim = len(lb)
    lb = np.asarray(lb, float); ub = np.asarray(ub, float)
    rango = (ub - lb)

    # Población y velocidades iniciales (más amplias)
    X = rng.uniform(lb, ub, size=(particulas, dim))
    V = rng.uniform(-vel_escala_ini*rango, vel_escala_ini*rango, size=(particulas, dim))

    f = np.array([objetivo(x) for x in X])
    pbest = X.copy(); pbest_val = f.copy()
    g_idx = int(np.argmin(f)); gbest = X[g_idx].copy(); gbest_val = float(f[g_idx])

    historial_gbest = [gbest.copy()]

    for it in range(1, iteraciones+1):
        r1 = rng.random((particulas, dim))
        r2 = rng.random((particulas, dim))
        ruido = rng.normal(0.0, jitter, size=(particulas, dim)) * rango

        V = w*V + c1*r1*(pbest - X) + c2*r2*(gbest - X) + ruido
        X = X + V
        X = np.clip(X, lb, ub)

        # Diversificación periódica: re-spawn de algunas partículas
        if respawn_cada and (it % respawn_cada == 0):
            m = max(1, int(particulas * frac_respawn))
            idx = rng.choice(particulas, size=m, replace=False)
            X[idx] = rng.uniform(lb, ub, size=(m, dim))
            V[idx] = rng.uniform(-vel_escala_ini*rango, vel_escala_ini*rango, size=(m, dim))

        f = np.array([objetivo(x) for x in X])
        mejora = f < pbest_val
        pbest[mejora] = X[mejora]
        pbest_val[mejora] = f[mejora]

        g_idx = int(np.argmin(f))
        if float(f[g_idx]) < gbest_val:
            gbest = X[g_idx].copy()
            gbest_val = float(f[g_idx])

        historial_gbest.append(gbest.copy())

    return gbest, gbest_val, np.array(historial_gbest)


def main():
    RUTA_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datos", "datos_guasave.csv"))
    K = 10
    r_km = 1.2
    alfa = 0.03
    beta = 8.0
    dmin_km = 0.6
    margen_caja_deg = 0.001

    # === Cargar datos y utilidad ===
    humedad, cultivo, elevacion, salinidad, temperatura, latitudes, longitudes = cargar_csv(RUTA_CSV)
    u = construir_utilidad(humedad, cultivo, elevacion, salinidad, temperatura)

    # === Límites (lb/ub) ===
    lat_min = min(latitudes) - margen_caja_deg
    lat_max = max(latitudes) + margen_caja_deg
    lon_min = min(longitudes) - margen_caja_deg
    lon_max = max(longitudes) + margen_caja_deg

    lb = np.array([lat_min, lon_min] * K, dtype=float)
    ub = np.array([lat_max, lon_max] * K, dtype=float)

    # === Objetivo ===
    def objetivo(x):
        return fitness(x, latitudes, longitudes, u, K, r_km, alfa, beta, dmin_km)

    # === Correr PSO simple con historial ===
    gbest, gbest_val, historial = pso_simple(
    objetivo, lb, ub,
    particulas=80,
    iteraciones=250,
    w=0.8, c1=1.6, c2=1.6,
    semilla=None,            # <-- aleatorio real
    vel_escala_ini=0.25,     # más movimiento inicial
    jitter=0.015,            # pequeñas sacudidas
    respawn_cada=50,         # re-spawn periódico
    frac_respawn=0.2         # 20% de partículas
    )


    # === Graficar movimiento de los K sensores del gbest en cada iteración ===
    # 'historial' tiene forma (iteraciones+1, 2K). Para cada sensor k tomamos (lat, lon) a lo largo del tiempo.
    iters = historial.shape[0]
    latitudes_hist = []
    longitudes_hist = []
    for k in range(K):
        lat_k = historial[:, 2*k]
        lon_k = historial[:, 2*k + 1]
        latitudes_hist.append(lat_k)
        longitudes_hist.append(lon_k)

    # Gráfico 1: trayectoria de cada sensor (líneas) y puntos del dataset (dispersión)
    plt.figure()
    # puntos de referencia del dataset
    plt.scatter(longitudes, latitudes, s=5, alpha=0.5, label="Puntos del dataset")
    # trayectorias
    for k in range(K):
        plt.plot(longitudes_hist[k], latitudes_hist[k], linewidth=1)
        plt.scatter(longitudes_hist[k][-1], latitudes_hist[k][-1], marker='x')  # posición final
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.title("Movimiento del mejor conjunto (gbest) por iteración")
    plt.legend()
    plt.show()

    # Imprimir resultados finales
    lat_s = gbest[0::2]; lon_s = gbest[1::2]
    print("Sensores óptimos (latitud, longitud):")
    for i, (la, lo) in enumerate(zip(lat_s, lon_s), 1):
        print(f"{i:02d} -> {la:.6f}, {lo:.6f}")
    print("fitness_minimizado:", gbest_val)

if __name__ == "__main__":
    main()
