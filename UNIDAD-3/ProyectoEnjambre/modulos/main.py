
# PSO que registra el movimiento del mejor (gbest) en cada iteración
# y genera una gráfica de las trayectorias de los sensores (lat, lon).

import numpy as np
import matplotlib.pyplot as plt
import os
from datos import cargar_csv
from utilidad import construir_utilidad
from fitness import fitness

"""
    Optimización por Enjambre de Partículas (PSO) con diversificación opcional.

    Minimiza la función 'objetivo' en un espacio acotado por 'limites_inf' y 'limites_sup'.
    Implementa la actualización clásica de velocidad/posición con componentes
    de inercia, atracción al mejor local (pbest) y atracción al mejor global (gbest).
    Opcionalmente re-inicializa (respawn) una fracción de partículas cada cierto número
    de iteraciones para evitar estancamientos.

    Parámetros:
    - objetivo : función que se desea MINIMIZAR.
    - limites_inf : arreglo con límites inferiores por dimensión.
    - limites_sup : arreglo con límites superiores por dimensión.
    - num_particulas : tamaño del enjambre.
    - num_iteraciones : número de ciclos de búsqueda.
    - w : inercia
    - c1, c2 : coeficientes local/global
    - vel_escala_ini : escala de velocidad inicial (qué tan rápido se mueven al inicio).
    - jitter : pequeño valor aleatorio para evitar que se queden estancadas.
    - respawn_cada : cada cuántas iteraciones re-inicializar algunas partículas
    - fraccion_respawn : fporcentaje de partículas que se reubican aleatoriamente.

    Retorna:
    - mejor_pos_global : vector (dim,) con la mejor posición encontrada (gbest).
    - mejor_valor_global : valor de 'objetivo' asociado a gbest.
    - mejor_trayectoria  : arreglo (num_iteraciones+1, dim) con el gbest por iteración.
    """

import numpy as np

def generar_enjambre(
    objetivo, limites_inf, limites_sup,
    num_particulas=100, num_iteraciones=200,
    w=0.8, c1=2, c2=2,
    vel_escala_inicial=0.2,
    jitter=0.01,
    respawn_cada=40,
    fraccion_respawn=0.15
):
    rng = np.random.default_rng()

    limites_inf = np.asarray(limites_inf, float)
    limites_sup = np.asarray(limites_sup, float)
    dimension = len(limites_inf)
    rango = (limites_sup - limites_inf)

    # Inicialización de posiciones y velocidades
    posiciones = rng.uniform(limites_inf, limites_sup, size=(num_particulas, dimension))
    velocidades = rng.uniform(-vel_escala_inicial*rango, vel_escala_inicial*rango, size=(num_particulas, dimension))

    # Evaluación inicial
    puntajes = np.array([objetivo(x) for x in posiciones])
    mejores_locales_pos = posiciones.copy()
    mejores_locales_val = puntajes.copy()
    idx_mejor = int(np.argmin(puntajes))
    mejor_pos_global = posiciones[idx_mejor].copy()
    mejor_valor_global = float(puntajes[idx_mejor])

    mejor_trayectoria = [mejor_pos_global.copy()]

    for iteracion in range(1, num_iteraciones + 1):
        r1 = rng.random((num_particulas, dimension))
        r2 = rng.random((num_particulas, dimension))
        ruido = rng.normal(0.0, jitter, size=(num_particulas, dimension)) * rango

        velocidades = (
            w * velocidades
            + c1 * r1 * (mejores_locales_pos - posiciones)
            + c2 * r2 * (mejor_pos_global - posiciones)
            + ruido
        )

        posiciones = posiciones + velocidades
        posiciones = np.clip(posiciones, limites_inf, limites_sup)

        # Diversificación periódica (respawn)
        if respawn_cada and (iteracion % respawn_cada == 0):
            n_respawn = max(1, int(num_particulas * fraccion_respawn))
            idx = rng.choice(num_particulas, size=n_respawn, replace=False)
            posiciones[idx] = rng.uniform(limites_inf, limites_sup, size=(n_respawn, dimension))
            velocidades[idx] = rng.uniform(-vel_escala_inicial*rango, vel_escala_inicial*rango, size=(n_respawn, dimension))

        # Evaluación y actualización de mejores
        puntajes = np.array([objetivo(x) for x in posiciones])
        mejora = puntajes < mejores_locales_val
        mejores_locales_pos[mejora] = posiciones[mejora]
        mejores_locales_val[mejora] = puntajes[mejora]

        idx_mejor = int(np.argmin(puntajes))
        if float(puntajes[idx_mejor]) < mejor_valor_global:
            mejor_pos_global = posiciones[idx_mejor].copy()
            mejor_valor_global = float(puntajes[idx_mejor])

        mejor_trayectoria.append(mejor_pos_global.copy())

    return mejor_pos_global, mejor_valor_global, np.array(mejor_trayectoria)


def main():
    #  Ruta del archivo CSV
    RUTA_CSV = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "datos", "datos_guasave.csv"
    ))

    # Parámetros del problema
    num_sensores = 10
    radio_km = 1.2
    peso_repulsion = 0.03
    peso_penalizacion = 8.0
    distancia_min_km = 0.6
    margen_grados = 0.001

    # Cargar datos y calcular utilidad
    humedad, cultivo, elevacion, salinidad, temperatura, latitudes, longitudes = cargar_csv(RUTA_CSV)
    utilidades = construir_utilidad(humedad, cultivo, elevacion, salinidad, temperatura)

    # Límites geográficos 
    lat_min = min(latitudes) - margen_grados
    lat_max = max(latitudes) + margen_grados
    lon_min = min(longitudes) - margen_grados
    lon_max = max(longitudes) + margen_grados

    limites_inf = np.array([lat_min, lon_min] * num_sensores, dtype=float)
    limites_sup = np.array([lat_max, lon_max] * num_sensores, dtype=float)

    # Función objetivo
    def funcion_objetivo(x):
        return fitness(x, latitudes, longitudes, utilidades, num_sensores, radio_km, peso_repulsion, peso_penalizacion, distancia_min_km)

    # Ejecución del PSO
    mejor_pos_global, mejor_valor_global, mejor_trayectoria = generar_enjambre(
        funcion_objetivo, limites_inf, limites_sup,
        num_particulas=100,
        num_iteraciones=200,
        w=0.8,
        c1=2.0,
        c2=2.0,
        vel_escala_inicial=0.25,
        jitter=0.015,
        respawn_cada=50,
        fraccion_respawn=0.2
    )

    # Gráfica de las trayectorias del mejor global 
    lat_hist = []
    lon_hist = []
    for i in range(num_sensores):
        lat_i = mejor_trayectoria[:, 2*i]
        lon_i = mejor_trayectoria[:, 2*i + 1]
        lat_hist.append(lat_i)
        lon_hist.append(lon_i)

    plt.figure()
    plt.scatter(longitudes, latitudes, s=5, alpha=0.5, label="_nolegend_")
    for i in range(num_sensores):
        plt.plot(lon_hist[i], lat_hist[i], linewidth=1)
        plt.scatter(lon_hist[i][-1], lat_hist[i][-1], marker='x')
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.title("PSO: Movimientos de los mejores sensores ")
    plt.legend()
    try:
        plt.show()
    except Exception:
        print("No se pudo mostrar la gráfica.")

    # --- Resultados finales ---
    lat_s = mejor_pos_global[0::2]
    lon_s = mejor_pos_global[1::2]
    print("\nSensores óptimos (latitud, longitud):")
    for i, (lat, lon) in enumerate(zip(lat_s, lon_s), 1):
        print(f"{i:02d} -> {lat:.6f}, {lon:.6f}")
    print("Valor del fitness:", mejor_valor_global)

if __name__ == "__main__":
    main()