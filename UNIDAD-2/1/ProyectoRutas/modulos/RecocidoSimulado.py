import math
import random
import numpy as np

def calcular_funcion_objetivo(tour, matriz_costos):
    matriz = np.asarray(matriz_costos, dtype=float)
    recorrido = [int(x) for x in tour]

    costo_total = 0
    n = len(recorrido)
    for i in range(n):
        origen = recorrido[i]
        destino = recorrido[(i + 1) % n]
        peso = matriz[origen, destino]
        if not np.isfinite(peso):
            return np.inf
        costo_total += peso
    return costo_total


def generar_vecino(tour, fijar_inicio=True):
    n = len(tour)
    if fijar_inicio and n > 2:
        i, j = random.sample(range(1, n), 2)
    else:
        i, j = random.sample(range(n), 2)
    nuevo_tour = tour[:]
    nuevo_tour[i], nuevo_tour[j] = nuevo_tour[j], nuevo_tour[i]
    return nuevo_tour, (min(i, j), max(i, j))

def recocido_simulado(tour_inicial, matriz_costos, T0, alpha, L, Tmin, fijar_inicio):
    matriz = np.asarray(matriz_costos, dtype=float)

    solucion_actual = [int(x) for x in tour_inicial]
    costo_actual = calcular_funcion_objetivo(solucion_actual, matriz)
    mejor_solucion, mejor_costo = solucion_actual[:], costo_actual
    historial_costos = [mejor_costo]

    temperatura = T0
    while temperatura > Tmin and np.isfinite(mejor_costo):
        for _ in range(L):
            vecino, _ = generar_vecino(solucion_actual, fijar_inicio=fijar_inicio)
            costo_vecino = calcular_funcion_objetivo(vecino, matriz)
            delta = costo_vecino - costo_actual

            if delta <= 0 or random.random() < math.exp(-delta / temperatura):
                solucion_actual, costo_actual = vecino, costo_vecino
                if costo_actual < mejor_costo:
                    mejor_solucion, mejor_costo = solucion_actual[:], costo_actual

            historial_costos.append(mejor_costo)
        temperatura *= alpha

    return mejor_solucion, mejor_costo, historial_costos
