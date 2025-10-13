from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from CargarDatos import cargar_entradas, cargar_rutas
from ConstruirGrafo import (construir_grafo, nodos_mayor_componente, matriz_clausura_metrica, mapa_nombre_a_indice)
from RecocidoSimulado import recocido_simulado

#Variables de configuración
MODO_OPTIMIZAR = "costo"   # "costo" | "distancia"
T0 = 1000.0
ALPHA = 0.995
L_ITERS = 50
T_MIN  = 0.1

CARPETA_ENTRADA = "./Topicos-de-IA/UNIDAD 2/1/ProyectoRutas/datos"
ARCHIVO_NODOS = "datos_distribucion_tiendas.xlsx"
ARCHIVO_DIST = "matriz_distancias.xlsx"
ARCHIVO_COSTO = "matriz_costos_combustible.xlsx"
ARCHIVO_RUTAS = "rutas3.csv"
CARPETA_SALIDA = Path("./Topicos-de-IA/UNIDAD 2/1/ProyectoRutas/resultados")

def main():
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

    #Cargar nodos y matrices
    nodos_df, matriz_distancias, matriz_costos = cargar_entradas(CARPETA_ENTRADA, ARCHIVO_NODOS, ARCHIVO_DIST, ARCHIVO_COSTO)

    #Cargar rutas (archivo exacto)
    rutas_df = cargar_rutas(CARPETA_ENTRADA, ARCHIVO_RUTAS)

    #Elegir matriz base según modo
    matriz_base = matriz_costos if MODO_OPTIMIZAR == "costo" else matriz_distancias

    #Construir grafo
    grafo = construir_grafo(nodos_df, rutas_df, matriz_base)

    #Mayor componente conectada
    indices_componente = nodos_mayor_componente(grafo)
    if len(indices_componente) < 3:
        raise RuntimeError("La mayor componente tiene muy pocos nodos para armar un tour.")

    #Clausura métrica (costos entre nodos de la componente usando caminos reales)
    W = matriz_clausura_metrica(grafo, indices_componente)
    W = np.asarray(W, dtype=float)

    #Elegir nodo inicial (si hay centros en la componente, usar uno al azar)
    nombre_a_indice = mapa_nombre_a_indice(nodos_df)
    nombres_centros = [f"Centro de Distribución {i}" for i in range(1, 11)]
    indices_centros_en_comp = [nombre_a_indice.get(n) for n in nombres_centros if nombre_a_indice.get(n) in indices_componente]
    if indices_centros_en_comp:
        indice_inicio_global = int(np.random.choice(indices_centros_en_comp))
    else:
        indice_inicio_global = int(np.random.choice(indices_componente))

    # Índice del nodo inicial dentro de la componente
    pos_local = {g: i for i, g in enumerate(indices_componente)}
    inicio_local = pos_local[indice_inicio_global]
    print(f"Nodo inicial: {nodos_df.loc[indice_inicio_global, 'Nombre_clean']}")

    # Crear tour inicial rotado para que inicie en el nodo elegido
    n_local = len(indices_componente)
    tour_inicial = list(range(n_local))
    # rotar para que inicio quede en 0
    tour_inicial = tour_inicial[inicio_local:] + tour_inicial[:inicio_local]

    #Se llama al metodo de Recocido Simulado
    mejor_tour, mejor_costo, historial = recocido_simulado(tour_inicial, W, T0=T0, alpha=ALPHA, L=L_ITERS, Tmin=T_MIN, fijar_inicio=True)

    # Convertir los nombres a índices globales
    nombres_en_comp = [nodos_df.loc[i, "Nombre_clean"] for i in indices_componente]
    mejor_tour_nombres = [nombres_en_comp[i] for i in mejor_tour]

    #Exportamos los resultados a archivos CSV y PNG y mostramos en consola el camino recorrido
    serie_costos = pd.Series(historial, name="costo").replace([np.inf, -np.inf], np.nan).ffill()

    ax = serie_costos.plot(figsize=(8, 4))
    ax.set_title("Evolución del mejor costo (Recocido Simulado)")
    ax.set_xlabel("Iteraciones")
    ax.set_ylabel("Costo")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(CARPETA_SALIDA / "evolucion_costo.png", dpi=100)
    plt.show()

    print("\nRuta óptima::")
    for i, nombre in enumerate(mejor_tour_nombres[:100], start=1):
        print(f"{i:02d}. {nombre}")
    print(f"\nMejor costo total: {mejor_costo:,.3f}")

if __name__ == "__main__":
    main()
