import numpy as np
import networkx as nx

#Crea un diccionario para mapear nombres a índices de nodos.
def mapa_nombre_a_indice(nodos_df):
    return {nombre: i for i, nombre in enumerate(nodos_df["Nombre_clean"])}

#Crea un grafo no dirigido con los pesos de los costos o distancias según se desee.
#El CSV tiene las columnas "origen_nombre" y "destino_nombre" con los nombres de los nodos.
#La matriz_pesos es la matriz de costos o distancias (numpy array).

def construir_grafo(nodos_df, rutas_df, matriz_pesos):
    nombre_a_indice = mapa_nombre_a_indice(nodos_df)

    indices_origen = rutas_df["origen_nombre"].map(lambda s: nombre_a_indice.get(s))
    indices_destino = rutas_df["destino_nombre"].map(lambda s: nombre_a_indice.get(s))

    if indices_origen.isna().any() or indices_destino.isna().any():
        faltantes = rutas_df[indices_origen.isna() | indices_destino.isna()][["origen_nombre", "destino_nombre"]].drop_duplicates()
        raise RuntimeError(f"Hay nombres en rutas que no existen en el Excel:\n{faltantes}")

    indices_origen = indices_origen.astype(int).to_numpy()
    indices_destino = indices_destino.astype(int).to_numpy()

    grafo = nx.Graph()
    for u, v in zip(indices_origen, indices_destino):
        if u == v:
            continue
        peso = matriz_pesos[u, v]
        if np.isfinite(peso):
            grafo.add_edge(u, v, weight=peso)

    return grafo

# Retorna los índices de los nodos en la mayor componente conectada del grafo.
def nodos_mayor_componente(grafo):
    componentes = list(nx.connected_components(grafo))
    if not componentes:
        return []
    mayor = max(componentes, key=len)
    return sorted(list(mayor))

# Matriz W tal que W[i, j] = costo del camino más corto utilizando el algoritmo de Dijkstra
# entre los nodos indices_nodos[i] e indices_nodos[j] dentro del grafo.

def matriz_clausura_metrica(grafo, indices_nodos):
    n = len(indices_nodos)
    W = np.full((n, n), np.inf)
    for i, u in enumerate(indices_nodos):
        distancias = nx.single_source_dijkstra_path_length(grafo, u, weight="weight")
        for j, v in enumerate(indices_nodos):
            if u == v:
                continue
            W[i, j] = distancias.get(v, np.inf)
    np.fill_diagonal(W, np.inf)
    return W
