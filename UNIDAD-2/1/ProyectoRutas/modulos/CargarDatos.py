import pandas as pd
from pathlib import Path

#Lee los datos de entrada, tanto nodos como matrices de distancias y costos
#Así mismo limpia los datos para que estén en el formato correcto (Nombre_clean, matrices n x n, etc.)
#También lee las rutas desde un archivo CSV.
# Las funciones devuelven dataframes y matrices numpy listas para usar.

def cargar_entradas(carpeta_base, archivo_nodos, archivo_distancias, archivo_costos):
    carpeta_base = Path(carpeta_base)
    ruta_nodos = (carpeta_base / archivo_nodos).resolve()
    ruta_dist  = (carpeta_base / archivo_distancias).resolve()
    ruta_cost  = (carpeta_base / archivo_costos).resolve()
    nodos_df = pd.read_excel(ruta_nodos, engine="openpyxl")
    if 'Nombre_clean' not in nodos_df.columns:
        fuente_nombre = None
        if 'Nombre' in nodos_df.columns:
            fuente_nombre = 'Nombre'
        else:
            fuente_nombre = nodos_df.columns[0]
        nodos_df['Nombre_clean'] = nodos_df[fuente_nombre].astype(str).str.strip()
    n = len(nodos_df)

    #Con este bloque se carga la matriz de distancias, limpiandola y recortandola para que los encabezados no afecten
    dist_raw = pd.read_excel(ruta_dist, header=None, engine="openpyxl")
    dist_raw = dist_raw.iloc[1:, :]  
    if dist_raw.shape[1] == n + 1:  
        dist_raw = dist_raw.iloc[:, 1:]
    if dist_raw.shape[0] < n or dist_raw.shape[1] < n:
        raise RuntimeError(f"Distancias tiene forma {dist_raw.shape} y no alcanza para n={n}.")
    dist_raw = dist_raw.iloc[:n, :n]
    matriz_distancias = dist_raw.apply(pd.to_numeric, errors="coerce").to_numpy(float)

    #Con este bloque se carga la matriz de costos, limpiandola igual que la de distancias
    cost_raw = pd.read_excel(ruta_cost, header=None, engine="openpyxl")
    cost_raw = cost_raw.iloc[1:, :]
    if cost_raw.shape[1] == n + 1:
        cost_raw = cost_raw.iloc[:, 1:]
    if cost_raw.shape[0] < n or cost_raw.shape[1] < n:
        raise RuntimeError(f"Costos tiene forma {cost_raw.shape} y no alcanza para n={n}.")
    cost_raw = cost_raw.iloc[:n, :n]
    matriz_costos = cost_raw.apply(pd.to_numeric, errors="coerce").to_numpy(float)

    return nodos_df, matriz_distancias, matriz_costos

#Lee el archivo de rutas CSV y devuelve un dataframe de pandas
def cargar_rutas(carpeta_base=".", nombre_archivo="rutas1.csv"):
    carpeta_base = Path(carpeta_base)
    ruta = (carpeta_base / nombre_archivo).resolve()
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo de rutas: {ruta}")
    rutas_df = pd.read_csv(ruta)

    # Limpiar espacios en blanco en columnas de texto
    for col in rutas_df.columns:
        if rutas_df[col].dtype == object:
            rutas_df[col] = rutas_df[col].astype(str).str.strip()
    return rutas_df
