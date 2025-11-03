
import csv

# Carga un CSV con encabezados: Humedad, Cultivo, Elevación, Salinidad, Temperatura, Latitud, Longitud
# Devuelve listas paralelas: humedad, cultivo, elevacion, salinidad, temperatura, latitudes, longitudes
def cargar_csv(ruta_csv):
    humedad = []
    cultivo = []
    elevacion = []
    salinidad = []
    temperatura = []
    latitudes = []
    longitudes = []
    with open(ruta_csv, newline='', encoding='utf-8') as f:
        lector = csv.DictReader(f)
        for fila in lector:
            humedad.append(float(str(fila['Humedad']).replace(',', '.')))
            cultivo.append(str(fila['Cultivo']).strip())
            elevacion.append(float(str(fila['Elevación']).replace(',', '.')))
            salinidad.append(float(str(fila['Salinidad']).replace(',', '.')))
            temperatura.append(float(str(fila['Temperatura']).replace(',', '.')))
            latitudes.append(float(str(fila['Latitud']).replace(',', '.')))
            longitudes.append(float(str(fila['Longitud']).replace(',', '.')))
    return humedad, cultivo, elevacion, salinidad, temperatura, latitudes, longitudes
