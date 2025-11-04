
import numpy as np

"""
Para calcular las distancias reales entre sensores y puntos del terreno se implementó la fórmula de Haversine.
Esta fórmula permite obtener la distancia más corta sobre la superficie de la Tierra a partir de coordenadas geográficas (latitud y longitud).
Es necesaria porque los datos del proyecto se encuentran en grados y no en coordenadas planas, 
por lo que la distancia produciría errores al no considerar la curvatura de la Tierra.

Parámetros:
- lat1, lon1 : coordenadas del primer punto (en grados decimales).
- lat2, lon2 : coordenadas del segundo punto (en grados decimales).


-Primero convierte las coordenadas de grados a radianes.
-Después calcula las diferencias angulares de latitud (dlat) y longitud (dlon).
-Se aplica la fórmula de Haversine para obtener el ángulo central (c) entre los dos puntos.
-Finalmente multiplica el ángulo central (c) por el radio de la Tierra para obtener la distancia final.

Retorna:
- Distancia entre los dos puntos en kilómetros (float).
"""

RADIO_TIERRA_KM = 6371.0088

def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arcsin(np.sqrt(a))
    return RADIO_TIERRA_KM * c
