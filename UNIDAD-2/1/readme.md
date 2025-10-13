# Proyecto para optimizar rutas de distribución entre centros de distribución y sucursales en un escenario ficticio de Culiacán, Sinaloa, usando Recocido Simulado (RS).

El problema se modela como grafo ponderado (distancia/costo de combustible). Si el CSV de rutas no conecta todo, se trabaja sobre la mayor componente y se usa clausura métrica (Dijkstra) para evaluar costos entre pares.

Bibliotecas usadas:
<ul>
  <li>Paquetes: <code>pandas</code>, <code>numpy</code>, <code>networkx</code>, <code>openpyxl</code>, <code>matplotlib</code></li>
</ul>

Archivos utilizados:
<ul>
  <li>ARCHIVO_NODOS    = "datos_distribucion_tiendas.xlsx"</li>
  <li>ARCHIVO_DIST     = "matriz_distancias.xlsx"</li>
  <li>ARCHIVO_COSTO    = "matriz_costos_combustible.xlsx"</li>
  <li>ARCHIVO_RUTAS    = "rutas.csv"</li>
</ul>
