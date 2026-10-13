# 07 · Avatar 3D en el Rival Fantasma

Pedido (13-10): los avatares 3D en el Rival Fantasma (pantalla 07).

Tarea: W18 (web). Esfuerzo: 1,5 días. Depende de B14 y W17 (`06`), y de B6 (altura y envergadura
cargadas).

---

## 1. Qué muestra

Hoy la pantalla 07 dibuja "Dimensiones reales" con siluetas 2D a escala por altura y envergadura
cargadas a mano. W18 reemplaza esas siluetas, cuando la bandera `avatar_3d` está encendida, por los
dos maniquíes de `06` enfrentados:

- La postura promedio de cada uno en sparring y combate (la misma del perfil, filtrada por esos
  contextos, como el resto de la pantalla).
- Escalados a sus medidas cargadas: la altura total a `altura_m` y los brazos a `envergadura_m`.
  La envergadura se reparte así: envergadura = ancho de hombros + 2 × (brazo + antebrazo + mano);
  el ancho de hombros y la proporción brazo/antebrazo salen de la postura, y el largo se ajusta
  para cerrar la envergadura. Si falta algún dato, ese maniquí se dibuja en unidades del modelo y
  la pantalla dice "sin medidas cargadas: no está a escala".
- Enfrentados, cada uno con su guardia (un zurdo con el pie derecho adelante), a una distancia
  fija de presentación.

Distancia entre los dos (decisión menor, tomada el 13-10 junto con las DA). No hay distancia de
trabajo medida (es B9, P2, con bandas a definir por Lucas). Se usa una distancia fija igual a la suma de los
dos alcances de brazo extendido, rotulada "distancia de presentación, no medida". Cuando exista
B9, se usa la distancia media de los dos en sparring.

## 2. Lo que no hace

- No anima un intercambio de golpes ni simula una pelea: el Rival Fantasma "no es un pronóstico",
  y una animación de dos maniquíes peleando se lee como uno.
- No colorea zonas sobre el 3D. Los cruces de zonas de `05` siguen en las siluetas 2D, que son las
  que tienen las cuatro zonas definidas. Poner zonas sobre un cuerpo 3D sugiere una precisión
  lateral que DA-4 justamente evita.

## 3. Interacción

- Una sola escena con los dos maniquíes. Arrastrar rota la cámara alrededor de la pareja; botones
  "De costado" (la vista que mejor compara alcance), "Desde A" y "Desde B".
- Superponer: el mismo toggle de hoy ("lado a lado / superpuestos") pone los dos en el mismo lugar
  mirando para el mismo lado, para comparar altura y alcance como en el diseño.
- Celular: la escena ocupa el ancho con proporción 4:3, sin zoom libre, y los botones debajo.

## 4. Carga y respaldo

- Mismo componente `Avatar3D` de W17 en modo pareja, mismo import dinámico: `/rival/` sigue por
  debajo de 380 KB (hoy 313 KB).
- Bandera apagada, sin WebGL o con movimiento reducido: las siluetas 2D de hoy, sin cambios.

## 5. Criterios de aceptación

- CA-18.1. Con las medidas cargadas, la diferencia de altura y de envergadura que se ve en la vista
  de costado es la de los datos (la prueba compara las cajas de los dos modelos en la escena con
  los datos, con tolerancia de 1 cm).
- CA-18.2. Con un dato faltante aparece el aviso "no está a escala" y la silueta de ese peleador no
  se escala.
- CA-18.3. La escena no tiene animación de golpes.
- CA-18.4. Con la bandera apagada, la pantalla es idéntica a la de hoy.
