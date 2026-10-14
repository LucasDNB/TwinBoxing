# Cambios a la tesis por la ampliación del 13-10

Fuente: `prototipo/specs/ampliacion-13-10/08-plan-criterios-y-tesis.md` §5. Los textos ANTES
se verificaron el 13-10 contra el PDF de la Entrega 6 en LaTeX (Drive, "03 - Benitez Lucas -
BoxTwin - Entrega 06 - LaTex.pdf"): los tres están tal cual, así que ninguno está aplicado
todavía. La fuente LaTeX no está en el repositorio ni en Drive; se aplican a mano en el `.tex`.

Uno de los DESPUÉS cambia respecto del spec porque C8 no pasó (fila de Fight-Card de la
Tabla 4.1): está marcado.

## 1. Glosario (Cap 3, marco conceptual)

ANTES: "Mapa de calor de exposición: representación visual que agrega estadísticamente, sobre
las zonas corporales de un peleador, la frecuencia con la que cada zona resultó alcanzada en las
sesiones registradas. Es descriptivo de lo observado y no predictivo."

DESPUÉS: "Mapa de calor de exposición: representación visual que agrega estadísticamente, sobre
las zonas corporales de un peleador, la frecuencia con la que sus rivales dirigieron golpes a
cada zona en las sesiones registradas. Cuando exista la heurística de conexión, suma la
frecuencia estimada con la que cada zona resultó alcanzada, con su margen. Es descriptivo de lo
observado y no predictivo."

QUÉ CAMBIÓ: "resultó alcanzada" afirmaba contacto, que un sistema monocular no establece. La
definición nueva describe lo que el mapa puede contar y deja el contacto estimado como
extensión. Vale aunque el mapa no se haya construido: la definición vieja estaba mal igual.

## 2. Cap 4, Tabla 4.1, fila "Clasificador de acciones"

ANTES: "Detección de eventos de golpe dentro de la secuencia y clasificación por tipo, lado y
altura"

DESPUÉS: "Detección de eventos de golpe dentro de la secuencia, con el brazo que lanza, y
clasificación por tipo"

QUÉ CAMBIÓ: el lado lo da el detector y la altura no la da ningún modelo de acciones.

## 3. Cap 4, Tabla 4.1, fila "Fight-Card y perfil"

ANTES: "Conteos por asalto, heurísticas de conexión con margen declarado y agregación en el
perfil táctico"

DESPUÉS (cambia respecto del spec): "Conteos por asalto, heurísticas de conexión con margen
declarado y agregación en el perfil táctico. La altura del golpe por posición de la mano
respecto del rival se midió y no alcanzó su criterio (apartado del Cap 7)"

QUÉ CAMBIÓ: el spec proponía sumar la altura como responsabilidad de la Fight-Card. C8 no pasó
(exactitud balanceada 0,726 contra 0,75), así que la Fight-Card no la muestra; se dice dónde se
midió en vez de listarla como algo que el sistema hace. Si se prefiere no tocar la fila, la
alternativa es dejar el ANTES y poner lo de la altura solo en el Cap 7.

## 4. Cap 6, apartado 8, y Cap 4, apartado 2.7

No se reescriben. Nota fechada para la entrega acumulativa, propuesta:

"El 13 de octubre especifiqué una ampliación de alcance con siete pedidos, ordenada en tres
tandas con una regla de corte. Al 13 de octubre entraron las combinaciones izquierda,
derecha, izquierda y la identificación de una sola persona en bolsa y sombra, esta última con
el código hecho y sin habilitar hasta medirla. La altura del golpe se midió y no pasó su
criterio, así que el mapa de exposición y el cruce de zonas del Rival Fantasma quedan como
diseño rotulado. El avatar 3D espera la regla de corte."

(Revisar antes del 03-11: si se mide B8 o arranca el avatar, la nota cambia.)

## 5. Cap 7 (entrega del 13-10)

Párrafos propuestos, en el registro del Cap 2. Los números salen de
`docs/experiments/2026-10-13-cabeza-cuerpo.md`.

Criterios pre-registrados nuevos:

"Con la ampliación del 13 de octubre sumé cuatro criterios, escritos antes de medir como los
siete anteriores. C8 pide que la altura del golpe, calculada con la posición de la mano
respecto de los hombros y el torso del rival, llegue a una exactitud balanceada de 0,75 y a un
recall de golpes al cuerpo de 0,60. C9 compara el recall del detector sobre golpes a la cabeza
y al cuerpo. C10 pide coherencia y plausibilidad al avatar. C11 registra, sin umbral, cuántas
combinaciones izquierda, derecha, izquierda encuentra el sistema de punta a punta."

Dato descriptivo del conjunto propio:

"En las seis fuentes con identidad anotada hay 972 golpes con altura marcada, 162 al cuerpo, un
16,7%. Con esa proporción la exactitud simple no sirve: decir que todo va a la cabeza da 0,83
sin saber nada, y su exactitud balanceada es 0,50. En todo el material hay 28 combinaciones
izquierda, derecha, izquierda con menos de medio segundo entre golpes, una cada 30 golpes, así
que ese bloque va a mostrar números chicos."

Resultado de C8:

"Elegí el umbral sobre dos fuentes y medí una sola vez sobre las otras cuatro, 806 golpes. La
regla llegó a 0,726 de exactitud balanceada y a 0,818 de recall al cuerpo. No pasa: le gana con
claridad a la línea de base, pero llama cuerpo a 204 de 674 golpes a la cabeza. Antes de medir
fijé que un golpe sin rival en cuadro cuenta como error; sin esos 53 golpes la cifra sería
0,781, y no la uso porque la miré después. La altura no se muestra en la Fight-Card, y el mapa
de exposición y el cruce de zonas, que dependían de ella, no se construyeron."

Nivel 2, punta a punta:

"Medí también lo que ve el entrenador: las cuatro fuentes procesadas por el producto, con la
altura calculada sobre los golpes que el sistema encontró y emparejó con uno anotado. Sobre
391 golpes la exactitud balanceada baja a 0,608 y el recall al cuerpo a 0,587, aun con un
detector que vio estas fuentes al entrenar. Hay un segundo dato que pesa más que la regla: el
detector encuentra la mitad de los golpes a la cabeza y un tercio de los golpes al cuerpo. Un
mapa de exposición hecho con este detector le restaría peso al cuerpo aunque la altura
estuviera bien medida."

## 6. Cap 8 (20-10)

Sin cambios por ahora: el experimento del avatar (`06` §6) solo entra si arranca la tanda 3.
