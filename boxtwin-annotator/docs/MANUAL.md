# Manual de uso — boxtwin-annotator

Guía de trabajo, ordenada por lo que hay que hacer. El *por qué* de cada decisión de diseño
está en el `README.md`; acá está el *cómo*.

---

## 1. El flujo completo

Cuatro fases. Las dos primeras se hacen una vez por video; la tercera es el trabajo real.

```
preprocesar  →  asignar identidad  →  anotar eventos  →  exportar
  (CLI, GPU)      (app, una vez)       (app, horas)      (CLI)
```

**Asignar identidad va antes de anotar, y no es opcional.** Si un track no tiene rol, los
eventos se marcan igual pero el export no puede sacar keypoints de esos cuadros. Es el
error más caro de descubrir tarde.

---

## 2. Abrir la aplicación

Desde una terminal del escritorio (por RDP o local):

```bash
cd ~/Proyectos/TwinBoxing/anotacion
~/miniforge3/envs/twinboxing_env/bin/boxtwin-annotator annotate videos/Sparring.mp4
```

Si no hay sesión gráfica, falla con una explicación en vez de abortar. En una máquina
headless se puede servir la ventana por VNC con `--platform vnc`, pero con RDP no hace
falta.

Si falta el cache de pose, el mensaje dice qué correr.

---

## 3. La pantalla

| Zona | Qué es |
|---|---|
| Centro | el cuadro con el overlay de pose |
| Barra angosta de abajo | el timeline, con dos carriles: arriba fighter_A, abajo fighter_B |
| Barra de estado | cuadro actual, tiempo, fps nativo, velocidad, peleador elegido, evento abierto |
| Panel derecho | cinco pestañas: Eventos, Identidad, Reanotación, Balance, Vista, Validación |

En el overlay, el **color va por rol**: rojo fighter_A, azul fighter_B, gris los ignorados,
**ámbar los que no tienen rol asignado**. Si ves ámbar, falta trabajo de identidad.

Los keypoints con poca confianza se dibujan atenuados en vez de ocultarse: una pose mala y
una pose incompleta son cosas distintas.

---

## 4. Moverse por el video

| Tecla | Qué hace |
|---|---|
| `Espacio` | reproducir / pausar |
| `Shift+Espacio` | reproducir hacia atrás |
| `←` `→` | ±1 cuadro |
| `Shift+←` `Shift+→` | ±5 cuadros |
| `Ctrl+←` `Ctrl+→` | ±1 segundo |
| `Inicio` `Fin` | principio / final del video |
| `G` | ir a un cuadro por número |
| `+` `-` | cambiar velocidad |

Las velocidades son 0,10x / 0,25x / 0,50x / 1,0x, y **arranca en 0,25x**. Es la velocidad a
la que se distingue el inicio de la extensión del codo, que es la definición de
`start_frame`.

El reproductor avanza exactamente un cuadro por tic y nunca descarta. Si la máquina no
llega, se pone lento en vez de saltear.

### Ver mejor

| Tecla | Qué hace |
|---|---|
| rueda del mouse | zoom sobre el puntero |
| arrastrar | desplazar |
| doble click, o `Ctrl+0` | ajustar a la ventana |
| `Ctrl++` `Ctrl+-` | zoom |
| `K` `X` `I` `L` | esqueleto, cajas, IDs, guantes |
| `O` | ver **solo** el peleador elegido |

`O` sirve en el clinch, donde los dos esqueletos se superponen y no se distingue qué
keypoint es de quién. Con el filtro puesto, `1` y `2` cambian a quién estás mirando.

Con zoom sobre 1,6x y en pausa, la imagen pasa a resolución completa del video original.
Abajo a la derecha dice qué fuente se está usando.

---

## 5. Asignar identidad (hacer esto primero)

Pestaña **Identidad**.

El tracker parte a cada peleador en muchos `track_id`: pierde el track en cada oclusión y lo
recupera con un id nuevo. En el video de sparring de prueba son 42 tracks para dos personas.

### Lo básico

1. Poné el cursor donde el track aparece por primera vez.
2. Elegí el track en la lista **Tracks en este cuadro**.
3. Botón `→ A`, `→ B`, o `→ ignorar` para el árbitro y los que miran.

La asignación vale **desde el cuadro actual hacia adelante**, hasta la próxima decisión
manual. No desde el principio del video: ese track pudo haber sido otra persona antes.

Conviene recorrer el video de principio a fin asignando cada track ámbar que aparezca. El
timeline no marca dónde aparecen los tracks nuevos, así que el método es avanzar de a
segundo con `Ctrl+→` mirando el color.

### Cuando A y B se intercambian

Ubicá el primer cuadro donde están cambiados y apretá **Intercambiar A ↔ B desde acá**. La
corrección llega hasta la próxima decisión manual posterior, así que no pisa lo que ya
hayas arreglado más adelante.

### Cuando el tracker pierde a alguien

**re-sembrar: A** o **B**, y dibujás una caja sobre el peleador en el cuadro actual.

- Si la caja cae sobre un track existente, le asigna el rol a ese track. Es lo normal: el
  tracker no perdió al peleador, le cambió el id.
- Si no cae sobre nada, crea una caja manual sin keypoints, y esos cuadros se marcan solos
  como no confiables porque no hay pose que exportar.

### Tramos que no sirven

Un clinch largo, alguien fuera de cuadro, pose claramente rota. Poné `desde` y `hasta` con
los botones **= cuadro actual**, elegí peleador y motivo, y **Marcar tramo**.

No borra la pose: la seguís viendo y juzgando. Solo la excluye del export por defecto.

### Rellenar huecos internos

Hacelo una vez al final, cuando ya asignaste todos los tracks.

Un track no es continuo: el mismo id aparece, desaparece y vuelve. Sobre `Sparring.mp4` los
42 tracks son en realidad 367 tramos, con 325 huecos de mediana 1 cuadro. En esos cuadros el
peleador existe pero no tiene pose, y si el hueco cae dentro de un golpe, ese golpe se exporta
incompleto.

El panel muestra cuántos huecos quedan y cuántos cuadros son. **Rellenar huecos internos** los
interpola todos de una vez.

Va en lote y las uniones no, y la diferencia no es de comodidad. Unir dos tracks afirma que
dos ids son la misma persona, y equivocarse mete keypoints del peleador equivocado en el
dataset. Un hueco interno no afirma nada: el id es el mismo a los dos lados y lo único que se
agrega son los cuadros del medio. No cambia ninguna asignación de rol.

Solo toca tracks que ya son un peleador, ignora los huecos de más de 20 cuadros —ahí
interpolar en línea recta sería inventar— y se puede apretar de nuevo sin duplicar nada.

> Medido sobre la anotación real de `Sparring.mp4`: 243 huecos, 559 cuadros. La cobertura de
> "los dos peleadores presentes" sube de 76,7% a 84,9%, y de los 118 eventos anotados **28
> tenían cuadros sin pose del peleador anotado; después quedan cero**.

### Unir tracks

**Buscar tracks para unir** propone pares que podrían ser la misma persona, ordenados por
parecido de caja. Se confirman de a uno, porque en un clinch las cajas de los dos peleadores
se superponen casi por completo y ahí es donde la heurística se equivoca.

No propone unir dos tracks que ya asignaste a peleadores distintos en el cuadro donde se
unirían. Ojo con el matiz: un mismo track puede ser fighter_B un rato y fighter_A después, que
es justo lo que deja un swap, así que el filtro mira el cuadro de la unión y no el track
entero.

> El umbral de hueco por defecto es de 60 cuadros. Con los 5 de la versión anterior el
> detector proponía **cero** uniones sobre material real. El valor queda congelado en cada
> `annot.json`, así que un proyecto empezado antes de este cambio conserva el 5: se cambia
> editando `settings_snapshot.interp_max_gap_frames` en el archivo.

Todo esto se deshace con `Ctrl+Z`.

---

## 6. Anotar eventos

El bucle es siempre el mismo:

1. `1` o `2` para elegir de qué peleador es el golpe.
2. Buscá el inicio y apretá **`[`**. La barra de estado pasa a `ABIERTO en N` y va contando
   los cuadros transcurridos.
3. Buscá el final y apretá **`]`**. Se abre el diálogo de clasificación.
4. Clasificá con el teclado y `Enter`.

Al confirmar, el foco vuelve al reproductor en el `end_frame` y **se guarda en el acto**.

Si te equivocaste antes de cerrar el evento, **`Z`** lo descarta. Es distinto de `Ctrl+Z`,
que deshace lo ya confirmado.

### El diálogo de clasificación

El clip corre en loop a 0,25x mientras clasificás. Distinguir un hook de un uppercut sobre
un cuadro congelado es adivinar: lo que los separa es la trayectoria.

| Tecla | Campo |
|---|---|
| `Q` `W` | lado izquierdo / derecho |
| `A` `S` `D` | recto / hook / uppercut |
| `H` `B` | cabeza / cuerpo |
| `F` | amague (se aprieta de nuevo y vuelve a completo) |
| `Shift+F` | abortado |
| `4` `5` `6` `7` `8` | conecta, bloqueado, esquivado, falla, sin datos |
| `C` `V` `N` | calidad limpia, con oclusión, ambigua |
| `P` | marcar el pico en el cuadro que muestra el preview |
| `[` `]` | corregir las fronteras sobre el preview |
| `Espacio` | pausar el loop |
| `Enter` | confirmar |
| `Esc` | cancelar |

Lado, tipo y altura **son obligatorios y no vienen preseleccionados**. Si falta alguno, el
panel lo dice y `Enter` no confirma. No hay default a propósito: preseleccionar el tipo más
frecuente sesgaría el dataset cada vez que confirmes sin mirar.

Resultado y calidad sí vienen puestos en *sin datos* y *limpia*, porque esos defaults son
afirmaciones honestas y no suposiciones.

El contador **vueltas del preview** queda registrado: es el indicador de cuán difícil fue la
decisión.

### Las definiciones que hay que respetar

Están siempre a la vista en la pestaña Vista. La consistencia entre sesiones depende de
aplicarlas igual el lunes y el jueves.

- **`start_frame`** — primer cuadro en que el puño inicia el desplazamiento hacia el
  objetivo, con el codo empezando a extenderse o el hombro rotando. **No** el cuadro en que
  se carga el peso.
- **`peak_frame`** — máxima extensión del brazo o contacto, lo que ocurra primero.
- **`end_frame`** — cuadro en que el puño retrocedió aproximadamente la mitad del camino de
  vuelta a la guardia.
- **`feint`** — el movimiento inicia pero se aborta antes del 60% de la extensión esperada y
  no hay retracción de recuperación completa.

### Combinaciones

Un golpe puede empezar antes de que termine el anterior, y hay que anotarlos como dos
eventos solapados. Es lo correcto y el sistema no se queja: el solapamiento entre golpes de
**lados distintos** es una combinación.

Sí avisa el solapamiento del **mismo brazo**, que salvo doble jab es sospechoso.

---

## 7. Revisar y corregir

Pestaña **Eventos**. La tabla es filtrable por peleador y por texto, ordenable por cualquier
columna, y **editable en la propia celda**: doble click y cambiás el valor. Click en una
fila salta a ese evento.

| Tecla | Qué hace |
|---|---|
| `Supr` | borrar el evento seleccionado |
| `Ctrl+Z` | deshacer |
| `Ctrl+Shift+Z` | rehacer |
| `Ctrl+[` `Ctrl+]` | ir al inicio / final del evento seleccionado |

Todo pasa por el historial, incluidas las correcciones de la tabla. Guarda 100 pasos.

La pestaña **Balance** muestra el conteo por clase mientras anotás y resalta la más escasa.
Sirve para ir a buscar material de las clases flacas antes de acumular más de las que sobran.

La pestaña **Validación** lista lo que el sistema encuentra raro. Son avisos, no bloqueos:
siempre podés guardar.

---

## 8. Guardado

Automático en cada evento confirmado y además cada 30 segundos. `Ctrl+S` fuerza. Un punto en
la barra de estado indica que hay cambios sin guardar.

Todo va a `annotations/<video>.annot.json`, que es la fuente de verdad: eventos, identidad,
tramos no confiables y métricas de proceso. Es texto legible y diffeable en git.

El `.npz` de pose **no se modifica nunca**.

---

## 9. Exportar

```bash
boxtwin-annotator export videos/Sparring.mp4 --format stats
boxtwin-annotator export videos/Sparring.mp4 --format mmaction --classes 12
boxtwin-annotator export videos/Sparring.mp4 --format sequence
boxtwin-annotator export videos/Sparring.mp4 --format clips
```

Empezá siempre por `stats`: dice cuántos eventos hay por clase, cuánto duran, qué porcentaje
de cuadros quedó no confiable y cuánto tiempo llevó anotar.

Las clases por defecto son por lado: `straight-left`, `straight-right`, `hook-left`,
`hook-right`, `uppercut-left`, `uppercut-right`, más la altura con `--classes 12`.

---

## 10. Reanotación ciega

Para medir cuánto se contradice tu propia anotación.

```bash
boxtwin-annotator reanno videos/Sparring.mp4 --fraction 0.10 --seed 42
```

Sortea una muestra y **la congela**. Después, desde la pestaña **Reanotación**, botón
*Empezar*. Mientras el modo está activo se ocultan las marcas del timeline y la lista de
eventos, y cada intento se presenta en una ventana con relleno aleatorio, para que nada
delate dónde está el golpe.

Marcás inicio y final con `[` y `]` como siempre, y clasificás. Al terminar:

```bash
boxtwin-annotator reanno videos/Sparring.mp4 --report
```

Da kappa de Cohen por dimensión y el error de fronteras en cuadros.

---

## 11. Cuando algo sale mal

| Síntoma | Qué pasa |
|---|---|
| El esqueleto sale ámbar | ese track no tiene rol; asignalo en Identidad |
| `]` no hace nada | no hay evento abierto; marcá el inicio con `[` |
| `Enter` no confirma en el diálogo | falta lado, tipo o altura; el panel dice cuál |
| El esqueleto no sigue bien al cuerpo | es la calidad de la estimación de pose, no un desfasaje: está verificado que imagen y keypoints son del mismo cuadro |
| La imagen se ve borrosa | estás sobre el proxy; en pausa y con zoom pasa sola a resolución completa |
| Una edición en la tabla se rechaza | violaba una regla del esquema, por ejemplo un final anterior al inicio |

Para revisar la anotación de corrido sin abrir la interfaz:

```bash
python tools/render_overlay.py videos/Sparring.mp4 --from 0 --to 500 --fps 8
```

---

## 12. Límites conocidos

- **El detector de uniones propone poco.** Sobre `Sparring.mp4` da 3 propuestas correctas en
  todo el video, así que el grueso de la asignación de tracks sigue siendo manual. Lo que sí
  rinde es el relleno de huecos internos; ver la sección 5.
- **El validador cuenta colisiones de rol que no existen.** `ID_ROLE_COLLISION` compara solo
  los rangos de las asignaciones, sin mirar si los dos tracks tienen detecciones en esos
  cuadros. Como al asignar un track el anterior no se trunca, los rangos se solapan y salen
  errores de a decenas. Medido sobre la anotación real: **147 colisiones, 0 cuadros y 0
  eventos afectados**. Son ruido, no datos rotos.
- **La guardia arranca en ortodoxa para los dos peleadores.** Si alguno es zurdo, hay que
  cambiarlo antes de anotar, porque de ahí se deriva el rol lead/rear en el export.
- **`landed` es una estimación.** Un sistema monocular no establece contacto físico; el
  campo existe para alimentar heurísticos que se reportan con margen de error.
- **Los guantes dibujados son derivados**, no observados: se extrapolan sobre el antebrazo.
  Sirven para juzgar a ojo, no aportan información a un modelo que ya ve codo y muñeca.
