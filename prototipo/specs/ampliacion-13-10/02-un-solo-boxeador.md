# 02 · Marcar un solo boxeador en bolsa y sombra

Pedido (13-10): en los videos de bolsa o sombra, que se marque un solo boxeador.

Tareas: B8b (cierre de B8: medición y ajustes del worker), W19 (pantalla de identificación para
una persona). Esfuerzo: 1,5 días más la medición, que necesita dos videos propios y la GPU.

---

## 1. De qué se parte (relevado en `main` 3a62a39)

Ya está hecho y probado:

- API: `POST /jobs/{id}/siembra` acepta `track_b: null` si el contexto es bolsa o sombra y lo
  rechaza en sparring y combate (`app.py`, l. 451 a 458).
- Worker: `completar` con una semilla llama a `proponer_individual` (`mvp/individual.py`): lo que
  coexiste con la persona asignada es otra persona; lo que no coexiste y tiene guantes de su color
  es ella; el resto, ninguno o sin asignar. La Fight-Card trae solo `peleadores.A`.
- Web: con la bandera `sesiones_individuales` encendida, Nueva sesión pide un solo rincón y la
  identificación manda una semilla (`individual.test.tsx`).

Lo que falta:

1. La bandera está apagada y bolsa y sombra salen "En construcción" (decisión del 13-10), porque
   el criterio de B8 no se midió.
2. La identificación reutiliza la pantalla de dos: muestra "Es A" y "Es B" con B deshabilitado.
   Para una persona eso es ruido. El pedido es exactamente esto: marcar uno.
3. La revisión de tramos (B3) no se probó con una sola persona.
4. El margen del detector que muestra la Fight-Card es el medido en sparring. En bolsa y sombra
   el detector nunca se midió, y el número no se puede trasladar.

## 2. B8b · Worker y medición

### Casos que la regla de coexistencia tiene que resolver

Se prueban con casos sintéticos antes de medir, porque son los que aparecen en un gimnasio:

| Caso | Qué pasa | Qué tiene que hacer |
|---|---|---|
| Espejo en sombra | El reflejo es una "persona" con los mismos guantes, siempre en cuadro con él | Coexiste: ninguno. Si el entrenador marcó el reflejo como semilla, el boxeador queda como ninguno y la sesión sale mal: la pantalla lo previene (§3) |
| Otro boxeador pasa por atrás | Track corto, más chico, a veces con guantes | Filtro de altura o coexistencia: ninguno |
| El boxeador sale de cuadro y vuelve | Track nuevo con su color | Color: es él |
| La bolsa tapa al boxeador | Cortes de track con guante tapado | Sin asignar; entra a la revisión de tramos |
| Entrenador con manoplas | Dos personas con guantes, coexisten | Fuera de alcance: es trabajo de dos; la pantalla de contexto no lo ofrece |

### Revisión de tramos con una persona

`GET /jobs/{id}/tramos` y `PUT /jobs/{id}/tramos` aceptan `rol: "A" | "ninguno"`; un `"B"` en una
sesión individual devuelve 422. `cobertura` trae solo `A` y `sin_asignar`.

### El margen del detector en bolsa y sombra

`detector.precision_medida` y `recall_medido` son de sparring. En una Fight-Card de bolsa o sombra
se agrega `detector.medido_en: "sparring"` y la web, si el contexto de la sesión no coincide,
muestra el margen como "medido en sparring; en bolsa todavía no se midió" en lugar del "acierta 9
de cada 10". El número de golpes se muestra igual: es lo que el sistema detectó.

Medir el detector en bolsa y sombra es una medición aparte (anotar golpes de los dos videos). Se
propone como opcional en `08` §2 porque cuesta horas de anotación.

### Medición para encender la bandera (criterio de B8, ya pre-registrado en `05`)

Sobre dos videos propios, uno de bolsa y uno de sombra, de al menos 3 minutos cada uno: la persona
queda asignada en al menos el 95% de los cuadros en que está en cuadro, mirado por Lucas sobre el
video procesado. Se agrega un detalle que el criterio original no fijaba, para que el "mirado a
ojo" sea repetible: Lucas recorre el video procesado en saltos de 2 s y anota, en cada salto, si
la persona en cuadro tiene la marca A. El porcentaje es saltos con marca sobre saltos con la
persona en cuadro. Se fija ahora, antes de mirar.

Si uno de los dos videos es de sombra frente al espejo, mejor: es el caso más difícil.

Resultado a `docs/experiments/AAAA-MM-DD-sesiones-individuales.md`. Si pasa, `BOXTWIN_SESIONES_INDIVIDUALES=1`
en `/etc/boxtwin.env` y la web deja de mostrar "En construcción" sin tocar código. Si no pasa,
queda como está y se declara en el Cap 7.

### Pruebas (B8b)

1. Espejo sintético: dos tracks coexistentes con el mismo perfil de color; la semilla queda A y
   el otro ninguno.
2. Reentrada: dos tracks no coexistentes con el mismo color quedan los dos A.
3. Tramos con una persona: `PUT` con `"B"` da 422; asignar un tramo a A sube `cobertura.A`.
4. La Fight-Card individual trae `detector.medido_en` y no trae `peleadores.B`.

## 3. W19 · Pantalla de identificación para una persona

Misma ruta (`/sesion/identificar/?id=`); la pantalla elige el modo por `job.contexto`.

- Título "¿Cuál es el boxeador?" y la indicación: "Tocá una vez al que está trabajando. El resto
  del video se sigue por el color de sus guantes."
- Cada candidato: recorte, color de guante, tramo en cuadro, altura relativa, y un solo botón
  "Es él" (grupo de radio de una opción por tarjeta, una sola elección en toda la pantalla).
- Orden de los candidatos: el de más tiempo en cuadro primero, con el rótulo "sugerido". En una
  sesión individual no hay `pareja_sugerida`.
- Aviso de espejo: si el candidato elegido coexiste con otro de color de guante parecido
  (separación de color menor que el piso de 0,55), antes de confirmar: "Hay otra figura con los
  mismos guantes al mismo tiempo. Si estás frente a un espejo, elegí al de verdad, no al
  reflejo." Los dos recortes lado a lado.
- Confirmación: alert-dialog con el recorte elegido: "¿Este es Tomás Ríos?" (o "¿Este es el
  boxeador?" si no hay peleador asignado). Manda `{track_a, track_b: null}`.
- Fase 2 (tramos): botones "Es él" e "Ignorar".
- Fight-Card, perfil y análisis ya dibujan una sola columna (hecho en B8).

### Criterios de aceptación

- CA-19.1. En bolsa y sombra la pantalla no muestra "Es A", "Es B" ni la palabra "rincón".
- CA-19.2. Con una sola persona marcada y confirmada, la sesión llega a `listo` sin otra
  intervención (C4 aplicado a sesiones individuales).
- CA-19.3. En una sesión de bolsa la Fight-Card no muestra el margen de sparring como si fuera de
  bolsa.
- CA-19.4. Con la bandera apagada todo sigue como hoy ("En construcción").
