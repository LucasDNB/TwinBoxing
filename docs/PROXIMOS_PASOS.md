# Próximos pasos — retomar acá

Estado al 22-09. La spec del MVP está aprobada (`docs/specs/MVP.md`), el esqueleto
está construido y testeado, y el circuito **ya corrió una vez sobre video real**:
`amateur_estatico`, 120 s, en 1,098x tiempo real, con 46 golpes detectados. Lo que
salió de ahí está en
[`docs/experiments/2026-09-22-mvp-sobre-video-real.md`](experiments/2026-09-22-mvp-sobre-video-real.md)
y cambia dos cosas de este plan: la siembra hace lo que se esperaba pero por otro
motivo, y el indicador de guardia no mide lo que dice medir.

## Lo que está hecho

| | |
|---|---|
| `boxtwin procesar` / `completar` | de video crudo a `fightcard.json`, sin proyecto de anotación |
| Siembra humana en la identidad | el par de tracks fija el ancla, la orientación y quién es A |
| `boxtwin_detector.inferencia` | carriles desde keypoints identificados, con remuestreo a 30 fps |
| Indicadores de guardia | los dos, con el umbral de retorno **sin calibrar** |
| `fightcard.json` v0.1 | golpes detectados con su recall al lado, tipo aparte y rotulado |
| `boxtwin-api` | cola en tabla, worker por subproceso, export, corrección de tipo |
| `boxtwin-web` | los seis pasos del flujo |
| `despliegue/` | Dockerfiles y compose, **nunca construidos** |

## 1. Correr el circuito sobre material real (primero, y bloquea todo)

`anotacion-amateur/` ya está preprocesado y es el caso más duro que hay: cámara
lejana, árbitro adentro del ring, ancho de hombros mediano de 22 px contra 132 en
`sparring-3`. Pone a prueba la invariancia a escala de las features, que nunca se
probó a ese tamaño.

```bash
cd ~/Proyectos/TwinBoxing
boxtwin-annotator procesar anotacion-amateur/videos/<video>.mp4 \
    --out sesiones/amateur --modelo-guantes boxtwin-guantes/modelos/guantes-v2.pt \
    --round 120
# mirar sesiones/amateur/candidatos/*.jpg y elegir
boxtwin-annotator completar sesiones/amateur --semilla-a <t> --semilla-b <t> \
    --detector modelos/detector-7fuentes.pt
```

Ojo con el amateur: los dos llevan guantes casi del mismo color (separación 0,133
contra un piso de 0,55), así que el color no va a decidir nada y todo el peso cae
en la coexistencia y en la siembra. Es exactamente el caso para el que se hizo la
siembra, y es la prueba de si alcanza.

El checkpoint de producción ya existe: `modelos/detector_7fuentes.ens.pt`, ensamble
de cinco semillas (42, 1, 2, 3, 4) sobre las siete fuentes, 20 features. Es el que
usó la corrida del 22-09.

## 2. Medir C1: identidad con siembra humana

Umbral pre-registrado ≥ 95% de tracks bien asignados sobre las seis fuentes de
gimnasio, sembrando con los dos tracks que elegiría el usuario. Ya existe
`puntuar_contra`, que compara una propuesta contra las asignaciones manuales del
documento, así que la medición es barata: correr `proponer` con semillas sobre
cada fuente anotada y puntuar.

Si no llega, el plan B está declarado: corrección manual por track en la web.

## 3. Medir C3: el indicador de guardia

100 golpes de `sparring-3` marcados a mano (volvió a guardia sí/no, mano opuesta
caída sí/no) contra lo que dice el indicador. Umbral de acuerdo 80% en cada uno.

De esta medición sale además el umbral de retorno lento, que **hoy no existe**: el
módulo reporta el tiempo medido y no marca nada, a propósito. Fijar el umbral antes
de mirar los datos, sobre la mediana.

**Y hay que revisar el radio de la zona antes de medir.** Sobre `amateur_estatico`,
con 0,6 anchos de hombro el indicador marca la mano opuesta caída en 40 de 41
golpes, y la muñeca está a 1,4–1,75 anchos de la nariz de mediana: el 0,6 no
describe «mano arriba» sino «mano tocándose la cara». El p10 medido anda entre 0,75
y 1,09. El default no se tocó porque la definición está pre-registrada y moverla
mirando esos datos sería elegir el criterio después de ver el resultado; corregirla
es una decisión de dominio y va antes de correr C3, no después.

Si no valida, F5 sale del MVP y se declara como línea futura. El volumen y la línea
de tiempo sostienen el producto solos.

## 4. Medir C5: tiempo de procesamiento

Umbral 2x la duración del video. La instrumentación ya está: `sesion.json` guarda
segundos por etapa y `factor_tiempo_real` los divide por la duración. El tiempo de
máquina no cuenta la espera humana de la siembra, que es lo correcto y lo que una
medición a reloj de pared mezclaría.

## 5. La imagen del worker

`despliegue/Dockerfile.worker` lleva los dos entornos conda. Está escrito y sin
construir, y ahí es donde el riesgo 3 de la spec dice que suelen irse los días:
mmcv pinneado contra CUDA 11.8 compilando adentro de una imagen.

Si no sale, el plan B no toca código: `BOXTWIN_CMD_CLASIFICADOR` apunta al entorno
conda local y la clasificación corre afuera del contenedor leyendo el mismo JSON.

## 6. Lo que queda abierto y hay que decidir

- **La nomenclatura rioplatense.** La spec fija que cross es el gancho y eso está
  puesto. Falta decidir cómo se llaman en pantalla el hook y el uppercut; hoy
  quedan con su nombre en inglés, que es lo que se usa en el gimnasio, en vez de
  inventarles una traducción. Es criterio de dominio, no de código.
- **El clasificador.** Entra al MVP sin umbral y se registra por versión de
  checkpoint (C7). Cada corrección del entrenador es una etiqueta nueva sobre
  material que el modelo no vio, y se guarda en `correcciones.jsonl` con el
  checkpoint que produjo la original. Eso es lo que le falta para generalizar.

## Aparte: los experimentos que no dependen de nada de esto

El benchmark de perfiles de despliegue sigue sin hacerse y el Capítulo 4 no se
escribe sin él (§6 de CLAUDE.md). Hoy es el verdadero cuello de botella para la
escritura, y se puede correr en paralelo a todo lo de arriba.
