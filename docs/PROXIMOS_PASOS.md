# Próximos pasos — retomar acá

Estado al 10-08. Todo lo automatizable está hecho y commiteado. Lo que sigue
arranca con anotación manual, que no se puede delegar.

## 1. Anotar la muestra de V8/V9/V10 (bloqueante)

```bash
cd ~/Proyectos/TwinBoxing/test-BoxingVI
python ../scripts/boxingvi_annot.py --csv ./clips/muestra_v8v9v10.csv \
    --videos V8 V9 V10 --out ./clips/verificacion_v8v9v10.csv
```

54 clips, ~3 min a 2,2 s por clip. Anotar ciego, sin tocar la tecla E. Los tres
videos vienen mezclados a propósito: no saber cuál estás juzgando evita que el
recuerdo de V1 roto contamine el juicio.

La muestra está congelada en el commit `7f6dcd5`, anterior a toda anotación. No
resortearla: el script se niega salvo `--force`, y hacerlo anula el criterio
pre-registrado.

**Criterio de decisión, fijado de antemano** (aciertos sobre 18):

| 16 o más | usar el video tal cual |
| 11 a 15  | reanotar completo |
| 10 o menos | descartar |

Después: contar aciertos por video contra la etiqueta original y aplicar. Si V9
o V10 caen, la validación se rehace entera y probablemente haya que pasar a
validación cruzada por video en vez de split fijo. Ojo que V5 también está hoy
en validación, no solo V9 y V10.

## 2. Reanotar V3 (decidido: directo, sin muestra previa)

```bash
python ../scripts/boxingvi_annot.py --csv ./clips/manifest_filtrado.csv \
    --videos V3 --shuffle --out ./clips/reanotado.csv
```

810 clips, ~30 min. **Chequeo temprano obligatorio:** a los ~50 clips, mirar la
tasa de "sin golpe" en `reanotado.csv`, que se escribe incrementalmente y se
puede leer sin frenar el server. Si se acerca al 42% de V2, las ventanas
temporales de V3 también están rotas, reclasificar no alcanza y hay que
resegmentar. Mejor descubrirlo a los dos minutos que a la media hora.

Cuidado: `reanotado.csv` ya tiene los 232 clips de V2. El script reanuda por
`clip`, así que no los repite, pero al analizar hay que filtrar por `video_key`.

## 3. Consolidación (después de 1 y 2)

- Excluir del dataset final los 209 clips de placas de V1 (`clips/placas.csv`,
  columnas `es_placa` y `placa_parcial`). Caen todos en train.
- Rehacer el split con lo que sobreviva.
- Recalcular pesos de clase. Sacar las placas los mueve 1,9% como máximo, así
  que el driver real va a ser qué videos entren, no las placas.
- Decidir V1 con la tasa de descarte que deje V3. Sigue pendiente a propósito.

## 4. Recién ahí, Fase D

Extracción de pose sobre el dataset final (`boxingvi_pose.py`).

## Aparte: experimentos que no dependen del dataset

El Capítulo 4 no se escribe sin el benchmark de perfiles de despliegue (§6 de
CLAUDE.md). Se puede correr en paralelo a toda la anotación y hoy es el
verdadero cuello de botella para la escritura.
