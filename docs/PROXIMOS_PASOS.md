# Próximos pasos — retomar acá

Estado al 11-08. La verificación por muestreo está hecha sobre V4, V5, V8, V9 y
V10. Lo que sigue es reanotación manual, que no se puede delegar.

## Resultado de la verificación

| Video | Aciertos /18 | Piso Wilson 95% | Decisión |
|---|---|---|---|
| V5 | 18 | 82,4% | usar tal cual |
| V8 | 17 | 74,2% | usar tal cual |
| V9 | 17 | 74,2% | usar tal cual |
| V10 | 15 | 60,8% | reanotar completo |
| V4 | 12 | 43,7% | reanotar completo |

V4 venía figurando como confiable con 5 de 5. Eso no era evidencia: el piso
Wilson de 5/5 es 56,6% y el 66,7% real cae adentro de ese intervalo.

Cero clips sin golpe en los 90 de muestra. La segmentación temporal de estos
cinco videos está intacta y el daño es solo de clase.

## 1. Chequeo temprano de V3 (primero, son 3 minutos)

```bash
cd ~/Proyectos/TwinBoxing/test-BoxingVI
python ../scripts/boxingvi_annot.py --csv ./clips/manifest_filtrado.csv \
    --videos V3 --shuffle --out ./clips/reanotado.csv
```

Frenar a los ~50 clips y mirar la tasa de "sin golpe". `reanotado.csv` se
escribe incrementalmente y se lee sin frenar el server:

```bash
python -c "import pandas as pd; d=pd.read_csv('clips/reanotado.csv'); d=d[d.video_key=='V3']; print(len(d), (d.nueva_cls=='sin golpe').mean())"
```

Si se acerca al 42% de V2, las ventanas de V3 también están rotas, reclasificar
no alcanza y hay que resegmentar. Es el único bloque grande cuya segmentación
sigue sin verificar, y son 810 clips en juego.

Cuidado: `reanotado.csv` ya tiene los 232 clips de V2. El script reanuda por
`clip` así que no los repite, pero al analizar hay que filtrar por `video_key`.

## 2. Reanotar V4 (559 clips, ~28 min)

```bash
python ../scripts/boxingvi_annot.py --csv ./clips/manifest_filtrado.csv \
    --videos V4 --shuffle --out ./clips/reanotado.csv
```

Prioridad sobre V10 porque V4 aporta 75 de los 136 Rear Hook que sobreviven al
descarte de V1 y V2, y sus 2 Rear Hook de muestra salieron los dos mal, los dos
como Rear Uppercut. Al terminar, recontar la clase.

## 3. Reanotar V10 (151 clips, ~9 min) y después V3 completo

Mismo comando cambiando `--videos`.

## 4. Consolidación

- Excluir los 209 clips de placas de V1 (`clips/placas.csv`). Discutible si V1
  se descarta entero, que es lo que hoy parece.
- Rehacer el split. La validación aguanta: 83% de ella es V5 y V9, que pasaron.
  El problema es train, donde tras descartar V1 y V2 el único bloque limpio son
  los 199 clips de V8.
- Recalcular pesos de clase.
- Decidir si Rear Hook sigue siendo viable como clase.
- Muestra de 18 para V7, que pasa como confiable con 3 clips mirados.
- Decidir V1 con la tasa de descarte que deje V3.

## 5. Recién ahí, Fase D

Extracción de pose sobre el dataset final (`boxingvi_pose.py`).

## Aparte: experimentos que no dependen del dataset

El Capítulo 4 no se escribe sin el benchmark de perfiles de despliegue (§6 de
CLAUDE.md). Se puede correr en paralelo a toda la anotación y hoy es el
verdadero cuello de botella para la escritura.
