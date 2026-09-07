# El sistema completo sobre video continuo: 88,5% de precisión al detectar, y la familia sigue siendo el techo

**07-09-2026 · `sparring-3` entero, 380 golpes · el detector reemplaza al disparador heurístico.**

## Qué se cerró

Las tres piezas estaban medidas por separado. El único número de punta a punta era el del
demo del 01-09, que usaba el disparador por extensión de muñeca: **14 de 21 golpes
encontrados, 66 disparos, 21% de precisión**. Después se midió que ese disparador **no supera
al azar**, así que aquel 21% describía al ruido que alimentaba al sistema, no al sistema.

Ahora el disparador es el detector entrenado, y se mide sobre el video entero con el mismo
emparejador que el resto del proyecto.

**El detector se entrenó SIN `sparring-3`**, sobre las otras seis fuentes. El clasificador no
puede decir lo mismo: se entrenó en distribución sobre `sparring-3` y vio su mitad de
entrenamiento.

## Etapa 1: detectar

| | |
|---|---|
| Golpes anotados | 380 (367 con pose medida) |
| Marcas del detector | 208 |
| Emparejados | 184 |
| **Recall** | **0,484** |
| **Precisión** | **0,885** |
| F1 | 0,626 |
| IoU medio | 0,746 |
| **Error de fronteras** | **1,14 al inicio, 0,97 al final** |

Contra el disparador heurístico, la precisión pasa de **0,21 a 0,885**. Los criterios no son
idénticos —el demo medía contención sobre 30 segundos y esto es IoU 0,3 sobre 9,6 minutos—
así que el factor exacto no es comparable al decimal, pero el orden de magnitud sí: de cuatro
marcas falsas por cada acierto a una cada ocho.

**Las fronteras están al nivel del humano.** El acuerdo intra-anotador da 1,12 cuadros al
inicio y 1,55 al final; el detector da 1,14 y **0,97**. En el borde final es *más consistente
con la anotación que el anotador consigo mismo*, que es lo que mide ese número. Cuando
encuentra un golpe, lo ubica tan bien como una persona.

## Etapa 2: qué golpe es

Sobre los 184 golpes que el detector encontró:

| Real | n | → straight | → hook | → uppercut |
|---|---|---|---|---|
| straight | 109 | **97** | 12 | 0 |
| hook | 68 | **29** | 39 | 0 |
| uppercut | 7 | 2 | 4 | **1** |

Acierto de familia **0,745**, y con el mismo patrón que este proyecto ya documentó tres veces
por caminos independientes: **29 de 68 hooks llamados straight**. El anotador los distingue
con kappa 0,881; ningún descriptor 2D pasó de AUC 0,64; y el clasificador los confunde en la
misma dirección. Uppercut sigue marginal: 1 de 7.

Y este 0,745 está **inflado**: el clasificador vio la mitad de entrenamiento de este mismo
video. Sobre una fuente nueva su desempeño está medido en o por debajo de su línea de base.

## De punta a punta

De los **380 golpes anotados**, el sistema encuentra y clasifica bien el **36,1%**.

```
0,484  encontrados          (detector, honestamente fuera de distribución)
0,745  familia correcta     (clasificador, en distribución e inflado)
─────
0,361  de punta a punta
```

## Qué dice sobre dónde invertir

**El detector ya no es el cuello de botella en precisión.** 0,885 significa que casi nueve de
cada diez marcas son un golpe real. Lo que le falta es recall: se pierde la mitad.

**El clasificador sí lo es, y peor de lo que este número muestra.** Su 0,745 es en
distribución; sobre `sparring-3` sin haberla visto, sería ruido. Así que el sistema **hoy no
puede clasificar golpes en un video nuevo**, aunque sí puede encontrarlos: con detección
generalizando a 0,885 de precisión y clasificación que no generaliza, el producto sobre una
fuente nueva colapsa a la parte de detección.

Eso invierte la prioridad que traía el proyecto. La pregunta ya no es *cuándo hay un golpe*
—eso está resuelto a nivel de precisión y le falta recall— sino **si la familia del golpe es
aprendible en pose monocular**, que es exactamente lo que tres mediciones independientes
vienen sugiriendo que no.

## Reproducir

```bash
# etapa 1, en twinboxing_env
python tools/pipeline.py sparring-3-rounds --out pipeline/

# etapa 2, en boxtwin_mmaction
~/miniforge3/envs/boxtwin_mmaction/bin/python tools/clasificar_segmentos.py \
    pipeline/sparring-3-rounds.segmentos.json \
    --annot ../anotacion-sparring-3/annotations/sparring-3-rounds.annot.json \
    --secuencia ../anotacion-sparring-3/exports/sparring-3-rounds.sequence.npz \
    --config ../modelos/poseC3D_sparring3_indist.py \
    --checkpoint ../modelos/poseC3D_sparring3_indist.pth
```

Van separadas por un JSON porque viven en entornos distintos —torch 2.6 el detector, torch
2.1 pinneado el clasificador— y juntarlas pediría mover un pin por una razón que no es
técnica. El JSON además deja auditar los segmentos sin correr nada.
