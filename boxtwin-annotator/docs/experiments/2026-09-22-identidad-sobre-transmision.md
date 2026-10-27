# La identidad automática sobre transmisión da 95,7%, el mejor de todas las fuentes, y el umbral de guante resulta depender del material

**22-09-2026 · `pacquiao_margarito`, 37,9 minutos y 9914 tracks · muestra visual independiente de 50, semilla 7.**

## Por qué

Las siete fuentes anteriores son gimnasio: cámara fija, cerca, poca gente. Una transmisión
es otro régimen — cortes de plano, repeticiones, primeros planos, público — y el sistema
estaba calibrado sobre el primero. Hacía falta saber si transfiere.

## Resultado

**44 de 46 tracks evaluables, 95,7%.** Es el mejor de todas las fuentes, y sobre el material
más difícil. De los 69 tracks de peleador de la anotación, 3 los descartan los filtros y 19
quedan sin decidir: fragmentos de pocos cuadros, con mediana de 8 recortes, que no juntan
guantes suficientes para votar ni coexisten con nadie ya repartido.

La asignación A contra B se sostiene a lo largo de los 37 minutos: A es Margarito, de
pantalón oscuro, y B es Pacquiao, de blanco y dorado.

## Los 462 que la anotación no cubre

La propuesta asigna rol a 508 tracks con el umbral de 0,30, y la anotación solo tiene 69
peleadores. Los 462 restantes no se pueden puntuar: la anotación de este video cubre la
identidad donde hizo falta y el resto lo marcó el lote por altura.

Se miraron **50 en el video**, muestra independiente y con semilla declarada. **46 son
peleadores legítimos**, o sea 92,0% con piso Wilson del 81,2%: al menos 374 de los 462 son
peleador. El 95,7% no escondía un problema — el sistema encuentra peleadores en muchos más
tracks de los que la anotación manual cubre.

Los cuatro fallos, que enseñan más que los aciertos:

| Track | Qué es | Fracción de guante |
|---|---|---|
| 12798 | **el árbitro** | **0,30**, clavada en el umbral |
| 10225 | un espectador entre fotógrafos | 1,00 |
| 1788 | el track se va a la pantalla gigante | 0,87 |
| 2000 | el track se va a un hombre de traje | 0,39 |

## El umbral de guante depende del material

Subirlo a 0,40 saca dos de esos cuatro sin costar nada:

| | 0,30 | 0,40 |
|---|---|---|
| Partición | 44/46 | **44/46** |
| Peleadores descartados | 3 | **3** |
| Peleadores sin decidir | 19 | **19** |
| Tracks asignados | 508 | 487 |
| Fallos en la muestra | 4 de 50 | **2 de 48** |
| **Piso Wilson** | 81,2% | **86,0%** |

Mejor calidad y mejor cobertura estimada a la vez: 379 peleadores legítimos estimados con 21
tracks menos asignados.

Pero en gimnasio es al revés: 0,30 deja las seis fuentes vivas y 0,40 mata una. **El valor
bueno depende del material, no solo del detector** —que ya lo había cambiado una vez, de 0,45
a 0,30 al reentrenarlo—. Queda en 0,30 porque seis de siete fuentes son de gimnasio, y el
flag documenta que en transmisión conviene 0,40.

## Lo que no se puede arreglar con un umbral

Los dos fallos que sobreviven a 0,40 son de otra naturaleza. El espectador 10225 tiene
fracción **1,00**: el detector está firmemente convencido, y ningún umbral de guante lo saca.
Su debilidad es que tiene 7 recortes. Y el 1788 es un problema de **tracking**, no de filtro:
el track cambia de sujeto a mitad de camino, así que ninguna propiedad agregada del track lo
describe bien.

## Una hipótesis que salió falsa

Se probó derivar el umbral de las dos poblaciones por el criterio de Otsu, como hace
`boxingvi_placas.py` con las placas de título. **No anda.** La distribución de fracciones no
es bimodal limpia y el corte lo domina la clase mayoritaria, que son los no peleadores: en
gimnasio dejaba 5 fuentes vivas en vez de 6, y en transmisión elegía 0,45 y bajaba de 44
aciertos sobre 46 a 42 sobre 44. El código se sacó del paquete.

## Reproducir

```bash
conda activate twinboxing_env
cd ~/Proyectos/TwinBoxing/boxtwin-annotator
python3 -m boxtwin.cli identidad-auto \
    ~/Proyectos/TwinBoxing/anotacion-pacquiao/videos/pacquiao_margarito.mp4 \
    --modelo ../boxtwin-guantes/modelos/guantes-v2.pt \
    --guardar-evidencia /tmp/pacquiao.json --umbral-guante 0.40 --contra-anotacion
```

La medición pesada se guarda con `--guardar-evidencia` y después los umbrales se barren con
`--evidencia` en segundos, sin volver a decodificar el video.
