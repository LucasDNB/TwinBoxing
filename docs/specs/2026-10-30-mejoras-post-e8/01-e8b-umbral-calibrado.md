# E8b · YOLO26 con el umbral de puntaje calibrado (pre-registro)

30-10-2026. El commit de este archivo es el pre-registro. Se puede corregir hasta antes de la
primera corrida de S5 (`10-pasos.md`), con la corrección anotada como decisión; después, no se
toca.

## 1. La pregunta

E8 midió YOLO26 con el umbral de puntaje pensado para YOLOv8 (`features.MIN_SCORE` = 0,3) y
declaró que no se puede escribir "YOLO26 estima peor la pose" (§5.7). E8b separa eso: **¿YOLO26
con un umbral calibrado a su propia distribución de puntajes iguala o supera a YOLOv8?**

| Id | Pregunta | Tipo |
|---|---|---|
| P13a | F1 por evento del detector, YOLO26 calibrado contra la base v8 de E8 | con regla |
| P13b | Familia de E4, YOLO26 calibrado contra la base v8 de E8 | con regla |
| P13c | Cuánto recupera la calibración respecto de YOLO26 con 0,3 (E8) | sin umbral, dirección declarada |
| P13d | Cuadros-carril usables y pasos usables de E4 por condición | sin umbral; es el mecanismo |

## 2. La regla de calibración (fija desde ahora)

**Igualar la fracción por articulación.** Para cada articulación que usan las features (nariz,
hombros, codos, muñecas, caderas; los pares izquierda y derecha juntos), el umbral de YOLO26 es
el cuantil de sus puntajes que deja **la misma fracción de keypoints aceptados** que tiene v8
con 0,3 en esa articulación.

- Población: todos los keypoints de las detecciones asignadas a A o a B en los siete cachés de
  `raiz-v8` (para la fracción objetivo) y de `raiz-y26` (para el cuantil), sobre los cuadros del
  denominador común de P12c (los que las dos condiciones tienen emparejados). Sin etiquetas de
  golpe: la calibración no ve la evaluación.
- Se calcula una vez, sobre las siete fuentes juntas, y se guarda en
  `~/boxtwin-mejoras/e8b/umbrales.json` con los cuantiles, las fracciones y el sha256 de los
  cachés. Es transductiva sin etiquetas, y se declara: un producto haría lo mismo sobre su
  propio material de calibración.
- Cinco umbrales (nariz, hombro, codo, muñeca, cadera). Si alguno sale por encima de 0,3, se
  usa igual: la regla es igualar fracciones, no bajar umbrales.
- La base v8 no cambia: sus cinco umbrales son 0,3 por definición.

Por qué por articulación y no uno solo sobre la muñeca: K3 midió la muñeca, pero la máscara de
`features_de` exige hombros, caderas, codo y muñeca, y la cabeza de YOLO26 puede desplazar cada
distribución distinto. Un solo umbral calibrado en la muñeca podría dejar a los hombros sub o
sobre filtrados.

## 3. Condiciones y controles

| | Base v8 | Y26 (E8) | Y26-cal (nueva) |
|---|---|---|---|
| Caché | `raiz-v8` | `raiz-y26` | `raiz-y26` |
| Umbrales | 0,3 | 0,3 | los de §2 |
| Corridas | las de E8, reusadas | las de E8, reusadas | nuevas: `folds` (5 semillas, 7 folds) y E4 (5 semillas, 7 folds) |

| Id | Qué | Si falla |
|---|---|---|
| K-reuso | Con los umbrales en 0,3, el código nuevo arma tensores del detector y `golpes.npz` de E4 con sha256 igual a los de E8, en las dos condiciones de E8 | E8b no corre hasta encontrar la diferencia; si no se encuentra, se re-entrenan las tres condiciones con el código nuevo y se declara |
| K-cal | Con los umbrales de §2, la fracción aceptada de cada articulación en Y26-cal queda a ±0,005 de la de v8 | la implementación de §2 está mal |
| K-mec | Cuadros-carril usables de Y26-cal, contra 147788 (base) y 111472 (Y26) | sin umbral: se informa; si no se acerca a la base, la calibración no está actuando donde K3 decía |

## 4. Reglas de lectura

**P13a.** Igual que P12a de E8: F1 por evento del ensamble de cinco semillas en el umbral de
decodificación 0,80, promedio sobre los seis folds comparables. **Mejora** si Y26-cal supera a
la base por 0,03 o más y gana en 5 de 6 folds. **Iguala** si queda a menos de 0,03 en
cualquier dirección. **Peor** si queda debajo por 0,03 o más. Al lado, sin decidir: el F1 en el
mejor umbral de cada condición y la curva de 0,70 a 0,90 (E8 mostró el pico de Y26 en 0,85).

**P13b.** La regla de E7 y E8: familia del ensamble, 1047 golpes, folds concatenados; **mejora**
si supera a la base por 0,03 o más y gana en 5 de 7 folds, con desvío entre semillas ≤ 0,03.
Mismas categorías de iguala y peor.

**P13c.** Diferencia Y26-cal menos Y26 en F1 y en familia. Dirección declarada: positiva. Si es
positiva y P13a o P13b no mejoran, la lectura es "el umbral explicaba parte de la caída".

**P13d.** Por fuente: cuadros-carril usables y fracción de pasos usables de E4; la correlación
por fuente entre pasos recuperados y familia recuperada, como el 0,82 de E8.

## 5. Qué decide cada resultado

- **Y26-cal mejora o iguala en P13a y P13b**: la frase del Capítulo 7 cambia de "actualizar el
  estimador de pose no mejoró el sistema" a "con el umbral calibrado, YOLO26 iguala (o supera) a
  YOLOv8 sobre este material", y la Tabla 4.5 lo dice. Cap 8: evaluar YOLO26 para A después del
  17-11, con C5 medido otra vez; YOLO26 es 8,4% más rápido y usa 36% menos memoria (P12e), lo
  que importa para el modo LIVE. La regla de §2 pasa a ser la receta para cualquier estimador
  nuevo (E9 la hereda).
- **Y26-cal sigue peor**: la frase de E8 queda y se le suma que la calibración no alcanza. La
  regla de §2 igual se hereda en E9, porque es lo que hay que hacer con cualquier estimador.
- **Mezclado**: se informa tal cual.
- Nada cambia A, ni el 0,3 de A.

## 6. Lo que E8b no separa

- Si una calibración por fuente o por peleador funcionaría mejor: es una sola global.
- Si conviene recalibrar también el umbral de decodificación 0,80: se informa la curva, no se
  decide con ella.
- El puente sigue sesgando a favor de v8 (E8 §5.2).

## 7. Auditoría

Auditor independiente (`.claude/agents/auditor-medicion.md`) antes del texto:

1. Commit de este archivo anterior a la primera corrida de S5.
2. `umbrales.json` recalculado con numpy a partir de los cachés, sin importar el código de T2.
3. K-reuso, K-cal y K-mec.
4. Las corridas de Y26-cal con la misma config que las de Y26 de E8 salvo los umbrales y el commit.
5. Recálculo de P13a y P13b; la regla aplicada como está escrita.
6. Originales y cachés de E8 con el mismo sha256 antes y después (CM3).

## 8. Salidas

`docs/experiments/<fecha>-e8b-umbral-calibrado.md`, `docs/experiments/datos/<fecha>-e8b-*.json`,
entrada del README raíz y `docs/tesis/<fecha>-e8b-cambios.md` (Caps 4, 7 y 8).
