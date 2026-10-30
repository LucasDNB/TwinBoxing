# E12 · El guante detectado como features del detector de golpes (pre-registro)

30-10-2026. El commit de este archivo es el pre-registro. Correcciones solo antes de la primera
corrida de S14 (`10-pasos.md`), anotadas como decisión.

## 1. La pregunta

El detector ve la mano solo a través del keypoint de muñeca. En la extensión de un golpe la
muñeca se borronea y su puntaje cae; con YOLO26 más (K3 de E8). El sistema ya tiene una segunda
fuente de la posición de la mano que no depende del estimador de pose: el detector de guantes
`guantes-v2.pt`, que hoy solo se usa para la identidad y solo uno de cada cinco cuadros.

**¿Sumar la posición del guante detectado, cuadro por cuadro, mejora el F1 del detector de
golpes sobre la línea v8?**

| Id | Pregunta | Tipo |
|---|---|---|
| P16a | F1 por evento del detector, v8+G contra v8 (base de E8) | con regla |
| P16b | Recall del detector en los golpes cuya muñeca está bajo umbral en el pico de la extensión | sin umbral, dirección declarada |
| P16c | Cobertura del guante: fracción de cuadros-carril con guante asignado a esa mano, por fuente | descriptiva |
| P16d | Costo: ms por cuadro de la detección de guantes por cuadro, y su proyección sobre C5 | sin umbral |
| P16e | Familia de E4 con las mismas features de guante (E4+G) | informativa, solo si sobra noche |

## 2. Extracción del guante por cuadro (fija desde ahora)

1. En cada cuadro de las siete fuentes, para cada peleador asignado (A y B según la anotación
   de `raiz-v8`), `DetectorGuantes.detectar_con_color(frame, caja)` sobre su caja, como en
   `analizar` pero **sin saltear cuadros** y sin el filtro de aislamiento (el aislamiento es
   para el color, no para la posición).
2. `_solo_los_propios` con `max_distancia_muneca` 1,2 (el de A): se queda con los guantes cerca
   de donde la pose dice que está la mano de esa persona.
3. **A qué mano va.** Cada guante que queda se asigna a la mano cuya posición esperada
   (`derive_gloves`) está más cerca. Si las dos manos esperadas no están disponibles, la que
   esté; si ninguna, el guante se descarta. Dos guantes a la misma mano: el de mayor confianza.
4. Se guarda por fuente en `~/boxtwin-mejoras/e12/guantes/<fuente>.npz`: `(peleador, T, mano,
   [x, y, conf])` en píxeles, con conf 0 si no hay guante. Reanudable por fuente.

Sin cambiar el umbral de confianza del detector de guantes ni reentrenarlo.

## 3. Las features nuevas

El centro del guante entra al esqueleto como dos keypoints más (índices 17 y 18, izquierdo y
derecho) con su confianza como puntaje, y `PARES_LR` suma el par (17, 18): así el espejado por
brazo de `features_de` lo trata igual que a la muñeca, sin una rama aparte.

Cinco features por brazo, centradas y escaladas como las otras 20:

| Feature | Qué |
|---|---|
| `guante_x`, `guante_y` | centro del guante de ese brazo, relativo |
| `guante_ok` | 1 si hay guante asignado a esa mano en ese cuadro |
| `d_guante_x`, `d_guante_y` | derivadas, con la misma regla de vecinos de `features_de` |

**La máscara.** Un cuadro es usable si hombros y caderas están bien y **(codo y muñeca están
sobre umbral, o hay guante en esa mano)**. Si la muñeca está bajo umbral y el guante no, el
cuadro sale como hoy. Si la muñeca está bajo umbral y el guante sí, las features de muñeca y
codo van en 0 y el guante lleva la mano. Esto son dos cambios (features y máscara) y se declaran
como una sola variable, "la fuente de la posición de la mano": sin el cambio de máscara el
guante no podría entrar justo en los cuadros que importan.

La TCN no cambia salvo la dimensión de entrada (20 a 25). Hiperparámetros, semillas, folds y el
umbral 0,80 de decodificación, los de E8.

## 4. Controles

| Id | Qué | Si falla |
|---|---|---|
| K-reuso | Con el guante apagado (`--guantes` sin pasar), sha256 de los tensores igual a los de la base de E8 | igual que en E8b |
| K-mano | En 200 cuadros de sparring-3-rounds elegidos al azar (semilla 0) con guante asignado, Lucas marca si el guante va a la mano correcta; ≥ 90% | si no, el criterio de §2.3 está mal y E12 no corre hasta revisarlo, con la corrección como decisión |
| K-golpes | Los mismos eventos en las dos condiciones | E12 no se lee |

## 5. Reglas de lectura

**P16a**: la regla de P12a (+0,03 y 5 de 6 folds comparables, ensamble de cinco semillas, umbral
0,80). Al lado: recall y precisión, y la curva de 0,70 a 0,90.

**P16b**: sobre los golpes anotados, el cuadro de pico es el de máxima distancia muñeca-centro
de hombros dentro del segmento anotado, con la pose de v8 y sin filtrar por puntaje. Se separan
los golpes con muñeca bajo 0,3 en ese cuadro. Recall del detector en ese subconjunto, por condición. Dirección declarada:
v8+G mayor.

**P16d**: mediana y p95 de ms por cuadro de la extracción de §2 en la 2080, y la proyección
sobre C5 (hoy 2,741× en el ensayo, 1,74–1,86× proyectado): si la suma pasa de 2×, se declara que
la mejora no entra en el producto **sin** muestrear (por ejemplo uno de cada dos cuadros), y
eso sería otro experimento.

## 6. Qué decide cada resultado

- **Mejora y entra en el tiempo**: candidato para A después del 17-11, con C5 medido otra vez.
  Cap 7 informa el recall nuevo.
- **Mejora y no entra en el tiempo**: Cap 7 lo informa como resultado de investigación; Cap 8
  como línea, con el costo.
- **No mejora**: Cap 7 en una línea; si P16b tampoco se mueve, refuerza que la pérdida en la
  extensión no es solo de la muñeca sino del golpe entero borroso, y apunta a E15 y a cámaras a
  más fps.

## 7. Lo que E12 no separa

- El efecto de la máscara del efecto de las features (se declaran juntas, §3).
- Un guante de otro detector o reentrenado.
- Guantes en cuadros sin peleador asignado.

## 8. Auditoría

Como E8b §7, más: recálculo con numpy de la asignación de mano en una fuente, revisión de
K-mano y del registro de tiempo de P16d.

## 9. Salidas

`docs/experiments/<fecha>-e12-guantes-en-detector.md`, JSON, entrada del README y
`docs/tesis/<fecha>-e12-cambios.md`.
