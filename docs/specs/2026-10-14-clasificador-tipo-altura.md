# Clasificador por tipo y altura: etiquetado asistido y reentrenamiento

14-10-2026 · pedido de Lucas: que el modelo clasifique cada tipo de golpe y si va a la cabeza o
al cuerpo, con el etiquetado y el entrenamiento a cargo de Claude Code en modo agente.

Orquestación: `/etiquetado-reentreno` (`.claude/skills/etiquetado-reentreno/SKILL.md`).
Scripts: `scripts/clasificador/`. Directorio de trabajo: `~/boxtwin/clasificador-2026-10/`.

**Estado: borrador.** Pasa a pre-registrado cuando los campos marcados [LUCAS] están
completos y el archivo está commiteado. Ninguna medición de este spec corre antes de ese
commit, y su hash va en cada documento de resultados.

---

## 1. Qué se pregunta

1. P1. ¿Claude, mirando tiras de cuadros a ciegas, le acierta a la anotación de Lucas lo
   suficiente como para preetiquetar material nuevo?
2. P2. ¿Cuánto acierta el clasificador de producción sobre una fuente que no vio? El 0,746
   de familia es en distribución y está inflado; nunca se midió dejando una fuente afuera con
   el modelo de siete fuentes.
3. P3. ¿Cambiar el espacio de clases a tipo por altura, sin lado, mejora la familia y además
   da la altura?
4. P4. ¿Sumar al rival en la entrada mejora la altura?

Lo que no se pregunta: el hook. Cinco mediciones independientes dicen que es proyección
monocular y no datos ni prior (`boxtwin-detector/docs/experiments/2026-09-07-hook-no-es-prior.md`).
La expectativa declarada de antemano es que su recall no se mueva en ninguna configuración.

## 2. De qué se parte

- Las siete fuentes ya tienen tipo y altura por golpe. El anotador obliga a elegir la altura,
  así que no es un valor por defecto. Golpes completos: 1.047 (593 recto a la cabeza, 223
  hook a la cabeza, 77 hook al cuerpo, 69 recto al cuerpo, 58 uppercut a la cabeza, 27
  uppercut al cuerpo). Para reentrenar no hace falta etiquetar nada.
- `boxtwin-annotator export --format mmaction` ya arma los espacios de 6, 12 y 14 clases, con
  o sin el rival (`--persons both`), y guarda en la meta el sha256 de la anotación.
- C8 (regla geométrica de altura) no pasó: balanceada 0,726 sobre 806 golpes de cuatro
  fuentes (`docs/experiments/2026-10-13-cabeza-cuerpo.md`).
- El etiquetado de Claude sirve para material sin anotar (el amateur, las sesiones del
  piloto), no para el reentrenamiento de este spec. Por eso primero se mide (E0).

## 3. Hallazgo a verificar antes de medir (H-F)

El config de producción (`modelos/poseC3D_7fuentes.py`) entrena con `Flip`, `flip_ratio=0.5`
y los índices izquierda/derecha intercambiados. Si la transformación no toca la etiqueta,
la mitad de los ejemplos entrena con el brazo cambiado: un jab espejado se ve como un cross
con la etiqueta de jab. Sería compatible con la distancia entre top-1 (0,534) y familia
(0,746) del modelo actual.

Se verifica leyendo código, sin correr nada:

1. La transformación `Flip` de poses en `test-poseConvBoxing/mmaction2/mmaction/datasets/transforms/`:
   ¿modifica `label`?
2. `boxtwin-detector/tools/clasificar_sesion.py` y el armado de la Fight-Card: ¿jab y cross
   salen del rol que predice el clasificador o del carril del detector más la guardia?

El resultado se escribe en el documento de E1 como hallazgo, con las líneas de código. No
cambia E1: E1 replica producción tal cual, justamente para medir lo que corre hoy. E2 y E3
no tienen lado en la etiqueta, así que el Flip deja de contradecirla en cualquier caso.

## 4. E0. Piloto del etiquetador

Muestra:

1. 150 golpes completos de las siete fuentes, estratificados por tipo y altura, mínimo 10 por
   celda (uppercut al cuerpo entra entero si tiene menos). Semilla 20261014. Con esa semilla
   la muestra queda: recto 63 cabeza y 15 cuerpo, hook 30 y 16, uppercut 14 y 12.
2. Una tira por golpe (`extraer_tiras.py anotado`): ocho cuadros del tramo más dos de margen,
   recortados alrededor de los dos peleadores, atacante en rojo, rival en celeste, brazo que
   golpea en amarillo. El brazo se da porque en el producto lo da el detector.

Cegado:

1. Ids opacos en orden aleatorio. La clave va a `clave/`, que el subagente `etiquetador-golpes`
   no puede abrir: un hook (`.claude/hooks/solo_tiras.py`) le bloquea toda lectura fuera de
   `tiras/` y `lote.jsonl` y toda escritura fuera de `respuestas/`. No tiene Bash.
2. El subagente no ve las anotaciones, el código ni este spec. Ve su guía de criterios, que
   está en su propio archivo (`.claude/agents/etiquetador-golpes.md`).
3. La guía la revisa Lucas antes del pre-registro [LUCAS]: tiene que decir lo mismo que
   aplicó él al anotar.

Lectura, fijada ahora:

1. Estricta: sin respuesta, `no_se` y `no_golpe` cuentan como error. La cobertura se reporta
   al lado, sin umbral.
2. Primarias: exactitud de familia y exactitud balanceada de altura, sobre los 150.
3. Al lado, sin umbral: balanceada de familia, kappa, recall de cuerpo con Wilson, todo por
   fuente, matrices de confusión.
4. Umbral para usarlo como preetiquetador [LUCAS confirma o cambia antes del commit]:
   familia ≥ 0,85 y altura balanceada ≥ 0,80. Propuesto porque la mitad de la anotación sería
   corregir: por debajo de esos valores corregir cuesta casi lo mismo que anotar.
5. Si pasa, se habilita E5. Si no, es un resultado negativo que va al Cap 7 como tal y no hay
   preetiquetado.

Una sola corrida sobre la muestra. Si se quiere repetir con otra guía o con otro dibujo de
tira, es otro experimento con otra muestra, no una segunda vuelta sobre esta.

## 5. E1 a E3. Reentrenamiento dejando una fuente afuera

| | Espacio de clases | Entrada | Variante de export |
|---|---|---|---|
| E1 | 6: tipo por rol (lead-rear), el de producción | atacante | `--classes 6 --label-space lead-rear --persons attacker` |
| E2 | 6: tipo por altura, sin lado | atacante | `--classes 12 --label-space side --persons attacker`, colapsado |
| E3 | 6: tipo por altura, sin lado | atacante y rival | `--classes 12 --label-space side --persons both`, colapsado |

De E1 a E2 cambia una cosa (el espacio de clases); de E2 a E3, una (el rival). Con
`--persons both` los dos peleadores se dibujan en el mismo volumen de mapas de calor, así
que el modelo no sabe quién es el atacante salvo por el movimiento. Se dice de antemano: E3
puede empeorar por eso.

Fijo para las tres:

1. Partición: una fuente afuera por vez, siete folds. La fuente de prueba no aparece en
   entrenamiento (`armar_folds.py` lo verifica y tiene prueba).
2. Config: `modelos/poseC3D_7fuentes.py` (producción, sin pesos de clase) con solo estos
   cambios (`generar_configs.py`, con prueba): pickle del fold, número de clases, semilla,
   `work_dir`, `load_from` absoluto, y sin validación durante el entrenamiento. Se entrenan
   las épocas del config y se prueba el último checkpoint. Elegir la mejor época mirando la
   fuente de prueba sería medir sobre lo que se usó para elegir.
3. Semillas 0, 1 y 2. Son 63 corridas. La campana corre primero todas las de semilla 0.
4. Sin pesos de clase en ninguna. Ya se sabe que arreglan el uppercut (0,200 a 0,520) y
   empeoran el hook; sumarlos acá cambiaría dos cosas a la vez. Queda para un E4 si hace falta.

## 6. Métricas y lectura

Número primario: ensamble de las tres semillas por fold (promedio de probabilidades), folds
concatenados (`evaluar.py`). La dispersión entre semillas se reporta al lado.

1. Familia: probabilidades sumadas por tipo. Exactitud, balanceada, recall por tipo, Wilson.
   Línea de base: siempre recto, y 1/3 de balanceada.
2. Altura (E2, E3): probabilidades sumadas por altura. Balanceada y recall de cuerpo con
   Wilson, en total y sobre las cuatro fuentes de C8. Línea de base: 0,50.
3. Eje (E1): exactitud de lead contra rear. Es el número que muestra si H-F pesa.

Lectura, fijada ahora:

1. P2 se contesta con la familia de E1, al lado del 0,746 en distribución.
2. P3, familia: E2 mejora a E1 si la diferencia del ensamble es ≥ 0,03 y E2 gana en al menos
   5 de los 7 folds. Si no, no hay diferencia demostrada, y se dice así.
3. P3, altura: criterio nuevo C8b, mismos umbrales que C8 (balanceada ≥ 0,75 y recall de
   cuerpo ≥ 0,60) sobre las cuatro fuentes de C8. No es C8 ni lo reemplaza: C8 midió una
   regla y quedó no aprobado. El denominador también es otro: C8 contó 806 golpes con
   amagues, abortados y sin dato como error; acá entran solo los completos (775, 128 al
   cuerpo). Las
   dos cosas se dicen al lado del número.
4. P4: E3 mejora a E2 en altura con la misma regla que el punto 2 (≥ 0,03 y 5 de 7 folds).
5. Sub-hipótesis 1. [LUCAS: copiar acá la definición textual del Cap 2 y decir qué número de
   esta campana la contesta.] Sin ese campo, el agente informa números y no escribe que la
   sub-hipótesis queda confirmada o refutada. La fecha de corte sigue siendo el 31/10.

## 7. Lo que no se hace

1. No se cambia nada de §4 a §6 después de ver un resultado. Una idea nueva es un
   experimento nuevo con fecha y commit propios.
2. No se tocan las anotaciones. Las huellas sha256 se toman antes de empezar y se verifican
   después de cada fase (`huellas.py`); si cambian, se frena.
3. Las etiquetas de Claude nunca entran a un conjunto de prueba.
4. No se reemplaza el checkpoint de producción. Si E2 o E3 pasan, se entrena el modelo final
   sobre las siete fuentes con las tres semillas y queda al lado; integrarlo a la Fight-Card
   es otro spec y lo decide Lucas.
5. No se entrena mientras corre una medición de tiempo (benchmark de perfiles, C5) ni con otro
   proceso de cómputo en la GPU. `correr_campana.sh` espera a que la placa esté libre.
6. No se actualiza nada en `boxtwin_mmaction`.

## 8. E5. Preetiquetado de material nuevo (solo si E0 pasa)

1. Tiras de una sesión del producto (`extraer_tiras.py sesion`), etiquetadas por el mismo
   subagente con la misma guía.
2. Sale un CSV de sugerencias por golpe (tiempo, brazo, tipo, altura, confianza) para que
   Lucas revise en el anotador o con la corrección de tipo de la web (O-01 del Cap 6). La
   etiqueta que entra a cualquier dato de entrenamiento es la de Lucas.
3. Sesgo de anclaje, declarado: si Lucas corrige sugerencias, su etiqueta puede quedar pegada
   a la sugerencia. Para medirlo, una de cada cinco tiras del lote va sin sugerencia y Lucas
   la anota de cero; se compara el acuerdo con y sin sugerencia.

## 9. Presupuesto y calendario

- E0: una tarde. 150 tiras se generan en minutos; el etiquetado en lotes de 25.
- E1 a E3: la primera corrida se cronometra (`correr_campana.sh --solo 1`) y se extrapola. Si
  la campana completa pasa de 14 horas, se frena y Lucas elige entre correr solo la semilla 0
  de las tres configuraciones o las tres semillas de dos.
- Fechas: pre-registro 15/10 (el mismo día que el benchmark de perfiles, que corre primero y
  sin la campana en la GPU); E0 15 o 16/10; campana las noches del 16 y 17/10; evaluación y
  documento el 18/10; Cap 7 y Declaración de uso de IA antes del 20/10. Corte de la
  sub-hipótesis 1: 31/10.

## 10. Lo que va a la tesis

1. Cap 7: E0 y E1 a E3 con sus números, lo que no pasó incluido. H-F como hallazgo si se
   confirma.
2. Cap 6: R-01 se actualiza con el número de E1 (generalización medida por primera vez con
   siete fuentes). Riesgo nuevo si se usa E5: sesgo de anclaje del preetiquetado.
3. Declaración de uso de IA: Claude Code corrió la campana en modo agente; si se usa E5, Claude
   propuso etiquetas que Lucas revisó.

## Anexo. Campos [LUCAS] a completar antes del commit

1. §4, guía de criterios del etiquetador: revisada y conforme, o con los cambios.
2. §4, umbrales de E0: 0,85 y 0,80, o los que fije.
3. §6.5, sub-hipótesis 1: definición textual y número que la contesta.
