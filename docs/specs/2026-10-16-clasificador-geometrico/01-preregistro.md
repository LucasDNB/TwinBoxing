# Pre-registro E4 a E6 · clasificador geométrico, 3D y rol por brazo

**Estado: aprobado por Lucas al lanzar `/noche-geometrico` el 15-10-2026.** El primer paso de
la noche commitea este archivo solo; su hash va en cada documento de resultados. Después de
ese commit no se edita.

Datos de base, iguales a E1 a E3: las siete fuentes anotadas, 1.047 golpes completos (amagues
y abortados afuera), una fuente afuera por vez (siete folds). Golpes por fuente: Sparring 114,
01-sparring 96, 02-sparring 62, 03-sparring 129, 04-sparring 120, sparring-3-rounds 381,
pacquiao_margarito 145.

## 1. Preguntas

1. **P5 (E4).** ¿Un modelo chico sobre features geométricas normalizadas generaliza mejor que
   PoseConv3D en familia? ¿Le gana en altura a la regla de C8?
2. **P6 (E5).** ¿Sumar features 3D (MotionBERT, marco del cuerpo) mejora la familia sobre E4, y
   en particular el hook?
3. **P7 (E6).** ¿El rol sacado del brazo del detector más la guardia acierta más que el
   clasificador de producción?
4. **Sub-hipótesis 1.** Ocho clases, con la misma lectura que el §6.5 del pre-registro del
   14-10 (`d359247`).

## 2. Configuraciones

| | Entrada por cuadro | Modelo |
|---|---|---|
| E4 | 20 features del brazo que golpea (`features_de`) + 6 del rival (§2.2 del diseño) = 26 | `ClasificadorGolpe` (diseño §3) |
| E4a | Solo las 20 del brazo | igual. Ablación sin umbral |
| E5 | Las 26 de E4 + 18 features 3D (diseño §4) = 44 | igual |
| F | Promedio de probabilidades de E4 y E3 (PoseConv3D con rival, `pred.pkl` del 15-10) | sin entrenamiento. Lectura secundaria sin umbral |
| E6 | Regla: rol = f(brazo del detector, guardia anotada) | sin entrenamiento |

Fijo para E4, E4a y E5, sin ajustar nada mirando la fuente de prueba:

1. Ventana: del inicio menos 8 cuadros al fin más 8, remuestreada a 32 pasos.
2. Arquitectura e hiperparámetros del diseño §3, tal cual: 60 épocas, sin early stopping, sin
   validación sobre la fuente de prueba.
3. Pesos de clase: inversa de la frecuencia en el fold de entrenamiento, normalizada a media 1,
   para tipo y para altura.
4. Semillas 0 a 4. Número primario: ensamble de las cinco (promedio de probabilidades) con los
   folds concatenados.
5. Estandarización ajustada solo sobre el fold de entrenamiento.

## 3. Métricas y lectura (fijadas ahora)

Las líneas de base salen de los JSON del 15-10 (`docs/experiments/datos/2026-10-15-clasificador-E*.json`):
E1 familia 0,656 total y por fold; E3 altura; siempre recto 0,632; siempre cabeza 0,50 de
balanceada.

1. **P5, familia.** E4 mejora a E1 si la exactitud de familia del ensamble supera a la de E1 en
   0,03 o más y E4 gana en al menos 5 de los 7 folds. Si no, no hay diferencia demostrada.
2. **P5, altura.** Primero se recalcula la regla de C8 (`altura_relativa_v1`, τ = 0,20, sin
   dato = error) sobre los mismos golpes completos de las cuatro fuentes de C8 (775, 128 al
   cuerpo). E4 le gana a la regla si su balanceada supera a la de la regla en 0,03 o más y
   gana en al menos 3 de las 4 fuentes. Al lado se dice si E4 cumple C8b (balanceada ≥ 0,75 y
   recall de cuerpo ≥ 0,60).
3. **P6.** E5 mejora a E4 en familia con la misma regla que el punto 1 (≥ 0,03 y 5 de 7). El
   recall de hook de E4 y E5 se reporta con Wilson; la expectativa declarada es que, si algo
   mejora, es el hook.
4. **P7.** Sobre los rectos anotados que el detector encontró en las cuatro sesiones de C8
   nivel 2, la regla mejora al clasificador si su exactitud de rol supera a la del
   clasificador en 0,05 o más. Se reporta además el acuerdo de brazo (detector contra
   anotación) sobre todos los golpes emparejados, y la regla con guardia "todos ortodoxos"
   como variante sin umbral. Advertencia de antemano: el detector de producción se entrenó con
   esas fuentes, así que el acuerdo de brazo es optimista.
5. **Sub-hipótesis 1.** Exactitud de ocho clases de E4 y E5 (tipo y altura del modelo; jab y
   cross por el rol anotado). Lectura a/b/c idéntica a la del 14-10: por debajo de 0,80 la
   parte de golpes no alcanza; en 0,80 o más no la confirma.
6. **Dispersión.** Media y desvío entre semillas de cada número primario. Si el desvío de la
   familia pasa de 0,03, cualquier "mejora" del punto 1 o 3 se informa como no concluyente
   aunque cumpla la regla.
7. **F y E4a.** Se reportan sin umbral, en una sección aparte. No deciden nada.

## 4. Lo que no se hace

1. No se cambia nada de §2 y §3 después de ver un número.
2. No se tocan anotaciones; huellas sha256 antes y después de cada fase.
3. No se reemplaza nada en producción: ni checkpoint, ni regla de altura, ni Fight-Card. Si E6
   pasa, el cambio en la Fight-Card es otro spec.
4. No se instala ni actualiza nada en ningún entorno. Si a E5 le falta algo, E5 queda como no
   realizado y se documenta.
5. No se usa la GPU si hay otro proceso de cómputo en ella (el modelo de E4 entrena en CPU si
   hace falta).

## 5. Lo que va a la tesis

Cap 7: E4 a E6 con sus números, lo que no pase incluido. Cap 8: la comparación con BoxMind como
referencia del estado del arte monocular. Declaración de uso de IA: Claude Code implementó y
corrió E4 a E6 por SDD durante la noche del 15-10.
