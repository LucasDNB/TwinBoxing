# E7 · Pre-registro · 3D de 4D-Humans en lugar de MotionBERT

**Estado: propuesto el 24-10-2026, pendiente de aprobación de Lucas.** Se commitea aprobado,
solo, antes de correr cualquier parte; su hash va en el documento de resultados. Después de ese
commit no se edita: un cambio que obligue el entorno se anota en `TAREAS.md` con el motivo,
antes de correr la parte afectada.

Datos de base, iguales a E4 y E5: las siete fuentes anotadas, 1047 golpes completos, una fuente
afuera por vez (siete folds). Referencias ya medidas (`docs/experiments/2026-10-16-clasificador-geometrico.md`):
E4 familia 0,727, recall de hook 0,620; E5 familia 0,738, recall de hook 0,590.

## 1. Preguntas

1. **P11a.** Con las mismas 16 features 3D en el marco del cuerpo de E5, ¿el 3D de HMR2.0
   (4D-Humans) mejora la familia sobre el de MotionBERT? Es la pregunta que aísla el estimador.
2. **P11b.** ¿E7 mejora la familia sobre E4, que no usa 3D?
3. **Sub-hipótesis 1.** Ocho clases de E7, con la lectura a/b/c del 14-10 (`d359247`).

La expectativa declarada, como en E5: si algo mejora, es el hook.

## 2. Condiciones de uso

4D-Humans tiene licencia MIT, pero su modelo de cuerpo es SMPL, con licencia solo para
investigación científica no comercial (`smpl.is.tue.mpg.de/modellicense.html`, verificado el
24-10; el uso comercial se licencia aparte por Meshcapade). Por eso:

1. E7 es académica. Ningún modelo ni feature de E7 entra al producto. Si E7 pasa, llevarlo al
   producto es otro spec y empieza por la licencia comercial de SMPL o por un estimador sin SMPL.
2. Ni el modelo SMPL, ni los pesos de HMR2, ni las poses, ni las features entran a git. Todo vive
   en `~/boxtwin/e7-4dhumans/`.
3. El código de 4D-Humans se usa desde un clon externo en un commit fijo y no se copia.
4. La tesis declara en una línea el uso de SMPL bajo su licencia de investigación.

## 3. Configuraciones

| | Entrada por cuadro | Modelo |
|---|---|---|
| E4 | 26 features 2D (sin cambios, se reutilizan sus `pred.npz`) | `ClasificadorGolpe` |
| E5 | 26 + 16 features 3D de MotionBERT (sin cambios, se reutilizan) | igual |
| E7 | 26 + las mismas 16 features 3D, calculadas sobre HMR2.0b | igual |

Fijo para E7, igual a E5 salvo el estimador:

1. Ventana: del inicio menos 8 cuadros al fin más 8, remuestreada a 32 pasos.
2. Arquitectura, hiperparámetros, pesos de clase, estandarización y semillas 0 a 4 de E5, sin
   cambios. Número primario: ensamble de las cinco semillas, folds concatenados.
3. Las 16 features con `features3d.features_de` sin modificar.

Lo que cambia, todo igual en los siete folds:

- **E1. Estimador.** HMR2.0b, el checkpoint publicado por defecto, en el commit del clon que
  quede anotado en `TAREAS.md` antes de correr. Un cuadro por vez: HMR2 no tiene contexto
  temporal y no se suaviza nada después (MotionBERT usa 243 cuadros; es parte de lo que se
  compara).
- **E2. Recorte.** La caja de cada cuadro sale de los keypoints COCO del atacante con score
  ≥ 0,3 (mínimo y máximo), después del relleno de huecos de hasta 10 cuadros de E5. HMR2 la
  agranda con su `ViTDetDataset` por defecto. No se usa el detector de 4D-Humans.
- **E3. Articulaciones.** Las 24 de SMPL, sacadas del modelo con los parámetros que predice
  HMR2, llevadas al orden H36M que espera `features3d`: pelvis ← 0, cadera derecha ← 2, cadera
  izquierda ← 1, tórax ← 12 (cuello, a la altura de los hombros, como el tórax de H36M),
  hombros ← 17 y 16, codos ← 19 y 18, muñecas ← 21 y 20. Las demás articulaciones H36M no las
  usa `features3d` y quedan en cero.
- **E4. Huecos.** Un cuadro sin caja válida queda con máscara 0, como en E5.

## 4. Lectura (fijada ahora)

Regla de familia de E4: mejora si la exactitud del ensamble supera a la base en 0,03 o más y
gana en al menos 5 de los 7 folds. Si el desvío entre semillas de la familia de cualquiera de los
dos pasa de 0,03, una mejora se informa como no concluyente.

1. **P11a.** E7 contra E5 con esa regla. El recall de hook de los dos va con Wilson.
2. **P11b.** E7 contra E4 con la misma regla.
3. **Sub-hipótesis 1.** Ocho clases de E7 (tipo y altura del modelo; jab y cross por el rol
   anotado), lectura a/b/c.
4. **D9 de E4.** Familia ≥ 0,90 u ocho clases ≥ 0,80 obliga a buscar fuga antes de escribir.
5. **Sin umbral.** Altura balanceada de E7 sobre las cuatro fuentes de C8 al lado de E4 y E5;
   cobertura de pasos con 3D; vibración del 3D (mediana por fuente del módulo de la segunda
   diferencia de la muñeca en el marco del cuerpo) para E5 y E7; tiempo de cómputo. El
   coeficiente de variación de los huesos de E5 no se repite: con SMPL el esqueleto es
   paramétrico y ese número no informaría nada.

## 5. Lo que no se hace

1. No se cambia nada de §3 y §4 después de ver un número.
2. No se tocan anotaciones, ni el trabajo de E4 y E5, ni el checkpoint de producción.
3. No se reentrena E4 ni E5: se comparan sus `pred.npz` del 16-10.
4. No se usa BoxingWeb.
5. No se instala nada en `boxtwin_mmaction` ni en `twinboxing_env`.

## 6. Lo que va a la tesis

Cap 7: E7 con su número, pase o no, al lado de E5. Cap 8: si P11a pasa, el estimador 3D es la
palanca y la línea futura nombra la licencia de SMPL como condición; si no pasa, el resultado de
BM-0 contra BM-0b se explica por identidad, 2D o dominio y no por el estimador 3D.
