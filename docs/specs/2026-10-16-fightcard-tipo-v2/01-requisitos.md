# Requisitos y aceptación

## 1. Requisitos funcionales

- RT1. Con `BOXTWIN_CLASIFICADOR=geometrico`, la etapa `clasificar` del worker corre un solo
  comando en `twinboxing_env` (`boxtwin-annotator clasificar <sesion>`) seguido del
  `boxtwin-annotator tipos <sesion>` de siempre. No invoca `boxtwin_mmaction`.
- RT2. Con `BOXTWIN_CLASIFICADOR=poseconv3d` o sin la variable, el comportamiento es el de hoy,
  byte a byte en los comandos que arma el worker. Es el valor por defecto hasta que Lucas
  encienda v2.
- RT3. Familia de cada golpe detectado: ensamble de las cinco semillas del modelo final de E4
  (promedio de probabilidades). Se guardan las tres probabilidades.
- RT4. Rol: adelantado o atrasado con `arm_role(guardia, brazo)`; un recto adelantado es jab y
  uno atrasado es cross. Hook y uppercut conservan el rol como dato, pero se muestran como hoy
  (sin mano en el nombre, decisión del 13-10).
- RT5. Guardia por lado: la del boxeador asignado a A o B en la sesión
  (`Boxeador.guardia`, ortodoxa o zurda). Sin boxeador o sin guardia: ortodoxa, con
  `guardia_supuesta: true` en `tipos.json` y en la Fight-Card.
- RT6. La altura no cambia: sigue saliendo de `enriquecer` con la regla de C8.
- RT7. Las correcciones del entrenador se preservan igual que hoy: `aplicar_tipos` no pisa un
  golpe corregido.
- RT8. Al cambiar la guardia de un boxeador o la asignación de boxeadores de una sesión en
  estado `listo`, y con v2 encendida, la API encola `clasificar` para las sesiones afectadas.
- RT9. La Fight-Card declara el método: `clasificador.metodo` = `geometrico-v1` o `poseconv3d`,
  el identificador del modelo, la exactitud de familia fuera de distribución que corresponde
  (0,727 para v2, con la referencia al documento) y, por lado, si la guardia fue supuesta.
- RT10. La web muestra, cuando la guardia de un lado fue supuesta, un aviso junto al tipo:
  "Jab y cross suponen guardia ortodoxa. Cargá la guardia de <nombre o A/B> para corregirlo."
  No se agrega ninguna pantalla nueva.

## 2. Requisitos no funcionales

- RNF1. La etapa `clasificar` v2 sobre una sesión de 120 s tarda menos de 10 s en CPU.
- RNF2. El modelo final no se versiona en git (`*.pt` está ignorado). Vive en
  `~/boxtwin/modelos/clasificador_geo_v1/` con un `meta.json` que tiene sha256 de cada archivo,
  commit, hiperparámetros, lista de features, estandarización y las métricas de E4 que se
  muestran.
- RNF3. Todas las suites existentes siguen en verde: anotador, detector, API, web (unidad,
  integración y contrato) y `scripts/clasificador*`.

## 3. Criterios de aceptación

| Id | Criterio | Cómo se verifica |
|---|---|---|
| CA1 | Las features del camino de producción son las mismas que las de E4 | Sobre los 1.047 golpes anotados, con la ventana anotada, la función de producción da el mismo `X` y `M` que `golpes.npz` (diferencia absoluta máxima ≤ 1e-5) |
| CA2 | La inferencia de producción reproduce las predicciones de E4 | Con un modelo de un fold guardado por `entrenar_final.py --excluir <fuente> --semillas 0`, la inferencia de producción sobre esa fuente da las mismas probabilidades que el `pred.npz` de E4 s0 de ese fold (≤ 1e-4). Una fuente basta; se elige Sparring |
| CA3 | El worker arma los comandos correctos con cada valor de la bandera | Prueba de `comando_de` con `geometrico`, `poseconv3d` y sin variable |
| CA4 | Guardia supuesta y reclasificación | Pruebas de API: sin guardia, `guardia_supuesta`; al hacer PATCH de la guardia o de la asignación, se encola `clasificar` solo si v2 está encendida y la sesión está `listo` |
| CA5 | El contrato admite los campos nuevos | `fightcard-0.3.schema.json` con los campos opcionales; las pruebas de contrato del anotador y de la web en verde con una Fight-Card v1 y una v2 |
| CA6 | Las correcciones sobreviven | Prueba: una Fight-Card con una corrección, reclasificada con v2, conserva la corrección y guarda `tipo_modelo_nuevo` |
| CA7 | De punta a punta en la estación | Una copia de una sesión real (la de `amateur_estatico` o una de C8 nivel 2) pasa por `clasificar` v2 y `tipos`: todos los golpes con familia, tiempo medido (RNF1), Fight-Card válida contra el contrato |
| CA8 | Nada cambió con la bandera apagada | Las suites existentes en verde sin tocar sus aserciones |

## 4. Medición pre-registrada: v2 de punta a punta (M1)

Se commitea con este archivo antes de la primera corrida y no se toca después.

Pregunta: sobre golpes que encontró el detector, ¿cuánto acierta v2 la familia y el tipo de
cuatro clases (jab, cross, hook, uppercut)?

1. Datos: copias de las cuatro sesiones de C8 nivel 2 (`~/boxtwin/mediciones/c8-nivel2/`),
   copiadas a `~/boxtwin/tipo-v2/m1/`, nunca las originales.
2. Clasificador fuera de distribución: para la sesión de cada fuente, el modelo final de E4
   entrenado sin esa fuente (`entrenar_final.py --excluir <fuente>`, cinco semillas). El
   detector sí vio esas fuentes, como en C8 nivel 2 y E6; se dice al lado.
3. Emparejamiento: como E6 (por peleador, sin separar por brazo, IoU 0,3, contra golpes
   completos).
4. Guardia: dos lecturas, la anotada del evento y todos ortodoxos (lo que pasa sin guardia
   cargada).
5. Métricas: exactitud de familia y de cuatro clases con Wilson, total y por fuente; recall por
   familia. Sin umbral: es la descripción de lo que el producto haría, no un criterio.
6. Comparación descriptiva: la familia de E4 sobre esos mismos golpes con ventana anotada (del
   `pred.npz` de E4) al lado, para ver cuánto cuestan las fronteras del detector.

## 5. Fuera de alcance

- Encender la bandera en la estación. Lo hace Lucas.
- Construir la imagen Docker. Se deja escrita la nota de que v2 la simplifica.
- Cambiar la altura o el detector.
- Reentrenar con más datos, cambiar la arquitectura o los hiperparámetros de E4.
