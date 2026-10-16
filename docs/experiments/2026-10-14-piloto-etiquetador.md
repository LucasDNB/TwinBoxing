# E0 · Piloto del etiquetador a ciegas (tipo y altura)

14-10-2026 · spec: `docs/specs/2026-10-14-clasificador-tipo-altura.md` §4, pre-registrado en
`d359247e39a9184880b85a545d83c7bc3c228931` · scripts `scripts/clasificador/extraer_tiras.py` y
`scripts/clasificador/comparar_etiquetas.py` · subagente `.claude/agents/etiquetador-golpes.md` ·
datos en `datos/2026-10-14-piloto-etiquetador.json`

## 1. Qué se mide (pre-registrado en el spec)

- P1: ¿Claude, mirando tiras de cuadros a ciegas, le acierta a la anotación de Lucas lo
  suficiente como para preetiquetar material nuevo?
- Muestra: 150 golpes completos de las siete fuentes, estratificados por tipo y altura, semilla
  20261014. Celdas: recto 63 cabeza y 15 cuerpo, hook 30 y 16, uppercut 14 y 12.
- Una tira por golpe: ocho cuadros, atacante en rojo, rival en celeste, brazo que golpea en
  amarillo. El brazo se da porque en el producto lo da el detector.
- Lectura estricta: sin respuesta, `no_se` y `no_golpe` cuentan como error.
- Primarias: exactitud de familia y exactitud balanceada de altura, sobre los 150.
- Umbral para preetiquetar: familia ≥ 0,85 y altura balanceada ≥ 0,80. Si no pasa, es un
  resultado negativo que va al Cap 7 y no hay preetiquetado (E5).
- Una sola corrida sobre esta muestra.

## 2. Lo que se fijó antes de correr

Nada fuera del spec. El etiquetado se hizo en seis tandas de 25 ids (T0001 a T0150) con el
subagente `etiquetador-golpes`, cada una con un pedido que decía solo el directorio del lote,
el rango de ids y dónde escribir. Ninguna tanda supo de las otras.

## 3. Resultado

**No pasa.** Las dos primarias quedan por debajo del umbral.

| Métrica | Valor | Wilson 95% | Umbral |
|---|---|---|---|
| Familia, exactitud | 0,400 (60/150) | 0,325–0,480 | ≥ 0,85 |
| Altura, balanceada | 0,644 | — | ≥ 0,80 |

Al lado, sin umbral:

- Cobertura: tipo 0,773, altura 0,887.
- Familia: balanceada 0,315, kappa 0,113. Línea de base: siempre recto 0,520 de exactitud y
  1/3 de balanceada.
- Recall por tipo: recto 0,564 (44/78, Wilson 0,454–0,669), hook 0,304 (14/46, 0,191–0,448),
  uppercut 0,077 (2/26, 0,021–0,241).
- Altura: recall de cabeza 0,776 (83/107), recall de cuerpo 0,512 (22/43, Wilson
  0,368–0,654), kappa 0,358. Línea de base 0,50.

Matriz de confusión de familia (filas: anotado; columnas: etiquetador):

| | recto | hook | uppercut | no_se | no_golpe |
|---|---|---|---|---|---|
| recto (78) | 44 | 17 | 2 | 14 | 1 |
| hook (46) | 19 | 14 | 4 | 9 | 0 |
| uppercut (26) | 3 | 11 | 2 | 10 | 0 |

Matriz de altura:

| | cabeza | cuerpo | no_se |
|---|---|---|---|
| cabeza (107) | 83 | 13 | 11 |
| cuerpo (43) | 15 | 22 | 6 |

Por fuente:

| Fuente | Golpes | Al cuerpo | Cobertura tipo | Familia (Wilson 95%) | Altura balanceada | Recall cuerpo |
|---|---|---|---|---|---|---|
| 01-sparring | 17 | 5 | 0,765 | 0,529 (0,310–0,738) | 0,417 | 0,000 |
| 02-sparring | 5 | 0 | 0,800 | 0,400 (0,118–0,769) | 1,000 | — |
| 03-sparring | 13 | 1 | 0,846 | 0,385 (0,177–0,645) | 0,458 | 0,000 |
| 04-sparring | 21 | 5 | 0,762 | 0,238 (0,106–0,451) | 0,544 | 0,400 |
| Sparring | 16 | 2 | 0,563 | 0,313 (0,142–0,556) | 0,571 | 0,500 |
| pacquiao_margarito | 16 | 4 | 0,938 | 0,500 (0,280–0,720) | 0,792 | 0,750 |
| sparring-3-rounds | 62 | 26 | 0,774 | 0,419 (0,305–0,543) | 0,683 | 0,615 |

Con muestras de 5 a 21 golpes por fuente, ninguna diferencia entre fuentes se puede afirmar.

## 4. Lectura

El etiquetador no se separa del azar en familia. No solo no llega a 0,85: su balanceada
(0,315) queda por debajo de 1/3 y su kappa es 0,11. Decir siempre recto le ganaría en
exactitud (0,520 contra 0,400).

La confusión tiene una forma clara. El hook se lee como recto (19 de 46), que es la trampa
monocular que la guía ya avisaba, y el uppercut se lee como hook (11 de 26). El recto tampoco
sale limpio: 17 de 78 se leen como hook. Esto no es solo un sesgo hacia una clase. En las tiras,
la trayectoria de la mano no le alcanza para separar las tres familias.

La altura está por encima del azar (0,644 contra 0,50, kappa 0,36), pero lejos de 0,80. El
recall de cuerpo (0,512) es el que la tira abajo.

La cobertura pesa: 34 de 150 tipos quedaron en `no_se` o `no_golpe`. Los reportes de las
tandas apuntan a lo mismo: atacante de espaldas, clinch, cuerdas tapando el brazo, pose que
salta al otro peleador y tomas lejanas de transmisión. Sobre los 116 tipos respondidos la
exactitud de familia es 0,517 (60/116). Esa lectura es posterior a ver el número, no es la
pre-registrada, y tampoco pasaría.

## 5. Auditoría

`auditor-medicion`, 14-10-2026, sobre `resultado.json` y el lote:

| # | Resultado | Evidencia |
|---|---|---|
| 1 | PASA | Spec commiteado en d359247 antes que todo el lote: tiras, clave, respuestas y resultado son posteriores. |
| 2 | PASA | `huellas.py verificar`: anotaciones intactas. |
| 3 | N/A al auditar | No había documento todavía; las cifras coinciden con `resultado.json`. |
| 4 | N/A al auditar | Umbrales usados: 0,85 y 0,80, los pre-registrados. |
| 5 | PASA | 150 filas en la clave, todas `completeness == "full"`; tipo, altura, peleador, lado y cuadros coinciden con la anotación; celdas iguales a §4. |
| 6 | PASA | Recálculo independiente: familia 60/150 = 0,4000 (Wilson 0,3250–0,4800), altura (83/107 + 22/43)/2 = 0,6437; coinciden balanceada, kappas, coberturas y las siete fuentes. |
| 7 | PASA | 150 respuestas, ninguna fuera del lote ni faltante; el brazo de `lote.jsonl` coincide con la anotación en las 150. |
| Cegado | PASA | Cada etiquetador leyó solo `lote.jsonl` y sus 25 tiras; sin Bash; los únicos bloqueos del hook fueron un Glob sobre `respuestas/` y un Write con ruta equivocada, ningún intento sobre la clave. |

Observaciones del auditor, que no son fallas:

1. El número de cuadro va en blanco sobre la imagen (`extraer_tiras.py:308`) y en algunas
   tiras no se lee: en T0002 el etiquetador lo anotó. Una heurística de brillo marca 73 de
   150 tiras con al menos la mitad de los rótulos sobre fondo claro. Puede haber pesado.
2. Los ocho cuadros se toman parejos sobre el tramo más los dos de margen de cada lado, no
   ocho del tramo más el margen. El spec admite las dos lecturas y el código es anterior al
   pre-registro. En golpes largos puede saltearse el cuadro de máxima extensión.

## 6. Cegado

Copiado tal cual de `<trabajo>/lotes/canario/veredicto.md`:

> ## Cegado: canario
>
> Veredicto: verde. Lo dio Lucas el 14-10-2026 con esta evidencia, sin una segunda prueba.
>
> 1. Al subagente `etiquetador-golpes` se le pidió leer `lotes/canario/clave/clave.csv`. Se negó
>    por lo que dice su guía y nunca llamó a Read sobre la clave. O sea que el hook no llegó a
>    frenar una lectura de la clave: no hubo intento.
> 2. El hook `.claude/hooks/solo_tiras.py` sí corre dentro del subagente: en esa misma corrida
>    le bloqueó 17 llamadas a `SubagentHandback`. Los hooks de subagente están activos.
> 3. Prueba manual del hook: Read de `clave/clave.csv` y de `tiras/../clave/clave.csv`
>    bloqueados (código 2); Read de una tira permitido (código 0).
>
> Después del canario se permitió `SubagentHandback` en el hook (no abre archivos), porque el
> bloqueo dejaba al etiquetador sin poder reportar. El caso está en `test_hook_cegado.py`.

## 7. Consecuencias

- P1 se contesta que no: el etiquetador a ciegas no sirve como preetiquetador con estas tiras.
  Va al Cap 7 como resultado negativo.
- E5 (`preetiquetar`) no se habilita.
- E1 a E3 no dependen de esto y siguen como estaban.
- Repetir con otra tira (rótulos legibles, cuadros densos en el tramo) o con otra guía es otro
  experimento, con su propia muestra y su propio pre-registro. Esta muestra ya se miró.
