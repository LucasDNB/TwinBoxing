# IE6 · Identidad estable: errores silenciosos, cola de revisión y usuario simulado

3-11-2026. Pre-registro: `docs/specs/2026-11-02-identidad-estable/00-README.md` §6 (commit
`5c88ea3`, 2-11). Código medido: `e93fa9d` (rama `identidad-estable-02-11`). Script:
`scripts/identidad/medir_identidad_estable.py`. JSON:
`docs/experiments/datos/2026-11-03-identidad-estable.json`. Diagnóstico aparte, no
pre-registrado: `scripts/identidad/diagnostico_identidad_estable.py`, JSON
`docs/experiments/datos/2026-11-03-identidad-estable-diagnostico.json`.

**Material de desarrollo: no confirma nada.** Son las seis fuentes de gimnasio de C1, donde se
diagnosticó el problema y se diseñó E11.

**En una línea: con la regla de §6, ninguna condición pasa E1 ni E2. El producto se declara no
estable en identidad, se elige `base` y las dos banderas quedan apagadas. El 98% de los errores
silenciosos está en una sola fuente, 03-sparring, donde los perfiles de color no separan y
ningún detector de sospecha puede dispararse.**

## 1. Auditoría antes de leer

Precisión y cobertura automáticas por cuadro, contra E11 S10 v2 (`~/boxtwin-mejoras/e11/desarrollo.json`):

| Condición | IE6 | E11 | Coincide |
|---|---|---|---|
| `base` | 0,850283 / 0,788429 | `sin_corte` 0,850283 / 0,788429 | sí, todos los decimales |
| `corte` | 0,922814 / 0,872321 | `con_corte` 0,922814 / 0,872321 | sí, todos los decimales |

Mismo denominador: 82 809 cuadros de peleador. Pruebas en verde antes de medir: anotador 807,
API 346, `scripts/identidad` 8, `scripts/mejoras` 63, web entrenador lint, tipos, 261 vitest y
build (este con Next 16.4.4, ver §7).

## 2. Resultados por condición (seis fuentes, 24,4 min de video)

| | `base` | `corte` | `corte+estricta` |
|---|---|---|---|
| Precisión automática | 0,850 | 0,923 | 0,973 |
| Cobertura automática | 0,788 | 0,872 | 0,562 |
| Errores silenciosos | 1275 (1,540%) | 1284 (1,551%) | 1284 (1,551%) |
| Peor fuente | 12,84% (03) | 12,84% (03) | 12,84% (03) |
| Errores visibles | 10 221 | 4758 | 0 |
| Cola: ítems | 537 | 535 | 535 |
| Cola: segundos a mirar | 2623 | 2472 | 2472 |
| Decisiones | 548 | 540 | 540 |
| Decisiones por minuto | 22,5 | 22,1 | 22,1 |
| Precisión tras revisar | 0,970 | 0,970 | 0,984 |
| Cobertura tras revisar | 0,925 | 0,929 | 0,906 |
| E1 (≤ 0,5% y ninguna > 2%) | no | no | no |
| E2 (≥ 0,99) | no | no | no |

## 3. Por fuente

Errores silenciosos (cuadros y % de los de peleador de la fuente), ítems de la cola, ítems y
decisiones por minuto de video, precisión tras revisar.

| Fuente | min | Condición | Silenciosos | Ítems | Ítems/min | Dec/min | Prec. tras revisar |
|---|---|---|---|---|---|---|---|
| Sparring | 3,1 | base | 0 (0,000%) | 15 | 4,8 | 5,8 | 1,000 |
| | | corte | 9 (0,093%) | 13 | 4,2 | 4,2 | 0,998 |
| | | corte+estricta | 9 (0,093%) | 13 | 4,2 | 4,2 | 0,998 |
| 01-sparring | 2,8 | las tres | 2 (0,020%) | 35 | 12,5 | 12,5 | 0,958 |
| 02-sparring | 2,8 | base, corte | 0 (0,000%) | 145 | 51,0 | 51,0 | 0,977 |
| | | corte+estricta | 0 | 145 | 51,0 | 51,0 | 0,974 |
| 03-sparring | 2,9 | base, corte | 1248 (12,843%) | 152 | 52,1 | 52,8 | 0,861 |
| | | corte+estricta | 1248 (12,843%) | 152 | 52,1 | 52,8 | 0,984 |
| 04-sparring | 3,0 | base | 4 (0,038%) | 112 | 36,8 | 37,1 | 0,983 |
| | | corte, estricta | 3 (0,028%) | 112 | 36,8 | 37,1 | 0,983 |
| sparring-3-rounds | 9,7 | base | 21 (0,064%) | 78 | 8,1 | 8,6 | 0,993 |
| | | corte, estricta | 22 (0,067%) | 78 | 8,1 | 8,3 | 0,993 |

Cola por motivo (`base`; en `corte` los `color_contradice` y `posible_mezcla` pasan casi todos a
`conflicto`): `corto` 431 de 537 (80%), `color_insuficiente` 55, `conflicto` 33,
`color_contradice` 10, `posible_mezcla` 8. Sin 03-sparring, los silenciosos son 27 de 73 092
(0,037%) en `base` y 36 (0,049%) en `corte`: las otras cinco fuentes pasarían E1.

## 4. Lectura con la regla de §6

- E1: ninguna condición. El total queda en 1,54% contra 0,5% y 03-sparring en 12,8% contra 2%.
- E2: ninguna. La mejor es `corte+estricta`, con 0,984 contra 0,99.
- Elección: como ninguna pasa E1, se elige la de menos errores silenciosos, que es `base` (1275
  contra 1284).
- **Veredicto: no estable.** `BOXTWIN_CORTE_MIXTOS=0` y `BOXTWIN_IDENTIDAD_ESTRICTA=0`. La
  Fight-Card muestra la advertencia y la spec no se da por cumplida.

## 5. De dónde salen los números (diagnóstico, no pre-registrado, no cambia la lectura)

### 5.1 Errores silenciosos por forma

| Forma | `base` | `corte` |
|---|---|---|
| Tramo con el rol del otro peleador entero (intercambiado) | 385 | 364 |
| Tramo mixto, la parte del otro peleador | 854 | 885 |
| Un tercero (`ignore` en C1) con rol de peleador | 36 | 35 |

En 03-sparring son 364 intercambiados, 854 mixtos y 30 de tercero. Todos están en tramos largos
asignados sin sospecha. La separación de perfiles de color en esa fuente es 0,348, debajo del
piso de 0,55 (01-sparring tiene 0,384, pero casi no tiene errores). Sin perfiles que separen,
`color_contradice` no mira, el corte de E11 no corta y `posible_mezcla` no aparece. Tampoco hay
conflicto: el tramo equivocado no coexiste con otro del mismo rol en los cuadros medidos. Los
detectores de la capa 2 dependen del color o de la coexistencia, y en 03 no hay ninguno de los
dos. Por eso `corte` y `corte+estricta` no cambian nada en esa fuente.

### 5.2 Por qué E2 no llega a 0,99 aunque haya pocos silenciosos

Después de la ronda del usuario simulado (`base`), los cuadros mal son 1275 silenciosos que
nadie vio y 1108 cuadros de tercero adentro de ítems ya contestados. El usuario simulado parte
un ítem solo si la verdad tiene A y B con al menos 5 cuadros cada uno. Un ítem que es A y
`ignore` (el track pasa del peleador a un técnico o a otra persona del gimnasio) se contesta con
la mayoría, y los cuadros del tercero se quedan con el rol. En 01-sparring eso da 428 cuadros y
explica entera su precisión de 0,958 con apenas 2 silenciosos. Lo mismo pasa con 227 cuadros en
02, 171 en 04 y 190 en sparring-3-rounds. Una persona que ve el video probablemente partiría el
ítem, pero la regla de §6 define así al usuario simulado y así se lee.

### 5.3 Una segunda ronda de la cola

El producto arma la cola buscando conflictos en los tramos ya corregidos, así que después de
contestar aparecen ítems nuevos. La medición de §6 modela una sola ronda. Si se simula una
segunda:

- En 03-sparring, en `base` y `corte`, salen 2 ítems nuevos (3 decisiones) que contienen 1211 de
  los 1248 cuadros silenciosos. Las correcciones de la primera ronda crean conflictos con los
  tramos equivocados que hasta ahí no se veían. La precisión de 03 pasa de 0,861 a 0,987.
- En `corte+estricta` pasa lo mismo ya en la primera ronda: esos tramos entran en conflicto y se
  degradan solos. De ahí sale su 0,984.
- En total, con dos rondas: precisión 0,985 y cobertura 0,939 (`base`), 554 decisiones
  (22,7 por minuto). E2 sigue sin pasar, por los cuadros de tercero de §5.2.

Que los silenciosos de 03 aparezcan en la segunda ronda depende de que la persona conteste bien
la primera. En la medición contesta la verdad. Una persona que acepta un rol equivocado en la
primera ronda no genera ese conflicto.

### 5.4 El costo

22 decisiones por minuto de video, unas 540 para 24 minutos. La cola suma 2623 s de video para
mirar, contra 1463 s de material: más que el video entero, porque los ítems de tracks distintos
se superponen en el tiempo. El 80% de los ítems son `corto`: tracks sin asignar de más de 1 s que no
juntaron `min_recortes` recortes de guante. 02 y 03 piden unas 51 decisiones por minuto. Sparring es la más barata, con 4 a
6 por minuto, y sparring-3-rounds, el video más largo, pide 8.

## 6. Limitaciones

- **Desarrollo.** Es el material donde se diagnosticó el problema y se diseñó E11. Ningún número
  de acá confirma nada. La confirmación necesita material nuevo anotado a mano con la propuesta
  apagada (L3 de E11, hoy suspendido).
- **El usuario simulado no se equivoca.** Contesta cada ítem con la verdad de C1 y, en una
  sola ronda, no parte A contra tercero. Una persona real se equivoca (eso es C6), puede partir
  lo que el simulado no parte y ve la cola recalculada.
- **La verdad es la de C1.** Los cuadros de tercero se cuentan como error solo porque C1 los
  anotó como `ignore`.
- **n = 6 fuentes, y una sola decide el veredicto.** E1 cae por 03-sparring. Sin esa fuente, el
  total de silenciosos es 0,04% a 0,05%.
- **Cuadros no independientes**, igual que en E11: los porcentajes no llevan intervalo.
- El diagnóstico de §5 se escribió después de ver los números. Explica, pero no decide.

## 7. Entorno

`twinboxing_env` (Python 3.12.13). Los paquetes de la copia principal están instalados en modo
editable; el código del worktree se cargó con `PYTHONPATH`, sin reinstalar, para no cambiar lo que
corre A. El build de la web cortaba con un `SecurityFatalError` de Next 16.4.0, que tiene avisos
críticos (RCE en Image Optimization con SVG, RCE en `use cache` y DoS en Server Actions). Se
subieron `next` y `eslint-config-next` a 16.4.4 (`c62ec7a`). Es solo un parche: el código no
cambió y lint, tipos, tests y build pasan.
