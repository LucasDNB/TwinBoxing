# IE7 · Verificador de apariencia (guante, remera, pantalón) y usuario simulado v2

3-11-2026. Pre-registro: `docs/specs/2026-11-02-identidad-estable/02-ie7-apariencia.md`
(commit `92e0105`, antes del código). Código medido: `ff49041`. Script:
`scripts/identidad/medir_identidad_apariencia.py`. JSON:
`docs/experiments/datos/2026-11-03-ie7-apariencia.json`. Diagnóstico de señales:
`docs/experiments/datos/2026-11-03-ie7-senales.json`.

**Material de desarrollo, y contaminado: el diseño se eligió después de ver estos números en
las mismas seis fuentes (se declara en el pre-registro). No confirma nada.**

**En una línea: con la medición pre-registrada de IE7 las dos condiciones pasan E1 y E2, y la
regla elige `base`. El producto queda estable en identidad sobre este material, con
`BOXTWIN_IDENTIDAD_APARIENCIA` apagada. Lo que cambió el veredicto respecto de IE6 no es el
verificador: es medir la cola en rondas, como funciona el producto, con un usuario que puede
separar a un peleador de un tercero. El verificador encuentra lo mismo en la primera ronda que
la cola sin verificador encuentra en la segunda.**

## 1. Señales (diagnóstico previo al pre-registro)

Acierto por muestra contra la verdad de C1, muestras aisladas (en contacto, entre paréntesis):

| Fuente | Guante | Remera | Pantalón | Altura |
|---|---|---|---|---|
| Sparring | 0,868 | 0,845 (0,743) | 0,600 (0,532) | 0,665 |
| 01-sparring | 0,742 | 0,973 (0,954) | 0,891 (0,848) | 0,720 |
| 02-sparring | 0,957 | 0,924 (0,889) | 0,545 (0,715) | 0,687 |
| 03-sparring | 0,860 | 0,777 (0,715) | 0,838 (0,795) | 0,520 |
| 04-sparring | 0,809 | 0,645 (0,591) | 0,655 (0,509) | 0,894 |
| sparring-3-rounds | 0,894 | 0,583 (0,545) | 0,731 (0,753) | 0,697 |

Ninguna señal sirve en todas las fuentes, y la ropa da señal también en contacto, que es
donde el tracker cambia de persona. En 03 el guante acierta 0,860 por muestra, pero el piso de
separación (0,55, sobre la distancia entre perfiles) lo deja afuera: los dos guantes tienen el
mismo tono y se distinguen por saturación y brillo. La altura separa solo en 04 y quedó afuera.

Pesos que el producto calcula con las semillas (logit del acierto por bloques de tiempo):

| Fuente | Guante | Remera | Pantalón |
|---|---|---|---|
| Sparring | 0,60 | 0,51 | 0,19 |
| 01-sparring | 1,11 | 2,88 | 1,76 |
| 02-sparring | 2,64 | 1,54 | 1,55 |
| 03-sparring | 1,75 | 1,57 | 1,23 |
| 04-sparring | 1,64 | 0,85 | 0,38 |
| sparring-3-rounds | 2,76 | 0,51 | 0,00 |

## 2. Auditoría

Con el simulado de IE6 y una sola ronda, `base` da 1275 silenciosos y 0,9698 después de
revisar: lo mismo que IE6. Las muestras de ropa regeneradas con las funciones del producto son
idénticas, byte a byte, a las del diagnóstico. Pruebas antes de medir: anotador 819, API 346,
`scripts/identidad` 17, `scripts/mejoras` 63, web: lint, tipos, 261 pruebas y build.

## 3. Resultados (medición pre-registrada: simulado v2 y hasta 3 rondas)

| | `base` | `apariencia` |
|---|---|---|
| Errores silenciosos | 43 (0,052%) | 43 (0,052%) |
| Peor fuente | 0,381% (03) | 0,381% (03) |
| Precisión después de revisar | 0,9993 | 0,9993 |
| Cobertura después de revisar | 0,941 | 0,941 |
| Ítems vistos | 540 | 562 |
| Segundos a mirar | 2684 | 3232 |
| Decisiones por minuto | 23,2 | 24,1 |
| E1 / E2 | sí / sí | sí / sí |

Por fuente (`base` → `apariencia`):

| Fuente | min | Silenciosos | Rondas | Decisiones/min | Precisión después |
|---|---|---|---|---|---|
| Sparring | 3,1 | 0 → 0 | 1 → 1 | 5,8 → 6,4 | 1,000 → 1,000 |
| 01-sparring | 2,8 | 2 → 2 | 1 → 1 | 13,5 → 13,5 | 1,000 → 1,000 |
| 02-sparring | 2,8 | 0 → 0 | 1 → 1 | 52,4 → 53,4 | 1,000 → 1,000 |
| 03-sparring | 2,9 | 37 → 37 | 2 → 1 | 54,2 → 54,5 | 0,996 → 0,996 |
| 04-sparring | 3,0 | 4 → 4 | 1 → 1 | 37,8 → 39,4 | 0,999 → 0,999 |
| sparring-3-rounds | 9,7 | 0 → 0 | 2 → 2 | 9,0 → 10,1 | 1,000 → 1,000 |

Motivos nuevos de `apariencia`: 6 `apariencia_contradice` y 18 `apariencia_mezcla`, 11 de ellos
en sparring-3-rounds.

## 4. Lectura con la regla

- E1 y E2 pasan en las dos condiciones.
- Se elige la de menos decisiones por minuto: `base` (23,2 contra 24,1, más de 0,1 de diferencia).
- **Estado: estable** sobre material de desarrollo. `BOXTWIN_IDENTIDAD_APARIENCIA=0`.
- No cambia nada de lo que decidió IE6: `BOXTWIN_CORTE_MIXTOS=0` y
  `BOXTWIN_IDENTIDAD_ESTRICTA=0`.

**Decisión posterior (3-11, Lucas, no la regla):** después de leer este informe, el verificador
pasa a venir encendido por defecto (`BOXTWIN_IDENTIDAD_APARIENCIA` sin definir vale 1; se apaga
con 0). La razón es la de §5: es la red si la persona se equivoca en la primera ronda, a cambio de
0,9 decisiones más por minuto. La lectura de la regla no cambia.

## 5. Qué dice esto (no cambia la lectura)

**Con el simulado de IE6 (una ronda, contesta la mayoría):**

| | `base` | `apariencia` |
|---|---|---|
| Silenciosos | 1275 (1,540%) | 64 (0,077%) |
| 03-sparring | 1248 | 37 |
| Precisión después | 0,970 | 0,985 |

El verificador hace lo que dice el diseño: en 03 encuentra en la primera ronda, con tres ítems,
los tramos que el color del guante no podía mirar. Pero con rondas no agrega nada. Cuando la
persona corrige un tramo en la primera ronda, el tramo equivocado que coexistía con él queda en
conflicto, y la segunda ronda lo muestra. En 03 eso cuesta 2 ítems más y una ronda más.

**Esa redundancia depende de que la persona conteste bien la primera ronda.** Sin verificador,
los errores de 03 aparecen recién en la segunda ronda, y solo porque las correcciones de la
primera fueron correctas. Con verificador aparecen solos en la primera. Si la persona se
equivoca, el verificador sigue mostrándolos y la segunda ronda no. La regla pre-registrada no
pesa eso: elige por costo, y el verificador cuesta 0,9 decisiones más por minuto y 548 s más de
video para mirar. Medir esa robustez necesita personas reales (C6).

**IE6 decía "no estable" por cómo estaba definido el usuario simulado**, no por el producto. La
web ya permitía separar a un peleador de un tercero y recalculaba la cola. El simulado de IE6
no hacía ninguna de las dos cosas.

**El costo es lo que queda.** 23 decisiones por minuto, el 80% de la cola son ítems `corto`, y
02 y 03 piden más de 50 por minuto. Que la identidad sea estable depende de que la persona
conteste unas 70 preguntas por cada 3 minutos de sparring.

## 6. Limitaciones

- **Desarrollo y contaminado.** Las señales, los pesos y los dos cambios al simulado se
  eligieron mirando estas seis fuentes.
- **El usuario simulado no se equivoca y siempre vuelve a la cola.** Una persona real puede
  hacer las dos cosas mal. Eso es C6.
- La verdad es la de C1.
- La ropa de la medición sale del diagnóstico. Son las mismas funciones y los mismos cuadros
  que `analizar`, pero el diagnóstico mide también las cajas de menos de 16 por 8 píxeles, que
  `analizar` saltea (nunca son peleadores).
- Los cuadros no son independientes: los porcentajes van sin intervalo.
