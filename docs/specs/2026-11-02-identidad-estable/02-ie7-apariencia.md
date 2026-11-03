# IE7 · Verificador de apariencia: guante, remera y pantalón (pre-registro)

3-11-2026. Sigue a IE6 (`docs/experiments/2026-11-03-identidad-estable.md`), que dio **no
estable**. El commit de este archivo es el pre-registro: va antes del código y de la medición.

## 1. De dónde sale

- IE6: 1275 cuadros con rol equivocado que no llegan a la cola (1,54%). 1248 son de
  03-sparring, donde los perfiles de guante separan 0,348, debajo del piso de 0,55. Con ese
  piso, `color_contradice`, `posible_mezcla` y el corte no miran nada.
- Propuesta de Lucas (3-11): seguir a cada peleador con un perfil de guantes, pantalón, remera y
  altura. Dato de dominio de Lucas: en sparring de gimnasio la ropa de los dos suele ser
  distinta y no cambia entre rounds.
- Diagnóstico de señales (`scripts/identidad/diagnostico_senales_apariencia.py`, 3-11, sobre
  las mismas seis fuentes). Acierto por muestra contra la verdad de C1, en muestras aisladas:

  | Fuente | Guante | Remera | Pantalón | Remera + pantalón | Altura (coexisten) |
  |---|---|---|---|---|---|
  | Sparring | 0,868 | 0,845 | 0,600 | 0,823 | 0,665 |
  | 01-sparring | 0,742 | 0,973 | 0,891 | 0,915 | 0,720 |
  | 02-sparring | 0,957 | 0,924 | 0,545 | 0,780 | 0,687 |
  | 03-sparring | 0,860 | 0,777 | 0,838 | 0,919 | 0,520 |
  | 04-sparring | 0,809 | 0,645 | 0,655 | 0,654 | 0,894 |
  | sparring-3-rounds | 0,894 | 0,583 | 0,731 | 0,578 | 0,697 |

  En contacto (cajas superpuestas, donde el tracker cambia de persona) el guante no se mide.
  Remera y pantalón sí: en 03, 0,879 juntos y en 01 la remera da 0,954.
- La confiabilidad de cada señal estimada **solo con las semillas** (perfiles con tres de cuatro
  bloques de tiempo, acierto en el cuarto) sigue al acierto real en guante y remera; en
  pantalón se equivoca en 02 y en sparring-3-rounds.

**Esto contamina el diseño, y se declara.** Las señales, el método para pesarlas y las reglas
de abajo se eligieron después de ver estos números sobre el mismo material en el que se mide.
IE7 es de **desarrollo**, igual que IE6: muestra si el mecanismo hace lo que dice, pero no
confirma nada. Para confirmar hace falta material nuevo (L3 de E11).

## 2. Diseño

**La asignación no cambia.** El verificador solo agrega motivos de sospecha a la cola de
revisión (capa 2 de la spec), detrás de `BOXTWIN_IDENTIDAD_APARIENCIA` (apagada por defecto) y
`completar --apariencia`.

1. **Evidencia.** `analizar` mide además, por detección y en los mismos cuadros que el guante,
   el color de la remera (caja de hombros a caderas) y el del pantalón (de las caderas hasta 0,4
   del camino a las rodillas), con `color_de_parche`. Los keypoints tienen que tener puntaje
   ≥ 0,3. Se guarda si la detección estaba aislada, con el mismo criterio que usa el guante. Es
   un campo nuevo, `ropa`, en la evidencia. Si la evidencia no lo trae, el verificador no actúa.
2. **Modelo por video, con las semillas.** Por cada señal c (guante, remera, pantalón):
   - Los perfiles A y B salen de las muestras aisladas de cada semilla (`perfil_de`).
   - La confiabilidad r_c sale de cuatro bloques de tiempo por semilla: perfiles con tres
     bloques, acierto en el cuarto y promedio de los cuatro.
   - El peso es w_c = logit(r_c), con r_c recortado a [0,5; 0,99]. Una señal que no separa pesa
     cero.
   - La altura queda afuera: separa en una sola de las seis fuentes (04) y en 03 da 0,520.
3. **Señal por cuadro.** Cada señal da el signo de d(c, perfil B) − d(c, perfil A), igual que
   en E11. Con dos guantes en el cuadro, vale el signo de la suma. La señal del cuadro es
   v = Σ w_c · signo_c / Σ w_c (de las señales presentes), entre −1 y 1. Se usan todas las
   muestras de ropa (aisladas o en contacto) y las de guante, que ya vienen aisladas.
4. **Dos motivos nuevos**, para tramos automáticos con A o B, cuando los de color existentes no
   se dispararon. Son las mismas reglas que `color_contradice` y `posible_mezcla`, con los
   mismos parámetros, pero sobre v. Ningún umbral nuevo fuera de los de los puntos 1 y 2.
   - `apariencia_contradice` (prioridad 2): al menos `min_guantes_voto` cuadros con v ≠ 0 y la
     mayoría del lado del otro rol.
   - `apariencia_mezcla` (prioridad 3): al menos 2 × `min_guantes_voto` cuadros y
     `_mejor_corte(v, min_guantes_voto, margen_orientacion / 2)` encuentra un corte.
5. La web aprende los textos de los dos motivos y la Fight-Card informa `identidad_apariencia`.

## 3. Medición

`scripts/identidad/medir_identidad_apariencia.py`, offline y sin GPU. Usa la evidencia de C1 más
la ropa medida con las mismas funciones del producto, y la misma siembra que C1, E11 e IE6.

**Condiciones.** `base` (IE6 base) y `apariencia` (base con el verificador). Sin corte ni modo
estricto: IE6 mostró que no bajan los errores silenciosos.

**Dos cambios al usuario simulado**, pre-registrados acá. Se informan también las medidas con el
simulado de IE6, para poder comparar.

- **Parte también a un peleador de un tercero.** Un ítem se parte cuando la verdad tiene al menos
  dos roles entre A, B y ninguno (`ignore` o sin anotar), con al menos 5 cuadros cada uno.
  Cuesta 2 decisiones, se parta en dos o en más pedazos, como en IE6. La web ya lo permite:
  "Acá cambia de persona" y después "Ignorar" en una mitad.
- **Rondas.** Después de contestar, la cola se recalcula con lo corregido, como hace el producto.
  Se repite hasta que no quedan ítems pendientes, o hasta 3 rondas. Las decisiones se suman.

**Medidas.** Son las de IE6, por cuadro y en los cuadros donde el track tiene detección:

- Error silencioso: un cuadro con A o B equivocado en la propuesta automática que no cae
  adentro de ningún ítem de ninguna ronda.
- Precisión y cobertura después de la última ronda.
- Decisiones por minuto de video.
- Ítems por motivo.

**Regla** (los mismos números de IE6 §6):

- E1: silenciosos ≤ 0,5% del total y ninguna fuente por encima del 2%.
- E2: precisión después de revisar ≥ 0,99, con el simulado de este documento.
- E3: decisiones por minuto, sin umbral.
- Entre las condiciones que pasan E1 y E2, se elige la de menos decisiones por minuto. Si
  empatan a 0,1, la que tenga menos banderas.
- Si ninguna pasa las dos: la que tenga menos silenciosos, y el producto sigue **no estable**.
  Si alguna pasa E1 pero no E2, se informa así ("E1 sí, E2 no") y tampoco se declara estable.

**Auditoría antes de leer.** Con el simulado de IE6 y una sola ronda, `base` tiene que dar los
números de IE6: 1275 silenciosos y precisión después de revisar 0,970. Si no los da, no se lee
nada.

## 4. Tareas

| Id | Qué | Prueba |
|---|---|---|
| IE7a | `boxtwin/core/apariencia.py`: zonas, color de ropa, modelo con las semillas, señal v, los dos motivos; `ropa` en la evidencia (analizar, guardar y cargar, compatible con la vieja) | tests sintéticos |
| IE7b | `revision.py` y `completar --apariencia`; API `BOXTWIN_IDENTIDAD_APARIENCIA` hasta el worker; web: textos y esquema | tests del anotador, de la API y vitest; sin bandera, todo igual que antes |
| IE7c | medición con el simulado v2 y rondas | tests con verdad sintética; auditoría contra IE6 |
| IE7d | correr y documentar | `docs/experiments/<fecha>-ie7-apariencia.md` |

Nada de esto entra a A antes del 17-11.
