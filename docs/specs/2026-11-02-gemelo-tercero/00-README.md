# G3 · Gemelo digital de un tercero en A

2-11-2026 · spec SDD · nada lanzado todavía · registro en `01-pasos.md`

Pedido de Lucas (2-11): "generemos un gemelo de tercero", leído como el gemelo digital (perfil
táctico) de un boxeador que no es él. Decisiones del 2-11, de Lucas:

| Id | Decisión |
|---|---|
| D1 | El tercero sale de Pacquiao–Margarito, la única transmisión del dataset: ya está en la estación, anotada, con la identidad al 95,7% y números en la tesis. Un solo combate, 12 rounds |
| D2 | El gemelo vive en **A** (producción), junto a las sesiones de Lucas y de Daniel. Así el Rival Fantasma tiene por fin un rival que no es el de siempre |
| D3 | Seudónimos, no nombres reales: `Tercero Z` (el zurdo) y `Tercero O` (el ortodoxo) por defecto, cambiables en la carga |

Contexto: el mismo 2-11 Lucas suspendió L3 y S16 de las mejoras post-E8. BoxTwin queda donde
está y se relanza como un producto de menor alcance. Esto no es un experimento ni mide nada nuevo
del sistema: es usar el producto tal como está sobre material de terceros, para mostrar qué arma.

## 1. El problema de la entrada

La única entrada de video de A es `POST /videos`, que pide que quien sube declare el
consentimiento de las personas filmadas (R-09). Para una transmisión ese consentimiento no
existe: marcar la casilla dejaría en la base una declaración falsa con el email de Lucas. Abrir
la ruta a otro origen es cambiar A, que se congela el 3-11.

Por eso la carga la hace un script de operador, `boxtwin-api/scripts/cargar_tercero.py`, que crea
la sesión directo en la base de A igual que la ruta (mismos campos, video en
`sesiones/<id>/videos/`, `procesar` encolado en el mismo commit) y escribe el origen del material
donde iría el consentimiento:

```
consentimiento_por = "material de terceros: transmision publica, uso academico (RE-06);
                      sin consentimiento de las personas filmadas"
```

No toca el código de A: corre con el paquete instalado en `twinboxing_env`, que es el de A
(`1058d5e`). Entre `1058d5e` y `main` no cambian `modelos.py`, `acceso.py`, `cola.py` ni
`config.py`, así que lo que prueban las pruebas de `main` es lo que corre en A.

## 2. Qué se carga

| | Valor | Por qué |
|---|---|---|
| Contexto | `combate` | Entra al perfil y al Rival (`solo_pelea` toma sparring y combate) |
| Rounds | 12, de 180 s | El combate |
| Descanso | 0 s, sobre el video de rounds netos que arma `cortar_rounds.py` | El video de Lucas trae los descansos (decisión 4) |
| Peleador A | `Tercero Z`, guardia zurda | El tipo v2 saca jab y cross del brazo y la guardia; sin guardia supone ortodoxa y el jab del zurdo saldría cross |
| Peleador B | `Tercero O`, guardia ortodoxa | |
| Nota en los dos | "Gemelo de un tercero (2-11-2026): material de transmisión, con seudónimo. No es un atleta de la cuenta" | Que se lea en la ficha sin abrir la base |
| Cuenta | la de Lucas (titular) | Nunca en las cuentas de prueba (`BOXTWIN_DEMO`) |

La siembra la hace Lucas en la web como en cualquier sesión, con A = el zurdo. Los peleadores ya
quedan asignados en la carga; si la siembra sale al revés, se corrige con `PUT
/sesiones/{id}/boxeadores` sin reprocesar.

### 2.1 El video trae los descansos

El video que pasó Lucas (`~/Proyectos/twinboxing-finales/pacman3x1.mp4`) trae los descansos. A
arma los rounds contando desde el segundo 0 con round y descanso fijos, y una transmisión no
cumple eso: hay presentación antes del primer round, descansos de largo variable y, sobre todo,
repeticiones en cámara lenta y desde otro ángulo. El detector ve golpes en las repeticiones y A
suma los golpes del descanso al round que acaba de terminar. Cargado tal cual, el volumen sale
inflado con golpes repetidos y, con un minuto de desfase al principio, los doce rounds quedan
corridos.

Por eso Lucas marca dónde empieza cada round (doce tiempos, mm:ss) y
`boxtwin-api/scripts/cortar_rounds.py` corta los rounds de 180 s, reencodeando para que el corte
caiga en el cuadro exacto, y los une en `pacman-netos.mp4`, con un JSON al lado con los inicios y
el sha256 del original. Se carga ese, con descanso 0. El original no se toca.

## 3. Qué produce

1. La Fight-Card del combate: golpes detectados por brazo y por round, tipo como estimación,
   altura por la regla de C8, línea de tiempo enlazada al video.
2. El perfil de `Tercero Z`: volumen y ritmo, mezcla de golpes, distribución mano × altura y
   mapa de exposición, todo de una sola sesión (la evolución tiene un punto).
3. El Rival Fantasma `Lucas` contra `Tercero Z`. Con DC-10, la exposición de cada uno sale sin
   las sesiones contra el otro: la de Lucas son sus sesiones con Daniel y la de Tercero Z es el
   combate. Las reglas disparan solo con 50 golpes con altura o más en cada mitad.

## 4. Criterios de aceptación (escritos antes de cargar)

| Id | Criterio | Cómo se verifica |
|---|---|---|
| CT1 | A queda igual | `GET /salud` con commit `1058d5e` antes y después; en la base, una sesión y dos peleadores más, nada más (conteos de S1 contra S5) |
| CT2 | La sesión llega a Fight-Card | `procesar`, `completar` y `clasificar` listos; `fightcard.json` en el directorio de la sesión |
| CT3 | El origen queda auditable | `consentimiento_por` de la sesión es el texto de §1; los dos peleadores llevan la nota |
| CT4 | La guardia del zurdo se usa | en la Fight-Card, el jab de `Tercero Z` sale del brazo derecho |
| CT5 | El Rival responde | `GET /rival?a=<Lucas>&b=<Tercero Z>` da 200 con lecturas o con lo que falta, nunca un error |
| CT6 | Nada del tercero en las cuentas de prueba | la sesión y los peleadores tienen el `cuenta_id` de Lucas |

## 5. Lo que hay que declarar

- **El seudónimo cubre los nombres, no las caras.** El video se ve en la línea de tiempo y la
  pelea es reconocible. En la tesis se dice de qué combate sale.
- **A está expuesta por Funnel.** El material queda en la cuenta de Lucas, con el registro por
  invitación; nadie fuera de la cuenta lo ve. RE-06 permite usar material de transmisión solo
  para investigación: mostrarlo en la presentación del 17-11 entra, publicarlo o abrirlo a
  cuentas de prueba no.
- **Está en distribución.** El detector (`detector_7fuentes.ens.pt`) y el clasificador E4 se
  entrenaron con las siete fuentes, Pacquiao incluido. La Fight-Card del tercero muestra lo que
  el sistema arma, no cuánto generaliza. Si se compara con la anotación, el número no es de
  fuente no vista y no se presenta como tal.
- **Una sesión no es un perfil.** La evolución tiene un solo punto; la comparación entre rounds
  sigue rotulada "sin validar" (C2).
- **El Rival cruza un amateur de gimnasio con un profesional, y el material es distinto**
  (transmisión con cortes de plano contra cámara fija). Las lecturas describen golpes dirigidos,
  no predicen nada (R-23), y acá además mezclan condiciones de captura.

## 6. Para la tesis

- Cap 7: un párrafo en la presentación del producto, "el producto sobre material de terceros",
  con la Fight-Card y el perfil como figura y las limitaciones de §5. No es un resultado medido.
- Cap 6: R-09 dice qué se hizo con material sin consentimiento (script de operador, origen
  auditable, sin ruta pública que lo acepte).
- Presentación del 17-11: el Rival `Lucas` contra `Tercero Z` como demostración, con la leyenda
  de §5.

## 7. Lo que esta spec no hace

- No cambia el código de A ni sus modelos. No agrega una ruta de subida de terceros: si el
  producto relanzado la necesita, es una función nueva con su propia spec.
- No carga más peleas del mismo boxeador. Si después se quiere evolución, cada pelea es una
  carga más con el mismo script y los mismos seudónimos (los peleadores se reusan por nombre).
- No mide nada contra umbral.
