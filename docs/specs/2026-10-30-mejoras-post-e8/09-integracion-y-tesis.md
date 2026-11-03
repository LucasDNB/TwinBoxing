# Integración a A y cambios de tesis

## 1. Cómo un resultado positivo llega a A

Nada de esta campaña llega a A antes del 17-11 (congelamiento del 3-11 y entrega final). Después:

1. PR propio por mejora, contra `main`, con la mejora detrás de una bandera apagada por defecto:

   | Mejora | Bandera | Qué cambia en A |
   |---|---|---|
   | E8b (si se migra a YOLO26) | `BOXTWIN_MODELO_POSE` + umbrales en el `meta.json` del modelo | pose, umbrales, detector y E4 reentrenados |
   | E10 | `BOXTWIN_RELLENO_KP=1` | detector y E4 reentrenados con relleno |
   | E11 | `BOXTWIN_CORTE_MIXTOS=1` | solo la identidad |
   | E12 | `BOXTWIN_GUANTES_DETECTOR=1` | extracción de guantes por cuadro, detector reentrenado |
   | E9, E13, E14 | ídem, uno por experimento | |

2. **Los umbrales de puntaje viajan con el modelo**, no en el código: el `meta.json` de cada
   detector y clasificador suma `umbrales_puntaje` (los cinco de E8b §2) y la inferencia los lee
   de ahí. Un modelo sin ese campo usa 0,3, que es lo de hoy. Así no se puede usar un detector
   entrenado con un umbral y servido con otro.
3. Antes de encender en A: C1 y C5 medidos otra vez con sus pre-registros, sobre material que
   no sea de desarrollo, y huella de A antes y después.
4. Encender, con vuelta atrás documentada (`despliegue/README.md`).

## 2. Cambios de tesis por resultado

Cada experimento escribe `docs/tesis/<fecha>-<exp>-cambios.md` en formato ANTES / DESPUÉS / QUÉ
CAMBIÓ, como E8. Lo que puede cambiar:

| Capítulo | Qué | Depende de |
|---|---|---|
| 4, Tabla 4.5 | Fila de YOLO26: "evaluado con el umbral de v8 (E8) y calibrado (E8b)" con los dos números; fila de RTMO si E9 corrió | E8b, E9 |
| 4, §3.1 | La edición pendiente de la licencia (memoria del 29-10) queda igual; E9 suma que existe una alternativa medida o en curso | E9 |
| 5 | Nada antes del 17-11. Si E9 y E13 dan positivo y las tres piezas de `05-e9-rtmo.md` §8 están cubiertas, un escenario sin licencia del estimador; con una sola pieza, no | E9, E13 |
| 6 | R-02 (identidad): baja de probabilidad solo si E11 mejora **y** se integra. R-01 (clasificador): sin cambios, la sub-hipótesis 1 ya cerró | E11 |
| 7 | Una subsección "Mejoras posteriores a E8", con un párrafo por experimento cerrado: qué se midió, el número, la regla y si pasó; los negativos igual | E8b, E10, E11, E12 |
| 8 | Líneas futuras con pre-registro escrito: E9, E13, E14, E15, y las de fase 1 que no hayan cerrado. Cada una con el link a su archivo de este spec | todos |
| Declaración de uso de IA | Suma esta campaña: specs y código por Claude Code en modo agente, con pre-registros y reglas fijados antes de medir, y las decisiones de Lucas anotadas | todos |

**Reglas de redacción para el Cap 7** (las de E8): no escribir "YOLO26 estima peor la pose";
separar desarrollo de confirmación en E11; un número sin documento en `docs/experiments/` no
entra.
