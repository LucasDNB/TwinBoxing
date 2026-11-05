# E11 · Corte de tracks mixtos (solo desarrollo)

4-11-2026 · pre-registro: `docs/specs/2026-10-30-mejoras-post-e8/03-e11-tracks-mixtos.md` · corrida:
S10 v2 del 31-10, `9782993` · datos: `docs/experiments/datos/2026-11-04-e11-desarrollo.json` · código:
`scripts/mejoras/medir_e11.py`

## Veredicto

**E11 queda sin decidir.** Las preguntas P15a y P15b se deciden sobre material de confirmación, y ese
material no llegó: `sparr` tiene dos tracks asignados y `sparr01` nada. Por §4 del pre-registro, E11
se informa solo con desarrollo, rotulado como exploratorio, y la confirmación pasa al Capítulo 8.

Sobre las seis fuentes donde se diseñó, el corte ayuda. Eso no confirma nada: es el mismo material.

## Resultados, exploratorios

Seis fuentes de gimnasio, cachés de A, 82809 cuadros de peleador.

| Condición | Precisión por cuadro | Cobertura por cuadro | C1 |
|---|---|---|---|
| Sin corte | 0,850 | 0,788 | 85/105 por track |
| Con corte | 0,923 | 0,872 | 111/119 por segmento |

Las unidades de C1 no son iguales: no se comparan directamente. La cifra de 85/105 reproduce el C1 del
20-10, como pedía la auditoría de S10.

| Fuente | Sin corte, precisión y cobertura | Con corte | Cortes |
|---|---|---|---|
| Sparring | 0,887 y 0,504 | 0,960 y 0,580 | 7 |
| 01-sparring | 0,980 y 0,897 | igual | 0 |
| 02-sparring | 0,927 y 0,955 | igual | 0 |
| 03-sparring | 0,728 y 0,709 | igual | 0 |
| 04-sparring | 0,855 y 0,786 | 0,966 y 0,888 | 1 |
| sparring-3-rounds | 0,819 y 0,813 | 0,942 y 0,969 | 10 |

Ninguna fuente empeora. 03-sparring no se mueve: ahí el color no separa, y el corte depende del color.
P15d: 16 tracks partidos, 10 de ellos mixtos según la verdad, de 18 mixtos.

## Limitaciones

1. **Diseñado y medido sobre lo mismo.** La regla se corrigió el 31-10 después de ver que invertía
   Sparring (decisión 16). Estos números son optimistas por construcción.
2. **Los cuadros no son independientes.** Por eso no cito los intervalos de Wilson del JSON.
3. **Lo que no separa, por §7:** árbitro mezclado con peleador y clinch largo sin muestras de color.
4. **El corte queda apagado** (`BOXTWIN_CORTE_MIXTOS`), como decidió IE6.
5. **Con BoostTrack el corte baja la precisión** de 0,977 a 0,960 (E13). El efecto depende del tracker.
